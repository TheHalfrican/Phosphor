"""
Phosphor — build-time prompt embedding bake.  DEV ONLY.

Runs once on a developer machine that has the full UMT5-XXL encoder. Produces
assets/embeddings.safetensors, which ships with the app in place of a 11.4 GB text
encoder. See CLAUDE.md §4.

THE CONTRACT (get this wrong and you get silent quality loss, not a crash)
-------------------------------------------------------------------------
diffusers' _get_t5_prompt_embeds does, in order:

    tokenize(padding="max_length", max_length=512, truncation=True, add_special_tokens=True)
    embeds = text_encoder(ids, mask).last_hidden_state
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]        # trim to real token count
    embeds = stack([cat([u, zeros(max_len - len(u), dim)]) for u in embeds])   # zero-pad back

The trailing pad is pure zeros, so we store the TRIMMED tensor and re-pad at load time.
That is what keeps this file ~4 MB instead of ~38 MB: padding every prompt to 512 tokens
would cost 512 x 4096 x 2 bytes = 4.19 MB *each*.

Prompt text is also run through diffusers' own prompt_clean() — imported, not
reimplemented, so it cannot drift from upstream.
"""

import json
import os
import sys
import urllib.request

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.pipelines.wan.pipeline_wan import prompt_clean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "wan-ti2v-5b-diffusers")
PRESETS = os.path.join(ROOT, "assets", "presets.json")
OUT = os.path.join(ROOT, "assets", "embeddings.safetensors")

MAX_SEQ_LEN = 512  # diffusers WanImageToVideoPipeline.__call__ default
SAVE_DTYPE = torch.bfloat16  # matches transformer compute dtype; no extra conversion loss

# Load the encoder in bf16, NOT fp32. This is fidelity, not a shortcut:
# the stock pipeline is built with torch_dtype=bfloat16, so its text_encoder is bf16 and
# _get_t5_prompt_embeds resolves `dtype = dtype or self.text_encoder.dtype` to bf16 too.
# Baking in fp32 would produce embeddings the live pipeline never would.
#
# It also halves the model from 11.4 GB to 5.7 GB. At fp32 the weights plus attention
# activations overflow 24 GB, and Windows WDDM silently pages VRAM to host RAM rather
# than raising OOM — which presents as 100% GPU utilisation at near-zero throughput.
LOAD_DTYPE = torch.bfloat16

NEG_URL = "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/shared_config.py"
NEG_CACHE = os.path.join(ROOT, "assets", ".negative_prompt.txt")


def fetch_negative_prompt() -> str:
    """Pull Wan's official negative prompt verbatim from upstream.

    Deliberately NOT transcribed into this file — it is a long CJK string and a silent
    copy/encoding error would degrade every generation in a way that is near-impossible
    to trace back. Cached locally so the bake stays reproducible offline.
    """
    try:
        with urllib.request.urlopen(NEG_URL, timeout=30) as r:
            src = r.read().decode("utf-8")
        for line in src.splitlines():
            if "sample_neg_prompt" in line and "=" in line:
                val = line.split("=", 1)[1].strip()
                if val and val[0] in "\"'":
                    neg = val[1:].rsplit(val[0], 1)[0]
                    with open(NEG_CACHE, "w", encoding="utf-8") as f:
                        f.write(neg)
                    print(f"  negative prompt : fetched from upstream ({len(neg)} chars)")
                    return neg
        raise RuntimeError("sample_neg_prompt not found in upstream config")
    except Exception as e:
        if os.path.exists(NEG_CACHE):
            with open(NEG_CACHE, encoding="utf-8") as f:
                neg = f.read()
            print(f"  negative prompt : upstream fetch failed ({e}); using cache")
            return neg
        raise SystemExit(
            f"\n  Could not fetch the official negative prompt and no cache exists.\n"
            f"  {e}\n  Refusing to invent one — see CLAUDE.md §4.\n"
        )


@torch.no_grad()
def encode(tokenizer, encoder, prompts, device):
    """Replicates diffusers _get_t5_prompt_embeds, returning TRIMMED (unpadded) tensors.

    Encodes ONE prompt at a time. UMT5-XXL has 64 attention heads, so a batch of B
    sequences at 512 tokens materialises B x 64 x 512 x 512 attention scores per layer —
    batching all nine presets at once is what pushed this over 24 GB. Batch size 1 is
    also bit-identical to what the pipeline does at inference, where B is 1 anyway.
    """
    embeds, lens = [], []
    for p in prompts:
        ti = tokenizer(
            [prompt_clean(p)],
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = ti.input_ids.to(device), ti.attention_mask.to(device)
        n = int(mask.gt(0).sum())

        out = encoder(ids, mask).last_hidden_state  # [1, 512, 4096]
        embeds.append(out[0, :n].to(SAVE_DTYPE).cpu())
        lens.append(n)
        del out, ids, mask
        if device == "cuda":
            torch.cuda.empty_cache()
    return embeds, lens


def main():
    if not os.path.isdir(os.path.join(MODEL_DIR, "text_encoder")):
        raise SystemExit(
            f"\n  Text encoder not found at {MODEL_DIR}\\text_encoder\n"
            f"  Run tools/download_models.py first (it is the 11.4 GB stage).\n"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n  Phosphor — embedding bake")
    print(f"  device          : {device}")

    with open(PRESETS, encoding="utf-8") as f:
        cfg = json.load(f)
    presets = cfg["presets"]
    negative = fetch_negative_prompt()

    print(f"  presets         : {len(presets)}")
    print(f"  max_seq_len     : {MAX_SEQ_LEN}")
    print(f"  load dtype      : {LOAD_DTYPE}  (matches stock pipeline)")
    print(f"  save dtype      : {SAVE_DTYPE}")
    print("\n  Loading UMT5-XXL as bf16 (~5.7 GB)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        MODEL_DIR, subfolder="text_encoder", dtype=LOAD_DTYPE
    ).to(device).eval()

    if device == "cuda":
        print(f"  encoder on GPU  : {torch.cuda.memory_allocated()/1024**3:.2f} GiB", flush=True)

    prompts = [p["prompt"] for p in presets] + [negative]
    keys = [p["id"] for p in presets] + ["__negative__"]

    print("  Encoding...\n", flush=True)
    embeds, seq_lens = encode(tokenizer, encoder, prompts, device)

    tensors, total = {}, 0
    for key, emb, n in zip(keys, embeds, seq_lens):
        tensors[key] = emb.contiguous()
        total += emb.numel() * emb.element_size()
        print(f"    {key:<20} {int(n):>4} tokens   {tuple(emb.shape)}   "
              f"{emb.numel() * emb.element_size() / 1024:>7.1f} KB")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        tensors,
        OUT,
        metadata={
            "max_sequence_length": str(MAX_SEQ_LEN),
            "dtype": str(SAVE_DTYPE).replace("torch.", ""),
            "hidden_dim": str(embeds[0].shape[-1]),
            "storage": "trimmed",  # consumer MUST zero-pad to max_sequence_length
            "negative_key": "__negative__",
            "preset_count": str(len(presets)),
        },
    )

    on_disk = os.path.getsize(OUT)
    print(f"\n  Wrote {OUT}")
    print(f"  tensor bytes : {total/1024/1024:.2f} MB")
    print(f"  file size    : {on_disk/1024/1024:.2f} MB")
    padded = len(keys) * MAX_SEQ_LEN * embeds[0].shape[-1] * 2
    print(f"  (padded would have been {padded/1024/1024:.1f} MB — "
          f"{padded/on_disk:.1f}x larger)")
    print(f"\n  Replaces an 11.4 GB text encoder at runtime.\n")


if __name__ == "__main__":
    sys.exit(main())
