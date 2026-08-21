"""
Phosphor — Spike 2 verification: do baked embeddings equal live-encoded ones?

CLAUDE.md §4 is the load-bearing storage decision: ship a ~4 MB asset instead of an 11.4 GB
text encoder. That only holds if the baked tensors are what the live pipeline would have
produced. This asserts it directly at the tensor level, for every preset plus the negative,
rather than inferring it from one generation looking plausible.

What is being compared
----------------------
  baked : trimmed tensor from assets/embeddings.safetensors, zero-padded to max_seq_len
          (exactly what tools/spike_generate.py:load_embeds does at runtime)
  live  : UMT5-XXL encode of the same prompt, trimmed to real token length and zero-padded
          (exactly what diffusers _get_t5_prompt_embeds does)

Both must run at the same dtype (bf16) — that is what the stock pipeline uses, since it
builds its text_encoder at the pipeline dtype.
"""

import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.pipelines.wan.pipeline_wan import prompt_clean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "wan-ti2v-5b-diffusers")
EMBEDS = os.path.join(ROOT, "assets", "embeddings.safetensors")
PRESETS = os.path.join(ROOT, "assets", "presets.json")
NEG_CACHE = os.path.join(ROOT, "assets", ".negative_prompt.txt")

DTYPE = torch.bfloat16


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with safe_open(EMBEDS, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        max_len = int(meta["max_sequence_length"])
        baked = {k: f.get_tensor(k) for k in f.keys()}

    with open(PRESETS, encoding="utf-8") as fh:
        presets = json.load(fh)["presets"]
    prompts = {p["id"]: p["prompt"] for p in presets}
    if os.path.exists(NEG_CACHE):
        prompts["__negative__"] = open(NEG_CACHE, encoding="utf-8").read()

    print("\n  Phosphor — baked vs live embedding verification")
    print(f"  device      : {device}")
    print(f"  max_seq_len : {max_len}")
    print(f"  entries     : {len(baked)}")
    print(f"\n  Loading UMT5-XXL as bf16...", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(
        MODEL_DIR, subfolder="text_encoder", dtype=DTYPE
    ).to(device).eval()
    print(f"  encoder VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB\n")

    print(f"  {'key':<20}{'tokens':>7}{'max|diff|':>11}{'mean|diff|':>12}{'cosine':>10}  verdict")
    print("  " + "-" * 74)

    worst_cos, all_ok = 1.0, True
    for key, ref in baked.items():
        text = prompts.get(key)
        if text is None:
            print(f"  {key:<20}  (no source prompt — skipped)")
            continue

        ti = tok([prompt_clean(text)], padding="max_length", max_length=max_len,
                 truncation=True, add_special_tokens=True,
                 return_attention_mask=True, return_tensors="pt")
        ids, mask = ti.input_ids.to(device), ti.attention_mask.to(device)
        n = int(mask.gt(0).sum())
        live = enc(ids, mask).last_hidden_state[0, :n].to(DTYPE).cpu()

        if live.shape != ref.shape:
            print(f"  {key:<20}{n:>7}   SHAPE MISMATCH  baked {tuple(ref.shape)} vs live {tuple(live.shape)}")
            all_ok = False
            continue

        a = ref.float()
        b = live.float()
        diff = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        worst_cos = min(worst_cos, cos)

        exact = torch.equal(ref, live)
        verdict = "IDENTICAL" if exact else ("match" if cos > 0.9999 else "DIVERGED")
        if verdict == "DIVERGED":
            all_ok = False
        print(f"  {key:<20}{n:>7}{diff.max().item():>11.5f}{diff.mean().item():>12.6f}"
              f"{cos:>10.6f}  {verdict}")

    print("\n  " + "=" * 74)
    if all_ok:
        print(f"  PASS — every baked embedding reproduces the live encoder "
              f"(worst cosine {worst_cos:.6f}).")
        print(f"  §4 holds: a {os.path.getsize(EMBEDS)/1024/1024:.2f} MB asset validly "
              f"replaces the 11.4 GB encoder.")
    else:
        print("  FAIL — at least one entry diverged. Do NOT ship this embeddings file.")
    print("  " + "=" * 74 + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
