"""
Phosphor — dev model fetch.

Pulls everything the end-to-end spike needs, smallest first so progress is visible
early. Every step is resumable: re-run after an interruption and completed files are
skipped (huggingface_hub verifies size + hash on resume).

NOTE ON WHAT IS *NOT* DOWNLOADED
--------------------------------
The Wan-AI diffusers repo ships the transformer as 5 fp32 shards (~20 GB). We skip all
of it — the transformer comes from the Q6_K GGUF instead. That skip is most of why this
is a ~19 GB fetch rather than a ~39 GB one.

The text encoder (11.4 GB, fp32 UMT5-XXL) is DEV-ONLY. It exists solely so
bake_embeddings.py can run once. The shipped app never downloads it — that is the whole
point of CLAUDE.md §4.
"""

import os
import sys
import time

# huggingface_hub >=1.x transfers via Xet; HF_HUB_ENABLE_HF_TRANSFER is deprecated
# and silently does nothing. HF_XET_HIGH_PERFORMANCE is the current knob.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
GGUF_REPO = "QuantStack/Wan2.2-TI2V-5B-GGUF"
LORA_REPO = "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF"

GGUF_FILE = "Wan2.2-TI2V-5B-Q6_K.gguf"
LORA_FILE = "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"


def banner(n, total, title, size):
    print()
    print("=" * 74)
    print(f"  [{n}/{total}]  {title}")
    print(f"           ~{size}")
    print("=" * 74, flush=True)


def main():
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()

    print()
    print("  Phosphor — model fetch")
    print(f"  destination : {MODELS}")
    print("  total       : ~19 GB  (11.4 GB of it is dev-only text encoder)")
    print("  resumable   : yes — safe to close and re-run")

    # ---- 1. configs + tokenizer -------------------------------------------------
    banner(1, 4, "Pipeline configs + tokenizer", "20 MB")
    snapshot_download(
        repo_id=WAN_REPO,
        local_dir=os.path.join(MODELS, "wan-ti2v-5b-diffusers"),
        allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json"],
    )

    # ---- 2. Turbo LoRA ----------------------------------------------------------
    banner(2, 4, "Turbo 4-step distill LoRA (rank 64)", "332 MB")
    hf_hub_download(
        repo_id=LORA_REPO,
        filename=LORA_FILE,
        local_dir=os.path.join(MODELS, "lora"),
    )

    # ---- 3. transformer (GGUF) + VAE --------------------------------------------
    banner(3, 4, "Transformer Q6_K GGUF", "4.21 GB")
    hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        local_dir=os.path.join(MODELS, "gguf"),
    )

    banner(3, 4, "VAE (full precision — do NOT quantize, see CLAUDE.md §3)", "2.82 GB")
    snapshot_download(
        repo_id=WAN_REPO,
        local_dir=os.path.join(MODELS, "wan-ti2v-5b-diffusers"),
        allow_patterns=["vae/*"],
    )

    # ---- 4. text encoder (dev only) ---------------------------------------------
    banner(4, 4, "UMT5-XXL text encoder  [DEV ONLY — baked away for release]", "11.4 GB")
    print("  This is the 3-shard fp32 encoder. It is needed exactly once, to run")
    print("  tools/bake_embeddings.py. The shipped app never fetches it.\n", flush=True)
    snapshot_download(
        repo_id=WAN_REPO,
        local_dir=os.path.join(MODELS, "wan-ti2v-5b-diffusers"),
        allow_patterns=["text_encoder/*"],
    )

    mins = (time.time() - t0) / 60
    print()
    print("=" * 74)
    print(f"  Done in {mins:.1f} min.")
    print("=" * 74)
    print()
    print("  Next:")
    print("    python tools/bake_embeddings.py      # one-time, needs the text encoder")
    print("    python tools/spike_generate.py ...   # image in, PNG frames out")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Re-run this script to resume where it stopped.\n")
        sys.exit(130)
