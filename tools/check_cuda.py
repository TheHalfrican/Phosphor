"""Environment sanity check. Exits non-zero if anything required is missing.

Lives in a file rather than a `python -c` string on purpose: Windows truncates
multi-line arguments passed to a native exe at the first newline, so a here-string
piped into `python -c` silently arrives as a broken fragment.
"""

import sys


def main():
    import torch

    print("  torch      : " + torch.__version__)
    print("  cuda build : " + str(torch.version.cuda))
    print("  available  : " + str(torch.cuda.is_available()))

    if not torch.cuda.is_available():
        print("\n  CUDA NOT AVAILABLE - stopping. Check driver / wheel variant.")
        return 1

    p = torch.cuda.get_device_properties(0)
    print("  device     : " + p.name)
    print("  vram       : %.1f GiB" % (p.total_memory / 1024**3))
    print("  capability : sm_%d%d" % (p.major, p.minor))

    import diffusers
    import transformers

    print("  diffusers  : " + diffusers.__version__)
    print("  transformers: " + transformers.__version__)

    # The four symbols the whole design depends on (CLAUDE.md §9).
    missing = []
    try:
        from diffusers import WanImageToVideoPipeline  # noqa: F401
    except ImportError:
        missing.append("WanImageToVideoPipeline")
    try:
        from diffusers import WanTransformer3DModel  # noqa: F401
    except ImportError:
        missing.append("WanTransformer3DModel")
    try:
        from diffusers import GGUFQuantizationConfig  # noqa: F401
    except ImportError:
        missing.append("GGUFQuantizationConfig")
    try:
        from diffusers import AutoencoderKLWan  # noqa: F401
    except ImportError:
        missing.append("AutoencoderKLWan")

    if missing:
        print("\n  MISSING from diffusers %s: %s" % (diffusers.__version__, ", ".join(missing)))
        return 1

    print("  wan symbols: all present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
