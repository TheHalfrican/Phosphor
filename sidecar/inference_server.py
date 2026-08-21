"""
Phosphor sidecar — newline-delimited JSON over stdin/stdout (CLAUDE.md §2).

Not HTTP: no port to conflict over, no socket bound on the user's machine, no firewall
prompt on first run.

Invocation:
    python inference_server.py [--models DIR] [--assets DIR]

    --models  directory holding wan-ti2v-5b-diffusers/, gguf/ and craft/
    --assets  directory holding embeddings.safetensors

    Both default to the repo layout when running from source. Both are REQUIRED when
    frozen, because the two live in different places once installed (see _resolve_roots).

Requests (one JSON object per line on stdin):
    {"op":"ping","id":"..."}
    {"op":"generate","id":"...","image":"...","preset":"ember_glow","guidance":2.0, ...}
    {"op":"detect_text","id":"...","image":"...","width":768,"height":1024}
    {"op":"protect","id":"...","frames_dir":"...","source":"...","mask":"..."}
    {"op":"shutdown","id":"..."}

Replies (one JSON object per line on stdout), every line carrying the originating id:
    {"id":"...","type":"progress","stage":"generate","step":3,"total":20}
    {"id":"...","type":"result", ...}
    {"id":"...","type":"error","message":"..."}

STDOUT DISCIPLINE
-----------------
stdout is the protocol channel and nothing else. diffusers, transformers and torch all
print freely — a single stray line corrupts the stream and the Rust side sees a parse
error instead of a reply. So the very first thing this module does, before importing
anything heavy, is stash the real stdout and rebind `sys.stdout` to stderr. Protocol
writes go through the stashed handle; every incidental `print()` anywhere in the process
lands harmlessly on stderr, which Rust forwards to the UI log.

Requests are handled one at a time. The GPU is the bottleneck and a 4090 has no headroom
to run two 768x1024 generations concurrently, so a queue would add complexity and buy
nothing.
"""

import json
import os
import sys
import traceback

# --- stdout discipline: do this BEFORE importing torch/diffusers ------------------------
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr

FROZEN = getattr(sys, "frozen", False)

# Where our own bundled modules live. Frozen, that is PyInstaller's extraction dir; from
# source, it is this file's directory. `vendor.craft` is imported off this.
HERE = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _resolve_roots(argv):
    """Work out where the models and the baked assets live.

    These are passed in rather than derived from `__file__`, for two reasons:

    1. Frozen, `__file__` points into PyInstaller's temp extraction directory, which is
       neither of these and disappears on exit.
    2. In a packaged build they are not even under a common root. Models are downloaded
       to the user's app data (they are 7 GB and must survive an app update), while
       `embeddings.safetensors` ships inside the installed app's resource directory. A
       single ROOT cannot name both.

    Running from source, the repo layout is the obvious default, so dev usage is
    unchanged: `python sidecar/inference_server.py` still works with no arguments.
    """
    models = assets = None
    i = 0
    while i < len(argv):
        a = argv[i]
        for flag, name in (("--models", "models"), ("--assets", "assets")):
            if a == flag and i + 1 < len(argv):
                if name == "models":
                    models = argv[i + 1]
                else:
                    assets = argv[i + 1]
                i += 1
            elif a.startswith(flag + "="):
                if name == "models":
                    models = a.split("=", 1)[1]
                else:
                    assets = a.split("=", 1)[1]
        i += 1

    if FROZEN and not (models and assets):
        # Refuse to guess. A frozen build that silently fell back to a path relative to
        # the executable would fail much later, as a confusing model-load error.
        raise SystemExit(
            "phosphor-sidecar: --models and --assets are required when frozen"
        )

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return (models or os.path.join(repo, "models"),
            assets or os.path.join(repo, "assets"))


MODELS_ROOT, ASSETS_ROOT = _resolve_roots(sys.argv[1:])


def send(obj):
    """Write one protocol line. Flush every time — the Rust reader is line-oriented and a
    buffered progress event is a progress event the user never sees."""
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


def log(msg):
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)


def main():
    from pipeline import PhosphorPipeline

    pipe = PhosphorPipeline(MODELS_ROOT, ASSETS_ROOT)
    log(f"ready (models={MODELS_ROOT}, assets={ASSETS_ROOT})")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send({"id": "", "type": "error", "message": f"malformed request: {e}"})
            continue

        rid = req.get("id", "")
        op = req.get("op", "")

        try:
            if op == "ping":
                send({"id": rid, "type": "result", "ok": True})

            elif op == "shutdown":
                send({"id": rid, "type": "result", "ok": True})
                break

            elif op == "generate":
                def progress(step, total):
                    send({"id": rid, "type": "progress", "stage": "generate",
                          "step": step, "total": total})

                send({"id": rid, "type": "progress", "stage": "load", "step": 0, "total": 1})
                import time
                t0 = time.time()
                out = pipe.generate(
                    req["image"],
                    req["preset"],
                    guidance=float(req.get("guidance", 2.0)),
                    steps=int(req.get("steps", 20)),
                    frames=int(req.get("frames", 33)),
                    seed=int(req.get("seed", 0)),
                    progress=progress,
                    log=log,
                )
                out["seconds"] = round(time.time() - t0, 1)
                send({"id": rid, "type": "result", **out})

            elif op == "detect_text":
                out = pipe.detect_text(
                    req["image"],
                    int(req["width"]),
                    int(req["height"]),
                    threshold=float(req.get("threshold", 0.30)),
                    log=log,
                )
                send({"id": rid, "type": "result", **out})

            elif op == "protect":
                out = pipe.protect(req["frames_dir"], req["source"], req["mask"], log=log)
                send({"id": rid, "type": "result", **out})

            else:
                send({"id": rid, "type": "error", "message": f"unknown op '{op}'"})

        except Exception as e:  # noqa: BLE001 - the sidecar must never die on one bad request
            log(traceback.format_exc())
            send({"id": rid, "type": "error", "message": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        send({"id": "", "type": "error", "message": f"sidecar fatal: {e}"})
        sys.exit(1)
