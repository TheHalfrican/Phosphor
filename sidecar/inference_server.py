"""
Phosphor sidecar — newline-delimited JSON over stdin/stdout (CLAUDE.md §2).

Not HTTP: no port to conflict over, no socket bound on the user's machine, no firewall
prompt on first run.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def send(obj):
    """Write one protocol line. Flush every time — the Rust reader is line-oriented and a
    buffered progress event is a progress event the user never sees."""
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


def log(msg):
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)


def main():
    from pipeline import PhosphorPipeline

    pipe = PhosphorPipeline(ROOT)
    log("ready")

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
