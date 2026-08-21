# Bundled sidecar binaries

Tauri's `externalBin` requires the **`<name>-<target-triple>.exe`** naming convention.
On this machine the triple is `x86_64-pc-windows-msvc`.

| binary | status |
|---|---|
| `ffmpeg-x86_64-pc-windows-msvc.exe` | present — BtbN **LGPL** build (`--enable-version3`, no `--enable-gpl`), webp + gif |
| `phosphor-sidecar-x86_64-pc-windows-msvc.exe` | **not yet built** — PyInstaller freeze of `sidecar/inference_server.py` |

## Two traps, both hit during scaffolding

1. **A missing `externalBin` entry fails every build, including `cargo check` in dev.**
   It is a build-script error, not a bundle-time one. Do not add the sidecar entry back to
   `tauri.conf.json` until the frozen binary actually exists here.
2. **`tauri.conf.json` rejects unknown fields.** JSON has no comments and Tauri validates
   the schema strictly, so notes like this one live in a README rather than in the config.

## ffmpeg size

The current LGPL build is 111 MB against CLAUDE.md §7's ~80 MB estimate, because it is a
full-featured build. A custom build configured for webp + gif only would be far smaller and
is worth doing before release.
