# Bundled sidecar binaries

Tauri's `externalBin` requires the **`<name>-<target-triple>.exe`** naming convention.
On this machine the triple is `x86_64-pc-windows-msvc`.

| binary | status |
|---|---|
| `ffmpeg-x86_64-pc-windows-msvc.exe` | present — BtbN **LGPL** build (`--enable-version3`, no `--enable-gpl`), webp + gif |

## The sidecar is NOT an externalBin

It is a PyInstaller **directory** build, not a single file, so it cannot ship through
`externalBin` at all. Onefile was rejected deliberately: the payload is ~2.9 GB of torch
and CUDA, and onefile re-extracts that to a temp directory on every single launch.

It ships as a bundle **resource** instead:

```jsonc
"resources": { "../sidecar-dist/phosphor-sidecar/": "sidecar/" }
```

so it installs to `<resources>/sidecar/phosphor-sidecar.exe`. `bundled_binary()` in
`src-tauri/src/lib.rs` looks there, and reports every path it tried when it comes up empty.

Rebuild it with:

```powershell
./tools/build_sidecar.ps1
```

Use that script rather than calling pyinstaller directly: its default output directory is
`dist/`, which Vite empties on every `npm run build`. `sidecar-dist/` and `build/` are
gitignored. The freeze is reproducible from `setup.ps1` plus that
one command, so it is never committed.

## Two traps, both hit during scaffolding

1. **A missing `externalBin` entry fails every build, including `cargo check` in dev.**
   It is a build-script error, not a bundle-time one. Only ffmpeg is listed there, and the
   sidecar never will be — see above.
2. **`tauri.conf.json` rejects unknown fields.** JSON has no comments and Tauri validates
   the schema strictly, so notes like this one live in a README rather than in the config.

## ffmpeg size

The current LGPL build is 111 MB against CLAUDE.md §7's ~80 MB estimate, because it is a
full-featured build. A custom build configured for webp + gif only would be far smaller and
is worth doing before release.
