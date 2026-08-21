//! ffmpeg invocation (CLAUDE.md §7).
//!
//! ffmpeg ships as a bundled sidecar binary, built **LGPL with only webp and gif** — no
//! x264, no x265. None of the GPL-encumbered codecs are needed and their absence keeps
//! commercial distribution clean.
//!
//! This stage also owns the **upscale to output resolution**. Frames arrive from the
//! Python sidecar at generation size (768x1024 / 768x1152) and are scaled here to
//! 1350x1800 / 1200x1800. The scale happens in the same pass as the encode — a separate
//! resize pass would write a second full set of frames to disk for nothing.
//!
//! Measured 2026-08-20, 64 frames at 1350x1800: WebP q75 lands at 6.18 MB (5.70 MB with
//! text protection, since static regions compress better). GIF is 44.41 MB — 7.2x — which
//! is why it is the compatibility option and warrants a size warning before it runs.

use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aspect {
    ThreeFour,
    TwoThree,
}

impl Aspect {
    /// Classify a source cover. 3:4 = 0.7500, 2:3 = 0.6667.
    pub fn classify(w: u32, h: u32) -> Self {
        let r = w as f32 / h as f32;
        if (r - 0.75).abs() < (r - 2.0 / 3.0).abs() {
            Aspect::ThreeFour
        } else {
            Aspect::TwoThree
        }
    }

    /// Generation size. Must be divisible by 32 — VAE spatial 16 x transformer patch 2.
    /// Note ffmpeg is not what enforces this; the diffusers pipeline only checks %16,
    /// which is too lax (496 passes and then yields a fractional patch grid).
    pub fn gen_size(self) -> (u32, u32) {
        match self {
            Aspect::ThreeFour => (768, 1024),
            Aspect::TwoThree => (768, 1152),
        }
    }

    /// Output size — SteamGridDB-standard grid dimensions.
    pub fn out_size(self) -> (u32, u32) {
        match self {
            Aspect::ThreeFour => (1350, 1800),
            Aspect::TwoThree => (1200, 1800),
        }
    }
}

pub const FPS: u32 = 24;

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("ffmpeg i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffmpeg exited {code:?}: {stderr}")]
    Failed { code: Option<i32>, stderr: String },
}

async fn run(ffmpeg: &Path, args: &[String]) -> Result<(), EncodeError> {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(args);

    #[cfg(windows)]
    {
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }

    let out = cmd.output().await?;
    if !out.status.success() {
        return Err(EncodeError::Failed {
            code: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr)
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .join("\n"),
        });
    }
    Ok(())
}

/// Animated WebP — the primary export.
///
/// `quality` 75 is the default and sits at the knee of the size/quality curve: q95 costs
/// 2.9x the size for 4.6 dB. It does not need to be a user-facing slider.
pub async fn webp(
    ffmpeg: &Path,
    pattern: &Path,
    out: &Path,
    aspect: Aspect,
    quality: u8,
) -> Result<PathBuf, EncodeError> {
    let (ow, oh) = aspect.out_size();
    let args: Vec<String> = vec![
        "-y".into(), "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-framerate".into(), FPS.to_string(),
        "-i".into(), pattern.to_string_lossy().into(),
        "-vf".into(), format!("scale={ow}:{oh}:flags=lanczos"),
        "-c:v".into(), "libwebp_anim".into(),
        "-lossless".into(), "0".into(),
        "-q:v".into(), quality.to_string(),
        "-loop".into(), "0".into(),
        "-preset".into(), "picture".into(),
        out.to_string_lossy().into(),
    ];
    run(ffmpeg, &args).await?;
    Ok(out.to_path_buf())
}

/// GIF — compatibility export only. Two passes, and the flags matter.
///
/// `stats_mode=diff` aims the 256-colour budget at the *changing* regions rather than the
/// static majority of the frame, which is exactly the content profile of an animated
/// cover. `dither=bayer` is smaller than error diffusion and aesthetically right for retro
/// cover art.
///
/// The scale must be applied identically in BOTH passes and BEFORE `palettegen` — the
/// palette has to be built from the same pixels it will be applied to, or the colour
/// selection is subtly wrong.
pub async fn gif(
    ffmpeg: &Path,
    pattern: &Path,
    palette: &Path,
    out: &Path,
    aspect: Aspect,
) -> Result<PathBuf, EncodeError> {
    let (ow, oh) = aspect.out_size();
    let scale = format!("scale={ow}:{oh}:flags=lanczos");

    run(ffmpeg, &[
        "-y".into(), "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-i".into(), pattern.to_string_lossy().into(),
        "-vf".into(), format!("{scale},palettegen=stats_mode=diff"),
        palette.to_string_lossy().into(),
    ]).await?;

    run(ffmpeg, &[
        "-y".into(), "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-framerate".into(), FPS.to_string(),
        "-i".into(), pattern.to_string_lossy().into(),
        "-i".into(), palette.to_string_lossy().into(),
        "-lavfi".into(), format!("{scale}[s];[s][1:v]paletteuse=dither=bayer:bayer_scale=5"),
        "-loop".into(), "0".into(),
        out.to_string_lossy().into(),
    ]).await?;

    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_classification() {
        assert_eq!(Aspect::classify(1350, 1800), Aspect::ThreeFour);
        assert_eq!(Aspect::classify(1200, 1800), Aspect::TwoThree);
        assert_eq!(Aspect::classify(1920, 2880), Aspect::TwoThree);
    }

    #[test]
    fn generation_sizes_divisible_by_32() {
        for a in [Aspect::ThreeFour, Aspect::TwoThree] {
            let (w, h) = a.gen_size();
            assert_eq!(w % 32, 0, "gen width {w} not divisible by 32");
            assert_eq!(h % 32, 0, "gen height {h} not divisible by 32");
        }
    }

    #[test]
    fn output_targets_are_steamgriddb_sizes() {
        assert_eq!(Aspect::ThreeFour.out_size(), (1350, 1800));
        assert_eq!(Aspect::TwoThree.out_size(), (1200, 1800));
    }
}
