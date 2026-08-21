//! ffmpeg invocation (CLAUDE.md §7).
//!
//! ffmpeg ships as a bundled sidecar binary, built **LGPL with only webp and gif** — no
//! x264, no x265. None of the GPL-encumbered codecs are needed and their absence keeps
//! commercial distribution clean.
//!
//! This stage also owns the **scale to output resolution**. Frames arrive from the Python
//! sidecar at generation size (768x1024 / 768x1152) and are scaled here. The scale happens
//! in the same pass as the encode; a separate resize pass would write a second full set of
//! frames to disk for nothing.
//!
//! Measured 2026-08-20, 64 frames at 1350x1800: WebP q75 lands at 6.18 MB (5.70 MB with
//! text protection, since static regions compress better). GIF is 44.41 MB, 7.2x, which is
//! why it is the compatibility option and warrants a size warning before it runs.
//!
//! **Two output scales (see `OutScale`).** Full is the SteamGridDB grid size; Half is
//! exactly half of each axis, added 2026-08-21 because RetroVoid hitches while scrolling a
//! grid of full-size animated covers. Halving each axis is a *quarter* of the pixels, so it
//! cuts per-frame decode work about 4x.

use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Output scale. The aspect ratio is decided by the source cover, never by the user, so
/// this is the only export-size choice the UI offers.
///
/// `Half` exists because a launcher scrolling a grid of these has to decode every frame:
/// halving each axis quarters the pixel count, which is where the cost actually is. It is
/// also *closer to native* than `Full` is. Generation is 768x1024 / 768x1152, so `Full`
/// upscales 1.76x / 1.56x while `Half` downscales to 0.88x / 0.78x. `Half` is discarding
/// resolution the model never produced, rather than detail. That is why it can be the
/// default without costing visible quality.
///
/// **No `Default` impl on purpose.** The default is a UI preference and lives in exactly
/// one place, `App.tsx`'s `half` state, currently `Half`. A derived default here would be
/// a second answer to the same question that nothing reads, and would silently go stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutScale {
    /// SteamGridDB grid size. 1350x1800 and 1200x1800.
    Full,
    /// Half of each axis. 675x900 and 600x900.
    Half,
}

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

    /// Output size. `Full` is the SteamGridDB-standard grid dimensions.
    ///
    /// Both full sizes are even on both axes, so `Half` divides exactly and there is no
    /// rounding to reason about. A test pins that, since an odd dimension would have to be
    /// rounded somewhere and would quietly shift the aspect ratio.
    pub fn out_size(self, scale: OutScale) -> (u32, u32) {
        let (w, h) = match self {
            Aspect::ThreeFour => (1350, 1800),
            Aspect::TwoThree => (1200, 1800),
        };
        match scale {
            OutScale::Full => (w, h),
            OutScale::Half => (w / 2, h / 2),
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
/// Split out so the argument list can be asserted in a test. The `-f webp` in here is
/// load-bearing and easy to delete as redundant; see the note below.
fn webp_args(
    pattern: &Path,
    out: &Path,
    aspect: Aspect,
    scale: OutScale,
    quality: u8,
) -> Vec<String> {
    let (ow, oh) = aspect.out_size(scale);
    vec![
        "-y".into(), "-hide_banner".into(), "-loglevel".into(), "error".into(),
        "-framerate".into(), FPS.to_string(),
        "-i".into(), pattern.to_string_lossy().into(),
        "-vf".into(), format!("scale={ow}:{oh}:flags=lanczos"),
        "-c:v".into(), "libwebp_anim".into(),
        "-lossless".into(), "0".into(),
        "-q:v".into(), quality.to_string(),
        "-loop".into(), "0".into(),
        "-preset".into(), "picture".into(),
        // Force the muxer instead of letting ffmpeg infer it from the output extension.
        // The Steam export writes these exact bytes to a file named `.png`, and without
        // this ffmpeg would see that name, select the image2 muxer, and emit a numbered
        // PNG sequence rather than one animated WebP.
        "-f".into(), "webp".into(),
        out.to_string_lossy().into(),
    ]
}

pub async fn webp(
    ffmpeg: &Path,
    pattern: &Path,
    out: &Path,
    aspect: Aspect,
    scale: OutScale,
    quality: u8,
) -> Result<PathBuf, EncodeError> {
    run(ffmpeg, &webp_args(pattern, out, aspect, scale, quality)).await?;
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
    scale: OutScale,
) -> Result<PathBuf, EncodeError> {
    let (ow, oh) = aspect.out_size(scale);
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

    /// The Steam export is the same bytes under a `.png` name, which only works because
    /// the muxer is stated explicitly. Drop `-f webp` and ffmpeg infers image2 from the
    /// extension and writes a PNG sequence, so this pins it.
    #[test]
    fn webp_forces_its_muxer_so_a_png_name_cannot_change_it() {
        let args = webp_args(
            Path::new("frames_%04d.png"),
            Path::new("cover_animated.png"),
            Aspect::ThreeFour,
            OutScale::Full,
            75,
        );
        let i = args.iter().position(|a| a == "-f").expect("-f must be present");
        assert_eq!(args[i + 1], "webp");
        assert!(args.iter().any(|a| a == "libwebp_anim"));
    }

    #[test]
    fn output_targets_are_steamgriddb_sizes() {
        assert_eq!(Aspect::ThreeFour.out_size(OutScale::Full), (1350, 1800));
        assert_eq!(Aspect::TwoThree.out_size(OutScale::Full), (1200, 1800));
    }

    #[test]
    fn half_is_the_four_sizes_the_ui_offers() {
        assert_eq!(Aspect::ThreeFour.out_size(OutScale::Half), (675, 900));
        assert_eq!(Aspect::TwoThree.out_size(OutScale::Half), (600, 900));
    }

    /// Integer division would silently shift the aspect ratio if a full size were ever odd
    /// on either axis, and the drift would be invisible until someone measured a cover.
    #[test]
    fn halving_is_exact_so_the_ratio_cannot_drift() {
        for a in [Aspect::ThreeFour, Aspect::TwoThree] {
            let (fw, fh) = a.out_size(OutScale::Full);
            assert_eq!(fw % 2, 0, "full width {fw} is odd, halving would round");
            assert_eq!(fh % 2, 0, "full height {fh} is odd, halving would round");

            let (hw, hh) = a.out_size(OutScale::Half);
            assert_eq!(
                (fw as f64 / fh as f64),
                (hw as f64 / hh as f64),
                "half changed the aspect ratio of {a:?}"
            );
        }
    }

    /// Half must reach ffmpeg as the scale filter, not merely exist on the enum. The whole
    /// point is the pixels RetroVoid has to decode.
    #[test]
    fn the_scale_choice_reaches_the_ffmpeg_filter() {
        for (scale, want) in [
            (OutScale::Full, "scale=1350:1800:flags=lanczos"),
            (OutScale::Half, "scale=675:900:flags=lanczos"),
        ] {
            let args = webp_args(
                Path::new("f_%04d.png"),
                Path::new("out.webp"),
                Aspect::ThreeFour,
                scale,
                75,
            );
            assert!(
                args.iter().any(|a| a == want),
                "{scale:?} did not produce `{want}`; args were {args:?}"
            );
        }
    }
}
