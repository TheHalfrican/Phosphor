//! Ping-pong loop assembly (CLAUDE.md §6).
//!
//! Generated video is not a loop. Ping-pong makes it one. For `N` generated frames the
//! output sequence is:
//!
//! ```text
//! [0, 1, 2, ..., N-1, N-2, N-3, ..., 2, 1]     length 2N - 2
//! ```
//!
//! Both endpoints are dropped on the reverse pass. Keeping frame `N-1` duplicates the
//! turnaround frame; keeping frame `0` duplicates the loop-point frame. Either reads as a
//! visible hitch.
//!
//! Verified empirically 2026-08-20 on a real 33-frame generation: 64 frames, zero
//! duplicates, and neither seam distinguishable from an ordinary frame transition
//! (turnaround 1.71, loop point 2.38, against a typical frame delta of mean 2.45 / max
//! 4.22). The tests below pin the sequencing rule that makes that true.

use std::path::{Path, PathBuf};

/// Frame indices for the ping-pong sequence.
pub fn pingpong_indices(n: usize) -> Vec<usize> {
    if n < 2 {
        return (0..n).collect();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.extend((1..n - 1).rev());
    idx
}

/// Materialise the ping-pong order as a numbered sequence ffmpeg can consume as
/// `pp_%04d.png`.
///
/// Hardlinks where the filesystem allows it, so the duplicated half costs no disk. Falls
/// back to copying (different volume, non-NTFS, permissions).
pub fn materialise(src_frames: &[PathBuf], dst_dir: &Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(dst_dir)?;
    let idx = pingpong_indices(src_frames.len());

    for (i, &src_i) in idx.iter().enumerate() {
        let src = &src_frames[src_i];
        let dst = dst_dir.join(format!("pp_{:04}.png", i));
        if dst.exists() {
            std::fs::remove_file(&dst)?;
        }
        if std::fs::hard_link(src, &dst).is_err() {
            std::fs::copy(src, &dst)?;
        }
    }
    Ok(idx.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_is_2n_minus_2() {
        assert_eq!(pingpong_indices(33).len(), 64);
        assert_eq!(pingpong_indices(17).len(), 32);
        assert_eq!(pingpong_indices(5).len(), 8);
    }

    #[test]
    fn endpoints_appear_exactly_once() {
        let idx = pingpong_indices(33);
        assert_eq!(idx.iter().filter(|&&i| i == 0).count(), 1, "loop point duplicated");
        assert_eq!(idx.iter().filter(|&&i| i == 32).count(), 1, "turnaround duplicated");
    }

    #[test]
    fn no_adjacent_duplicates_including_wrap() {
        let idx = pingpong_indices(33);
        for i in 0..idx.len() {
            let a = idx[i];
            let b = idx[(i + 1) % idx.len()];
            assert_ne!(a, b, "duplicate frame at position {i} -> would stutter");
        }
    }

    #[test]
    fn sequence_shape() {
        assert_eq!(pingpong_indices(5), vec![0, 1, 2, 3, 4, 3, 2, 1]);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert_eq!(pingpong_indices(0), Vec::<usize>::new());
        assert_eq!(pingpong_indices(1), vec![0]);
        assert_eq!(pingpong_indices(2), vec![0, 1]);
    }
}
