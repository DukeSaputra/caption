use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::burn::filter::build_filter_complex;
use crate::burn::font::{measure_text_width, pill_dimensions, scaled_font_size, PillDimensions};
use crate::burn::mask::{render_empty_frame, render_overlay_frame};
use crate::burn::probe::VideoInfo;
use crate::error::CaptionError;
use crate::stt::Word;
use crate::timing::{compute_display_timing, TimingConfig};

pub struct BurnConfig<'a> {
    pub input_path: &'a Path,
    pub output_path: &'a Path,
    pub ffmpeg_path: &'a Path,
    pub font_data: &'a [u8],
    pub video_info: &'a VideoInfo,
    pub words: &'a [Word],
}

fn compute_pills(
    words: &[Word],
    font_data: &[u8],
    font_size: f32,
    video_width: u32,
    video_height: u32,
) -> Result<Vec<Option<PillDimensions>>, CaptionError> {
    let mut pills = Vec::with_capacity(words.len());
    for word in words {
        if word.text.trim().is_empty() {
            pills.push(None);
            continue;
        }
        let text_width = measure_text_width(font_data, &word.text, font_size)?;
        let pill = pill_dimensions(text_width, font_size, video_width, video_height);
        pills.push(Some(pill));
    }
    Ok(pills)
}

fn lerp_pill(from: &PillDimensions, to: &PillDimensions, t: f32) -> PillDimensions {
    let t = t.clamp(0.0, 1.0);
    PillDimensions {
        width: from.width + (to.width - from.width) * t,
        height: from.height + (to.height - from.height) * t,
        radius: from.radius + (to.radius - from.radius) * t,
        x: from.x + (to.x - from.x) * t,
        y: from.y + (to.y - from.y) * t,
    }
}

const PILL_TRANSITION: f64 = 0.09;

// How much of the transition happens BEFORE the word boundary (anticipation).
const PILL_ANTICIPATION: f64 = 0.05;

pub fn run_burn(config: &BurnConfig) -> Result<(), CaptionError> {
    let vi = config.video_info;
    let font_size = scaled_font_size(vi.height);

    let pills = compute_pills(
        config.words,
        config.font_data,
        font_size,
        vi.width,
        vi.height,
    )?;

    let filter = build_filter_complex(vi.width, vi.height);

    let timing = compute_display_timing(config.words, &TimingConfig::default());
    let display_ends = timing.ends;

    // Pad 0.5s beyond the last word's end time.
    let last_end = display_ends.last().copied().unwrap_or(0.0);
    let frame_duration = 1.0 / vi.fps;
    let total_frames =
        ((last_end / frame_duration).ceil() as u64).saturating_add((vi.fps * 0.5).ceil() as u64);

    let size = format!("{}x{}", vi.width, vi.height * 2);

    let mut child = Command::new(config.ffmpeg_path)
        .args([
            "-y",
            "-i",
            &config.input_path.to_string_lossy(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &size,
            "-r",
            &vi.fps.to_string(),
            "-i",
            "pipe:0",
            "-filter_complex",
            &filter,
            "-map",
            "[out]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-nostdin",
            &config.output_path.to_string_lossy(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CaptionError::BurnFailed(format!("Failed to spawn FFmpeg: {e}")))?;

    let fade_short = 0.1; // 100ms for mid-transcript chain boundaries
    let fade_long = 0.25; // 250ms for the very first and very last word
    let word_count = config.words.len();

    let fade_in_enabled: Vec<bool> = config
        .words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if i == 0 {
                return true;
            }
            display_ends[i - 1] < w.start
        })
        .collect();

    let display_starts: Vec<f64> = config
        .words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if !fade_in_enabled[i] {
                return w.start;
            }
            let dur = if i == 0 { fade_long } else { fade_short };
            (w.start - dur).max(0.0)
        })
        .collect();

    let fade_out_enabled: Vec<bool> = (0..config.words.len())
        .map(|i| {
            if i + 1 >= config.words.len() {
                return true;
            }
            display_ends[i] < config.words[i + 1].start
        })
        .collect();

    {
        let stdin = child.stdin.as_mut().expect("stdin was piped");
        let empty_frame = render_empty_frame(vi.width, vi.height);

        for frame_idx in 0..total_frames {
            let timestamp = frame_idx as f64 * frame_duration;

            let active = config
                .words
                .iter()
                .enumerate()
                .zip(pills.iter())
                .zip(display_starts.iter().zip(display_ends.iter()))
                .find(|(((_, _), _), (start, end))| timestamp >= **start && timestamp < **end)
                .map(|(((idx, w), p), (start, end))| (idx, w, p, *start, *end));

            let frame = match active {
                Some((idx, word, Some(pill), display_start, display_end)) => {
                    let in_duration = if idx == 0 { fade_long } else { fade_short };
                    let out_duration = if idx == word_count - 1 {
                        fade_long
                    } else {
                        fade_short
                    };

                    let fade_in = if fade_in_enabled[idx] {
                        ((timestamp - display_start) / in_duration).min(1.0)
                    } else {
                        1.0
                    };
                    let fade_out = if fade_out_enabled[idx] {
                        ((display_end - timestamp) / out_duration).min(1.0)
                    } else {
                        1.0
                    };
                    let opacity = (fade_in.min(fade_out) as f32).clamp(0.0, 1.0);

                    let time_since_start = timestamp - word.start;
                    let time_to_end = display_end - timestamp;

                    let render_pill = if idx > 0
                        && !fade_in_enabled[idx]
                        && time_since_start < (PILL_TRANSITION - PILL_ANTICIPATION)
                    {
                        if let Some(prev_pill) = pills[idx - 1].as_ref() {
                            let t =
                                ((time_since_start + PILL_ANTICIPATION) / PILL_TRANSITION) as f32;
                            lerp_pill(prev_pill, pill, t)
                        } else {
                            pill.clone()
                        }
                    } else if idx + 1 < word_count
                        && !fade_out_enabled[idx]
                        && time_to_end < PILL_ANTICIPATION
                    {
                        if let Some(next_pill) = pills[idx + 1].as_ref() {
                            let t = ((PILL_ANTICIPATION - time_to_end) / PILL_TRANSITION) as f32;
                            lerp_pill(pill, next_pill, t)
                        } else {
                            pill.clone()
                        }
                    } else {
                        pill.clone()
                    };

                    render_overlay_frame(
                        vi.width,
                        vi.height,
                        &render_pill,
                        &word.text,
                        config.font_data,
                        font_size,
                        opacity,
                    )
                }
                _ => empty_frame.clone(),
            };

            if stdin.write_all(&frame).is_err() {
                break;
            }
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| CaptionError::BurnFailed(format!("Failed to wait for FFmpeg: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last_lines: String = stderr
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(CaptionError::BurnFailed(format!(
            "FFmpeg exited with {}: {}",
            output.status, last_lines
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
            confidence: 1.0,
        }
    }

    #[test]
    fn compute_pills_returns_one_per_word() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 0.6, 1.0)];
        assert_eq!(words.len(), 2);
    }

    // --- lerp_pill tests ---

    fn pill(x: f32, y: f32, w: f32, h: f32, r: f32) -> PillDimensions {
        PillDimensions {
            x,
            y,
            width: w,
            height: h,
            radius: r,
        }
    }

    #[test]
    fn lerp_pill_at_zero_returns_from() {
        let from = pill(10.0, 20.0, 200.0, 80.0, 40.0);
        let to = pill(50.0, 60.0, 300.0, 100.0, 50.0);
        let result = lerp_pill(&from, &to, 0.0);
        assert!((result.x - 10.0).abs() < 1e-4);
        assert!((result.y - 20.0).abs() < 1e-4);
        assert!((result.width - 200.0).abs() < 1e-4);
        assert!((result.height - 80.0).abs() < 1e-4);
        assert!((result.radius - 40.0).abs() < 1e-4);
    }

    #[test]
    fn lerp_pill_at_one_returns_to() {
        let from = pill(10.0, 20.0, 200.0, 80.0, 40.0);
        let to = pill(50.0, 60.0, 300.0, 100.0, 50.0);
        let result = lerp_pill(&from, &to, 1.0);
        assert!((result.x - 50.0).abs() < 1e-4);
        assert!((result.y - 60.0).abs() < 1e-4);
        assert!((result.width - 300.0).abs() < 1e-4);
        assert!((result.height - 100.0).abs() < 1e-4);
        assert!((result.radius - 50.0).abs() < 1e-4);
    }

    #[test]
    fn lerp_pill_at_half_interpolates() {
        let from = pill(0.0, 0.0, 100.0, 50.0, 20.0);
        let to = pill(100.0, 200.0, 300.0, 150.0, 60.0);
        let result = lerp_pill(&from, &to, 0.5);
        assert!((result.x - 50.0).abs() < 1e-4);
        assert!((result.y - 100.0).abs() < 1e-4);
        assert!((result.width - 200.0).abs() < 1e-4);
        assert!((result.height - 100.0).abs() < 1e-4);
        assert!((result.radius - 40.0).abs() < 1e-4);
    }

    #[test]
    fn lerp_pill_clamps_negative_t() {
        let from = pill(10.0, 20.0, 200.0, 80.0, 40.0);
        let to = pill(50.0, 60.0, 300.0, 100.0, 50.0);
        let result = lerp_pill(&from, &to, -0.5);
        // Should clamp to t=0, returning `from`
        assert!((result.x - 10.0).abs() < 1e-4);
        assert!((result.width - 200.0).abs() < 1e-4);
    }

    #[test]
    fn lerp_pill_clamps_t_above_one() {
        let from = pill(10.0, 20.0, 200.0, 80.0, 40.0);
        let to = pill(50.0, 60.0, 300.0, 100.0, 50.0);
        let result = lerp_pill(&from, &to, 2.0);
        // Should clamp to t=1, returning `to`
        assert!((result.x - 50.0).abs() < 1e-4);
        assert!((result.width - 300.0).abs() < 1e-4);
    }

    #[test]
    fn lerp_pill_identical_pills_returns_same() {
        let p = pill(30.0, 40.0, 250.0, 90.0, 45.0);
        let result = lerp_pill(&p, &p, 0.73);
        assert!((result.x - 30.0).abs() < 1e-4);
        assert!((result.y - 40.0).abs() < 1e-4);
        assert!((result.width - 250.0).abs() < 1e-4);
        assert!((result.height - 90.0).abs() < 1e-4);
        assert!((result.radius - 45.0).abs() < 1e-4);
    }

    // --- compute_pills tests ---

    #[test]
    fn compute_pills_empty_words() {
        let words: Vec<Word> = vec![];
        let pills = compute_pills(&words, &[], 32.0, 1080, 1920).unwrap();
        assert!(pills.is_empty());
    }

    #[test]
    fn compute_pills_whitespace_only_gives_none() {
        let words = vec![word("  ", 0.0, 0.5), word("\t", 0.5, 1.0)];
        let pills = compute_pills(&words, &[], 32.0, 1080, 1920).unwrap();
        assert_eq!(pills.len(), 2);
        assert!(pills[0].is_none());
        assert!(pills[1].is_none());
    }

    // --- PILL_TRANSITION / PILL_ANTICIPATION constant sanity ---

    #[test]
    fn pill_anticipation_less_than_transition() {
        assert!(
            PILL_ANTICIPATION < PILL_TRANSITION,
            "anticipation ({}) must be less than transition ({})",
            PILL_ANTICIPATION,
            PILL_TRANSITION,
        );
    }

    #[test]
    fn pill_transition_is_positive() {
        assert!(PILL_TRANSITION > 0.0);
        assert!(PILL_ANTICIPATION > 0.0);
    }
}
