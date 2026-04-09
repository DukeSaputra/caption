use crate::error::CaptionError;
use crate::stt::Word;
use crate::timing::{compute_display_timing, TimingConfig};

use super::{FormatConfig, SubtitleFormatter};

pub struct SrtFormatter;

// 1ms buffer to prevent two-line subtitle display
const OVERLAP_BUFFER: f64 = 0.001;

fn format_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    // SRT uses comma as decimal separator (Premiere Pro rejects periods)
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

impl SubtitleFormatter for SrtFormatter {
    fn format(&self, words: &[Word], config: &FormatConfig) -> Result<String, CaptionError> {
        if words.is_empty() {
            return Ok(String::new());
        }

        let timing_config = TimingConfig {
            bridge_threshold: config.pause_threshold,
            min_display: config.min_cue_duration,
            word_length_scaling: true,
        };
        let timing = compute_display_timing(words, &timing_config);

        let mut output = String::new();

        for (i, word) in words.iter().enumerate() {
            let start = word.start;
            let mut end = timing.ends[i];

            if i + 1 < words.len() {
                let next_start = words[i + 1].start;
                if end > next_start {
                    end = next_start - OVERLAP_BUFFER;
                }
            }

            let index = i + 1;
            let ts_start = format_timestamp(start);
            let ts_end = format_timestamp(end);

            output.push_str(&format!(
                "{index}\n{ts_start} --> {ts_end}\n{}\n\n",
                word.text
            ));
        }

        Ok(output)
    }

    fn file_extension(&self) -> &str {
        "srt"
    }
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

    // --- Timestamp formatting ---

    #[test]
    fn timestamp_zero() {
        assert_eq!(format_timestamp(0.0), "00:00:00,000");
    }

    #[test]
    fn timestamp_fractional_seconds() {
        assert_eq!(format_timestamp(1.5), "00:00:01,500");
    }

    #[test]
    fn timestamp_minutes_and_millis() {
        assert_eq!(format_timestamp(61.001), "00:01:01,001");
    }

    #[test]
    fn timestamp_hours() {
        assert_eq!(format_timestamp(3661.999), "01:01:01,999");
    }

    #[test]
    fn timestamp_uses_comma_not_period() {
        let ts = format_timestamp(1.5);
        assert!(ts.contains(','), "SRT timestamps must use comma: {ts}");
        assert!(!ts.ends_with(".500"), "Must not use period separator");
    }

    #[test]
    fn timestamp_zero_padded() {
        let ts = format_timestamp(1.001);
        assert_eq!(ts, "00:00:01,001");
    }

    // --- RSVP algorithm ---

    #[test]
    fn short_gap_bridges() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 0.6, 1.0)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.contains("00:00:00,000 --> 00:00:00,600"));
    }

    #[test]
    fn long_gap_does_not_bridge() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 1.0, 1.5)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.contains("00:00:00,000 --> 00:00:00,500"));
    }

    #[test]
    fn gap_exactly_at_threshold_does_not_bridge() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 1.0, 1.5)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.contains("00:00:00,000 --> 00:00:00,500"));
    }

    #[test]
    fn overlapping_timestamps_clamped() {
        let words = vec![word("hello", 0.0, 0.8), word("world", 0.5, 1.0)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.contains("00:00:00,000 --> 00:00:00,499"));
    }

    #[test]
    fn min_cue_duration_enforced() {
        let words = vec![word("I", 1.0, 1.05)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.contains("00:00:01,000 --> 00:00:01,200"));
    }

    #[test]
    fn min_cue_duration_after_overlap_clamp() {
        let words = vec![word("a", 0.0, 0.5), word("b", 0.05, 1.0)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.contains("00:00:00,000 --> 00:00:00,049"));
    }

    #[test]
    fn single_word() {
        let words = vec![word("hello", 0.5, 1.0)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert_eq!(srt, "1\n00:00:00,500 --> 00:00:01,000\nhello\n\n");
    }

    #[test]
    fn empty_word_list() {
        let words: Vec<Word> = vec![];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();
        assert_eq!(srt, "");
    }

    // --- SRT structure ---

    #[test]
    fn cue_indices_start_at_one() {
        let words = vec![word("a", 0.0, 0.5), word("b", 1.0, 1.5)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        assert!(srt.starts_with("1\n"));
        assert!(srt.contains("\n2\n"));
    }

    #[test]
    fn no_overlapping_cue_ranges() {
        let words = vec![
            word("the", 0.0, 0.3),
            word("quick", 0.35, 0.6),
            word("fox", 0.65, 1.0),
        ];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        let timestamps: Vec<(f64, f64)> = srt
            .split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let lines: Vec<&str> = block.lines().collect();
                let times: Vec<&str> = lines[1].split(" --> ").collect();
                (parse_ts(times[0]), parse_ts(times[1]))
            })
            .collect();

        for window in timestamps.windows(2) {
            assert!(
                window[0].1 <= window[1].0,
                "Overlap: cue ends at {} but next starts at {}",
                window[0].1,
                window[1].0
            );
        }
    }

    #[test]
    fn output_ends_with_trailing_newline() {
        let words = vec![word("hello", 0.0, 0.5)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();
        assert!(srt.ends_with('\n'));
    }

    #[test]
    fn no_bom_in_output() {
        let words = vec![word("hello", 0.0, 0.5)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();
        let bytes = srt.as_bytes();
        // BOM bytes: EF BB BF
        assert!(
            bytes.len() < 3 || !(bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF),
            "Output must not contain a BOM"
        );
    }

    // --- Edge cases ---

    #[test]
    fn one_hour_video_timestamps() {
        let words = vec![word("late", 3661.5, 3662.0)];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();
        assert!(srt.contains("01:01:01,500 --> 01:01:02,000"));
    }

    #[test]
    fn all_words_same_start_no_negative_durations() {
        let words = vec![
            word("a", 0.0, 0.1),
            word("b", 0.0, 0.1),
            word("c", 0.0, 0.1),
        ];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        for block in srt.split("\n\n").filter(|b| !b.is_empty()) {
            let lines: Vec<&str> = block.lines().collect();
            let times: Vec<&str> = lines[1].split(" --> ").collect();
            let start = parse_ts(times[0]);
            let end = parse_ts(times[1]);
            assert!(end >= start, "Negative duration: start={start}, end={end}");
        }
    }

    // --- Round-trip test ---

    #[test]
    fn round_trip_words_match() {
        let words = vec![
            word("hello", 0.0, 0.5),
            word("beautiful", 0.6, 1.2),
            word("world", 1.8, 2.5),
        ];
        let config = FormatConfig::default();
        let srt = SrtFormatter.format(&words, &config).unwrap();

        let extracted: Vec<&str> = srt
            .split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let lines: Vec<&str> = block.lines().collect();
                lines[2]
            })
            .collect();

        assert_eq!(extracted.len(), words.len());
        for (extracted_text, original) in extracted.iter().zip(words.iter()) {
            assert_eq!(*extracted_text, original.text);
        }
    }

    #[test]
    fn file_extension_is_srt() {
        assert_eq!(SrtFormatter.file_extension(), "srt");
    }

    fn parse_ts(ts: &str) -> f64 {
        // Format: HH:MM:SS,mmm
        let parts: Vec<&str> = ts.split(|c| c == ':' || c == ',').collect();
        let h: f64 = parts[0].parse().unwrap();
        let m: f64 = parts[1].parse().unwrap();
        let s: f64 = parts[2].parse().unwrap();
        let ms: f64 = parts[3].parse().unwrap();
        h * 3600.0 + m * 60.0 + s + ms / 1000.0
    }
}
