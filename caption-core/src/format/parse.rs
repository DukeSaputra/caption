use crate::error::CaptionError;
use crate::stt::Word;

pub struct ParsedSrt {
    pub words: Vec<Word>,
    pub had_multi_word_cues: bool,
}

pub fn parse_srt(content: &str) -> Result<ParsedSrt, CaptionError> {
    let mut words = Vec::new();
    let mut had_multi_word_cues = false;

    for block in content.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            continue;
        }

        let timestamp_line = lines[1];
        let (start, end) = parse_timestamp_range(timestamp_line)?;

        let text: String = lines[2..].join(" ");
        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.len() == 1 {
            words.push(Word {
                text: parts[0].to_string(),
                start,
                end,
                confidence: 1.0,
            });
        } else {
            had_multi_word_cues = true;
            let total_chars: usize = parts.iter().map(|p| p.len()).sum();
            let span = end - start;
            let mut cursor = start;

            for (i, part) in parts.iter().enumerate() {
                let is_last = i == parts.len() - 1;
                let word_end = if is_last {
                    end
                } else if total_chars == 0 {
                    cursor + span / parts.len() as f64
                } else {
                    cursor + span * (part.len() as f64 / total_chars as f64)
                };

                words.push(Word {
                    text: part.to_string(),
                    start: cursor,
                    end: word_end,
                    confidence: 1.0,
                });

                cursor = word_end;
            }
        }
    }

    if words.is_empty() {
        return Err(CaptionError::BurnFailed(
            "SRT file contains no cues".to_string(),
        ));
    }

    Ok(ParsedSrt {
        words,
        had_multi_word_cues,
    })
}

fn parse_timestamp_range(line: &str) -> Result<(f64, f64), CaptionError> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return Err(CaptionError::BurnFailed(format!(
            "Invalid SRT timestamp line: {line}"
        )));
    }

    let start = parse_srt_timestamp(parts[0].trim())?;
    let end = parse_srt_timestamp(parts[1].trim())?;
    Ok((start, end))
}

fn parse_srt_timestamp(ts: &str) -> Result<f64, CaptionError> {
    // HH:MM:SS,mmm or HH:MM:SS.mmm
    let parts: Vec<&str> = ts.split(|c| c == ':' || c == ',' || c == '.').collect();
    if parts.len() != 4 {
        return Err(CaptionError::BurnFailed(format!(
            "Invalid SRT timestamp: {ts}"
        )));
    }

    let h: f64 = parts[0]
        .parse()
        .map_err(|_| CaptionError::BurnFailed(format!("Invalid hours in timestamp: {ts}")))?;
    let m: f64 = parts[1]
        .parse()
        .map_err(|_| CaptionError::BurnFailed(format!("Invalid minutes in timestamp: {ts}")))?;
    let s: f64 = parts[2]
        .parse()
        .map_err(|_| CaptionError::BurnFailed(format!("Invalid seconds in timestamp: {ts}")))?;
    let ms: f64 = parts[3]
        .parse()
        .map_err(|_| CaptionError::BurnFailed(format!("Invalid millis in timestamp: {ts}")))?;

    Ok(h * 3600.0 + m * 60.0 + s + ms / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_word_cues() {
        let srt = "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n2\n00:00:00,600 --> 00:00:01,000\nworld\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert!(!parsed.had_multi_word_cues);
        let words = &parsed.words;
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!((words[0].end - 0.5).abs() < 1e-9);
        assert_eq!(words[1].text, "world");
        assert!((words[1].start - 0.6).abs() < 1e-9);
        assert!((words[1].end - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_multi_word_cue_splits_proportionally() {
        let srt = "1\n00:00:00,000 --> 00:00:01,000\nhello world\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert!(parsed.had_multi_word_cues);
        let words = &parsed.words;
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].end - 0.5).abs() < 1e-9);
        assert!((words[1].start - 0.5).abs() < 1e-9);
        assert!((words[1].end - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_timestamp_with_period_separator() {
        let srt = "1\n00:00:01.500 --> 00:00:02.000\nhi\n\n";
        let words = &parse_srt(srt).unwrap().words;
        assert_eq!(words.len(), 1);
        assert!((words[0].start - 1.5).abs() < 1e-9);
        assert!((words[0].end - 2.0).abs() < 1e-9);
    }

    #[test]
    fn parse_hour_timestamps() {
        let srt = "1\n01:02:03,456 --> 01:02:04,000\nlate\n\n";
        let words = &parse_srt(srt).unwrap().words;
        assert_eq!(words.len(), 1);
        let expected = 3600.0 + 120.0 + 3.456;
        assert!((words[0].start - expected).abs() < 1e-9);
    }

    #[test]
    fn parse_empty_srt_returns_error() {
        let result = parse_srt("");
        assert!(result.is_err());
    }

    #[test]
    fn parse_skips_blank_blocks() {
        let srt = "\n\n1\n00:00:00,000 --> 00:00:00,500\nhello\n\n\n\n";
        let words = &parse_srt(srt).unwrap().words;
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
    }

    #[test]
    fn parse_multi_line_cue_text() {
        let srt = "1\n00:00:00,000 --> 00:00:01,000\nhello\nbeautiful world\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert!(parsed.had_multi_word_cues);
        let words = &parsed.words;
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "beautiful");
        assert_eq!(words[2].text, "world");
    }

    #[test]
    fn round_trip_single_words() {
        use crate::format::srt::SrtFormatter;
        use crate::format::{FormatConfig, SubtitleFormatter};

        let original = vec![
            Word {
                text: "hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 1.0,
            },
            Word {
                text: "world".to_string(),
                start: 0.6,
                end: 1.0,
                confidence: 1.0,
            },
        ];

        let config = FormatConfig::default();
        let srt_text = SrtFormatter.format(&original, &config).unwrap();
        let parsed = parse_srt(&srt_text).unwrap();
        assert!(!parsed.had_multi_word_cues);

        assert_eq!(parsed.words.len(), original.len());
        for (orig, parsed) in original.iter().zip(parsed.words.iter()) {
            assert_eq!(orig.text, parsed.text);
            assert!((orig.start - parsed.start).abs() < 0.002);
        }
    }

    #[test]
    fn confidence_defaults_to_one() {
        let srt = "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n";
        let words = &parse_srt(srt).unwrap().words;
        assert!((words[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    // --- Malformed SRT: missing --> separator ---

    #[test]
    fn parse_missing_arrow_separator() {
        let srt = "1\n00:00:00,000 00:00:00,500\nhello\n\n";
        let result = parse_srt(srt);
        assert!(result.is_err(), "expected error for missing --> separator");
    }

    #[test]
    fn parse_wrong_arrow_separator() {
        let srt = "1\n00:00:00,000 -> 00:00:00,500\nhello\n\n";
        let result = parse_srt(srt);
        assert!(result.is_err(), "expected error for -> instead of -->");
    }

    // --- Malformed SRT: out-of-order indices ---

    #[test]
    fn parse_out_of_order_indices_still_works() {
        // Parser ignores cue index values and just processes blocks sequentially
        let srt = "5\n00:00:00,000 --> 00:00:00,500\nhello\n\n2\n00:00:00,600 --> 00:00:01,000\nworld\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 2);
        assert_eq!(parsed.words[0].text, "hello");
        assert_eq!(parsed.words[1].text, "world");
    }

    // --- Malformed SRT: non-numeric cue IDs ---

    #[test]
    fn parse_non_numeric_cue_id() {
        // The parser treats line[0] as the cue ID but never parses it as a number,
        // so a non-numeric ID should not cause an error
        let srt = "abc\n00:00:00,000 --> 00:00:00,500\nhello\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 1);
        assert_eq!(parsed.words[0].text, "hello");
    }

    // --- Malformed SRT: timestamps with wrong format ---

    #[test]
    fn parse_timestamp_missing_hours() {
        // MM:SS,mmm instead of HH:MM:SS,mmm -- only 3 parts instead of 4
        let srt = "1\n00:00,000 --> 00:00,500\nhello\n\n";
        let result = parse_srt(srt);
        assert!(
            result.is_err(),
            "expected error for timestamp missing hours"
        );
    }

    #[test]
    fn parse_timestamp_extra_colons() {
        let srt = "1\n00:00:00:00,000 --> 00:00:00:00,500\nhello\n\n";
        let result = parse_srt(srt);
        assert!(
            result.is_err(),
            "expected error for timestamp with extra colons"
        );
    }

    #[test]
    fn parse_timestamp_non_numeric_value() {
        let srt = "1\nAA:BB:CC,DDD --> 00:00:00,500\nhello\n\n";
        let result = parse_srt(srt);
        assert!(result.is_err(), "expected error for non-numeric timestamp");
    }

    #[test]
    fn parse_timestamp_missing_millis() {
        // HH:MM:SS without milliseconds -- only 3 parts
        let srt = "1\n00:00:00 --> 00:00:01\nhello\n\n";
        let result = parse_srt(srt);
        assert!(
            result.is_err(),
            "expected error for timestamp without millis"
        );
    }

    // --- Malformed SRT: empty file variants ---

    #[test]
    fn parse_whitespace_only_returns_error() {
        let result = parse_srt("   \n\n  \n  ");
        assert!(result.is_err());
    }

    #[test]
    fn parse_newlines_only_returns_error() {
        let result = parse_srt("\n\n\n\n");
        assert!(result.is_err());
    }

    // --- Malformed SRT: block too short (fewer than 3 lines) ---

    #[test]
    fn parse_block_with_only_two_lines_skipped() {
        // A block with just ID and timestamp but no text should be skipped
        let srt = "1\n00:00:00,000 --> 00:00:00,500\n\n2\n00:00:01,000 --> 00:00:01,500\nhello\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 1);
        assert_eq!(parsed.words[0].text, "hello");
    }

    #[test]
    fn parse_block_with_only_id_skipped() {
        let srt = "1\n\n2\n00:00:00,000 --> 00:00:00,500\nhello\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 1);
    }

    // --- Edge case: cue with blank text line ---

    #[test]
    fn parse_cue_with_whitespace_text_skipped() {
        let srt =
            "1\n00:00:00,000 --> 00:00:00,500\n   \n\n2\n00:00:01,000 --> 00:00:01,500\nhello\n\n";
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 1);
        assert_eq!(parsed.words[0].text, "hello");
    }

    // --- Edge case: multiple arrow separators ---

    #[test]
    fn parse_multiple_arrows_returns_error() {
        let srt = "1\n00:00:00,000 --> 00:00:00,500 --> 00:00:01,000\nhello\n\n";
        let result = parse_srt(srt);
        assert!(
            result.is_err(),
            "expected error for multiple --> separators"
        );
    }

    // --- Edge case: no trailing newline ---

    #[test]
    fn parse_no_trailing_newline() {
        let srt = "1\n00:00:00,000 --> 00:00:00,500\nhello";
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 1);
        assert_eq!(parsed.words[0].text, "hello");
    }

    // --- Edge case: Windows-style line endings ---

    #[test]
    fn parse_windows_line_endings() {
        let srt = "1\r\n00:00:00,000 --> 00:00:00,500\r\nhello\r\n\r\n";
        // The parser splits on "\n\n" so \r\n\r\n may split correctly
        // and .trim() handles residual \r
        let parsed = parse_srt(srt).unwrap();
        assert_eq!(parsed.words.len(), 1);
        assert_eq!(parsed.words[0].text, "hello");
    }
}
