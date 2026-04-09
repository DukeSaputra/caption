use crate::error::CaptionError;
use crate::stt::Word;

use super::{FormatConfig, SubtitleFormatter};

pub struct AssFormatter;

const ASS_HEADER: &str = "\
[Script Info]
Title: Caption
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

fn format_timestamp(seconds: f64) -> String {
    let total_cs = (seconds * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_secs = total_cs / 100;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    // ASS uses single-digit hour and centiseconds with period separator
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

fn duration_to_centiseconds(seconds: f64) -> u64 {
    let cs = (seconds * 100.0).round() as u64;
    // At least 1cs to avoid zero-duration karaoke tags
    cs.max(1)
}

struct Phrase<'a> {
    words: Vec<&'a Word>,
}

fn group_into_phrases<'a>(words: &'a [Word], config: &FormatConfig) -> Vec<Phrase<'a>> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut phrases: Vec<Phrase<'a>> = Vec::new();
    let mut current_words: Vec<&'a Word> = vec![&words[0]];

    for i in 1..words.len() {
        let prev = &words[i - 1];
        let prev_end = prev.end.max(prev.start + config.min_cue_duration);
        let gap = words[i].start - prev_end;

        if gap >= config.pause_threshold {
            phrases.push(Phrase {
                words: current_words,
            });
            current_words = Vec::new();
        }

        current_words.push(&words[i]);
    }

    if !current_words.is_empty() {
        phrases.push(Phrase {
            words: current_words,
        });
    }

    phrases
}

impl SubtitleFormatter for AssFormatter {
    fn format(&self, words: &[Word], config: &FormatConfig) -> Result<String, CaptionError> {
        if words.is_empty() {
            return Ok(String::new());
        }

        let phrases = group_into_phrases(words, config);
        let mut output = String::from(ASS_HEADER);

        for phrase in &phrases {
            let first = phrase.words[0];
            let last = phrase.words[phrase.words.len() - 1];

            let phrase_start = first.start;
            let phrase_end = last.end.max(last.start + config.min_cue_duration);

            let ts_start = format_timestamp(phrase_start);
            let ts_end = format_timestamp(phrase_end);

            let mut text = String::new();
            for (i, word) in phrase.words.iter().enumerate() {
                let word_duration =
                    (word.end.max(word.start + config.min_cue_duration)) - word.start;
                let cs = duration_to_centiseconds(word_duration);

                text.push_str(&format!("{{\\kf{cs}}}"));
                text.push_str(&word.text);
                if i + 1 < phrase.words.len() {
                    text.push(' ');
                }
            }

            output.push_str(&format!(
                "Dialogue: 0,{ts_start},{ts_end},Default,,0,0,0,,{text}\n"
            ));
        }

        Ok(output)
    }

    fn file_extension(&self) -> &str {
        "ass"
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

    // --- ASS header ---

    #[test]
    fn output_contains_script_info_section() {
        let words = vec![word("hello", 0.5, 1.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        assert!(ass.contains("[Script Info]"));
        assert!(ass.contains("ScriptType: v4.00+"));
        assert!(ass.contains("PlayResX: 1920"));
        assert!(ass.contains("PlayResY: 1080"));
    }

    #[test]
    fn output_contains_v4_styles_section() {
        let words = vec![word("hello", 0.5, 1.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        assert!(ass.contains("[V4+ Styles]"));
        assert!(ass.contains("Style: Default,Arial,72"));
    }

    #[test]
    fn output_contains_events_section() {
        let words = vec![word("hello", 0.5, 1.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        assert!(ass.contains("[Events]"));
        assert!(ass.contains(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ));
    }

    #[test]
    fn header_sections_in_correct_order() {
        let words = vec![word("hello", 0.5, 1.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let script_info_pos = ass.find("[Script Info]").unwrap();
        let styles_pos = ass.find("[V4+ Styles]").unwrap();
        let events_pos = ass.find("[Events]").unwrap();

        assert!(
            script_info_pos < styles_pos,
            "Script Info must come before V4+ Styles"
        );
        assert!(
            styles_pos < events_pos,
            "V4+ Styles must come before Events"
        );
    }

    #[test]
    fn empty_word_list_returns_empty_string() {
        let words: Vec<Word> = vec![];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        assert_eq!(ass, "");
    }

    #[test]
    fn empty_word_list_has_no_header() {
        let words: Vec<Word> = vec![];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        assert!(!ass.contains("[Script Info]"));
    }

    // --- Timestamp formatting ---

    #[test]
    fn timestamp_zero() {
        assert_eq!(format_timestamp(0.0), "0:00:00.00");
    }

    #[test]
    fn timestamp_fractional_seconds() {
        assert_eq!(format_timestamp(1.5), "0:00:01.50");
    }

    #[test]
    fn timestamp_minutes_and_centiseconds() {
        assert_eq!(format_timestamp(61.01), "0:01:01.01");
    }

    #[test]
    fn timestamp_hours() {
        assert_eq!(format_timestamp(3661.99), "1:01:01.99");
    }

    #[test]
    fn timestamp_single_digit_hour() {
        let ts = format_timestamp(3600.0);
        assert_eq!(ts, "1:00:00.00");
        assert!(
            !ts.starts_with("01:"),
            "ASS timestamps must use single-digit hour, not {ts}"
        );
    }

    #[test]
    fn timestamp_double_digit_hour() {
        let ts = format_timestamp(36000.0);
        assert_eq!(ts, "10:00:00.00");
    }

    #[test]
    fn timestamp_uses_period_not_comma() {
        let ts = format_timestamp(1.5);
        assert!(ts.contains('.'), "ASS timestamps must use period: {ts}");
        assert!(
            !ts.contains(','),
            "ASS timestamps must NOT use comma separator: {ts}"
        );
    }

    #[test]
    fn timestamp_centiseconds_not_milliseconds() {
        let ts = format_timestamp(1.5);
        assert!(
            ts.ends_with(".50"),
            "Expected centiseconds (.50), got: {ts}"
        );
    }

    // --- Karaoke tags ---

    #[test]
    fn single_word_produces_kf_tag() {
        let words = vec![word("Hello", 1.0, 1.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        // 500ms = 50 centiseconds
        assert!(
            ass.contains("{\\kf50}Hello"),
            "Expected {{\\kf50}}Hello in output: {ass}"
        );
    }

    #[test]
    fn kf_duration_is_centiseconds() {
        // 800ms = 80 centiseconds
        let words = vec![word("test", 0.0, 0.8)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        assert!(
            ass.contains("{\\kf80}test"),
            "Expected 80cs for 800ms word: {ass}"
        );
    }

    #[test]
    fn multiple_words_in_one_phrase() {
        let words = vec![
            word("Hello", 0.0, 0.3),
            word("beautiful", 0.35, 0.8),
            word("world", 0.85, 1.2),
        ];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            1,
            "All words within pause_threshold should be one phrase"
        );

        let line = dialogue_lines[0];
        assert!(line.contains("{\\kf30}Hello"), "Missing Hello tag: {line}");
        assert!(
            line.contains("{\\kf45}beautiful"),
            "Missing beautiful tag: {line}"
        );
        assert!(line.contains("{\\kf35}world"), "Missing world tag: {line}");
    }

    #[test]
    fn words_separated_by_spaces_in_dialogue() {
        let words = vec![word("Hello", 0.0, 0.3), word("world", 0.35, 0.7)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.contains("{\\kf30}Hello {\\kf35}world"),
            "Words should be space-separated: {dialogue_line}"
        );
    }

    #[test]
    fn no_trailing_space_after_last_word() {
        let words = vec![word("Hello", 0.0, 0.3), word("world", 0.35, 0.7)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.ends_with("world"),
            "No trailing space after last word: {dialogue_line}"
        );
    }

    // --- Phrase grouping ---

    #[test]
    fn long_pause_splits_into_two_phrases() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 1.0, 1.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            2,
            "Gap >= pause_threshold should create two phrases"
        );
    }

    #[test]
    fn gap_exactly_at_threshold_splits() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 1.0, 1.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            2,
            "Gap == pause_threshold should split into two phrases"
        );
    }

    #[test]
    fn short_gap_keeps_words_in_same_phrase() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 0.6, 1.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            1,
            "Gap < pause_threshold should keep words in same phrase"
        );
    }

    #[test]
    fn three_phrases_with_two_pauses() {
        let words = vec![
            word("one", 0.0, 0.3),
            word("two", 0.35, 0.6),
            word("three", 1.5, 1.8),
            word("four", 2.5, 2.8),
        ];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(dialogue_lines.len(), 3, "Should produce three phrases");
    }

    // --- Dialogue line format ---

    #[test]
    fn dialogue_line_has_correct_prefix() {
        let words = vec![word("hello", 1.0, 1.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.starts_with("Dialogue: 0,"),
            "Dialogue line should start with 'Dialogue: 0,': {dialogue_line}"
        );
    }

    #[test]
    fn dialogue_line_timestamps_match_phrase_bounds() {
        let words = vec![word("Hello", 1.0, 1.3), word("world", 1.35, 1.7)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.contains("0:00:01.00,0:00:01.70"),
            "Timestamps should span phrase: {dialogue_line}"
        );
    }

    #[test]
    fn dialogue_line_uses_default_style() {
        let words = vec![word("hello", 0.0, 0.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.contains(",Default,"),
            "Dialogue should reference Default style: {dialogue_line}"
        );
    }

    // --- Min cue duration ---

    #[test]
    fn min_cue_duration_enforced_for_short_word() {
        // 50ms word extended to 200ms = 20cs
        let words = vec![word("I", 1.0, 1.05)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        assert!(
            ass.contains("{\\kf20}I"),
            "Short word should get min_cue_duration (20cs): {ass}"
        );
    }

    #[test]
    fn min_cue_duration_affects_phrase_end() {
        let words = vec![word("I", 1.0, 1.05)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.contains("0:00:01.20"),
            "Phrase end should respect min_cue_duration: {dialogue_line}"
        );
    }

    // --- Edge cases ---

    #[test]
    fn single_word_output() {
        let words = vec![word("hello", 0.5, 1.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(dialogue_lines.len(), 1);
        assert!(dialogue_lines[0].contains("{\\kf50}hello"));
    }

    #[test]
    fn one_hour_video_timestamps() {
        let words = vec![word("late", 3661.5, 3662.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(
            dialogue_line.contains("1:01:01.50,1:01:02.00"),
            "Hour+ timestamps should use single-digit hour: {dialogue_line}"
        );
    }

    #[test]
    fn output_ends_with_trailing_newline() {
        let words = vec![word("hello", 0.0, 0.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        assert!(ass.ends_with('\n'));
    }

    #[test]
    fn file_extension_is_ass() {
        assert_eq!(AssFormatter.file_extension(), "ass");
    }

    // --- Duration centisecond conversion ---

    #[test]
    fn duration_to_cs_standard() {
        assert_eq!(duration_to_centiseconds(0.5), 50);
        assert_eq!(duration_to_centiseconds(0.3), 30);
        assert_eq!(duration_to_centiseconds(0.8), 80);
        assert_eq!(duration_to_centiseconds(1.0), 100);
    }

    #[test]
    fn duration_to_cs_minimum_is_one() {
        assert_eq!(duration_to_centiseconds(0.0), 1);
        assert_eq!(duration_to_centiseconds(0.001), 1);
    }

    #[test]
    fn duration_to_cs_rounds() {
        // 0.155s = 15.5cs, rounds to 16
        assert_eq!(duration_to_centiseconds(0.155), 16);
        // 0.154s = 15.4cs, rounds to 15
        assert_eq!(duration_to_centiseconds(0.154), 15);
    }

    // --- Phrase boundary with min_cue_duration ---

    #[test]
    fn min_cue_duration_considered_for_phrase_gap() {
        let words = vec![word("a", 0.4, 0.42), word("b", 1.11, 1.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            2,
            "Gap after min_cue_duration enforcement (0.3) should split: counted {}",
            dialogue_lines.len()
        );
    }

    #[test]
    fn min_cue_duration_keeps_phrase_together() {
        let words = vec![word("a", 0.4, 0.42), word("b", 0.89, 1.2)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            1,
            "Gap < pause_threshold (after min_cue_duration) should keep in same phrase"
        );
    }

    // --- Full output verification ---

    #[test]
    fn full_output_single_phrase() {
        let words = vec![word("Hello", 1.0, 1.5), word("world", 1.55, 2.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_line = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert_eq!(
            dialogue_line,
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\kf50}Hello {\\kf45}world"
        );
    }

    #[test]
    fn full_output_two_phrases() {
        let words = vec![word("Hello", 1.0, 1.5), word("world", 2.5, 3.0)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(dialogue_lines.len(), 2);
        assert_eq!(
            dialogue_lines[0],
            "Dialogue: 0,0:00:01.00,0:00:01.50,Default,,0,0,0,,{\\kf50}Hello"
        );
        assert_eq!(
            dialogue_lines[1],
            "Dialogue: 0,0:00:02.50,0:00:03.00,Default,,0,0,0,,{\\kf50}world"
        );
    }

    // --- No BOM ---

    #[test]
    fn no_bom_in_output() {
        let words = vec![word("hello", 0.0, 0.5)];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();
        let bytes = ass.as_bytes();
        assert!(
            bytes.len() < 3 || !(bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF),
            "Output must not contain a BOM"
        );
    }

    // --- Parse helper for validation ---

    fn parse_ass_ts(ts: &str) -> f64 {
        // Format: H:MM:SS.CC
        let parts: Vec<&str> = ts.split(|c| c == ':' || c == '.').collect();
        let h: f64 = parts[0].parse().unwrap();
        let m: f64 = parts[1].parse().unwrap();
        let s: f64 = parts[2].parse().unwrap();
        let cs: f64 = parts[3].parse().unwrap();
        h * 3600.0 + m * 60.0 + s + cs / 100.0
    }

    #[test]
    fn parse_ass_timestamp_roundtrip() {
        let ts = format_timestamp(3661.5);
        let parsed = parse_ass_ts(&ts);
        assert!(
            (parsed - 3661.5).abs() < 0.01,
            "Round-trip failed: formatted={ts}, parsed={parsed}"
        );
    }

    #[test]
    fn dialogue_start_before_end() {
        let words = vec![
            word("the", 0.0, 0.3),
            word("quick", 0.35, 0.6),
            word("brown", 0.65, 0.9),
            word("fox", 2.0, 2.3),
            word("jumps", 2.35, 2.7),
        ];
        let config = FormatConfig::default();
        let ass = AssFormatter.format(&words, &config).unwrap();

        for line in ass.lines().filter(|l| l.starts_with("Dialogue:")) {
            let after_layer = line.strip_prefix("Dialogue: 0,").unwrap();
            let parts: Vec<&str> = after_layer.splitn(3, ',').collect();
            let start = parse_ass_ts(parts[0]);
            let end = parse_ass_ts(parts[1]);
            assert!(end >= start, "Dialogue line has end < start: {line}");
        }
    }
}
