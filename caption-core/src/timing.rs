use crate::stt::Word;

#[derive(Debug, Clone)]
pub struct TimingConfig {
    pub bridge_threshold: f64,
    pub min_display: f64,
    pub word_length_scaling: bool,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            bridge_threshold: 1.0,
            min_display: 0.5,
            word_length_scaling: false,
        }
    }
}

pub struct DisplayTiming {
    pub ends: Vec<f64>,
}

fn min_display_for_word(text: &str, config: &TimingConfig) -> f64 {
    if !config.word_length_scaling {
        return config.min_display;
    }
    let char_count = text.chars().count();
    // Spritz RSVP formula: 200ms base + 25ms per character over 3
    let scaled = 0.200 + 0.025 * (char_count.saturating_sub(3) as f64);
    config.min_display.max(scaled)
}

pub fn compute_display_timing(words: &[Word], config: &TimingConfig) -> DisplayTiming {
    let ends = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if let Some(next) = words.get(i + 1) {
                let gap = next.start - w.end;
                if gap >= 0.0 && gap < config.bridge_threshold {
                    return next.start;
                }
            }
            let natural_end = w.end;
            let min_end = w.start + min_display_for_word(&w.text, config);
            natural_end.max(min_end)
        })
        .collect();

    DisplayTiming { ends }
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
    fn bridge_short_gap() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 0.7, 1.0)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert_eq!(timing.ends.len(), 2);
        assert!((timing.ends[0] - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn no_bridge_long_gap() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 2.5, 3.0)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn bridge_at_threshold_boundary() {
        let words = vec![word("hello", 0.0, 0.5), word("world", 1.5, 2.0)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn min_display_extends_short_word() {
        let words = vec![word("I", 1.0, 1.1)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn min_display_does_not_shorten_long_word() {
        let words = vec![word("extraordinary", 1.0, 3.0)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_words() {
        let words: Vec<Word> = vec![];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!(timing.ends.is_empty());
    }

    #[test]
    fn single_word_gets_min_display() {
        let words = vec![word("ok", 2.0, 2.1)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn custom_config() {
        let config = TimingConfig {
            bridge_threshold: 0.3,
            min_display: 0.2,
            word_length_scaling: false,
        };
        let words = vec![
            word("a", 0.0, 0.5),
            word("b", 0.7, 1.0),
            word("c", 2.0, 2.05),
        ];
        let timing = compute_display_timing(&words, &config);

        assert!((timing.ends[0] - 0.7).abs() < f64::EPSILON);
        assert!((timing.ends[1] - 1.0).abs() < f64::EPSILON);
        assert!((timing.ends[2] - 2.2).abs() < f64::EPSILON);
    }

    #[test]
    fn negative_gap_not_bridged() {
        let words = vec![word("a", 0.0, 0.8), word("b", 0.5, 1.0)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn gap_just_below_threshold_bridges() {
        let words = vec![word("a", 0.0, 0.5), word("b", 1.499, 2.0)];
        let timing = compute_display_timing(&words, &TimingConfig::default());
        assert!((timing.ends[0] - 1.499).abs() < f64::EPSILON);
    }

    #[test]
    fn chain_of_bridged_words() {
        let words = vec![
            word("the", 0.0, 0.3),
            word("quick", 0.35, 0.6),
            word("fox", 0.65, 1.0),
        ];
        let timing = compute_display_timing(&words, &TimingConfig::default());

        assert!((timing.ends[0] - 0.35).abs() < f64::EPSILON);
        assert!((timing.ends[1] - 0.65).abs() < f64::EPSILON);
        assert!((timing.ends[2] - 1.15).abs() < f64::EPSILON);
    }

    // --- Word-length scaling ---

    fn scaling_config(min_display: f64) -> TimingConfig {
        TimingConfig {
            bridge_threshold: 1.0,
            min_display,
            word_length_scaling: true,
        }
    }

    #[test]
    fn scaling_short_word_gets_200ms() {
        // "I" (1 char): 0.200 + 0.025 * max(0, 1-3) = 0.200
        let words = vec![word("I", 1.0, 1.05)];
        let timing = compute_display_timing(&words, &scaling_config(0.2));
        assert!((timing.ends[0] - 1.2).abs() < f64::EPSILON);
    }

    #[test]
    fn scaling_three_char_word_gets_200ms() {
        // "the" (3 chars): 0.200 + 0.025 * 0 = 0.200
        let words = vec![word("the", 1.0, 1.05)];
        let timing = compute_display_timing(&words, &scaling_config(0.2));
        assert!((timing.ends[0] - 1.2).abs() < f64::EPSILON);
    }

    #[test]
    fn scaling_five_char_word_gets_250ms() {
        // "hello" (5 chars): 0.200 + 0.025 * 2 = 0.250
        let words = vec![word("hello", 1.0, 1.05)];
        let timing = compute_display_timing(&words, &scaling_config(0.2));
        assert!((timing.ends[0] - 1.25).abs() < f64::EPSILON);
    }

    #[test]
    fn scaling_thirteen_char_word_gets_450ms() {
        // "unfortunately" (13 chars): 0.200 + 0.025 * 10 = 0.450
        let words = vec![word("unfortunately", 1.0, 1.05)];
        let timing = compute_display_timing(&words, &scaling_config(0.2));
        assert!((timing.ends[0] - 1.45).abs() < f64::EPSILON);
    }

    #[test]
    fn scaling_does_not_shorten_long_natural_end() {
        let words = vec![word("hello", 1.0, 2.0)];
        let timing = compute_display_timing(&words, &scaling_config(0.2));
        assert!((timing.ends[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn scaling_min_display_floor_respected() {
        // "I" (1 char): scaled = 0.200, min_display = 0.3, max = 0.3
        let words = vec![word("I", 1.0, 1.05)];
        let timing = compute_display_timing(
            &words,
            &TimingConfig {
                bridge_threshold: 1.0,
                min_display: 0.3,
                word_length_scaling: true,
            },
        );
        assert!((timing.ends[0] - 1.3).abs() < f64::EPSILON);
    }

    #[test]
    fn burn_pipeline_ignores_word_length() {
        let config = TimingConfig::default();
        let words = vec![word("I", 0.0, 0.05), word("unfortunately", 2.0, 2.05)];
        let timing = compute_display_timing(&words, &config);
        assert!((timing.ends[0] - 0.5).abs() < f64::EPSILON);
        assert!((timing.ends[1] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn scaling_bridged_words_unaffected() {
        let words = vec![word("unfortunately", 0.0, 0.3), word("I", 0.35, 0.6)];
        let timing = compute_display_timing(&words, &scaling_config(0.2));
        assert!((timing.ends[0] - 0.35).abs() < f64::EPSILON);
    }
}
