use crate::postprocess::PostProcessor;
use crate::stt::Word;

#[derive(Debug, Clone, Copy, Default)]
pub enum FillerMode {
    KeepAll,
    #[default]
    RemoveConfident,
    RemoveAll,
}

const CONFIDENT_FILLERS: &[&str] = &[
    "um", "uh", "hmm", "mm", "mhm", "mmhmm", "er", "erm", "ah", "huh",
];

const EXTENDED_SINGLE_FILLERS: &[&str] = &[
    "like",
    "so",
    "right",
    "basically",
    "literally",
    "actually",
    "well",
    "okay",
    "ok",
    "anyway",
    "anyways",
    "obviously",
    "honestly",
    "essentially",
];

const MULTI_WORD_FILLERS: &[(&str, &str)] = &[
    ("you", "know"),
    ("i", "mean"),
    ("kind", "of"),
    ("sort", "of"),
];

pub struct FillerRemover {
    pub mode: FillerMode,
}

impl PostProcessor for FillerRemover {
    fn process(&self, words: &mut Vec<Word>) {
        match self.mode {
            FillerMode::KeepAll => {}
            FillerMode::RemoveConfident => {
                remove_fillers(words, CONFIDENT_FILLERS, &[]);
            }
            FillerMode::RemoveAll => {
                let all_single: Vec<&str> = CONFIDENT_FILLERS
                    .iter()
                    .chain(EXTENDED_SINGLE_FILLERS.iter())
                    .copied()
                    .collect();
                remove_fillers(words, &all_single, MULTI_WORD_FILLERS);
            }
        }
    }
}

fn is_single_filler(word: &str, fillers: &[&str]) -> bool {
    let lower = word.to_lowercase();
    fillers.iter().any(|f| *f == lower)
}

fn remove_fillers(words: &mut Vec<Word>, single_fillers: &[&str], multi_fillers: &[(&str, &str)]) {
    let mut i = 0;
    while i < words.len() {
        if i + 1 < words.len() {
            let first_lower = words[i].text.to_lowercase();
            let second_lower = words[i + 1].text.to_lowercase();
            let is_multi = multi_fillers
                .iter()
                .any(|(a, b)| *a == first_lower && *b == second_lower);
            if is_multi {
                if i > 0 && i + 2 < words.len() {
                    words[i - 1].end = words[i + 2].start;
                } else if i > 0 {
                    words[i - 1].end = words[i + 1].end;
                }
                words.remove(i + 1);
                words.remove(i);
                continue;
            }
        }

        if is_single_filler(&words[i].text, single_fillers) {
            if i > 0 && i + 1 < words.len() {
                words[i - 1].end = words[i + 1].start;
            } else if i > 0 {
                words[i - 1].end = words[i].end;
            }
            words.remove(i);
            continue;
        }

        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::Word;

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
            confidence: 0.99,
        }
    }

    #[test]
    fn remove_confident_removes_um_and_uh() {
        let mut words = vec![
            word("um", 0.0, 0.3),
            word("hello", 0.4, 0.8),
            word("uh", 0.9, 1.1),
            word("world", 1.2, 1.6),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn remove_confident_keeps_like_and_so() {
        let mut words = vec![
            word("like", 0.0, 0.3),
            word("hello", 0.4, 0.8),
            word("so", 0.9, 1.1),
            word("yeah", 1.2, 1.5),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 4);
        assert_eq!(words[0].text, "like");
        assert_eq!(words[2].text, "so");
    }

    #[test]
    fn remove_all_removes_like() {
        let mut words = vec![word("like", 0.0, 0.3), word("hello", 0.4, 0.8)];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
    }

    #[test]
    fn keep_all_modifies_nothing() {
        let mut words = vec![
            word("um", 0.0, 0.3),
            word("like", 0.4, 0.8),
            word("hello", 0.9, 1.3),
        ];
        let remover = FillerRemover {
            mode: FillerMode::KeepAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "um");
        assert_eq!(words[1].text, "like");
        assert_eq!(words[2].text, "hello");
    }

    #[test]
    fn removing_filler_closes_timestamp_gap() {
        let mut words = vec![
            word("hello", 0.0, 0.5),
            word("um", 0.6, 0.9),
            word("world", 1.0, 1.5),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].end - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn removing_first_word_filler_does_not_adjust_timeline() {
        let mut words = vec![word("um", 0.0, 0.3), word("hello", 0.4, 0.8)];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
        assert!((words[0].start - 0.4).abs() < f64::EPSILON);
        assert!((words[0].end - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn case_insensitive_filler_removal() {
        let mut words = vec![
            word("Um", 0.0, 0.3),
            word("hello", 0.4, 0.8),
            word("UM", 0.9, 1.1),
            word("world", 1.2, 1.5),
            word("um", 1.6, 1.8),
            word("yeah", 1.9, 2.2),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert_eq!(words[2].text, "yeah");
    }

    #[test]
    fn empty_word_list_stays_empty() {
        let mut words: Vec<Word> = vec![];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert!(words.is_empty());
    }

    #[test]
    fn multi_word_filler_you_know_removed_as_unit() {
        let mut words = vec![
            word("hey", 0.0, 0.3),
            word("you", 0.4, 0.6),
            word("know", 0.7, 0.9),
            word("it", 1.0, 1.2),
            word("works", 1.3, 1.6),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "hey");
        assert_eq!(words[1].text, "it");
        assert_eq!(words[2].text, "works");
        assert!((words[0].end - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn multi_word_filler_i_mean_removed_as_unit() {
        let mut words = vec![
            word("so", 0.0, 0.2),
            word("I", 0.3, 0.4),
            word("mean", 0.5, 0.7),
            word("it", 0.8, 1.0),
            word("works", 1.1, 1.4),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "it");
        assert_eq!(words[1].text, "works");
    }

    #[test]
    fn multi_word_filler_i_mean_isolated() {
        let mut words = vec![
            word("hello", 0.0, 0.3),
            word("I", 0.4, 0.5),
            word("mean", 0.6, 0.8),
            word("world", 0.9, 1.2),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].end - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn multi_word_filler_case_insensitive() {
        let mut words = vec![
            word("hello", 0.0, 0.3),
            word("You", 0.4, 0.6),
            word("Know", 0.7, 0.9),
            word("world", 1.0, 1.3),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn multi_word_filler_at_end_of_list() {
        let mut words = vec![
            word("hello", 0.0, 0.5),
            word("you", 0.6, 0.8),
            word("know", 0.9, 1.1),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
        assert!((words[0].end - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn multi_word_filler_at_start_of_list() {
        let mut words = vec![
            word("you", 0.0, 0.2),
            word("know", 0.3, 0.5),
            word("hello", 0.6, 1.0),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveAll,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
        assert!((words[0].start - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn removing_last_word_filler_extends_previous() {
        let mut words = vec![word("hello", 0.0, 0.5), word("um", 0.6, 0.9)];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
        assert!((words[0].end - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn default_mode_is_remove_confident() {
        let mode = FillerMode::default();
        assert!(matches!(mode, FillerMode::RemoveConfident));
    }

    #[test]
    fn consecutive_fillers_all_removed() {
        let mut words = vec![
            word("hello", 0.0, 0.5),
            word("um", 0.6, 0.8),
            word("uh", 0.9, 1.1),
            word("world", 1.2, 1.6),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].end - 1.2).abs() < f64::EPSILON);
    }

    #[test]
    fn all_words_are_fillers() {
        let mut words = vec![
            word("um", 0.0, 0.3),
            word("uh", 0.4, 0.6),
            word("hmm", 0.7, 1.0),
        ];
        let remover = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        remover.process(&mut words);

        assert!(words.is_empty());
    }
}
