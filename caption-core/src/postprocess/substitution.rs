use std::collections::HashMap;

use crate::postprocess::PostProcessor;
use crate::stt::Word;

pub struct SubstitutionMap {
    pub replacements: HashMap<String, String>,
}

impl SubstitutionMap {
    pub fn with_defaults() -> Self {
        Self {
            replacements: default_substitutions(),
        }
    }

    pub fn empty() -> Self {
        Self {
            replacements: HashMap::new(),
        }
    }
}

impl PostProcessor for SubstitutionMap {
    fn process(&self, words: &mut Vec<Word>) {
        if self.replacements.is_empty() {
            return;
        }

        let mut result: Vec<Word> = Vec::with_capacity(words.len());

        for word in words.drain(..) {
            let key = word.text.to_lowercase();
            match self.replacements.get(&key) {
                None => result.push(word),
                Some(replacement) => {
                    let parts: Vec<&str> = replacement.split_whitespace().collect();
                    match parts.len() {
                        0 => {
                            result.push(word);
                        }
                        1 => {
                            result.push(Word {
                                text: replacement.clone(),
                                start: word.start,
                                end: word.end,
                                confidence: word.confidence,
                            });
                        }
                        _ => {
                            let expanded = split_into_words(&word, &parts);
                            result.extend(expanded);
                        }
                    }
                }
            }
        }

        *words = result;
    }
}

fn split_into_words(original: &Word, parts: &[&str]) -> Vec<Word> {
    let total_chars: usize = parts.iter().map(|p| p.len()).sum();
    let span = original.end - original.start;

    let mut result = Vec::with_capacity(parts.len());
    let mut cursor = original.start;

    for (i, part) in parts.iter().enumerate() {
        let is_last = i == parts.len() - 1;

        let end = if is_last {
            original.end
        } else {
            let weight = if total_chars == 0 {
                1.0 / parts.len() as f64
            } else {
                part.len() as f64 / total_chars as f64
            };
            cursor + span * weight
        };

        result.push(Word {
            text: part.to_string(),
            start: cursor,
            end,
            confidence: original.confidence,
        });

        cursor = end;
    }

    result
}

fn default_substitutions() -> HashMap<String, String> {
    let mut map = HashMap::new();

    map.insert("gonna".to_string(), "going to".to_string());
    map.insert("wanna".to_string(), "want to".to_string());
    map.insert("gotta".to_string(), "got to".to_string());
    map.insert("kinda".to_string(), "kind of".to_string());
    map.insert("sorta".to_string(), "sort of".to_string());
    map.insert("coulda".to_string(), "could have".to_string());
    map.insert("woulda".to_string(), "would have".to_string());
    map.insert("shoulda".to_string(), "should have".to_string());
    map.insert("dunno".to_string(), "don't know".to_string());
    map.insert("lemme".to_string(), "let me".to_string());
    map.insert("gimme".to_string(), "give me".to_string());
    map.insert("outta".to_string(), "out of".to_string());
    map.insert("tryna".to_string(), "trying to".to_string());
    map.insert("hafta".to_string(), "have to".to_string());

    map
}

pub struct HotWords {
    pub vocab_hints: Vec<String>,
    pub substitutions: HashMap<String, String>,
}

pub fn parse_hot_words(content: &str) -> HotWords {
    let mut vocab_hints = Vec::new();
    let mut substitutions = HashMap::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(arrow_pos) = trimmed.find("->") {
            let source = trimmed[..arrow_pos].trim().to_lowercase();
            let replacement = trimmed[arrow_pos + 2..].trim().to_string();

            if !source.is_empty() {
                substitutions.insert(source, replacement);
            }
        } else {
            let hint = trimmed.to_string();
            if !hint.is_empty() {
                vocab_hints.push(hint);
            }
        }
    }

    HotWords {
        vocab_hints,
        substitutions,
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
            confidence: 0.9,
        }
    }

    // -------------------------------------------------------------------------
    // SubstitutionMap tests
    // -------------------------------------------------------------------------

    #[test]
    fn simple_replacement_works() {
        let mut map = HashMap::new();
        map.insert("hello".to_string(), "hi".to_string());
        let processor = SubstitutionMap { replacements: map };

        let mut words = vec![word("hello", 0.0, 0.5), word("world", 0.6, 1.0)];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hi");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn replacement_is_case_insensitive_on_source() {
        let mut map = HashMap::new();
        map.insert("hello".to_string(), "hi".to_string());
        let processor = SubstitutionMap { replacements: map };

        let mut words = vec![
            word("Hello", 0.0, 0.5),
            word("HELLO", 0.6, 1.0),
            word("hElLo", 1.1, 1.5),
        ];
        processor.process(&mut words);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "hi");
        assert_eq!(words[1].text, "hi");
        assert_eq!(words[2].text, "hi");
    }

    #[test]
    fn unknown_words_are_unchanged() {
        let mut map = HashMap::new();
        map.insert("hello".to_string(), "hi".to_string());
        let processor = SubstitutionMap { replacements: map };

        let mut words = vec![word("goodbye", 0.0, 0.5), word("world", 0.6, 1.0)];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "goodbye");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn empty_map_is_no_op() {
        let processor = SubstitutionMap::empty();

        let mut words = vec![word("gonna", 0.0, 0.4), word("run", 0.5, 0.8)];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "gonna");
        assert_eq!(words[1].text, "run");
    }

    #[test]
    fn gonna_expands_to_two_words_with_proportional_timestamps() {
        let processor = SubstitutionMap::with_defaults();

        let mut words = vec![word("gonna", 0.0, 0.7)];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "going");
        assert_eq!(words[1].text, "to");

        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!((words[1].end - 0.7).abs() < 1e-9);

        assert!((words[0].end - words[1].start).abs() < 1e-9);

        // "going" = 5 chars, "to" = 2 chars, total = 7
        let expected_going_end = 0.7 * 5.0 / 7.0;
        assert!((words[0].end - expected_going_end).abs() < 1e-9);
    }

    #[test]
    fn wanna_expands_to_want_to() {
        let processor = SubstitutionMap::with_defaults();

        let mut words = vec![word("wanna", 0.0, 1.0)];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "want");
        assert_eq!(words[1].text, "to");
        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!((words[1].end - 1.0).abs() < 1e-9);
    }

    #[test]
    fn gotta_expands_to_got_to() {
        let processor = SubstitutionMap::with_defaults();

        let mut words = vec![word("gotta", 0.0, 0.6)];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "got");
        assert_eq!(words[1].text, "to");
        assert!((words[1].end - 0.6).abs() < 1e-9);
    }

    #[test]
    fn expansion_timestamps_are_contiguous_and_non_overlapping() {
        let mut map = HashMap::new();
        map.insert("one".to_string(), "alpha beta gamma".to_string());
        let processor = SubstitutionMap { replacements: map };

        let mut words = vec![word("one", 0.0, 1.5)];
        processor.process(&mut words);

        assert_eq!(words.len(), 3);

        assert!((words[0].end - words[1].start).abs() < 1e-9);
        assert!((words[1].end - words[2].start).abs() < 1e-9);

        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!((words[2].end - 1.5).abs() < 1e-9);
    }

    #[test]
    fn confidence_is_inherited_from_original() {
        let mut map = HashMap::new();
        map.insert("gonna".to_string(), "going to".to_string());
        let processor = SubstitutionMap { replacements: map };

        let orig = Word {
            text: "gonna".to_string(),
            start: 0.0,
            end: 0.5,
            confidence: 0.75,
        };
        let mut words = vec![orig];
        processor.process(&mut words);

        assert_eq!(words.len(), 2);
        assert!((words[0].confidence - 0.75).abs() < f32::EPSILON);
        assert!((words[1].confidence - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn mixed_replacements_and_unknowns() {
        let processor = SubstitutionMap::with_defaults();

        let mut words = vec![
            word("I", 0.0, 0.1),
            word("gonna", 0.2, 0.5),
            word("run", 0.6, 0.9),
        ];
        processor.process(&mut words);

        assert_eq!(words.len(), 4);
        assert_eq!(words[0].text, "I");
        assert_eq!(words[1].text, "going");
        assert_eq!(words[2].text, "to");
        assert_eq!(words[3].text, "run");
    }

    #[test]
    fn empty_word_list_stays_empty() {
        let processor = SubstitutionMap::with_defaults();

        let mut words: Vec<Word> = vec![];
        processor.process(&mut words);

        assert!(words.is_empty());
    }

    // -------------------------------------------------------------------------
    // parse_hot_words tests
    // -------------------------------------------------------------------------

    #[test]
    fn hot_words_parses_vocab_hints() {
        let content = "arc\nRSVP\nwhisper\n";
        let hot = parse_hot_words(content);

        assert_eq!(hot.vocab_hints, vec!["arc", "RSVP", "whisper"]);
        assert!(hot.substitutions.is_empty());
    }

    #[test]
    fn hot_words_parses_substitution_rules() {
        let content = "ARK -> arc\ngonna -> going to\n";
        let hot = parse_hot_words(content);

        assert!(hot.vocab_hints.is_empty());
        assert_eq!(
            hot.substitutions.get("ark").map(String::as_str),
            Some("arc")
        );
        assert_eq!(
            hot.substitutions.get("gonna").map(String::as_str),
            Some("going to")
        );
    }

    #[test]
    fn hot_words_ignores_comments() {
        let content = "# This is a comment\narc\n# Another comment\nRSVP\n";
        let hot = parse_hot_words(content);

        assert_eq!(hot.vocab_hints, vec!["arc", "RSVP"]);
        assert!(hot.substitutions.is_empty());
    }

    #[test]
    fn hot_words_ignores_blank_lines() {
        let content = "arc\n\n\nRSVP\n\n";
        let hot = parse_hot_words(content);

        assert_eq!(hot.vocab_hints, vec!["arc", "RSVP"]);
    }

    #[test]
    fn hot_words_mixed_hints_and_rules() {
        let content = r"
# Domain-specific vocabulary biasing
arc
RSVP

# Substitution rules
ARK -> arc
gonna -> going to
";
        let hot = parse_hot_words(content);

        assert_eq!(hot.vocab_hints, vec!["arc", "RSVP"]);
        assert_eq!(
            hot.substitutions.get("ark").map(String::as_str),
            Some("arc")
        );
        assert_eq!(
            hot.substitutions.get("gonna").map(String::as_str),
            Some("going to")
        );
    }

    #[test]
    fn hot_words_source_is_lowercased_for_matching() {
        let content = "ARK -> arc\n";
        let hot = parse_hot_words(content);

        assert!(hot.substitutions.contains_key("ark"));
        assert!(!hot.substitutions.contains_key("ARK"));
    }

    #[test]
    fn hot_words_empty_content_returns_empty() {
        let hot = parse_hot_words("");

        assert!(hot.vocab_hints.is_empty());
        assert!(hot.substitutions.is_empty());
    }

    #[test]
    fn hot_words_whitespace_only_content_returns_empty() {
        let hot = parse_hot_words("   \n   \n\t\n");

        assert!(hot.vocab_hints.is_empty());
        assert!(hot.substitutions.is_empty());
    }

    #[test]
    fn hot_words_comments_only_returns_empty() {
        let hot = parse_hot_words("# comment one\n# comment two\n");

        assert!(hot.vocab_hints.is_empty());
        assert!(hot.substitutions.is_empty());
    }

    #[test]
    fn hot_words_substitution_map_is_case_insensitive_at_process_time() {
        let content = "ARK -> arc\n";
        let hot = parse_hot_words(content);

        let processor = SubstitutionMap {
            replacements: hot.substitutions,
        };

        let mut words = vec![word("Ark", 0.0, 0.5)];
        processor.process(&mut words);

        assert_eq!(words[0].text, "arc");
    }
}
