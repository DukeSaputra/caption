use crate::postprocess::PostProcessor;
use crate::stt::Word;

const STRIP_CHARS: &[char] = &['.', ',', ';', ':'];

pub struct PunctuationStripper;

impl PostProcessor for PunctuationStripper {
    fn process(&self, words: &mut Vec<Word>) {
        for word in words.iter_mut() {
            let trimmed = word.text.trim_end_matches(STRIP_CHARS);
            if trimmed.len() != word.text.len() {
                word.text = trimmed.to_string();
            }
        }
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
    fn strips_trailing_period() {
        let mut words = vec![word("hello.", 0.0, 0.5)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "hello");
    }

    #[test]
    fn strips_trailing_comma() {
        let mut words = vec![word("so,", 0.0, 0.3)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "so");
    }

    #[test]
    fn strips_multiple_trailing() {
        let mut words = vec![word("ok..", 0.0, 0.3)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "ok");
    }

    #[test]
    fn keeps_exclamation() {
        let mut words = vec![word("wow!", 0.0, 0.3)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "wow!");
    }

    #[test]
    fn keeps_question_mark() {
        let mut words = vec![word("what?", 0.0, 0.3)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "what?");
    }

    #[test]
    fn keeps_hyphen() {
        let mut words = vec![
            word("8", 0.0, 0.2),
            word("-", 0.3, 0.4),
            word("10", 0.5, 0.7),
        ];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[1].text, "-");
    }

    #[test]
    fn preserves_timestamps() {
        let mut words = vec![word("end.", 1.5, 2.0)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "end");
        assert!((words[0].start - 1.5).abs() < f64::EPSILON);
        assert!((words[0].end - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn no_change_for_clean_word() {
        let mut words = vec![word("hello", 0.0, 0.5)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "hello");
    }

    #[test]
    fn empty_word_list() {
        let mut words: Vec<Word> = vec![];
        PunctuationStripper.process(&mut words);
        assert!(words.is_empty());
    }

    #[test]
    fn strips_trailing_semicolon() {
        let mut words = vec![word("right;", 0.0, 0.3)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "right");
    }

    #[test]
    fn strips_trailing_colon() {
        let mut words = vec![word("here:", 0.0, 0.3)];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "here");
    }

    #[test]
    fn mixed_punctuation() {
        let mut words = vec![
            word("first,", 0.0, 0.3),
            word("second.", 0.4, 0.7),
            word("third!", 0.8, 1.1),
            word("fourth?", 1.2, 1.5),
        ];
        PunctuationStripper.process(&mut words);
        assert_eq!(words[0].text, "first");
        assert_eq!(words[1].text, "second");
        assert_eq!(words[2].text, "third!");
        assert_eq!(words[3].text, "fourth?");
    }
}
