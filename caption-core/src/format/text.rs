use crate::error::CaptionError;
use crate::stt::Word;

use super::{FormatConfig, SubtitleFormatter};

pub struct TextFormatter;

impl SubtitleFormatter for TextFormatter {
    fn format(&self, words: &[Word], _config: &FormatConfig) -> Result<String, CaptionError> {
        if words.is_empty() {
            return Ok(String::new());
        }

        let mut output = String::new();
        let mut after_newline = false;

        for (i, word) in words.iter().enumerate() {
            if i > 0 && !after_newline {
                output.push(' ');
            }
            after_newline = false;

            output.push_str(&word.text);

            if word.text.ends_with('.') || word.text.ends_with('?') || word.text.ends_with('!') {
                output.push('\n');
                after_newline = true;
            }
        }

        if !output.ends_with('\n') {
            output.push('\n');
        }

        Ok(output)
    }

    fn file_extension(&self) -> &str {
        "txt"
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

    #[test]
    fn simple_sentence() {
        let words = vec![word("Hello", 0.0, 0.3), word("world.", 0.4, 0.8)];
        let config = FormatConfig::default();
        let text = TextFormatter.format(&words, &config).unwrap();
        assert_eq!(text, "Hello world.\n");
    }

    #[test]
    fn multiple_sentences() {
        let words = vec![
            word("Hi.", 0.0, 0.3),
            word("How", 0.5, 0.7),
            word("are", 0.8, 1.0),
            word("you?", 1.1, 1.5),
        ];
        let config = FormatConfig::default();
        let text = TextFormatter.format(&words, &config).unwrap();
        assert_eq!(text, "Hi.\nHow are you?\n");
    }

    #[test]
    fn no_sentence_ending() {
        let words = vec![word("hello", 0.0, 0.3), word("world", 0.4, 0.8)];
        let config = FormatConfig::default();
        let text = TextFormatter.format(&words, &config).unwrap();
        assert_eq!(text, "hello world\n");
    }

    #[test]
    fn empty_words() {
        let words: Vec<Word> = vec![];
        let config = FormatConfig::default();
        let text = TextFormatter.format(&words, &config).unwrap();
        assert_eq!(text, "");
    }

    #[test]
    fn single_word() {
        let words = vec![word("hello", 0.0, 0.5)];
        let config = FormatConfig::default();
        let text = TextFormatter.format(&words, &config).unwrap();
        assert_eq!(text, "hello\n");
    }

    #[test]
    fn exclamation_creates_newline() {
        let words = vec![word("Wow!", 0.0, 0.3), word("Cool.", 0.5, 0.8)];
        let config = FormatConfig::default();
        let text = TextFormatter.format(&words, &config).unwrap();
        assert_eq!(text, "Wow!\nCool.\n");
    }

    #[test]
    fn file_extension_is_txt() {
        assert_eq!(TextFormatter.file_extension(), "txt");
    }
}
