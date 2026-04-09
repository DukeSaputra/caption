pub mod model;
#[cfg(feature = "parakeet")]
pub mod parakeet_backend;
pub mod whisper_backend;

use crate::error::CaptionError;

#[derive(Debug, Clone)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct Transcription {
    pub words: Vec<Word>,
    pub language: String,
}

#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    pub language: String,
    pub initial_prompt: Option<String>,
    pub temperature: f32,
    pub vad_model_path: Option<String>,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            initial_prompt: None,
            temperature: 0.0,
            vad_model_path: None,
        }
    }
}

pub trait SpeechRecognizer: Send + Sync {
    fn transcribe(
        &self,
        audio: &[f32],
        config: &TranscribeConfig,
    ) -> Result<Transcription, CaptionError>;

    fn supports_word_timestamps(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_config_defaults() {
        let config = TranscribeConfig::default();
        assert_eq!(config.language, "en");
        assert!(config.initial_prompt.is_none());
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert!(config.vad_model_path.is_none());
    }

    #[test]
    fn transcribe_config_with_vad_path() {
        let config = TranscribeConfig {
            vad_model_path: Some("/path/to/silero_vad.onnx".to_string()),
            ..TranscribeConfig::default()
        };
        assert_eq!(
            config.vad_model_path.as_deref(),
            Some("/path/to/silero_vad.onnx")
        );
        assert_eq!(config.language, "en");
        assert!(config.initial_prompt.is_none());
    }

    #[test]
    fn word_clone_preserves_fields() {
        let word = Word {
            text: "hello".to_string(),
            start: 1.5,
            end: 2.0,
            confidence: 0.95,
        };
        let cloned = word.clone();
        assert_eq!(cloned.text, "hello");
        assert!((cloned.start - 1.5).abs() < f64::EPSILON);
        assert!((cloned.end - 2.0).abs() < f64::EPSILON);
        assert!((cloned.confidence - 0.95).abs() < f32::EPSILON);
    }

    struct MockRecognizer {
        words: Vec<Word>,
    }

    impl SpeechRecognizer for MockRecognizer {
        fn transcribe(
            &self,
            _audio: &[f32],
            _config: &TranscribeConfig,
        ) -> Result<Transcription, CaptionError> {
            Ok(Transcription {
                words: self.words.clone(),
                language: "en".to_string(),
            })
        }

        fn supports_word_timestamps(&self) -> bool {
            true
        }
    }

    #[test]
    fn mock_recognizer_returns_fixed_words() {
        let mock = MockRecognizer {
            words: vec![
                Word {
                    text: "hello".to_string(),
                    start: 0.0,
                    end: 0.5,
                    confidence: 0.99,
                },
                Word {
                    text: "world".to_string(),
                    start: 0.5,
                    end: 1.0,
                    confidence: 0.98,
                },
            ],
        };

        let config = TranscribeConfig::default();
        let result = mock.transcribe(&[], &config).unwrap();
        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[0].text, "hello");
        assert_eq!(result.words[1].text, "world");
        assert_eq!(result.language, "en");
        assert!(mock.supports_word_timestamps());
    }

    #[test]
    fn mock_recognizer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockRecognizer>();
    }

    #[test]
    fn speech_recognizer_trait_is_object_safe() {
        let mock: Box<dyn SpeechRecognizer> = Box::new(MockRecognizer { words: vec![] });
        let config = TranscribeConfig::default();
        let result = mock.transcribe(&[], &config).unwrap();
        assert!(result.words.is_empty());
    }
}
