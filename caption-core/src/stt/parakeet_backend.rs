use std::path::Path;

use log::debug;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineRecognizerResult,
    OfflineTransducerModelConfig,
};

use super::{SpeechRecognizer, TranscribeConfig, Transcription, Word};
use crate::error::CaptionError;

pub struct ParakeetBackend {
    recognizer: OfflineRecognizer,
}

// SAFETY: sherpa-onnx's OfflineRecognizer wraps a C pointer to the ONNX Runtime,
// which uses internal locking for thread safety. The recognizer is safe to share
// across threads. The raw pointer in OfflineRecognizer is an opaque handle that
// sherpa-onnx manages; we never dereference it directly.
unsafe impl Send for ParakeetBackend {}
unsafe impl Sync for ParakeetBackend {}

impl std::fmt::Debug for ParakeetBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParakeetBackend").finish_non_exhaustive()
    }
}

impl ParakeetBackend {
    pub fn new(model_dir: &Path) -> Result<Self, CaptionError> {
        if !model_dir.is_dir() {
            return Err(CaptionError::ModelNotFound(model_dir.display().to_string()));
        }

        let encoder = find_onnx_file(model_dir, "encoder")?;
        let decoder = find_onnx_file(model_dir, "decoder")?;
        let joiner = find_onnx_file(model_dir, "joiner")?;

        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.is_file() {
            return Err(CaptionError::ModelNotFound(format!(
                "tokens.txt not found in {}",
                model_dir.display()
            )));
        }

        let tokens_str = path_to_string(&tokens_path)?;
        let encoder_str = path_to_string(&encoder)?;
        let decoder_str = path_to_string(&decoder)?;
        let joiner_str = path_to_string(&joiner)?;

        debug!("Loading Parakeet TDT model from: {}", model_dir.display());
        debug!("  encoder: {encoder_str}");
        debug!("  decoder: {decoder_str}");
        debug!("  joiner:  {joiner_str}");
        debug!("  tokens:  {tokens_str}");

        let config = OfflineRecognizerConfig {
            model_config: OfflineModelConfig {
                transducer: OfflineTransducerModelConfig {
                    encoder: Some(encoder_str),
                    decoder: Some(decoder_str),
                    joiner: Some(joiner_str),
                },
                tokens: Some(tokens_str),
                num_threads: 4,
                debug: false,
                provider: Some("cpu".to_string()),
                model_type: Some("transducer".to_string()),
                ..Default::default()
            },
            decoding_method: Some("greedy_search".to_string()),
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            CaptionError::ModelLoadFailed(format!(
                "sherpa-onnx failed to create recognizer from '{}'",
                model_dir.display()
            ))
        })?;

        Ok(Self { recognizer })
    }
}

impl SpeechRecognizer for ParakeetBackend {
    fn transcribe(
        &self,
        audio: &[f32],
        config: &TranscribeConfig,
    ) -> Result<Transcription, CaptionError> {
        debug!(
            "Running Parakeet TDT transcription on {} samples ({:.2}s of audio)",
            audio.len(),
            audio.len() as f64 / 16_000.0
        );

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16_000, audio);
        self.recognizer.decode(&stream);

        let result = stream.get_result().ok_or_else(|| {
            CaptionError::TranscriptionFailed(
                "sherpa-onnx returned no result after decoding".to_string(),
            )
        })?;

        if result.tokens.is_empty() {
            debug!("Parakeet TDT: no tokens produced (no speech detected)");
            return Ok(Transcription {
                words: Vec::new(),
                language: config.language.clone(),
            });
        }

        let words = join_parakeet_tokens_into_words(&result);

        debug!("Extracted {} words from Parakeet token data", words.len());

        Ok(Transcription {
            words,
            language: config.language.clone(),
        })
    }

    fn supports_word_timestamps(&self) -> bool {
        true
    }
}

// U+2581 (SentencePiece lower one-eighth block) marks word boundaries in Parakeet's tokenizer
const WORD_BOUNDARY_CHAR: char = '\u{2581}';

pub(crate) fn join_parakeet_tokens_into_words(result: &OfflineRecognizerResult) -> Vec<Word> {
    let timestamps = result.timestamps.as_deref().unwrap_or(&[]);
    let durations = result.durations.as_deref().unwrap_or(&[]);

    let mut words: Vec<Word> = Vec::new();
    let mut current_text = String::new();
    let mut word_start: f64 = 0.0;
    let mut word_end: f64 = 0.0;

    for (i, token) in result.tokens.iter().enumerate() {
        let starts_new_word = token.starts_with(WORD_BOUNDARY_CHAR);

        if starts_new_word && !current_text.is_empty() {
            let trimmed = current_text.trim().to_string();
            if !trimmed.is_empty() {
                words.push(Word {
                    text: trimmed,
                    start: word_start,
                    end: word_end,
                    confidence: 1.0,
                });
            }
            current_text.clear();
        }

        let clean = token.trim_start_matches(WORD_BOUNDARY_CHAR);

        if current_text.is_empty() {
            if i < timestamps.len() {
                word_start = timestamps[i] as f64;
            }
        }

        if i < timestamps.len() && i < durations.len() {
            word_end = (timestamps[i] + durations[i]) as f64;
        } else if i < timestamps.len() {
            word_end = timestamps[i] as f64;
        }

        current_text.push_str(clean);
    }

    let trimmed = current_text.trim().to_string();
    if !trimmed.is_empty() {
        words.push(Word {
            text: trimmed,
            start: word_start,
            end: word_end,
            confidence: 1.0,
        });
    }

    words
}

fn find_onnx_file(dir: &Path, prefix: &str) -> Result<std::path::PathBuf, CaptionError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        CaptionError::ModelNotFound(format!(
            "cannot read model directory '{}': {e}",
            dir.display()
        ))
    })?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(prefix) && name_str.ends_with(".onnx") && entry.path().is_file() {
            debug!("Found {prefix} ONNX file: {name_str}");
            return Ok(entry.path());
        }
    }

    Err(CaptionError::ModelNotFound(format!(
        "no {prefix}*.onnx file found in {}",
        dir.display()
    )))
}

fn path_to_string(path: &Path) -> Result<String, CaptionError> {
    path.to_str().map(|s| s.to_string()).ok_or_else(|| {
        CaptionError::ModelLoadFailed(format!("path '{}' contains invalid UTF-8", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ParakeetBackend Send + Sync ---

    #[test]
    fn parakeet_backend_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ParakeetBackend>();
    }

    // --- Token-to-word joining tests ---

    #[test]
    fn simple_words_with_sentencepiece_boundary() {
        let result = OfflineRecognizerResult {
            text: "Hello world".to_string(),
            tokens: vec!["\u{2581}Hello".to_string(), "\u{2581}world".to_string()],
            timestamps: Some(vec![0.0, 0.5]),
            durations: Some(vec![0.5, 0.5]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((words[0].end - 0.5).abs() < f64::EPSILON);
        assert!((words[1].start - 0.5).abs() < f64::EPSILON);
        assert!((words[1].end - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn subword_tokens_join_into_single_word() {
        let result = OfflineRecognizerResult {
            text: "unbelievable".to_string(),
            tokens: vec![
                "\u{2581}un".to_string(),
                "believ".to_string(),
                "able".to_string(),
            ],
            timestamps: Some(vec![1.0, 1.2, 1.6]),
            durations: Some(vec![0.2, 0.4, 0.4]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "unbelievable");
        assert!((words[0].start - 1.0).abs() < f64::EPSILON);
        assert!((words[0].end - 2.0).abs() < f64::EPSILON);
        assert!((words[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn single_token_word() {
        let result = OfflineRecognizerResult {
            text: "I".to_string(),
            tokens: vec!["\u{2581}I".to_string()],
            timestamps: Some(vec![2.0]),
            durations: Some(vec![0.3]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "I");
        assert!((words[0].start - 2.0).abs() < f64::EPSILON);
        assert!((words[0].end - 2.3).abs() < 1e-6);
    }

    #[test]
    fn empty_tokens_produce_no_words() {
        let result = OfflineRecognizerResult {
            text: String::new(),
            tokens: vec![],
            timestamps: Some(vec![]),
            durations: Some(vec![]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert!(words.is_empty());
    }

    #[test]
    fn mixed_simple_and_subword_tokens() {
        let result = OfflineRecognizerResult {
            text: "Hello unbelievable world".to_string(),
            tokens: vec![
                "\u{2581}Hello".to_string(),
                "\u{2581}un".to_string(),
                "believ".to_string(),
                "able".to_string(),
                "\u{2581}world".to_string(),
            ],
            timestamps: Some(vec![0.0, 0.5, 0.7, 1.0, 1.3]),
            durations: Some(vec![0.5, 0.2, 0.3, 0.3, 0.5]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "unbelievable");
        assert_eq!(words[2].text, "world");
        assert!((words[1].start - 0.5).abs() < f64::EPSILON);
        assert!((words[1].end - 1.3).abs() < 1e-6);
    }

    #[test]
    fn missing_timestamps_and_durations_graceful() {
        let result = OfflineRecognizerResult {
            text: "Hello world".to_string(),
            tokens: vec!["\u{2581}Hello".to_string(), "\u{2581}world".to_string()],
            timestamps: None,
            durations: None,
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
        assert!((words[0].start - 0.0).abs() < f64::EPSILON);
        assert!((words[0].end - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn durations_missing_but_timestamps_present() {
        let result = OfflineRecognizerResult {
            text: "Hello".to_string(),
            tokens: vec!["\u{2581}Hello".to_string()],
            timestamps: Some(vec![1.5]),
            durations: None,
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "Hello");
        assert!((words[0].start - 1.5).abs() < f64::EPSILON);
        assert!((words[0].end - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn all_confidence_values_are_one() {
        let result = OfflineRecognizerResult {
            text: "one two three".to_string(),
            tokens: vec![
                "\u{2581}one".to_string(),
                "\u{2581}two".to_string(),
                "\u{2581}three".to_string(),
            ],
            timestamps: Some(vec![0.0, 0.3, 0.6]),
            durations: Some(vec![0.3, 0.3, 0.4]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        for word in &words {
            assert!(
                (word.confidence - 1.0).abs() < f32::EPSILON,
                "Expected confidence 1.0, got {}",
                word.confidence
            );
        }
    }

    #[test]
    fn tokens_without_boundary_marker_at_start() {
        let result = OfflineRecognizerResult {
            text: "hello world".to_string(),
            tokens: vec!["hello".to_string(), "\u{2581}world".to_string()],
            timestamps: Some(vec![0.0, 0.5]),
            durations: Some(vec![0.5, 0.5]),
        };
        let words = join_parakeet_tokens_into_words(&result);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    // --- ONNX file discovery tests ---

    #[test]
    fn find_onnx_file_finds_matching_file() {
        let dir = tempfile::tempdir().unwrap();
        let encoder_path = dir.path().join("encoder-epoch-999-avg-1.int8.onnx");
        std::fs::write(&encoder_path, b"fake").unwrap();

        let result = find_onnx_file(dir.path(), "encoder");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), encoder_path);
    }

    #[test]
    fn find_onnx_file_errors_on_missing() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_onnx_file(dir.path(), "encoder");
        assert!(matches!(result, Err(CaptionError::ModelNotFound(_))));
    }

    #[test]
    fn find_onnx_file_ignores_non_onnx_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("encoder-readme.txt"), b"not onnx").unwrap();

        let result = find_onnx_file(dir.path(), "encoder");
        assert!(matches!(result, Err(CaptionError::ModelNotFound(_))));
    }

    #[test]
    fn find_onnx_file_ignores_wrong_prefix() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("decoder-epoch-1.onnx"), b"fake").unwrap();

        let result = find_onnx_file(dir.path(), "encoder");
        assert!(matches!(result, Err(CaptionError::ModelNotFound(_))));
    }

    // --- Model directory validation tests ---

    #[test]
    fn new_nonexistent_dir_returns_model_not_found() {
        let result = ParakeetBackend::new(Path::new("/tmp/no-such-parakeet-dir"));
        assert!(
            matches!(result, Err(CaptionError::ModelNotFound(_))),
            "Expected ModelNotFound, got: {result:?}"
        );
    }

    #[test]
    fn new_missing_encoder_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("decoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(dir.path().join("joiner-epoch-999-avg-1.int8.onnx"), b"fake").unwrap();
        std::fs::write(dir.path().join("tokens.txt"), b"fake").unwrap();

        let result = ParakeetBackend::new(dir.path());
        match result {
            Err(CaptionError::ModelNotFound(msg)) => {
                assert!(
                    msg.contains("encoder"),
                    "Error should mention 'encoder', got: {msg}"
                );
            }
            other => panic!("Expected ModelNotFound with 'encoder', got: {other:?}"),
        }
    }

    #[test]
    fn new_missing_decoder_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("encoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(dir.path().join("joiner-epoch-999-avg-1.int8.onnx"), b"fake").unwrap();
        std::fs::write(dir.path().join("tokens.txt"), b"fake").unwrap();

        let result = ParakeetBackend::new(dir.path());
        match result {
            Err(CaptionError::ModelNotFound(msg)) => {
                assert!(
                    msg.contains("decoder"),
                    "Error should mention 'decoder', got: {msg}"
                );
            }
            other => panic!("Expected ModelNotFound with 'decoder', got: {other:?}"),
        }
    }

    #[test]
    fn new_missing_joiner_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("encoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("decoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(dir.path().join("tokens.txt"), b"fake").unwrap();

        let result = ParakeetBackend::new(dir.path());
        match result {
            Err(CaptionError::ModelNotFound(msg)) => {
                assert!(
                    msg.contains("joiner"),
                    "Error should mention 'joiner', got: {msg}"
                );
            }
            other => panic!("Expected ModelNotFound with 'joiner', got: {other:?}"),
        }
    }

    #[test]
    fn new_missing_tokens_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("encoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("decoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(dir.path().join("joiner-epoch-999-avg-1.int8.onnx"), b"fake").unwrap();

        let result = ParakeetBackend::new(dir.path());
        match result {
            Err(CaptionError::ModelNotFound(msg)) => {
                assert!(
                    msg.contains("tokens.txt"),
                    "Error should mention 'tokens.txt', got: {msg}"
                );
            }
            other => panic!("Expected ModelNotFound with 'tokens.txt', got: {other:?}"),
        }
    }

    #[test]
    #[ignore] // sherpa-onnx's C++ ONNX Runtime aborts the process on invalid model files
              // instead of returning None, so this test cannot run in-process.
    fn new_all_files_present_but_fake_returns_model_load_failed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("encoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("decoder-epoch-999-avg-1.int8.onnx"),
            b"fake",
        )
        .unwrap();
        std::fs::write(dir.path().join("joiner-epoch-999-avg-1.int8.onnx"), b"fake").unwrap();
        std::fs::write(dir.path().join("tokens.txt"), b"fake").unwrap();

        let result = ParakeetBackend::new(dir.path());
        assert!(
            matches!(result, Err(CaptionError::ModelLoadFailed(_))),
            "Expected ModelLoadFailed for garbage model files, got: {result:?}"
        );
    }

    // --- Integration test (requires real model files) ---

    #[test]
    #[ignore] // Requires actual Parakeet TDT model files on disk
    fn integration_silence_produces_empty_transcription() {
        use crate::stt::model::find_parakeet_model;

        let model_dir = match find_parakeet_model(None) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping integration test: no Parakeet model directory found");
                return;
            }
        };

        let backend = ParakeetBackend::new(&model_dir).expect("failed to load Parakeet model");
        let silence = vec![0.0f32; 16_000 * 2];
        let config = TranscribeConfig::default();
        let result = backend
            .transcribe(&silence, &config)
            .expect("transcription should succeed");

        assert!(
            result.words.len() <= 2,
            "Expected 0-2 words from silence, got {}",
            result.words.len()
        );
    }
}
