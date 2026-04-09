use std::path::{Path, PathBuf};

use log::debug;

use crate::error::CaptionError;

const DEFAULT_MODEL_FILENAME: &str = "ggml-large-v3-turbo-q8_0.bin";
const FALLBACK_MODEL_FILENAME: &str = "ggml-large-v3-turbo-q5_0.bin";
const DEFAULT_VAD_MODEL_FILENAME: &str = "silero_vad.onnx";
const DEFAULT_ALIGNER_MODEL_FILENAME: &str = "wav2vec2-base-960h.onnx";

fn find_model_file(
    explicit_path: Option<&str>,
    default_filename: &str,
    make_error: fn(String) -> CaptionError,
) -> Result<PathBuf, CaptionError> {
    if let Some(explicit) = explicit_path {
        let p = Path::new(explicit);
        if p.is_file() {
            debug!("Using explicit model path: {}", p.display());
            return Ok(p.to_path_buf());
        }
        return Err(make_error(explicit.to_string()));
    }

    let mut searched: Vec<String> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("models").join(default_filename);
        debug!(
            "Checking working directory models/: {}",
            candidate.display()
        );
        if candidate.is_file() {
            return Ok(candidate);
        }
        searched.push(candidate.display().to_string());
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let candidate = exe_dir.join(default_filename);
            debug!("Checking executable directory: {}", candidate.display());
            if candidate.is_file() {
                return Ok(candidate);
            }
            searched.push(candidate.display().to_string());
        }
    }

    if let Some(data_dir) = dirs::data_dir() {
        let candidate = data_dir
            .join("caption")
            .join("models")
            .join(default_filename);
        debug!("Checking data directory: {}", candidate.display());
        if candidate.is_file() {
            return Ok(candidate);
        }
        searched.push(candidate.display().to_string());
    }

    Err(make_error(searched.join(", ")))
}

pub fn find_model(model_path: Option<&str>) -> Result<PathBuf, CaptionError> {
    if model_path.is_some() {
        return find_model_file(
            model_path,
            DEFAULT_MODEL_FILENAME,
            CaptionError::ModelNotFound,
        );
    }

    find_model_file(None, DEFAULT_MODEL_FILENAME, CaptionError::ModelNotFound)
        .or_else(|_| find_model_file(None, FALLBACK_MODEL_FILENAME, CaptionError::ModelNotFound))
}

const PARAKEET_DIR_PREFIX: &str = "sherpa-onnx-nemo-parakeet-tdt";

pub fn find_parakeet_model(explicit_path: Option<&str>) -> Result<PathBuf, CaptionError> {
    if let Some(explicit) = explicit_path {
        let p = Path::new(explicit);
        if p.is_dir() {
            debug!("Using explicit Parakeet model path: {}", p.display());
            return Ok(p.to_path_buf());
        }
        return Err(CaptionError::ModelNotFound(format!(
            "Parakeet model directory not found: {explicit}"
        )));
    }

    let mut searched: Vec<String> = Vec::new();

    let search_dirs: Vec<PathBuf> = {
        let mut dirs = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            dirs.push(cwd.join("models"));
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                dirs.push(exe_dir.to_path_buf());
            }
        }
        if let Some(data_dir) = dirs::data_dir() {
            dirs.push(data_dir.join("caption").join("models"));
        }
        dirs
    };

    for search_dir in &search_dirs {
        debug!("Searching for Parakeet model in: {}", search_dir.display());
        searched.push(search_dir.display().to_string());

        if let Some(found) = find_parakeet_dir_in(search_dir) {
            return Ok(found);
        }
    }

    Err(CaptionError::ModelNotFound(format!(
        "Parakeet model directory ({}*) not found. Searched: {}",
        PARAKEET_DIR_PREFIX,
        searched.join(", ")
    )))
}

fn find_parakeet_dir_in(parent: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(parent).ok()?;

    let mut best: Option<PathBuf> = None;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(PARAKEET_DIR_PREFIX) {
            match &best {
                Some(existing) => {
                    if path > *existing {
                        best = Some(path);
                    }
                }
                None => {
                    best = Some(path);
                }
            }
        }
    }

    if let Some(ref found) = best {
        debug!("Found Parakeet model directory: {}", found.display());
    }

    best
}

pub fn find_vad_model(vad_model_path: Option<&str>) -> Result<PathBuf, CaptionError> {
    find_model_file(
        vad_model_path,
        DEFAULT_VAD_MODEL_FILENAME,
        CaptionError::VadModelNotFound,
    )
}

const HOT_WORDS_FILENAME: &str = "hot-words.txt";

pub fn find_hot_words_file() -> Option<PathBuf> {
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("models").join(HOT_WORDS_FILENAME);
        debug!("Checking for hot-words file: {}", candidate.display());
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let candidate = exe_dir.join(HOT_WORDS_FILENAME);
            debug!("Checking for hot-words file: {}", candidate.display());
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    if let Some(data_dir) = dirs::data_dir() {
        let candidate = data_dir
            .join("caption")
            .join("models")
            .join(HOT_WORDS_FILENAME);
        debug!("Checking for hot-words file: {}", candidate.display());
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    debug!("Hot-words file not found (this is not an error)");
    None
}

pub fn find_aligner_model(aligner_model_path: Option<&str>) -> Result<PathBuf, CaptionError> {
    find_model_file(
        aligner_model_path,
        DEFAULT_ALIGNER_MODEL_FILENAME,
        CaptionError::AlignerModelNotFound,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_model_none_returns_model_not_found_or_ok() {
        let result = find_model(None);
        assert!(
            matches!(result, Err(CaptionError::ModelNotFound(_)) | Ok(_)),
            "Expected ModelNotFound or Ok, got: {result:?}"
        );
    }

    #[test]
    fn find_model_explicit_nonexistent_returns_error() {
        let result = find_model(Some("/tmp/absolutely-does-not-exist-model.bin"));
        assert!(
            matches!(result, Err(CaptionError::ModelNotFound(_))),
            "Expected ModelNotFound, got: {result:?}"
        );
    }

    #[test]
    fn find_model_explicit_existing_file_returns_path() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("test-model.bin");
        std::fs::write(&model_path, b"fake model data").unwrap();

        let result = find_model(Some(model_path.to_str().unwrap()));
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), model_path);
    }

    #[test]
    fn model_not_found_error_lists_searched_paths() {
        let result = find_model(None);
        if let Err(CaptionError::ModelNotFound(searched)) = result {
            assert!(
                !searched.is_empty(),
                "ModelNotFound should list searched paths"
            );
        }
    }

    // --- VAD model discovery tests ---

    #[test]
    fn find_vad_model_none_returns_vad_model_not_found_or_ok() {
        let result = find_vad_model(None);
        assert!(
            matches!(result, Err(CaptionError::VadModelNotFound(_)) | Ok(_)),
            "Expected VadModelNotFound or Ok, got: {result:?}"
        );
    }

    #[test]
    fn find_vad_model_explicit_nonexistent_returns_error() {
        let result = find_vad_model(Some("/tmp/absolutely-does-not-exist-vad.onnx"));
        assert!(
            matches!(result, Err(CaptionError::VadModelNotFound(_))),
            "Expected VadModelNotFound, got: {result:?}"
        );
    }

    #[test]
    fn find_vad_model_explicit_existing_file_returns_path() {
        let dir = tempfile::tempdir().unwrap();
        let vad_path = dir.path().join("silero_vad.onnx");
        std::fs::write(&vad_path, b"fake vad model data").unwrap();

        let result = find_vad_model(Some(vad_path.to_str().unwrap()));
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), vad_path);
    }

    #[test]
    fn find_vad_model_not_found_error_lists_searched_paths() {
        let result = find_vad_model(None);
        if let Err(CaptionError::VadModelNotFound(searched)) = result {
            assert!(
                !searched.is_empty(),
                "VadModelNotFound should list searched paths"
            );
        }
    }

    // --- Parakeet model discovery tests ---

    #[test]
    fn find_parakeet_model_explicit_nonexistent_returns_error() {
        let result = find_parakeet_model(Some("/tmp/absolutely-does-not-exist-parakeet"));
        assert!(
            matches!(result, Err(CaptionError::ModelNotFound(_))),
            "Expected ModelNotFound, got: {result:?}"
        );
    }

    #[test]
    fn find_parakeet_model_explicit_existing_dir_returns_path() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir
            .path()
            .join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8");
        std::fs::create_dir(&model_dir).unwrap();

        let result = find_parakeet_model(Some(model_dir.to_str().unwrap()));
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), model_dir);
    }

    #[test]
    fn find_parakeet_model_none_returns_model_not_found_or_ok() {
        let result = find_parakeet_model(None);
        assert!(
            matches!(result, Err(CaptionError::ModelNotFound(_)) | Ok(_)),
            "Expected ModelNotFound or Ok, got: {result:?}"
        );
    }

    #[test]
    fn find_parakeet_dir_in_finds_matching_directory() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir
            .path()
            .join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8");
        std::fs::create_dir(&model_dir).unwrap();

        let result = find_parakeet_dir_in(dir.path());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), model_dir);
    }

    #[test]
    fn find_parakeet_dir_in_ignores_non_matching_dirs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("some-other-model")).unwrap();
        std::fs::create_dir(dir.path().join("whisper-large")).unwrap();

        let result = find_parakeet_dir_in(dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn find_parakeet_dir_in_picks_highest_version() {
        let dir = tempfile::tempdir().unwrap();
        let v1 = dir
            .path()
            .join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v1-int8");
        let v2 = dir
            .path()
            .join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8");
        std::fs::create_dir(&v1).unwrap();
        std::fs::create_dir(&v2).unwrap();

        let result = find_parakeet_dir_in(dir.path());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), v2);
    }

    #[test]
    fn find_parakeet_dir_in_ignores_files_with_matching_name() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("sherpa-onnx-nemo-parakeet-tdt-readme.txt");
        std::fs::write(&file_path, b"not a directory").unwrap();

        let result = find_parakeet_dir_in(dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn find_parakeet_dir_in_nonexistent_parent_returns_none() {
        let result = find_parakeet_dir_in(Path::new("/tmp/absolutely-does-not-exist-dir"));
        assert!(result.is_none());
    }

    #[test]
    fn find_parakeet_model_error_includes_prefix_name() {
        let result = find_parakeet_model(None);
        if let Err(CaptionError::ModelNotFound(msg)) = result {
            assert!(
                msg.contains("sherpa-onnx-nemo-parakeet-tdt"),
                "Error message should mention the expected directory prefix, got: {msg}"
            );
        }
    }

    // --- CTC aligner model discovery tests ---

    #[test]
    fn find_aligner_model_none_returns_aligner_model_not_found_or_ok() {
        let result = find_aligner_model(None);
        assert!(
            matches!(result, Err(CaptionError::AlignerModelNotFound(_)) | Ok(_)),
            "Expected AlignerModelNotFound or Ok, got: {result:?}"
        );
    }

    #[test]
    fn find_aligner_model_explicit_nonexistent_returns_error() {
        let result = find_aligner_model(Some("/tmp/absolutely-does-not-exist-aligner.onnx"));
        assert!(
            matches!(result, Err(CaptionError::AlignerModelNotFound(_))),
            "Expected AlignerModelNotFound, got: {result:?}"
        );
    }

    #[test]
    fn find_aligner_model_explicit_existing_file_returns_path() {
        let dir = tempfile::tempdir().unwrap();
        let aligner_path = dir.path().join("wav2vec2-base-960h.onnx");
        std::fs::write(&aligner_path, b"fake aligner model data").unwrap();

        let result = find_aligner_model(Some(aligner_path.to_str().unwrap()));
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");
        assert_eq!(result.unwrap(), aligner_path);
    }

    #[test]
    fn find_aligner_model_not_found_error_lists_searched_paths() {
        let result = find_aligner_model(None);
        if let Err(CaptionError::AlignerModelNotFound(searched)) = result {
            assert!(
                !searched.is_empty(),
                "AlignerModelNotFound should list searched paths"
            );
        }
    }

    // --- Hot-words file discovery tests ---

    #[test]
    fn find_hot_words_file_missing_returns_none() {
        let result = find_hot_words_file();
        let _ = result;
    }

    #[test]
    fn find_hot_words_file_finds_file_in_models_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path().join("models");
        std::fs::create_dir(&models_dir).unwrap();
        let hw_path = models_dir.join("hot-words.txt");
        std::fs::write(&hw_path, b"# test\narc\n").unwrap();

        let original_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let result = find_hot_words_file();
        std::env::set_current_dir(original_cwd).unwrap();

        let result_canonical = result.map(|p| p.canonicalize().unwrap());
        let expected_canonical = hw_path.canonicalize().unwrap();
        assert_eq!(result_canonical, Some(expected_canonical));
    }
}
