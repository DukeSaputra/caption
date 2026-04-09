use thiserror::Error;

#[derive(Error, Debug)]
pub enum CaptionError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("No audio track found in file")]
    NoAudioTrack,

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Audio extraction failed: {0}")]
    ExtractionFailed(String),

    #[error("Resampling failed: {0}")]
    ResamplingFailed(String),

    #[error("Model file not found. Searched: {0}")]
    ModelNotFound(String),

    #[error("Failed to load STT model: {0}")]
    ModelLoadFailed(String),

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    #[error("VAD model file not found. Searched: {0}")]
    VadModelNotFound(String),

    #[error("VAD processing failed: {0}")]
    VadProcessingFailed(String),

    #[error("{0}")]
    FfmpegNotFound(String),

    #[error("Aligner model file not found. Searched: {0}")]
    AlignerModelNotFound(String),

    #[error("Alignment failed: {0}")]
    AlignmentFailed(String),

    #[error("Font file not found. Searched: {0}")]
    FontNotFound(String),

    #[error("Failed to probe video: {0}")]
    VideoProbeError(String),

    #[error("Burn failed: {0}")]
    BurnFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CaptionError>();
    }

    #[test]
    fn error_messages_are_human_readable() {
        let err = CaptionError::FileNotFound("/tmp/missing.mp4".to_string());
        assert_eq!(err.to_string(), "File not found: /tmp/missing.mp4");

        let err = CaptionError::NoAudioTrack;
        assert_eq!(err.to_string(), "No audio track found in file");

        let err = CaptionError::UnsupportedFormat("Opus".to_string());
        assert_eq!(err.to_string(), "Unsupported audio format: Opus");

        let err = CaptionError::ExtractionFailed("decode error".to_string());
        assert_eq!(err.to_string(), "Audio extraction failed: decode error");

        let err = CaptionError::ResamplingFailed("invalid ratio".to_string());
        assert_eq!(err.to_string(), "Resampling failed: invalid ratio");

        let err = CaptionError::ModelNotFound("/usr/local/models, ~/.local/share".to_string());
        assert_eq!(
            err.to_string(),
            "Model file not found. Searched: /usr/local/models, ~/.local/share"
        );

        let err = CaptionError::ModelLoadFailed("invalid ggml header".to_string());
        assert_eq!(
            err.to_string(),
            "Failed to load STT model: invalid ggml header"
        );

        let err = CaptionError::TranscriptionFailed("out of memory".to_string());
        assert_eq!(err.to_string(), "Transcription failed: out of memory");

        let err = CaptionError::VadModelNotFound("/usr/local/models".to_string());
        assert_eq!(
            err.to_string(),
            "VAD model file not found. Searched: /usr/local/models"
        );

        let err = CaptionError::VadProcessingFailed("ONNX session error".to_string());
        assert_eq!(err.to_string(), "VAD processing failed: ONNX session error");

        let err = CaptionError::FfmpegNotFound("FFmpeg is required".to_string());
        assert_eq!(err.to_string(), "FFmpeg is required");

        let err = CaptionError::AlignerModelNotFound("/usr/local/models".to_string());
        assert_eq!(
            err.to_string(),
            "Aligner model file not found. Searched: /usr/local/models"
        );

        let err = CaptionError::AlignmentFailed("CTC Viterbi error".to_string());
        assert_eq!(err.to_string(), "Alignment failed: CTC Viterbi error");

        let err = CaptionError::FontNotFound("/usr/local/fonts".to_string());
        assert_eq!(
            err.to_string(),
            "Font file not found. Searched: /usr/local/fonts"
        );

        let err = CaptionError::VideoProbeError("no video stream".to_string());
        assert_eq!(err.to_string(), "Failed to probe video: no video stream");

        let err = CaptionError::BurnFailed("FFmpeg exited with code 1".to_string());
        assert_eq!(err.to_string(), "Burn failed: FFmpeg exited with code 1");
    }
}
