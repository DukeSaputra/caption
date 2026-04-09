pub mod ass;
pub mod parse;
pub mod srt;
pub mod text;
pub mod vtt;

use crate::error::CaptionError;
use crate::stt::Word;

#[derive(Debug, Clone)]
pub struct FormatConfig {
    pub min_cue_duration: f64,
    pub pause_threshold: f64,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            min_cue_duration: 0.2,
            pause_threshold: 0.5,
        }
    }
}

pub trait SubtitleFormatter {
    fn format(&self, words: &[Word], config: &FormatConfig) -> Result<String, CaptionError>;

    fn file_extension(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = FormatConfig::default();
        assert!((config.min_cue_duration - 0.2).abs() < f64::EPSILON);
        assert!((config.pause_threshold - 0.5).abs() < f64::EPSILON);
    }
}
