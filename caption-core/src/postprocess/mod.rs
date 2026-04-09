pub mod fillers;
pub mod hallucination;
pub mod profanity;
pub mod punctuation;
pub mod substitution;

use crate::stt::Word;

pub trait PostProcessor: Send + Sync {
    fn process(&self, words: &mut Vec<Word>);
}

pub fn run_pipeline(words: &mut Vec<Word>, processors: &[&dyn PostProcessor]) {
    for processor in processors {
        processor.process(words);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::fillers::{FillerMode, FillerRemover};
    use crate::postprocess::profanity::{ProfanityFilter, ProfanityMode};

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
            confidence: 0.99,
        }
    }

    #[test]
    fn pipeline_runs_filler_then_profanity() {
        let mut words = vec![
            word("um", 0.0, 0.3),
            word("shit", 0.4, 0.8),
            word("happens", 0.9, 1.3),
        ];

        let filler = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        let profanity = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };

        run_pipeline(&mut words, &[&filler, &profanity]);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "s**t");
        assert_eq!(words[1].text, "happens");
    }

    #[test]
    fn pipeline_with_empty_processors() {
        let mut words = vec![word("hello", 0.0, 0.5)];
        let processors: Vec<&dyn PostProcessor> = vec![];
        run_pipeline(&mut words, &processors);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
    }

    #[test]
    fn post_processor_trait_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FillerRemover>();
        assert_send_sync::<ProfanityFilter>();
        assert_send_sync::<crate::postprocess::punctuation::PunctuationStripper>();
    }

    // -----------------------------------------------------------------------
    // Post-processing order: hallucination -> fillers -> punctuation -> profanity -> substitution
    // -----------------------------------------------------------------------

    #[test]
    fn post_processing_order_hallucination_before_fillers() {
        use crate::postprocess::hallucination::{HallucinationFilter, HallucinationMode};

        let mut words = vec![
            word("um", 0.0, 0.3),
            word("thank", 0.4, 0.6),
            word("you", 0.7, 0.9),
            word("for", 1.0, 1.2),
            word("watching", 1.3, 1.5),
        ];

        let hallucination = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        let filler = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };

        run_pipeline(&mut words, &[&hallucination, &filler]);

        assert!(
            words.is_empty(),
            "Expected all words removed (phrase + filler), got: {words:?}"
        );
    }

    #[test]
    fn post_processing_order_full_chain_hallucination_fillers_punctuation_profanity_substitution() {
        use crate::postprocess::hallucination::{HallucinationFilter, HallucinationMode};
        use crate::postprocess::punctuation::PunctuationStripper;
        use crate::postprocess::substitution::SubstitutionMap;

        let mut words = vec![
            word("um", 0.0, 0.2),
            word("I", 0.3, 0.4),
            word("gonna", 0.5, 0.7),
            word("shit.", 0.8, 1.0),
            word("thank", 1.1, 1.3),
            word("you", 1.4, 1.6),
            word("for", 1.7, 1.9),
            word("watching", 2.0, 2.2),
        ];

        let hallucination = HallucinationFilter {
            allowed_phrases: vec![],
            mode: HallucinationMode::Moderate,
        };
        let filler = FillerRemover {
            mode: FillerMode::RemoveConfident,
        };
        let punctuation = PunctuationStripper;
        let profanity = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };
        let substitution = SubstitutionMap::with_defaults();

        run_pipeline(
            &mut words,
            &[
                &hallucination,
                &filler,
                &punctuation,
                &profanity,
                &substitution,
            ],
        );

        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["I", "going", "to", "s**t"]);
    }

    #[test]
    fn post_processor_trait_is_object_safe() {
        let filler: Box<dyn PostProcessor> = Box::new(FillerRemover {
            mode: FillerMode::default(),
        });
        let mut words = vec![word("um", 0.0, 0.3), word("hello", 0.4, 0.8)];
        filler.process(&mut words);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
    }
}
