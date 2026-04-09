use crate::postprocess::PostProcessor;
use crate::stt::Word;

#[derive(Debug, Clone, Copy, Default)]
pub enum ProfanityMode {
    #[default]
    KeepAll,
    Mask,
    Replace,
}

pub struct ProfanityFilter {
    pub mode: ProfanityMode,
}

impl PostProcessor for ProfanityFilter {
    fn process(&self, words: &mut Vec<Word>) {
        match self.mode {
            ProfanityMode::KeepAll => {}
            ProfanityMode::Mask => {
                for word in words.iter_mut() {
                    if is_profane(&word.text) {
                        word.text = mask_word(&word.text);
                    }
                }
            }
            ProfanityMode::Replace => {
                for word in words.iter_mut() {
                    if is_profane(&word.text) {
                        word.text = "[****]".to_string();
                    }
                }
            }
        }
    }
}

fn is_profane(word: &str) -> bool {
    let lower = word.to_lowercase();
    PROFANITY_LIST.iter().any(|p| *p == lower)
}

fn mask_word(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let len = chars.len();

    if len <= 1 {
        return word.to_string();
    }

    if len <= 3 {
        let mut result = String::with_capacity(len);
        result.push(chars[0]);
        for _ in 1..len {
            result.push('*');
        }
        return result;
    }

    let mut result = String::with_capacity(len);
    result.push(chars[0]);
    for _ in 1..len - 1 {
        result.push('*');
    }
    result.push(chars[len - 1]);
    result
}

const PROFANITY_LIST: &[&str] = &[
    "ass",
    "asshole",
    "assholes",
    "bastard",
    "bastards",
    "bitch",
    "bitches",
    "bitching",
    "bitchy",
    "bollocks",
    "bullshit",
    "cock",
    "cocks",
    "cocksucker",
    "cocksuckers",
    "crap",
    "crappy",
    "cum",
    "cunt",
    "cunts",
    "damn",
    "dammit",
    "damned",
    "dick",
    "dicks",
    "dickhead",
    "dickheads",
    "douchebag",
    "douchebags",
    "douche",
    "fag",
    "faggot",
    "faggots",
    "fags",
    "fuck",
    "fucked",
    "fucker",
    "fuckers",
    "fucking",
    "fucks",
    "goddamn",
    "goddamned",
    "goddamnit",
    "hell",
    "jackass",
    "jackasses",
    "jerkoff",
    "motherfucker",
    "motherfuckers",
    "motherfucking",
    "nigga",
    "niggas",
    "nigger",
    "niggers",
    "piss",
    "pissed",
    "pissing",
    "prick",
    "pricks",
    "pussy",
    "pussies",
    "retard",
    "retarded",
    "retards",
    "shit",
    "shits",
    "shitty",
    "shitting",
    "shithead",
    "shitheads",
    "slut",
    "sluts",
    "slutty",
    "twat",
    "twats",
    "wanker",
    "wankers",
    "whore",
    "whores",
    "arse",
    "arsehole",
    "arseholes",
    "ballsack",
    "biatch",
    "blowjob",
    "blowjobs",
    "boner",
    "boob",
    "boobs",
    "buttplug",
    "cameltoe",
    "chink",
    "chinks",
    "clit",
    "clitoris",
    "coochie",
    "coon",
    "coons",
    "cornhole",
    "deepthroat",
    "dildo",
    "dildos",
    "dyke",
    "dykes",
    "ejaculate",
    "erection",
    "fellatio",
    "fleshlight",
    "foreskin",
    "gangbang",
    "goatse",
    "goddammit",
    "gringo",
    "gringos",
    "handjob",
    "horny",
    "humping",
    "jizz",
    "kike",
    "kikes",
    "knobhead",
    "lesbo",
    "milf",
    "minge",
    "muff",
    "nonce",
    "nutsack",
    "orgasm",
    "pansy",
    "peckerhead",
    "penis",
    "pube",
    "pubes",
    "queef",
    "queer",
    "rimjob",
    "scrotum",
    "skank",
    "skanks",
    "skanky",
    "smegma",
    "spic",
    "spics",
    "spunk",
    "testicle",
    "testicles",
    "tit",
    "tits",
    "titties",
    "titty",
    "tosser",
    "tossers",
    "tranny",
    "vagina",
    "vulva",
    "wetback",
    "wetbacks",
    "wop",
    "wops",
];

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
    fn mask_mode_five_char_word() {
        let result = mask_word("shits");
        assert_eq!(result, "s***s");
    }

    #[test]
    fn mask_mode_four_char_word() {
        let result = mask_word("shit");
        assert_eq!(result, "s**t");
    }

    #[test]
    fn mask_mode_three_char_word() {
        let result = mask_word("ass");
        assert_eq!(result, "a**");
    }

    #[test]
    fn mask_mode_two_char_word() {
        let result = mask_word("ho");
        assert_eq!(result, "h*");
    }

    #[test]
    fn mask_mode_one_char_word() {
        let result = mask_word("f");
        assert_eq!(result, "f");
    }

    #[test]
    fn mask_mode_long_word() {
        let result = mask_word("motherfucker");
        assert_eq!(result, "m**********r");
    }

    #[test]
    fn replace_mode_replaces_with_fixed_string() {
        let mut words = vec![
            word("hello", 0.0, 0.5),
            word("fuck", 0.6, 0.9),
            word("world", 1.0, 1.4),
        ];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Replace,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "[****]");
        assert_eq!(words[2].text, "world");
    }

    #[test]
    fn keep_all_no_modifications() {
        let mut words = vec![word("fuck", 0.0, 0.3), word("shit", 0.4, 0.7)];
        let filter = ProfanityFilter {
            mode: ProfanityMode::KeepAll,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "fuck");
        assert_eq!(words[1].text, "shit");
    }

    #[test]
    fn case_insensitive_matching() {
        let mut words = vec![
            word("FUCK", 0.0, 0.3),
            word("Shit", 0.4, 0.7),
            word("DaMn", 0.8, 1.1),
        ];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "F**K");
        assert_eq!(words[1].text, "S**t");
        assert_eq!(words[2].text, "D**n");
    }

    #[test]
    fn non_profane_words_untouched() {
        let mut words = vec![
            word("hello", 0.0, 0.5),
            word("beautiful", 0.6, 1.2),
            word("world", 1.3, 1.7),
        ];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "beautiful");
        assert_eq!(words[2].text, "world");
    }

    #[test]
    fn mask_preserves_timestamps() {
        let mut words = vec![word("shit", 1.5, 2.0)];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "s**t");
        assert!((words[0].start - 1.5).abs() < f64::EPSILON);
        assert!((words[0].end - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn replace_preserves_timestamps() {
        let mut words = vec![word("shit", 1.5, 2.0)];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Replace,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "[****]");
        assert!((words[0].start - 1.5).abs() < f64::EPSILON);
        assert!((words[0].end - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_word_list_stays_empty() {
        let mut words: Vec<Word> = vec![];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };
        filter.process(&mut words);

        assert!(words.is_empty());
    }

    #[test]
    fn default_mode_is_keep_all() {
        let mode = ProfanityMode::default();
        assert!(matches!(mode, ProfanityMode::KeepAll));
    }

    #[test]
    fn multiple_profane_words_all_masked() {
        let mut words = vec![
            word("fuck", 0.0, 0.3),
            word("this", 0.4, 0.7),
            word("shit", 0.8, 1.1),
        ];
        let filter = ProfanityFilter {
            mode: ProfanityMode::Mask,
        };
        filter.process(&mut words);

        assert_eq!(words[0].text, "f**k");
        assert_eq!(words[1].text, "this");
        assert_eq!(words[2].text, "s**t");
    }
}
