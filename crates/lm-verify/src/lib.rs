//! Deterministic verifier for drafted answers.
//!
//! Given a [`DraftFile`] (structured claims), the [`RetrievedFile`] it was
//! drafted from and the user's [`Vehicle`], [`verify`] checks every claim:
//!
//! * every cite must be a retrieved chunk id, or `"vehicle"`;
//! * every number-like token (numbers, decimals, ranges such as `0.8-1.1`,
//!   thousands separators, values with units, DTC codes, identifiers such as
//!   `V6`) must appear verbatim in the [`lm_types::Chunk::context_text`] of a cited
//!   chunk (or the vehicle's context text), after only NFC normalization and
//!   whitespace collapse, and not as part of a longer number;
//! * a claim with numbers but no cites fails;
//! * a claim citing an OCR chunk is tagged `needs_image` so the UI shows the
//!   source image (its numbers must still match the OCR text);
//! * the next step's chunk must be retrieved and its label's numbers must be
//!   in that chunk.
//!
//! [`claims_from_inline`] converts an answer written with inline `[id]`
//! markers into claims; it is a fallback, not the main path.

use std::collections::HashMap;
use std::sync::LazyLock;

use lm_types::{
    Claim, DraftFile, NumberCheck, RetrievedFile, SCHEMA_VERSION, Source, VEHICLE_SOURCE_ID,
    Vehicle, Verdict, VerifiedClaim, VerifiedFile, VerifiedNextStep,
};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

static RE_CITE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\s*([0-9a-f]{16}(?:\s*[,;]\s*[0-9a-f]{16})*)\s*\]").unwrap());
static RE_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9a-f]{16}").unwrap());
static RE_DTC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[PBCU][0-9]{4}\b").unwrap());

/// Number, optional range, optional unit. Units are tried longest-first.
static RE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    let num = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+";
    let unit = r"N·m|N\.m|Nm|lb[ -]ft|lb[ -]in|lb\.? ?ft|lb\.? ?in|kPa|psi|RPM|rpm|mm|in\.|in\b|°C|°F|°|%|V\b|degrees";
    Regex::new(&format!(
        r"(?:{num})(?:\s*[-–]\s*(?:{num}))?(?:\s?(?:{unit}))?"
    ))
    .unwrap()
});

/// Abbreviations whose trailing period does not end a sentence.
const ABBREVIATIONS: [&str; 12] = [
    "in", "ft", "lb", "approx", "e.g", "i.e", "etc", "vs", "No", "Fig", "min", "max",
];

/// NFC + collapse all whitespace runs to one space + trim.
pub fn normalize(s: &str) -> String {
    let nfc: String = s.nfc().collect();
    nfc.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A sentence of the draft with its inline citations removed.
#[derive(Debug, Clone, PartialEq)]
pub struct Sentence {
    pub text: String,
    pub citations: Vec<String>,
}

/// Split the draft answer into sentences, attaching citation markers that
/// directly follow a sentence's final punctuation to that sentence.
pub fn split_sentences(answer: &str) -> Vec<Sentence> {
    let mut raw: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = answer.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        if c == '\n' {
            raw.push(std::mem::take(&mut cur));
        } else if matches!(c, '.' | '!' | '?') && ends_sentence(&chars, i, &cur) {
            // absorb following citation markers: ". [id] [id]"
            let mut j = i + 1;
            loop {
                let mut k = j;
                while k < chars.len() && chars[k] == ' ' {
                    k += 1;
                }
                if k < chars.len()
                    && chars[k] == '['
                    && let Some(end) = chars[k..].iter().position(|&ch| ch == ']')
                {
                    let marker: String = chars[k..=k + end].iter().collect();
                    if RE_CITE.is_match(&marker) {
                        cur.push(' ');
                        cur.push_str(&marker);
                        j = k + end + 1;
                        continue;
                    }
                }
                break;
            }
            raw.push(std::mem::take(&mut cur));
            i = j;
            continue;
        }
        i += 1;
    }
    raw.push(cur);

    let mut out: Vec<Sentence> = Vec::new();
    for r in raw {
        let citations: Vec<String> = RE_CITE
            .captures_iter(&r)
            .flat_map(|c| {
                RE_ID
                    .find_iter(c.get(1).map_or("", |m| m.as_str()))
                    .map(|m| m.as_str().to_string())
                    .collect::<Vec<_>>()
            })
            .fold(Vec::new(), |mut v, id| {
                if !v.contains(&id) {
                    v.push(id);
                }
                v
            });
        let text = normalize(&RE_CITE.replace_all(&r, " "));
        let text = text.replace(" .", ".").replace(" ,", ",");
        if text.is_empty() {
            // a line of only citations supports the previous sentence
            if let Some(prev) = out.last_mut() {
                for c in citations {
                    if !prev.citations.contains(&c) {
                        prev.citations.push(c);
                    }
                }
            }
            continue;
        }
        out.push(Sentence { text, citations });
    }
    out
}

fn ends_sentence(chars: &[char], i: usize, cur: &str) -> bool {
    let next = chars.get(i + 1).copied();
    // decimal point or "1.2" style
    if chars[i] == '.' && next.is_some_and(|n| n.is_ascii_digit()) {
        return false;
    }
    if next.is_some_and(|n| !n.is_whitespace()) {
        return false;
    }
    let word = cur[..cur.len() - 1]
        .rsplit(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("");
    if chars[i] == '.' && ABBREVIATIONS.contains(&word) {
        return false;
    }
    true
}

/// Number-like tokens in a sentence, in order: DTC codes, numbers with an
/// optional range and unit, and identifiers that glue digits to letters (V6, 3.5L).
pub fn extract_numbers(text: &str) -> Vec<String> {
    let mut spans: Vec<(usize, usize)> = RE_DTC
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();
    for m in RE_NUMBER.find_iter(text) {
        if spans.iter().any(|&(s, e)| m.start() < e && m.end() > s) {
            continue;
        }
        let (mut s, mut e) = (m.start(), m.end());
        let glued_left = text[..s]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphabetic());
        let glued_right = !m
            .as_str()
            .ends_with(|c: char| c.is_alphabetic() || c == '%' || c == '°' || c == '.')
            && text[e..].chars().next().is_some_and(|c| c.is_alphabetic());
        if glued_left || glued_right {
            while let Some(c) = text[..s]
                .chars()
                .next_back()
                .filter(|c| c.is_alphanumeric())
            {
                s -= c.len_utf8();
            }
            while let Some(c) = text[e..].chars().next().filter(|c| c.is_alphanumeric()) {
                e += c.len_utf8();
            }
        }
        if text[s..e].chars().any(|c| c.is_ascii_digit()) {
            spans.push((s, e));
        }
    }
    spans.sort_unstable();
    spans.dedup();
    let mut out = Vec::new();
    for (s, e) in spans {
        let mut tok = text[s..e].trim_end_matches([',']).trim().to_string();
        if tok.ends_with('.') && !tok.ends_with("in.") {
            tok.pop();
        }
        // a negative value: "-7" at a word start (not the "-" of a range)
        let neg = text[..s].ends_with('-')
            && text[..s - 1]
                .chars()
                .next_back()
                .is_none_or(|c| c.is_whitespace() || c == '(');
        if neg {
            tok.insert(0, '-');
        }
        out.push(tok);
    }
    out
}

/// True when `token` occurs in `hay` and is not part of a longer number or word.
pub fn contains_exact(hay: &str, token: &str) -> bool {
    let mut from = 0;
    while let Some(pos) = hay[from..].find(token) {
        let s = from + pos;
        let e = s + token.len();
        let before: Vec<char> = hay[..s].chars().rev().take(2).collect();
        let after: Vec<char> = hay[e..].chars().take(2).collect();
        let first = token.chars().next().unwrap_or(' ');
        let last = token.chars().next_back().unwrap_or(' ');
        let bad_before = match before.first() {
            Some(c) if c.is_ascii_digit() => true,
            Some(c) if c.is_alphabetic() && first.is_alphanumeric() => true,
            Some('.' | ',') => before.get(1).is_some_and(|c| c.is_ascii_digit()),
            Some('-') if first.is_ascii_digit() => false,
            _ => false,
        };
        let bad_after = match after.first() {
            Some(c) if c.is_ascii_digit() => last.is_ascii_digit() || last == '.' || last == ',',
            Some(c) if c.is_alphabetic() => last.is_alphanumeric(),
            Some('.' | ',') => {
                after.get(1).is_some_and(|c| c.is_ascii_digit()) && last.is_ascii_digit()
            }
            _ => false,
        };
        if !bad_before && !bad_after {
            return true;
        }
        from = s + token.len().max(1);
        while !hay.is_char_boundary(from) {
            from += 1;
        }
    }
    false
}

/// Fallback: turn an answer with inline `[id]` citations into claims, one per sentence.
pub fn claims_from_inline(answer: &str) -> Vec<Claim> {
    split_sentences(answer)
        .into_iter()
        .map(|s| Claim {
            text: s.text,
            cites: s.citations,
        })
        .collect()
}

/// A citable source: normalized verifiable text and whether it is OCR.
struct Src {
    text: String,
    ocr: bool,
}

fn check_tokens(
    text: &str,
    cites: &[String],
    sources: &HashMap<&str, Src>,
    reasons: &mut Vec<String>,
) -> Vec<NumberCheck> {
    let known: Vec<(&String, &Src)> = cites
        .iter()
        .filter_map(|id| sources.get(id.as_str()).map(|s| (id, s)))
        .collect();
    extract_numbers(text)
        .into_iter()
        .map(|t| {
            let tn = normalize(&t);
            let found_in = known
                .iter()
                .find(|(_, s)| contains_exact(&s.text, &tn))
                .map(|(id, _)| (*id).clone());
            if found_in.is_none() && !cites.is_empty() {
                reasons.push(format!("\"{t}\" not found in cited sources"));
            }
            NumberCheck { token: t, found_in }
        })
        .collect()
}

/// Run all checks. Pure and deterministic.
pub fn verify(
    draft: &DraftFile,
    retrieved: &RetrievedFile,
    vehicle: Option<&Vehicle>,
) -> VerifiedFile {
    let mut sources: HashMap<&str, Src> = retrieved
        .chunks
        .iter()
        .map(|r| {
            (
                r.chunk.id.as_str(),
                Src {
                    text: normalize(&r.chunk.context_text()),
                    ocr: r.chunk.source == Source::Ocr,
                },
            )
        })
        .collect();
    if let Some(v) = vehicle {
        sources.insert(
            VEHICLE_SOURCE_ID,
            Src {
                text: normalize(&v.context_text()),
                ocr: false,
            },
        );
    }

    let mut claims = Vec::new();
    for c in &draft.answer {
        let text = normalize(&c.text);
        let mut reasons = Vec::new();
        for id in &c.cites {
            if !sources.contains_key(id.as_str()) {
                reasons.push(format!("cites unknown source {id}"));
            }
        }
        let has_numbers = !extract_numbers(&text).is_empty();
        if has_numbers && c.cites.is_empty() {
            reasons.push("contains numbers but cites no source".into());
        }
        let numbers = check_tokens(&text, &c.cites, &sources, &mut reasons);
        let needs_image = c
            .cites
            .iter()
            .any(|id| sources.get(id.as_str()).is_some_and(|s| s.ocr));
        claims.push(VerifiedClaim {
            verdict: if reasons.is_empty() {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
            text,
            cites: c.cites.clone(),
            numbers,
            reasons,
            needs_image,
        });
    }

    let next_step = draft.next_step.as_ref().map(|n| {
        let mut reasons = Vec::new();
        if n.chunk_id == VEHICLE_SOURCE_ID || !sources.contains_key(n.chunk_id.as_str()) {
            reasons.push(format!("next step cites unknown chunk {}", n.chunk_id));
        } else {
            check_tokens(
                &normalize(&n.step_label),
                std::slice::from_ref(&n.chunk_id),
                &sources,
                &mut reasons,
            );
        }
        VerifiedNextStep {
            chunk_id: n.chunk_id.clone(),
            step_label: n.step_label.clone(),
            verdict: if reasons.is_empty() {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
            reasons,
        }
    });

    VerifiedFile {
        schema_version: SCHEMA_VERSION,
        query: retrieved.query.clone(),
        ok: claims.iter().all(|c| c.verdict == Verdict::Pass)
            && next_step
                .as_ref()
                .is_none_or(|n| n.verdict == Verdict::Pass),
        claims,
        next_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_tokens() {
        assert_eq!(
            extract_numbers("Gap to 1.52 mm (0.060 in.) now."),
            vec!["1.52 mm", "0.060 in."]
        );
        assert_eq!(
            extract_numbers("between 525-5,700 RPM and 9-18 volts"),
            vec!["525-5,700 RPM", "9-18"]
        );
        assert_eq!(
            extract_numbers("DTC P0300 sets after 2 trips"),
            vec!["P0300", "2"]
        );
        assert_eq!(
            extract_numbers("ECT below -7°C (+19°F)"),
            vec!["-7°C", "19°F"]
        );
        assert_eq!(
            extract_numbers("Torque to 20 N.m (15 lb ft)."),
            vec!["20 N.m", "15 lb ft"]
        );
        assert_eq!(extract_numbers("The 3.5L V6 engine"), vec!["3.5L", "V6"]);
        assert_eq!(
            extract_numbers("fuel level above 10 percent or 10%"),
            vec!["10", "10%"]
        );
        assert!(extract_numbers("Remove the spark plugs.").is_empty());
    }

    #[test]
    fn exact_containment_respects_number_boundaries() {
        let hay = "Spark Plug Gap 1.52 mm (0.060 in.)";
        assert!(contains_exact(hay, "1.52 mm"));
        assert!(contains_exact(hay, "0.060 in."));
        assert!(!contains_exact(hay, "1.5 mm"));
        assert!(!contains_exact(hay, "1.5"));
        assert!(!contains_exact(hay, "52 mm"));
        assert!(!contains_exact(hay, "0.06 in."));
        let rpm = "The engine speed is between 525-5,700 RPM.";
        assert!(contains_exact(rpm, "525-5,700 RPM"));
        assert!(!contains_exact(rpm, "5700 RPM"));
        assert!(!contains_exact(rpm, "700 RPM"));
        assert!(!contains_exact("DTC P0300 sets", "P030"));
        assert!(!contains_exact("DTC P03001", "P0300"));
    }

    #[test]
    fn sentence_split_keeps_abbreviations_and_trailing_citations() {
        let a = "Gap the plugs to 1.52 mm (0.060 in.) before install. [4184c22de9df6a3b] Torque them to 20 N.m [f3598b8bea83afe7].\nDone.";
        let s = split_sentences(a);
        assert_eq!(s.len(), 3, "{s:?}");
        assert_eq!(
            s[0].text,
            "Gap the plugs to 1.52 mm (0.060 in.) before install."
        );
        assert_eq!(s[0].citations, vec!["4184c22de9df6a3b"]);
        assert_eq!(s[1].text, "Torque them to 20 N.m.");
        assert_eq!(s[1].citations, vec!["f3598b8bea83afe7"]);
        let multi =
            split_sentences("See both [4184c22de9df6a3b, f3598b8bea83afe7][3b5f9e721f88ebc9].");
        assert_eq!(multi[0].citations.len(), 3);
    }

    #[test]
    fn normalize_is_nfc_and_whitespace() {
        assert_eq!(
            normalize("1.52\u{00a0}mm\n\t(0.060  in.)"),
            "1.52 mm (0.060 in.)"
        );
        assert_eq!(normalize("e\u{0301}"), "\u{00e9}");
    }
}
