//! Shared data types for LocalMotive.
//!
//! [`Chunk`] is one line of `chunks.jsonl` / `chunks.ocr.jsonl`. The stage
//! files ([`RouteFile`], [`RetrievedFile`], [`DraftFile`], [`VerifiedFile`])
//! are the JSON hand-offs between the answer pipeline's stages:
//! route → retrieve → draft → verify.

use serde::{Deserialize, Serialize};

/// Current schema version written into every stage file.
pub const SCHEMA_VERSION: u32 = 1;

/// Where a chunk's text came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// Text extracted from the manual's HTML.
    #[default]
    Html,
    /// Text recognised by OCR from an image; numbers must be checked against the image.
    Ocr,
}

/// Coarse content type, from path/title heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Procedure,
    Dtc,
    /// OCR of a DTC diagnostic step table image.
    DtcSteps,
    Spec,
    Description,
    Tsb,
    Diagram,
    Labor,
    Other,
}

impl Kind {
    pub const ALL: [Kind; 9] = [
        Kind::Procedure,
        Kind::Dtc,
        Kind::DtcSteps,
        Kind::Spec,
        Kind::Description,
        Kind::Tsb,
        Kind::Diagram,
        Kind::Labor,
        Kind::Other,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Procedure => "procedure",
            Kind::Dtc => "dtc",
            Kind::DtcSteps => "dtc_steps",
            Kind::Spec => "spec",
            Kind::Description => "description",
            Kind::Tsb => "tsb",
            Kind::Diagram => "diagram",
            Kind::Labor => "labor",
            Kind::Other => "other",
        }
    }
}

/// One retrievable chunk of the manual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// First 16 hex chars of sha256(`"{page_id}:{chunk_index}"`) (OCR: of `"ocr:{image}"`).
    pub id: String,
    pub page_id: u32,
    pub title: String,
    /// Primary path, joined with `" > "`.
    pub breadcrumb: String,
    /// Every path at which this (deduplicated) page body appears.
    pub aliases: Vec<String>,
    pub section_heading: String,
    /// Exact extracted text (HTML) or OCR output.
    pub text: String,
    /// Image paths relative to the manual root, e.g. `images/DM13Q313/gm60/1.png`.
    pub images: Vec<String>,
    /// Page ids linked from this chunk.
    pub links: Vec<u32>,
    pub kind: Kind,
    pub dtc_codes: Vec<String>,
    pub image_only: bool,
    #[serde(default)]
    pub source: Source,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_confidence: Option<f32>,
    /// Raw tesseract output before the numeric-context fix (OCR chunks only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_raw: Option<String>,
    /// OCR tables: line indexes of rows whose metric and English values agree
    /// by unit conversion (or whose angle is corroborated by HTML text).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ocr_checked_rows: Vec<u32>,
    /// OCR tables: values replaced by "[see image]" because they could not be cross-checked.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ocr_masked_values: u32,
    /// Kept in the file but not indexed or shown (e.g. OCR confidence < 60).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl Chunk {
    /// The text the answer model sees for this chunk, and the only text the
    /// verifier accepts numbers from: title, breadcrumb, body.
    pub fn context_text(&self) -> String {
        format!("{}\n{}\n{}", self.title, self.breadcrumb, self.text)
    }
}

/// Citation id of the vehicle pseudo-source.
pub const VEHICLE_SOURCE_ID: &str = "vehicle";

/// The user's vehicle (`~/localmotive/vehicle.json`). Claims may cite it as
/// [`VEHICLE_SOURCE_ID`]; its numbers are checked against [`Vehicle::context_text`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vehicle {
    pub vin: String,
    pub year: u16,
    pub make: String,
    pub model: String,
    pub trim: String,
    pub drive: String,
    pub engine: String,
    /// GM RPO engine code, e.g. LX9.
    pub engine_rpo: String,
    /// 8th VIN character.
    pub vin_engine_code: String,
}

impl Vehicle {
    /// Pretty JSON: what the model sees and what the verifier checks.
    pub fn context_text(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Approximate token count used everywhere in LocalMotive: words × 1.3, rounded.
pub fn approx_tokens(text: &str) -> usize {
    (text.split_whitespace().count() as f64 * 1.3).round() as usize
}

// ------------------------------------------------------------------ stages

/// Stage 1 output: what the question is about and where to look.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteFile {
    pub schema_version: u32,
    pub query: String,
    /// DTC codes mentioned by the user or read from the vehicle.
    #[serde(default)]
    pub dtc_codes: Vec<String>,
    /// Kinds to prefer when retrieving (empty = any).
    #[serde(default)]
    pub kinds: Vec<Kind>,
    /// Breadcrumb prefixes to restrict retrieval to (empty = whole manual).
    #[serde(default)]
    pub breadcrumb_prefixes: Vec<String>,
    /// The user asked what to do next in a procedure/diagnostic table.
    #[serde(default)]
    pub wants_next_step: bool,
}

/// One chunk returned by retrieval, with its score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedChunk {
    pub score: f32,
    pub chunk: Chunk,
}

/// Stage 2 output: the evidence the drafter may cite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedFile {
    pub schema_version: u32,
    pub query: String,
    pub chunks: Vec<RetrievedChunk>,
}

/// One claim of a draft answer and the sources it rests on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    /// Chunk ids from the retrieved set, or [`VEHICLE_SOURCE_ID`].
    #[serde(default)]
    pub cites: Vec<String>,
}

/// The single next diagnostic/repair step the answer points the user to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NextStep {
    pub chunk_id: String,
    /// e.g. "Step 3" or "Removal Procedure, step 2".
    pub step_label: String,
}

/// Stage 3 output: the model's draft answer as structured claims
/// (emitted through Ollama's JSON-schema `format`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DraftFile {
    pub schema_version: u32,
    pub answer: Vec<Claim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<NextStep>,
}

/// Outcome for one checked number-like token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumberCheck {
    pub token: String,
    /// Id of the first cited chunk whose text contains the token, if any.
    pub found_in: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
}

/// Verification result for one claim of the draft.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedClaim {
    pub text: String,
    pub cites: Vec<String>,
    pub numbers: Vec<NumberCheck>,
    pub verdict: Verdict,
    /// Human-readable reasons for a failure (empty on pass).
    pub reasons: Vec<String>,
    /// True when the claim relies on an OCR chunk: show the source image.
    pub needs_image: bool,
}

/// Verification result for the draft's next step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedNextStep {
    pub chunk_id: String,
    pub step_label: String,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
}

/// Stage 4 output: the draft, claim by claim, with verdicts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedFile {
    pub schema_version: u32,
    pub query: String,
    /// True when every claim (and the next step, if any) passed.
    pub ok: bool,
    pub claims: Vec<VerifiedClaim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<VerifiedNextStep>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_chunk_line_deserializes_with_defaults() {
        let line = r#"{"id":"abc","page_id":15449,"title":"Spark Plug Gap","breadcrumb":"A > B","aliases":["A > B"],"section_heading":"","text":"Spark Plug Gap 1.52 mm (0.060 in.)","images":[],"links":[9453],"kind":"spec","dtc_codes":[],"image_only":false}"#;
        let c: Chunk = serde_json::from_str(line).unwrap();
        assert_eq!(c.source, Source::Html);
        assert_eq!(c.ocr_confidence, None);
        assert_eq!(c.kind, Kind::Spec);
        let back = serde_json::to_string(&c).unwrap();
        assert!(!back.contains("ocr_confidence"));
        assert!(back.contains(r#""source":"html""#));
    }

    #[test]
    fn context_text_is_title_breadcrumb_text() {
        let line = r#"{"id":"abc","page_id":1,"title":"Spark Plug Gap","breadcrumb":"A > B","aliases":[],"section_heading":"","text":"1.52 mm","images":[],"links":[],"kind":"spec","dtc_codes":[],"image_only":false}"#;
        let c: Chunk = serde_json::from_str(line).unwrap();
        assert_eq!(c.context_text(), "Spark Plug Gap\nA > B\n1.52 mm");
        assert!(!serde_json::to_string(&c).unwrap().contains("hidden"));
    }

    #[test]
    fn draft_file_shape() {
        let d: DraftFile = serde_json::from_str(
            r#"{"schema_version":1,"answer":[{"text":"Gap is 1.52 mm.","cites":["4184c22de9df6a3b"]}],"next_step":{"chunk_id":"ba27a8b1130b8680","step_label":"Step 2"}}"#,
        )
        .unwrap();
        assert_eq!(d.answer[0].cites, vec!["4184c22de9df6a3b"]);
        assert_eq!(d.next_step.unwrap().step_label, "Step 2");
    }

    #[test]
    fn kind_names_match_serde() {
        for k in Kind::ALL {
            assert_eq!(
                serde_json::to_string(&k).unwrap(),
                format!("\"{}\"", k.as_str())
            );
        }
    }

    #[test]
    fn approx_tokens_rounds() {
        assert_eq!(approx_tokens("one two three"), 4);
        assert_eq!(approx_tokens(""), 0);
    }
}
