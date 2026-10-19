//! One LocalMotive turn, file by file:
//!
//! ```text
//! ~/localmotive/sessions/<session>/turn-<n>/
//!   route.json  retrieved.json  draft.json  verified.json   (+ *.attempt1.json on retry)
//!   answer.json answer.md  timings.json
//! ```
//!
//! The answer model gets only the retrieved chunks (as [`Chunk::context_text`])
//! and the vehicle, replies through a JSON-schema `format` whose cite ids are
//! restricted to the retrieved ids, and is verified deterministically. On a
//! verifier failure it gets one retry with the reasons; then only passing
//! claims are shown, with "not verified" notes for the rest.

pub mod rss;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use lm_ollama::{ChatRequest, Message, Options};
use lm_types::{
    Chunk, Claim, DraftFile, NextStep, RetrievedFile, RouteFile, SCHEMA_VERSION, Source,
    VEHICLE_SOURCE_ID, Vehicle, Verdict, VerifiedFile, approx_tokens,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const ANSWER_MODEL: &str = "qwen3:4b-instruct-2507-q4_K_M";
pub const SMALL_MODEL: &str = "qwen3:1.7b";

/// Answer-stage settings; see [`Config::preset`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub model: String,
    pub max_chunks: usize,
    /// Estimated-token budget for all chunk text in the prompt.
    pub chunk_budget: usize,
    /// Keep only the lines around the query hits in each chunk.
    pub trim: bool,
    pub num_thread: Option<u32>,
    pub num_ctx: u32,
    pub keep_alive: String,
    pub stream: bool,
    pub num_predict: i32,
}

impl Config {
    /// a: 4B, 6 chunks, full text (the first version)
    /// b: 4B, 4 chunks trimmed to ~900 prompt tokens
    /// c: 1.7B with b's context
    pub fn preset(name: &str) -> Option<Self> {
        let base = Self {
            name: name.into(),
            model: ANSWER_MODEL.into(),
            max_chunks: 6,
            chunk_budget: CONTEXT_BUDGET,
            trim: false,
            num_thread: None,
            num_ctx: NUM_CTX,
            keep_alive: "10m".into(),
            stream: true,
            num_predict: 700,
        };
        Some(match name {
            "a" => base,
            "b" => Self {
                max_chunks: 4,
                chunk_budget: 480,
                trim: true,
                num_ctx: 2048,
                ..base
            },
            "c" => Self {
                model: SMALL_MODEL.into(),
                max_chunks: 4,
                chunk_budget: 480,
                trim: true,
                num_ctx: 2048,
                ..base
            },
            _ => return None,
        })
    }

    fn thinking_model(&self) -> bool {
        !self.model.contains("instruct")
    }
}
pub const NUM_CTX: u32 = 4096;
/// Estimated-token budget for chunk context inside NUM_CTX (system prompt,
/// question, vehicle and the reply need the rest).
pub const CONTEXT_BUDGET: usize = 2600;
/// Per-chunk cap (long OCR step tables), in estimated tokens.
pub const CHUNK_CAP: usize = 700;

pub const SYSTEM_PROMPT: &str = "You are LocalMotive, a repair assistant for one vehicle. Answer ONLY from the manual chunks in the user message and the vehicle record below.

Rules:
- Split the answer into short claims. Every claim that states a fact from the manual must cite the id(s) of the chunk(s) it comes from; facts about the owner's vehicle cite \"vehicle\".
- Copy every number, range, unit and DTC code exactly as written in the chunk (for example \"1.52 mm (0.060 in.)\", \"20 N.m (15 lb ft)\", \"525-5,700 RPM\"). Never convert units, round, or compute new numbers.
- Text shown as [see image] could not be read reliably: do not guess it; say the value is in the manual's table image and cite that chunk.
- If the chunks do not cover part of the question, add a claim saying so (no citation needed when it contains no numbers).
- When a chunk contains a diagnostic table or procedure with a next step, set next_step to that chunk id and the step label (for example \"Step 2\"); otherwise set next_step to null.
- Answer the question itself from the most relevant chunk. Do not answer only with facts about the vehicle record.
- When listing procedure steps, keep the manual's order.
- Be brief and practical. No safety boilerplate beyond what the manual says.";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelDraft {
    answer: Vec<Claim>,
    next_step: Option<NextStep>,
}

/// JSON schema for the model's reply; cite ids are limited to `ids`.
pub fn draft_schema(ids: &[String]) -> serde_json::Value {
    let mut cite_ids: Vec<String> = ids.to_vec();
    cite_ids.push(VEHICLE_SOURCE_ID.to_string());
    json!({
        "type": "object",
        "properties": {
            "answer": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "cites": {"type": "array", "items": {"type": "string", "enum": cite_ids}}
                    },
                    "required": ["text", "cites"]
                }
            },
            "next_step": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string", "enum": ids},
                            "step_label": {"type": "string"}
                        },
                        "required": ["chunk_id", "step_label"]
                    }
                ]
            }
        },
        "required": ["answer", "next_step"]
    })
}

/// Truncate to roughly `cap` estimated tokens at a line boundary.
fn cap_text(text: &str, cap: usize) -> String {
    if approx_tokens(text) <= cap {
        return text.to_string();
    }
    let mut out = String::new();
    for line in text.lines() {
        if approx_tokens(&out) + approx_tokens(line) > cap {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("[…truncated]");
    out
}

/// Keep the title and breadcrumb lines, then the window of body lines around
/// the line with the most query-term hits, within `budget` estimated tokens.
/// Lines are copied verbatim, so every number stays a substring of the chunk.
pub fn trim_to_hits(chunk: &Chunk, terms: &[String], budget: usize) -> String {
    let lines: Vec<&str> = chunk.text.lines().collect();
    let head = format!("{}\n{}", chunk.title, chunk.breadcrumb);
    let score = |l: &str| {
        let l = l.to_lowercase();
        terms
            .iter()
            .filter(|t| l.contains(t.as_str()) || (t.len() > 5 && l.contains(&t[..5])))
            .count()
    };
    let best = (0..lines.len())
        .max_by_key(|&i| (score(lines[i]), std::cmp::Reverse(i)))
        .unwrap_or(0);
    let room = budget.saturating_sub(approx_tokens(&head));
    let (mut lo, mut hi) = (best, best); // inclusive window
    let mut used = lines.get(best).map_or(0, |l| approx_tokens(l));
    if used > room {
        return format!(
            "{head}\n{}",
            cap_text(lines.get(best).copied().unwrap_or(""), room)
        );
    }
    loop {
        let mut grew = false;
        for next in [hi + 1, lo.wrapping_sub(1)] {
            if let Some(l) = lines.get(next) {
                let t = approx_tokens(l);
                if used + t <= room {
                    used += t;
                    if next > hi {
                        hi = next
                    } else {
                        lo = next
                    }
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    let mut out = head;
    if lo > 0 {
        out.push_str("\n[…]");
    }
    for l in &lines[lo..=hi.min(lines.len().saturating_sub(1))] {
        out.push('\n');
        out.push_str(l);
    }
    if hi + 1 < lines.len() {
        out.push_str("\n[…]");
    }
    out
}

/// Chunks that fit the context budget, in retrieval order.
pub fn context_chunks(r: &RetrievedFile) -> Vec<(Chunk, String)> {
    let cfg = Config::preset("a").expect("preset a");
    context_chunks_with(r, &cfg)
}

pub fn context_chunks_with(r: &RetrievedFile, cfg: &Config) -> Vec<(Chunk, String)> {
    let terms = lm_retrieve::query_terms(&r.query);
    let n = r.chunks.len().min(cfg.max_chunks).max(1);
    let per_chunk = if cfg.trim {
        cfg.chunk_budget / n
    } else {
        CHUNK_CAP
    };
    let mut used = 0;
    let mut out = Vec::new();
    for rc in r.chunks.iter().take(cfg.max_chunks) {
        let c = &rc.chunk;
        let body = if cfg.trim {
            trim_to_hits(c, &terms, per_chunk)
        } else {
            cap_text(&c.context_text(), CHUNK_CAP)
        };
        let t = approx_tokens(&body);
        if used + t > cfg.chunk_budget && !out.is_empty() {
            continue;
        }
        used += t;
        out.push((c.clone(), body));
    }
    out
}

/// Fixed per-vehicle prefix (identical across turns, so Ollama reuses its
/// prompt cache): rules, reply format, vehicle record.
pub fn system_prompt(vehicle: &Vehicle) -> String {
    format!(
        "{SYSTEM_PROMPT}\n\nReply format: JSON {{\"answer\": [{{\"text\": string, \"cites\": [chunk id or \"vehicle\"]}}], \"next_step\": {{\"chunk_id\": string, \"step_label\": string}} or null}}.\n\nVEHICLE (cite as \"vehicle\"):\n{}",
        vehicle.context_text()
    )
}

pub fn user_prompt(question: &str, chunks: &[(Chunk, String)]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "MANUAL CHUNKS:");
    for (c, body) in chunks {
        let ocr = if c.source == Source::Ocr {
            " (OCR of a table image; [see image] marks unreadable values)"
        } else {
            ""
        };
        let _ = writeln!(s, "\n=== chunk {}{ocr} ===\n{body}", c.id);
    }
    let _ = writeln!(s, "\nQUESTION: {question}");
    s
}

/// Wall time and memory per stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timings {
    pub route_ms: u64,
    pub retrieve_ms: u64,
    pub draft_ms: Vec<u64>,
    pub verify_ms: Vec<u64>,
    pub total_ms: u64,
    /// Peak resident memory of the Ollama server + model runner while drafting, in MiB.
    pub ollama_peak_rss_mib: f64,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    /// Time to first streamed token per draft (prompt processing + load).
    #[serde(default)]
    pub first_token_ms: Vec<u64>,
    #[serde(default)]
    pub config: String,
}

/// What the user is shown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub question: String,
    pub claims: Vec<ShownClaim>,
    pub next_step: Option<ShownNextStep>,
    pub not_verified: Vec<String>,
    pub retried: bool,
    pub all_verified: bool,
    /// The retrieved page that best matches the question (title terms, then rank).
    #[serde(default)]
    pub best_match: Option<BestMatch>,
    /// The verified claims probably do not answer the question; point to `best_match`.
    #[serde(default)]
    pub weak: bool,
    #[serde(default)]
    pub weak_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestMatch {
    pub chunk_id: String,
    pub page_id: u32,
    pub title: String,
    pub breadcrumb: String,
    /// The page has OCR'd step tables (open it in the procedure walker).
    pub has_steps: bool,
}

fn term_hits(text: &str, terms: &[String]) -> usize {
    let t = text.to_lowercase();
    terms
        .iter()
        .filter(|w| t.contains(w.as_str()) || (w.len() > 5 && t.contains(&w[..5])))
        .count()
}

/// Retrieved page whose title (and section heading) has the most query-term
/// hits; ties go to the higher-ranked chunk.
pub fn best_match(retrieved: &RetrievedFile) -> Option<BestMatch> {
    let terms = lm_retrieve::query_terms(&retrieved.query);
    let (_, best) = retrieved.chunks.iter().enumerate().max_by_key(|(i, r)| {
        (
            term_hits(
                &format!("{} {}", r.chunk.title, r.chunk.section_heading),
                &terms,
            ),
            std::cmp::Reverse(*i),
        )
    })?;
    let c = &best.chunk;
    let has_steps = retrieved.chunks.iter().any(|r| {
        r.chunk.page_id == c.page_id
            && r.chunk.source == Source::Ocr
            && r.chunk.section_heading.starts_with("Step")
    });
    Some(BestMatch {
        chunk_id: c.id.clone(),
        page_id: c.page_id,
        title: c.title.clone(),
        breadcrumb: c.breadcrumb.clone(),
        has_steps,
    })
}

/// Deterministic relevance check. Weak when no verified claim cites a manual
/// chunk, or the cited pages miss the best-matching page while their titles
/// share no query term and the best page's title does.
pub fn relevance(
    retrieved: &RetrievedFile,
    v: &VerifiedFile,
    best: Option<&BestMatch>,
) -> (bool, String) {
    let terms = lm_retrieve::query_terms(&retrieved.query);
    let cited: Vec<&Chunk> = v
        .claims
        .iter()
        .filter(|c| c.verdict == Verdict::Pass)
        .flat_map(|c| c.cites.iter())
        .filter_map(|id| {
            retrieved
                .chunks
                .iter()
                .find(|r| &r.chunk.id == id)
                .map(|r| &r.chunk)
        })
        .collect();
    if cited.is_empty() {
        return (true, "no verified statement from the manual".into());
    }
    let Some(best) = best else {
        return (false, String::new());
    };
    let cites_best = cited.iter().any(|c| c.page_id == best.page_id);
    let best_hits = term_hits(&best.title, &terms);
    let cited_hits = cited
        .iter()
        .map(|c| term_hits(&c.title, &terms))
        .max()
        .unwrap_or(0);
    if !cites_best && best_hits > 0 && cited_hits < best_hits {
        return (
            true,
            format!(
                "the answer does not use the best-matching page \"{}\"",
                best.title
            ),
        );
    }
    (false, String::new())
}

/// Where a claim sits in its source chunk: position of its most specific
/// token (a number if any, else its rarest long word in that chunk).
fn anchor(claim: &str, chunk: &Chunk) -> usize {
    let hay = chunk.text.to_lowercase();
    let nums = lm_verify::extract_numbers(claim);
    if let Some(p) = nums
        .iter()
        .filter_map(|n| hay.find(&n.to_lowercase()))
        .min()
    {
        return p;
    }
    let words: Vec<String> = claim
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 5)
        .map(str::to_lowercase)
        .collect();
    words
        .iter()
        .filter(|w| hay.contains(w.as_str()))
        .min_by_key(|w| hay.matches(w.as_str()).count())
        .and_then(|w| hay.find(w.as_str()))
        .unwrap_or(usize::MAX)
}

/// Order claims as the manual does: vehicle facts first, then by the rank of
/// the cited chunk, then by position inside it; uncited notes last.
pub fn order_claims<'a>(
    claims: Vec<&'a lm_types::VerifiedClaim>,
    retrieved: &RetrievedFile,
) -> Vec<&'a lm_types::VerifiedClaim> {
    let mut keyed: Vec<((u8, usize, usize), usize, &lm_types::VerifiedClaim)> = claims
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let first = c.cites.iter().find_map(|id| {
                retrieved
                    .chunks
                    .iter()
                    .position(|r| &r.chunk.id == id)
                    .map(|rank| (rank, &retrieved.chunks[rank].chunk))
            });
            let key = match first {
                Some((rank, chunk)) => (1, rank, anchor(&c.text, chunk)),
                None if c.cites.iter().any(|id| id == VEHICLE_SOURCE_ID) => (0, 0, i),
                None => (2, 0, i),
            };
            (key, i, c)
        })
        .collect();
    keyed.sort_by_key(|(k, i, _)| (*k, *i));
    keyed.into_iter().map(|(_, _, c)| c).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShownClaim {
    pub text: String,
    pub sources: Vec<String>,
    /// Images to show beside the claim (OCR sources).
    pub images: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShownNextStep {
    pub step_label: String,
    pub source: String,
    pub images: Vec<String>,
}

pub struct Turn {
    pub dir: PathBuf,
    pub route: RouteFile,
    pub retrieved: RetrievedFile,
    pub verified: VerifiedFile,
    pub answer: Answer,
    pub timings: Timings,
}

pub fn sessions_root() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join("localmotive/sessions")
}

pub fn vehicle_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join("localmotive/vehicle.json")
}

/// New session id: `s<unix seconds>-<pid>`.
pub fn new_session_id() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("s{secs}-{}", std::process::id())
}

fn next_turn_dir(session: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(session)?;
    let n = std::fs::read_dir(session)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("turn-"))
        .count();
    let dir = session.join(format!("turn-{}", n + 1));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn write_json<T: Serialize>(dir: &Path, name: &str, v: &T) -> Result<()> {
    std::fs::write(dir.join(name), serde_json::to_string_pretty(v)?)
        .with_context(|| format!("writing {name}"))
}

fn ms(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

pub fn retry_message(v: &VerifiedFile) -> String {
    let mut s = String::from(
        "Your answer failed verification. Fix ONLY these problems and reply with the full corrected JSON. \
         Copy numbers and units exactly as they appear in the cited chunk, cite the chunk that contains each number, \
         or drop the claim if no chunk supports it.\n",
    );
    for c in v.claims.iter().filter(|c| c.verdict == Verdict::Fail) {
        let _ = writeln!(s, "- claim {:?}: {}", c.text, c.reasons.join("; "));
    }
    if let Some(n) = v.next_step.as_ref().filter(|n| n.verdict == Verdict::Fail) {
        let _ = writeln!(
            s,
            "- next_step {:?}: {}",
            n.step_label,
            n.reasons.join("; ")
        );
    }
    s
}

/// Build what the user sees: passing claims only, plus not-verified notes.
pub fn shown_answer(
    question: &str,
    retrieved: &RetrievedFile,
    v: &VerifiedFile,
    retried: bool,
) -> Answer {
    let find = |id: &str| {
        retrieved
            .chunks
            .iter()
            .find(|r| r.chunk.id == id)
            .map(|r| &r.chunk)
    };
    let label = |id: &str| -> String {
        if id == VEHICLE_SOURCE_ID {
            return "your vehicle record".into();
        }
        find(id).map_or_else(
            || id.to_string(),
            |c| format!("{} (page {})", c.title, c.page_id),
        )
    };
    let images = |ids: &[String]| -> Vec<String> {
        ids.iter()
            .filter_map(|id| find(id))
            .filter(|c| c.source == Source::Ocr)
            .flat_map(|c| c.images.clone())
            .collect()
    };
    let mut claims = Vec::new();
    let mut not_verified = Vec::new();
    let passing = order_claims(
        v.claims
            .iter()
            .filter(|c| c.verdict == Verdict::Pass)
            .collect(),
        retrieved,
    );
    for c in passing {
        claims.push(ShownClaim {
            text: c.text.clone(),
            sources: c.cites.iter().map(|id| label(id)).collect(),
            images: images(&c.cites),
        });
    }
    for c in &v.claims {
        if c.verdict == Verdict::Pass {
            continue;
        } else {
            let srcs: Vec<String> = c
                .cites
                .iter()
                .filter(|id| find(id).is_some() || *id == VEHICLE_SOURCE_ID)
                .map(|id| label(id))
                .collect();
            not_verified.push(if srcs.is_empty() {
                "A statement was removed because it could not be checked against the manual.".to_string()
            } else {
                format!("A statement was removed because its details could not be verified; see {} directly.", srcs.join(", "))
            });
        }
    }
    let next_step = v
        .next_step
        .as_ref()
        .filter(|n| n.verdict == Verdict::Pass)
        .map(|n| ShownNextStep {
            step_label: n.step_label.clone(),
            source: label(&n.chunk_id),
            images: images(std::slice::from_ref(&n.chunk_id)),
        });
    let best = best_match(retrieved);
    let (weak, weak_reason) = relevance(retrieved, v, best.as_ref());
    Answer {
        question: question.into(),
        all_verified: v.ok,
        claims,
        next_step,
        not_verified,
        retried,
        best_match: best,
        weak,
        weak_reason,
    }
}

pub fn answer_markdown(a: &Answer) -> String {
    let mut s = format!("**Q:** {}\n\n", a.question);
    for (i, c) in a.claims.iter().enumerate() {
        let _ = writeln!(
            s,
            "{}. {}  \n   _source: {}_",
            i + 1,
            c.text,
            c.sources.join("; ")
        );
        for img in &c.images {
            let _ = writeln!(
                s,
                "   ⚠ read from a table image — check the picture: `{img}`"
            );
        }
    }
    if let Some(n) = &a.next_step {
        let _ = writeln!(s, "\n**Next step:** {} — _{}_", n.step_label, n.source);
        for img in &n.images {
            let _ = writeln!(s, "   table image: `{img}`");
        }
    }
    for n in &a.not_verified {
        let _ = writeln!(s, "\n_Not verified:_ {n}");
    }
    if let Some(b) = a
        .best_match
        .as_ref()
        .filter(|_| a.weak || a.claims.is_empty())
    {
        {
            let _ = writeln!(
                s,
                "\n**This may not fully answer the question** ({}). Best match in the manual: **{}** (page {}){}.",
                a.weak_reason,
                b.title,
                b.page_id,
                if b.has_steps {
                    " — open it in the procedure walker for the step tables"
                } else {
                    ""
                }
            );
        }
    }
    s
}

pub struct Ctx {
    pub conn: rusqlite::Connection,
    pub client: lm_ollama::Client,
    pub vehicle: Vehicle,
    pub cfg: Config,
    pub use_vectors: bool,
}

impl Ctx {
    pub fn open(cfg: Config) -> Result<Self> {
        let vehicle: Vehicle = serde_json::from_str(
            &std::fs::read_to_string(vehicle_path())
                .context("reading ~/localmotive/vehicle.json")?,
        )?;
        Ok(Self {
            conn: lm_index::open_readonly(&lm_index::default_db_path())?,
            client: lm_ollama::Client::new()?,
            vehicle,
            cfg,
            use_vectors: true,
        })
    }
}

async fn draft(
    ctx: &Ctx,
    messages: &[Message],
    schema: &serde_json::Value,
    t: &mut Timings,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<(DraftFile, String)> {
    let cfg = &ctx.cfg;
    let mut req = ChatRequest::new(&cfg.model, messages.to_vec());
    req.format = Some(schema.clone());
    req.options = Some(Options {
        num_ctx: Some(cfg.num_ctx),
        temperature: Some(0.0),
        seed: Some(0),
        num_thread: cfg.num_thread,
        num_predict: Some(cfg.num_predict),
    });
    req.keep_alive = Some(cfg.keep_alive.clone());
    if cfg.thinking_model() {
        req.think = Some(false);
    }
    let start = Instant::now();
    let sampler = rss::Sampler::start(rss::OLLAMA_PROCS);
    let first_token: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);
    let resp = if cfg.stream {
        let mut on_token = |tok: &str| {
            let mut ft = first_token.lock().unwrap();
            if ft.is_none() {
                *ft = Some(ms(start));
            }
            progress(Progress::Token {
                text: tok.to_string(),
            });
        };
        ctx.client.chat_stream(&req, &mut on_token).await
    } else {
        ctx.client.chat(&req).await
    };
    let peak = sampler.stop();
    t.draft_ms.push(ms(start));
    if let Some(ft) = *first_token.lock().unwrap() {
        t.first_token_ms.push(ft);
    }
    t.ollama_peak_rss_mib = t.ollama_peak_rss_mib.max(peak);
    let resp = resp?;
    t.prompt_tokens.push(resp.prompt_eval_count);
    t.output_tokens.push(resp.eval_count);
    let raw = resp.message.content;
    let m: ModelDraft =
        serde_json::from_str(&raw).with_context(|| format!("model reply is not a draft: {raw}"))?;
    Ok((
        DraftFile {
            schema_version: SCHEMA_VERSION,
            answer: m.answer,
            next_step: m.next_step,
        },
        raw,
    ))
}

async fn query_embedding(ctx: &Ctx, q: &str) -> Option<Vec<f32>> {
    if !ctx.use_vectors || lm_index::vec_count(&ctx.conn).ok()? == 0 {
        return None;
    }
    let input = vec![format!("{}{q}", lm_retrieve::QUERY_INSTRUCTION)];
    ctx.client
        .embed(lm_index::EMBED_MODEL, &input, Some("10m"))
        .await
        .ok()?
        .pop()
}

/// Run one full turn and write every stage file into a new turn directory.
pub async fn run_turn(ctx: &Ctx, session: &Path, question: &str) -> Result<Turn> {
    run_turn_with(ctx, session, question, &|_| {}).await
}

/// Progress of a turn, for "thinking aloud" UIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Progress {
    StageStart {
        stage: String,
    },
    StageEnd {
        stage: String,
        ms: u64,
        detail: String,
    },
    Retrieved {
        chunks: Vec<RetrievedSummary>,
    },
    Verdicts {
        attempt: u32,
        passed: usize,
        failed: usize,
        reasons: Vec<String>,
    },
    Retry {
        reasons: usize,
    },
    Token {
        text: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedSummary {
    pub id: String,
    pub page_id: u32,
    pub title: String,
    pub kind: String,
    pub ocr: bool,
}

fn verdict_event(attempt: u32, v: &VerifiedFile) -> Progress {
    Progress::Verdicts {
        attempt,
        passed: v
            .claims
            .iter()
            .filter(|c| c.verdict == Verdict::Pass)
            .count(),
        failed: v
            .claims
            .iter()
            .filter(|c| c.verdict == Verdict::Fail)
            .count(),
        reasons: v.claims.iter().flat_map(|c| c.reasons.clone()).collect(),
    }
}

/// [`run_turn`] with a progress callback.
pub async fn run_turn_with(
    ctx: &Ctx,
    session: &Path,
    question: &str,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<Turn> {
    let start_stage = |s: &str| progress(Progress::StageStart { stage: s.into() });
    let end_stage = |s: &str, ms: u64, detail: String| {
        progress(Progress::StageEnd {
            stage: s.into(),
            ms,
            detail,
        })
    };
    let total = Instant::now();
    let dir = next_turn_dir(session)?;
    let mut t = Timings::default();

    start_stage("route");
    let s = Instant::now();
    let route = lm_retrieve::route(question);
    t.route_ms = ms(s);
    write_json(&dir, "route.json", &route)?;
    end_stage(
        "route",
        t.route_ms,
        format!("codes {:?}, kinds {:?}", route.dtc_codes, route.kinds),
    );

    start_stage("retrieve");
    let s = Instant::now();
    let emb = query_embedding(ctx, question).await;
    let hybrid = emb.is_some();
    let retrieved = lm_retrieve::retrieve(&ctx.conn, &route, emb.as_deref())?;
    t.retrieve_ms = ms(s);
    write_json(&dir, "retrieved.json", &retrieved)?;
    end_stage(
        "retrieve",
        t.retrieve_ms,
        format!(
            "{} chunks ({})",
            retrieved.chunks.len(),
            if hybrid { "hybrid" } else { "BM25" }
        ),
    );
    progress(Progress::Retrieved {
        chunks: retrieved
            .chunks
            .iter()
            .map(|r| RetrievedSummary {
                id: r.chunk.id.clone(),
                page_id: r.chunk.page_id,
                title: r.chunk.title.clone(),
                kind: r.chunk.kind.as_str().into(),
                ocr: r.chunk.source == Source::Ocr,
            })
            .collect(),
    });

    t.config = ctx.cfg.name.clone();
    let chunks = context_chunks_with(&retrieved, &ctx.cfg);
    let ids: Vec<String> = chunks.iter().map(|(c, _)| c.id.clone()).collect();
    let schema = draft_schema(&ids);
    let mut messages = vec![
        Message::system(system_prompt(&ctx.vehicle)),
        Message::user(user_prompt(question, &chunks)),
    ];

    start_stage("draft");
    let (d1, raw1) = draft(ctx, &messages, &schema, &mut t, progress).await?;
    end_stage(
        "draft",
        *t.draft_ms.last().unwrap_or(&0),
        format!("{} claims", d1.answer.len()),
    );
    start_stage("verify");
    let s = Instant::now();
    let v1 = lm_verify::verify(&d1, &retrieved, Some(&ctx.vehicle));
    t.verify_ms.push(ms(s));
    end_stage(
        "verify",
        *t.verify_ms.last().unwrap_or(&0),
        if v1.ok {
            "all claims verified".into()
        } else {
            "some claims failed".into()
        },
    );
    progress(verdict_event(1, &v1));

    let (final_draft, verified, retried) = if v1.ok {
        (d1, v1, false)
    } else {
        write_json(&dir, "draft.attempt1.json", &d1)?;
        write_json(&dir, "verified.attempt1.json", &v1)?;
        messages.push(Message {
            role: "assistant".into(),
            content: raw1,
        });
        messages.push(Message::user(retry_message(&v1)));
        progress(Progress::Retry {
            reasons: v1.claims.iter().map(|c| c.reasons.len()).sum(),
        });
        start_stage("draft");
        let (d2, _) = draft(ctx, &messages, &schema, &mut t, progress).await?;
        end_stage(
            "draft",
            *t.draft_ms.last().unwrap_or(&0),
            format!("{} claims (retry)", d2.answer.len()),
        );
        start_stage("verify");
        let s = Instant::now();
        let v2 = lm_verify::verify(&d2, &retrieved, Some(&ctx.vehicle));
        t.verify_ms.push(ms(s));
        end_stage(
            "verify",
            *t.verify_ms.last().unwrap_or(&0),
            if v2.ok {
                "all claims verified".into()
            } else {
                "some claims failed".into()
            },
        );
        progress(verdict_event(2, &v2));
        (d2, v2, true)
    };
    write_json(&dir, "draft.json", &final_draft)?;
    write_json(&dir, "verified.json", &verified)?;

    let answer = shown_answer(question, &retrieved, &verified, retried);
    write_json(&dir, "answer.json", &answer)?;
    std::fs::write(dir.join("answer.md"), answer_markdown(&answer))?;
    t.total_ms = ms(total);
    write_json(&dir, "timings.json", &t)?;
    Ok(Turn {
        dir,
        route,
        retrieved,
        verified,
        answer,
        timings: t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lm_types::{Kind, RetrievedChunk};

    fn chunk(id: &str, text: &str, source: Source) -> Chunk {
        Chunk {
            id: id.into(),
            page_id: 8668,
            title: "Spark Plug Gap".into(),
            breadcrumb: "Specifications > Spark Plug Gap".into(),
            aliases: vec![],
            section_heading: String::new(),
            text: text.into(),
            images: vec!["images/x.png".into()],
            links: vec![],
            kind: Kind::Spec,
            dtc_codes: vec![],
            image_only: false,
            source,
            ocr_confidence: None,
            ocr_raw: None,
            ocr_checked_rows: vec![],
            ocr_masked_values: 0,
            hidden: false,
        }
    }

    #[test]
    fn schema_restricts_cites_to_retrieved_ids() {
        let s = draft_schema(&["aaaa".into()]);
        assert_eq!(
            s["properties"]["answer"]["items"]["properties"]["cites"]["items"]["enum"],
            json!(["aaaa", "vehicle"])
        );
        assert_eq!(
            s["properties"]["next_step"]["anyOf"][1]["properties"]["chunk_id"]["enum"],
            json!(["aaaa"])
        );
    }

    #[test]
    fn shown_answer_drops_failures_and_flags_images() {
        let retrieved = RetrievedFile {
            schema_version: 1,
            query: "gap".into(),
            chunks: vec![
                RetrievedChunk {
                    score: 1.0,
                    chunk: chunk("aaaa", "Spark Plug Gap 1.52 mm (0.060 in.)", Source::Html),
                },
                RetrievedChunk {
                    score: 0.5,
                    chunk: chunk("bbbb", "Spark Plug 15 N.m 11 lb ft", Source::Ocr),
                },
            ],
        };
        let draft = DraftFile {
            schema_version: 1,
            answer: vec![
                Claim {
                    text: "Gap the plugs to 1.52 mm (0.060 in.).".into(),
                    cites: vec!["aaaa".into()],
                },
                Claim {
                    text: "Gap the plugs to 1.5 mm.".into(),
                    cites: vec!["aaaa".into()],
                },
                Claim {
                    text: "Torque them to 15 N.m.".into(),
                    cites: vec!["bbbb".into()],
                },
            ],
            next_step: None,
        };
        let v = lm_verify::verify(&draft, &retrieved, None);
        let a = shown_answer("gap", &retrieved, &v, false);
        assert_eq!(a.claims.len(), 2);
        assert_eq!(a.not_verified.len(), 1);
        assert!(
            !a.not_verified[0].contains("1.5"),
            "wrong number must not be shown"
        );
        assert_eq!(a.claims[1].images, vec!["images/x.png"]);
        let md = answer_markdown(&a);
        assert!(md.contains("check the picture"));
    }

    #[test]
    fn claims_follow_manual_order_and_weak_answers_point_to_best_page() {
        let mut steps = chunk(
            "aaaa",
            "1. Remove the caliper bolts.\n2. Pivot the caliper up.\n3. Install the lower guide pin and tighten to 54 Nm (40 ft. lbs.).",
            Source::Html,
        );
        steps.title = "Brake Pads Replacement - Front".into();
        let mut other = chunk("bbbb", "Lubricate the caliper bolt boots.", Source::Html);
        other.title = "Brake Caliper Bracket Replacement".into();
        other.page_id = 14257;
        let retrieved = RetrievedFile {
            schema_version: 1,
            query: "replace front brake pads".into(),
            chunks: vec![
                RetrievedChunk {
                    score: 1.0,
                    chunk: steps,
                },
                RetrievedChunk {
                    score: 0.5,
                    chunk: other,
                },
            ],
        };
        let claim = |t: &str, id: &str| Claim {
            text: t.into(),
            cites: vec![id.into()],
        };
        let draft = DraftFile {
            schema_version: 1,
            answer: vec![
                claim("Lubricate the caliper bolt boots.", "bbbb"),
                claim(
                    "Tighten the lower guide pin to 54 Nm (40 ft. lbs.).",
                    "aaaa",
                ),
                claim("Pivot the caliper up.", "aaaa"),
            ],
            next_step: None,
        };
        let v = lm_verify::verify(&draft, &retrieved, None);
        let a = shown_answer("replace front brake pads", &retrieved, &v, false);
        let order: Vec<&str> = a.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            order,
            vec![
                "Pivot the caliper up.",
                "Tighten the lower guide pin to 54 Nm (40 ft. lbs.).",
                "Lubricate the caliper bolt boots."
            ]
        );
        assert_eq!(
            a.best_match.as_ref().unwrap().title,
            "Brake Pads Replacement - Front"
        );
        assert!(!a.weak);

        // only a claim citing the weaker page: weak, points at the best page
        let draft = DraftFile {
            schema_version: 1,
            answer: vec![claim("Lubricate the caliper bolt boots.", "bbbb")],
            next_step: None,
        };
        let v = lm_verify::verify(&draft, &retrieved, None);
        let a = shown_answer("replace front brake pads", &retrieved, &v, false);
        assert!(a.weak, "{a:?}");
        assert!(
            answer_markdown(&a)
                .contains("Best match in the manual: **Brake Pads Replacement - Front**")
        );

        // vehicle-only answer: weak
        let draft = DraftFile {
            schema_version: 1,
            answer: vec![],
            next_step: None,
        };
        let v = lm_verify::verify(&draft, &retrieved, None);
        assert!(shown_answer("replace front brake pads", &retrieved, &v, false).weak);
    }

    #[test]
    fn context_respects_budget() {
        let big = "word ".repeat(2000);
        let retrieved = RetrievedFile {
            schema_version: 1,
            query: "q".into(),
            chunks: (0..6)
                .map(|i| RetrievedChunk {
                    score: 1.0,
                    chunk: chunk(&format!("c{i}"), &big, Source::Html),
                })
                .collect(),
        };
        let ctx = context_chunks(&retrieved);
        let used: usize = ctx.iter().map(|(_, b)| approx_tokens(b)).sum();
        assert!(used <= CONTEXT_BUDGET + CHUNK_CAP);
        assert!(ctx.iter().all(|(_, b)| approx_tokens(b) <= CHUNK_CAP + 2));
    }
}
