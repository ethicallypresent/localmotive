//! Routing and retrieval: `route.json` → `retrieved.json`.
//!
//! Router v0 is regex-only: DTC codes, spec words (torque, gap, capacity…)
//! and "next step" phrasing. Retrieval:
//!
//! 1. exact-DTC chunks first (the code's own page, then TSBs, then others);
//! 2. ranked lists fused with reciprocal rank fusion (k = 60): FTS5 BM25 over
//!    everything, BM25 restricted to the routed kinds (if any), and vector
//!    KNN when `vec_chunks` has rows and a query embedding is supplied;
//! 3. one chunk per page, at most [`MAX_RESULTS`].

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use anyhow::Result;
use lm_index::{CHUNK_COLUMNS, f32_blob, row_to_chunk};
use lm_types::{Chunk, Kind, RetrievedChunk, RetrievedFile, RouteFile, SCHEMA_VERSION};
use regex::Regex;
use rusqlite::{Connection, params_from_iter};

pub mod golden;

pub const MAX_RESULTS: usize = 6;
pub const RRF_K: f64 = 60.0;
const LIST_DEPTH: usize = 60;
/// Pages per DTC code taken from the exact-match list before fused results.
pub const DTC_EXACT_PAGES: usize = 3;
/// FTS5 bm25 column weights: title, breadcrumb, text, dtc.
const BM25_WEIGHTS: &str = "6.0, 2.0, 1.0, 8.0";
/// Query instruction for qwen3-embedding (documents are embedded without one).
pub const QUERY_INSTRUCTION: &str = "Instruct: Given a car owner's question about a 2006 Chevrolet Uplander, retrieve the service manual passage that answers it\nQuery: ";

static RE_DTC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b([pbcu])\s?([0-9]{4})\b").unwrap());
static RE_SPEC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(torque|torques|tighten|tightening|gap|spec|specs|specification|specifications|capacity|capacities|how much (oil|fluid|coolant)|ft[- ]?lbs?|lb[- ]?ft|n[·.]?m|firing order|pressure spec|clearance)\b").unwrap()
});
static RE_NEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(next step|what('s| is)? next|what now|then what|step \d+|go to step)\b")
        .unwrap()
});

/// General automotive vocabulary → the manual's wording, added as extra OR
/// terms. Written from common owner phrasing, not from golden-set misses.
static SYNONYMS: LazyLock<Vec<(Regex, &'static [&'static str])>> = LazyLock::new(|| {
    let rules: &[(&str, &[&str])] = &[
        (
            r"check engine|service engine soon|engine light|\bcel\b",
            &["malfunction", "indicator", "mil"],
        ),
        (r"serpentine|fan belt|accessory belt", &["drive", "belt"]),
        (r"headlights?\b", &["headlamp", "headlamps"]),
        (r"tail ?lights?\b", &["tail", "lamp"]),
        (r"brake lights?\b", &["stop", "lamp"]),
        (r"blinkers?\b|turn signals?\b", &["turn", "signal", "lamp"]),
        (
            r"how much (\w+ ){0,2}(oil|coolant|fluid|antifreeze|refrigerant|freon)|how many (quarts|liters|litres)",
            &["capacity", "capacities"],
        ),
        (r"antifreeze", &["coolant"]),
        (r"\babs\b", &["antilock"]),
        (
            r"\b(warning )?lights? (is |stays |comes |keeps )?(on|lit)\b",
            &["indicator"],
        ),
        (
            r"battery (keeps )?(dying|dies|dead|drains?|draining)|(dead|drained|dying) battery",
            &["parasitic", "drain", "discharged"],
        ),
        (
            r"won'?t start|will not start|doesn'?t start|no start",
            &["cranks", "start"],
        ),
        (r"\ba/?c\b|air con\b", &["air", "conditioning"]),
        (r"freon", &["refrigerant"]),
        (r"\btranny\b", &["transmission"]),
        (r"\bo2 sensor", &["oxygen", "sensor", "ho2s"]),
        (r"catalytic converter|\bcat\b", &["catalyst", "catalytic"]),
        (r"gas cap", &["fuel", "fill", "cap"]),
        (r"\bmaf\b", &["mass", "air", "flow"]),
        (r"squeal|squeak|grind(s|ing)?\b", &["noise"]),
        (r"shak(e|es|ing)\b|shimmy", &["vibration"]),
    ];
    rules
        .iter()
        .map(|(re, terms)| (Regex::new(&format!("(?i){re}")).unwrap(), *terms))
        .collect()
});

/// Extra OR terms for a query from [`SYNONYMS`].
pub fn synonym_terms(query: &str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for (re, terms) in SYNONYMS.iter() {
        if re.is_match(query) {
            for t in *terms {
                if !out.contains(t) {
                    out.push(t);
                }
            }
        }
    }
    out
}

const STOPWORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "my",
    "is",
    "it",
    "its",
    "it's",
    "at",
    "be",
    "do",
    "does",
    "how",
    "what",
    "when",
    "why",
    "where",
    "which",
    "i",
    "me",
    "can",
    "should",
    "would",
    "will",
    "are",
    "was",
    "there",
    "this",
    "that",
    "from",
    "by",
    "as",
    "get",
    "got",
    "have",
    "has",
    "van",
    "car",
    "vehicle",
    "uplander",
    "chevy",
    "chevrolet",
    "2006",
    "please",
    "need",
    "much",
    "keeps",
    "keep",
    "up",
    "out",
    "off",
    "so",
    "if",
    "about",
    "after",
];

/// Router v0: regex only.
pub fn route(query: &str) -> RouteFile {
    let mut codes: Vec<String> = Vec::new();
    for c in RE_DTC.captures_iter(query) {
        let code = format!("{}{}", c[1].to_uppercase(), &c[2]);
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    let kinds = if RE_SPEC.is_match(query) {
        vec![Kind::Spec]
    } else {
        Vec::new()
    };
    RouteFile {
        schema_version: SCHEMA_VERSION,
        query: query.to_string(),
        dtc_codes: codes,
        kinds,
        breadcrumb_prefixes: Vec::new(),
        wants_next_step: RE_NEXT.is_match(query),
    }
}

/// FTS5 MATCH expression: quoted non-stopword terms joined with OR.
pub fn fts_query(query: &str) -> Option<String> {
    let terms = query_terms(query);
    if terms.is_empty() {
        None
    } else {
        Some(
            terms
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" OR "),
        )
    }
}

/// Lowercased search terms: non-stopword words, DTC codes, synonym terms.
pub fn query_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for t in query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let t = t.to_lowercase();
        if (t.len() < 2 && !t.chars().all(|c| c.is_ascii_digit()))
            || STOPWORDS.contains(&t.as_str())
        {
            continue;
        }
        if !terms.contains(&t) {
            terms.push(t);
        }
    }
    // re-join split codes ("p 0300")
    for c in RE_DTC.captures_iter(query) {
        let code = format!("{}{}", c[1].to_lowercase(), &c[2]);
        if !terms.contains(&code) {
            terms.push(code);
        }
    }
    for t in synonym_terms(query) {
        if !terms.iter().any(|x| x == t) {
            terms.push(t.to_string());
        }
    }
    terms
}

/// Ranked rids from BM25, optionally restricted to kinds.
pub fn bm25(conn: &Connection, query: &str, kinds: &[Kind], limit: usize) -> Result<Vec<i64>> {
    let Some(q) = fts_query(query) else {
        return Ok(Vec::new());
    };
    let kind_clause = if kinds.is_empty() {
        String::new()
    } else {
        format!(
            " AND c.kind IN ({})",
            kinds
                .iter()
                .map(|k| format!("'{}'", k.as_str()))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let sql = format!(
        "SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.rid = f.rowid
         WHERE chunks_fts MATCH ?1 AND c.hidden = 0{kind_clause}
         ORDER BY bm25(chunks_fts, {BM25_WEIGHTS}) LIMIT {limit}"
    );
    let mut st = conn.prepare(&sql)?;
    let rows = st
        .query_map([q], |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<i64>>>()?;
    Ok(rows)
}

/// Ranked rids from vector KNN.
pub fn knn(conn: &Connection, embedding: &[f32], limit: usize) -> Result<Vec<i64>> {
    let mut st = conn.prepare(&format!(
        "SELECT rowid FROM vec_chunks WHERE embedding MATCH ?1 AND k = {limit} ORDER BY distance"
    ))?;
    let rows = st
        .query_map([f32_blob(embedding)], |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<i64>>>()?;
    Ok(rows)
}

/// Exact-DTC chunks for each code: the code's own page first (title starts
/// with the code), then TSBs, then other mentions; one chunk per page (the
/// page's best BM25 chunk for this query), at most [`DTC_EXACT_PAGES`] pages.
pub fn dtc_exact(conn: &Connection, codes: &[String], query: &str) -> Result<Vec<i64>> {
    let bm: HashMap<i64, usize> = bm25(conn, query, &[], 500)?
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r, i))
        .collect();
    let mut out = Vec::new();
    for code in codes {
        let mut st = conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE hidden = 0 AND rid IN (SELECT rid FROM chunk_dtc WHERE code = ?1)"
        ))?;
        let mut rows: Vec<(i64, Chunk)> = st
            .query_map([code], row_to_chunk)?
            .collect::<rusqlite::Result<_>>()?;
        let tier = |c: &Chunk| -> u8 {
            let own = c.title == *code
                || c.title.starts_with(&format!("{code} "))
                || c.title.starts_with(&format!("DTC {code}"));
            match (own, c.kind) {
                (true, Kind::Dtc) => 0,
                (true, _) => 1,
                (_, Kind::Dtc) => 3,
                (_, Kind::Tsb) => 2,
                _ => 4,
            }
        };
        rows.sort_by_key(|(rid, c)| (tier(c), bm.get(rid).copied().unwrap_or(usize::MAX), *rid));
        let mut pages = HashSet::new();
        out.extend(
            rows.into_iter()
                .filter(|(_, c)| pages.insert(c.page_id))
                .take(DTC_EXACT_PAGES)
                .map(|(rid, _)| rid),
        );
    }
    Ok(out)
}

pub fn rrf(lists: &[Vec<i64>]) -> Vec<(i64, f64)> {
    let mut score: HashMap<i64, f64> = HashMap::new();
    let mut first_seen: HashMap<i64, usize> = HashMap::new();
    let mut n = 0;
    for l in lists {
        for (rank, rid) in l.iter().enumerate() {
            *score.entry(*rid).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
            first_seen.entry(*rid).or_insert_with(|| {
                n += 1;
                n
            });
        }
    }
    let mut v: Vec<(i64, f64)> = score.into_iter().collect();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(first_seen[&a.0].cmp(&first_seen[&b.0]))
    });
    v
}

/// Full retrieval. `query_embedding` is used only when vectors exist.
pub fn retrieve(
    conn: &Connection,
    route: &RouteFile,
    query_embedding: Option<&[f32]>,
) -> Result<RetrievedFile> {
    let mut lists = vec![bm25(conn, &route.query, &[], LIST_DEPTH)?];
    if !route.kinds.is_empty() {
        lists.push(bm25(conn, &route.query, &route.kinds, LIST_DEPTH)?);
    }
    if let Some(e) = query_embedding
        && lm_index::vec_count(conn)? > 0
    {
        lists.push(knn(conn, e, LIST_DEPTH)?);
    }
    let fused = rrf(&lists);
    let exact = dtc_exact(conn, &route.dtc_codes, &route.query)?;

    // exact DTC chunks first (scored above any RRF score), then fused
    let mut ordered: Vec<(i64, f64)> = exact
        .iter()
        .enumerate()
        .map(|(i, r)| (*r, 1.0 + 1.0 / (i as f64 + 1.0)))
        .collect();
    ordered.extend(fused);

    let mut seen_rid = HashSet::new();
    let mut seen_page = HashSet::new();
    let mut chunks = Vec::new();
    for (rid, score) in ordered {
        if chunks.len() >= MAX_RESULTS || !seen_rid.insert(rid) {
            continue;
        }
        let Some(c) = lm_index::chunk_by_rid(conn, rid)? else {
            continue;
        };
        if c.hidden
            || !route
                .breadcrumb_prefixes
                .iter()
                .all(|p| c.breadcrumb.starts_with(p.as_str()))
            || !seen_page.insert(c.page_id)
        {
            continue;
        }
        chunks.push(RetrievedChunk {
            score: score as f32,
            chunk: c,
        });
    }

    // A second chunk from the same page when it is the page's OCR step table:
    // for a DTC's own page, and for the top page when the user wants the next step.
    let mut step_pages: Vec<u32> = chunks
        .iter()
        .filter(|r| route.dtc_codes.contains(&r.chunk.title) && r.chunk.kind == Kind::Dtc)
        .map(|r| r.chunk.page_id)
        .collect();
    if route.wants_next_step
        && let Some(top) = chunks.first()
    {
        step_pages.push(top.chunk.page_id);
    }
    for page in uniq_u32(step_pages) {
        let Some(pos) = chunks.iter().position(|r| r.chunk.page_id == page) else {
            continue;
        };
        if chunks[pos].chunk.section_heading.starts_with("Step") {
            continue;
        }
        if let Some(step) = step_table(conn, page, &route.query)? {
            let score = chunks[pos].score;
            chunks.insert(pos + 1, RetrievedChunk { score, chunk: step });
        }
    }
    chunks.truncate(MAX_RESULTS);
    Ok(RetrievedFile {
        schema_version: SCHEMA_VERSION,
        query: route.query.clone(),
        chunks,
    })
}

fn uniq_u32(v: Vec<u32>) -> Vec<u32> {
    let mut out = Vec::new();
    for x in v {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}

/// The page's step-table chunk (OCR, heading "Step N - Step M") best matching
/// the query, or its first one.
pub fn step_table(conn: &Connection, page: u32, query: &str) -> Result<Option<Chunk>> {
    let mut st = conn.prepare(&format!(
        "SELECT {CHUNK_COLUMNS} FROM chunks WHERE page_id = ?1 AND source = 'ocr' AND hidden = 0 AND section_heading LIKE 'Step %' ORDER BY rid"
    ))?;
    let rows: Vec<(i64, Chunk)> = st
        .query_map([page], row_to_chunk)?
        .collect::<rusqlite::Result<_>>()?;
    if rows.is_empty() {
        return Ok(None);
    }
    let rank: HashMap<i64, usize> = bm25(conn, query, &[], 500)?
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r, i))
        .collect();
    let best = rows
        .iter()
        .min_by_key(|(rid, _)| (rank.get(rid).copied().unwrap_or(usize::MAX), *rid))
        .map(|(_, c)| c.clone());
    Ok(best)
}

/// Rows of `chunks` for a set of rids, preserving order (used by tools).
pub fn chunks_by_rids(conn: &Connection, rids: &[i64]) -> Result<Vec<Chunk>> {
    if rids.is_empty() {
        return Ok(Vec::new());
    }
    let marks = vec!["?"; rids.len()].join(",");
    let mut st = conn.prepare(&format!(
        "SELECT {CHUNK_COLUMNS} FROM chunks WHERE rid IN ({marks})"
    ))?;
    let mut map: HashMap<i64, Chunk> = st
        .query_map(params_from_iter(rids), row_to_chunk)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rids.iter().filter_map(|r| map.remove(r)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_v0() {
        let r = route("p0300 misfire at idle, also P 0171?");
        assert_eq!(r.dtc_codes, vec!["P0300", "P0171"]);
        assert!(r.kinds.is_empty());
        assert!(!r.wants_next_step);
        assert_eq!(
            route("front brake caliper bolt torque").kinds,
            vec![Kind::Spec]
        );
        assert_eq!(route("spark plug gap").kinds, vec![Kind::Spec]);
        assert_eq!(route("engine coolant capacity").kinds, vec![Kind::Spec]);
        assert!(route("I did step 3, what's next?").wants_next_step);
        assert!(route("rough idle when cold").kinds.is_empty());
    }

    #[test]
    fn fts_query_quotes_and_drops_stopwords() {
        assert_eq!(
            fts_query("What is the spark plug gap?").unwrap(),
            r#""spark" OR "plug" OR "gap""#
        );
        assert_eq!(
            fts_query("P 0300 at idle").unwrap(),
            r#""0300" OR "idle" OR "p0300""#
        );
        assert!(fts_query("what is it").is_none());
        // FTS syntax characters can't leak through
        assert_eq!(
            fts_query("bolt\" OR NEAR(x)*").unwrap(),
            r#""bolt" OR "near""#
        );
    }

    #[test]
    fn synonyms_expand_owner_vocabulary() {
        assert_eq!(
            synonym_terms("check engine light flashing"),
            vec!["malfunction", "indicator", "mil"]
        );
        assert_eq!(
            synonym_terms("headlight bulb"),
            vec!["headlamp", "headlamps"]
        );
        assert!(synonym_terms("how much engine oil does it take").contains(&"capacity"));
        assert!(synonym_terms("ABS warning light stays on").contains(&"antilock"));
        assert!(synonym_terms("spark plug gap").is_empty());
        assert!(
            fts_query("serpentine belt replacement")
                .unwrap()
                .contains("\"drive\"")
        );
    }

    #[test]
    fn rrf_sums_reciprocal_ranks() {
        let f = rrf(&[vec![1, 2, 3], vec![3, 1]]);
        assert_eq!(f[0].0, 1);
        assert_eq!(f[1].0, 3);
        assert!((f[0].1 - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
    }
}
