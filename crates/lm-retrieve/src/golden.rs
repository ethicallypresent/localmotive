//! Golden queries (`tests/golden.yaml`) and recall@k evaluation.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Deserialize;

use crate::{MAX_RESULTS, retrieve, route};

#[derive(Debug, Clone, Deserialize)]
pub struct GoldenQuery {
    pub query: String,
    /// Page ids (dedupe representatives, as in `chunks.page_id`) that answer it.
    pub expect: Vec<u32>,
    #[serde(default)]
    pub note: String,
}

pub fn load(path: &Path) -> Result<Vec<GoldenQuery>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_norway::from_str(&s).with_context(|| format!("parsing {}", path.display()))
}

pub struct Eval {
    pub recall: f64,
    pub hit_rate: f64,
    pub mrr: f64,
    pub report: String,
}

/// BM25 (+ exact DTC) only: no query embeddings are passed.
pub fn eval_bm25(conn: &Connection, golden: &[GoldenQuery]) -> Result<Eval> {
    let mut report = String::new();
    let (mut recall_sum, mut hits, mut rr_sum) = (0.0, 0usize, 0.0);
    for g in golden {
        let r = route(&g.query);
        let got = retrieve(conn, &r, None)?;
        let pages: Vec<u32> = got.chunks.iter().map(|c| c.chunk.page_id).collect();
        let found = g.expect.iter().filter(|p| pages.contains(p)).count();
        let recall = found as f64 / g.expect.len().max(1) as f64;
        let first = pages.iter().position(|p| g.expect.contains(p));
        recall_sum += recall;
        hits += usize::from(found > 0);
        rr_sum += first.map_or(0.0, |i| 1.0 / (i as f64 + 1.0));
        let mark = if found == g.expect.len() {
            "ok  "
        } else if found > 0 {
            "part"
        } else {
            "MISS"
        };
        let _ = writeln!(report, "{mark} recall {recall:.2}  {:?}", g.query);
        for c in &got.chunks {
            let star = if g.expect.contains(&c.chunk.page_id) {
                "*"
            } else {
                " "
            };
            let _ = writeln!(
                report,
                "       {star} {:5} [{}] {}",
                c.chunk.page_id,
                c.chunk.kind.as_str(),
                c.chunk.title
            );
        }
        if found < g.expect.len() {
            let missing: Vec<&u32> = g.expect.iter().filter(|p| !pages.contains(p)).collect();
            let _ = writeln!(report, "         missing: {missing:?}");
        }
    }
    let n = golden.len().max(1) as f64;
    let (recall, hit_rate, mrr) = (recall_sum / n, hits as f64 / n, rr_sum / n);
    let _ = writeln!(
        report,
        "\n{} queries — recall@{MAX_RESULTS} {recall:.3}, hit@{MAX_RESULTS} {hit_rate:.3}, MRR {mrr:.3}",
        golden.len()
    );
    Ok(Eval {
        recall,
        hit_rate,
        mrr,
        report,
    })
}
