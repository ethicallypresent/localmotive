//! lm-retrieve — route a question and retrieve manual chunks.
//!
//! ```text
//! lm-retrieve route "<question>" [-o route.json]
//! lm-retrieve retrieve route.json [-o retrieved.json] [--bm25-only]
//! lm-retrieve ask "<question>" [--bm25-only]       # route + retrieve, prints a summary
//! lm-retrieve eval [tests/golden.yaml]              # recall@6, BM25 + exact DTC only
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use lm_types::{RetrievedFile, RouteFile};

fn write_or_print(out: Option<&Path>, json: &str) -> Result<()> {
    match out {
        Some(p) => std::fs::write(p, json).with_context(|| format!("writing {}", p.display())),
        None => {
            println!("{json}");
            Ok(())
        }
    }
}

/// Query embedding, only when the index has vectors and Ollama has the model.
fn query_embedding(conn: &rusqlite::Connection, query: &str, bm25_only: bool) -> Option<Vec<f32>> {
    if bm25_only || lm_index::vec_count(conn).ok()? == 0 {
        return None;
    }
    let client = lm_ollama::Client::new().ok()?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    let input = vec![format!("{}{query}", lm_retrieve::QUERY_INSTRUCTION)];
    match rt.block_on(client.embed(lm_index::EMBED_MODEL, &input, Some("10m"))) {
        Ok(mut v) => v.pop(),
        Err(e) => {
            eprintln!("vector search unavailable ({e}); using BM25 only");
            None
        }
    }
}

fn summary(r: &RetrievedFile) {
    for c in &r.chunks {
        let ch = &c.chunk;
        println!(
            "{:6.4}  {:5} [{}{}] {} — {}",
            c.score,
            ch.page_id,
            ch.kind.as_str(),
            if ch.source == lm_types::Source::Ocr {
                ",ocr"
            } else {
                ""
            },
            ch.title,
            ch.breadcrumb
        );
    }
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let bm25_only = args.iter().any(|a| a == "--bm25-only");
    args.retain(|a| a != "--bm25-only");
    let out = args.iter().position(|a| a == "-o").map(|i| {
        let p = args.get(i + 1).map(PathBuf::from);
        args.drain(i..(i + 2).min(args.len()));
        p
    });
    let out = out.flatten();
    let db = lm_index::default_db_path();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let arg = args.get(1).cloned();
    match cmd {
        "route" => {
            let q = arg.context("route needs a question")?;
            write_or_print(
                out.as_deref(),
                &serde_json::to_string_pretty(&lm_retrieve::route(&q))?,
            )?;
        }
        "retrieve" => {
            let p = arg.context("retrieve needs route.json")?;
            let route: RouteFile = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
            let conn = lm_index::open_readonly(&db)?;
            let e = query_embedding(&conn, &route.query, bm25_only);
            let r = lm_retrieve::retrieve(&conn, &route, e.as_deref())?;
            write_or_print(out.as_deref(), &serde_json::to_string_pretty(&r)?)?;
        }
        "ask" => {
            let q = arg.context("ask needs a question")?;
            let route = lm_retrieve::route(&q);
            println!(
                "route: codes {:?}, kinds {:?}, next_step {}",
                route.dtc_codes, route.kinds, route.wants_next_step
            );
            let conn = lm_index::open_readonly(&db)?;
            let e = query_embedding(&conn, &q, bm25_only);
            let r = lm_retrieve::retrieve(&conn, &route, e.as_deref())?;
            summary(&r);
            if let Some(p) = out {
                std::fs::write(p, serde_json::to_string_pretty(&r)?)?;
            }
        }
        "eval" => {
            let p = arg
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("tests/golden.yaml"));
            let golden = lm_retrieve::golden::load(&p)?;
            let conn = lm_index::open_readonly(&db)?;
            let e = lm_retrieve::golden::eval_bm25(&conn, &golden)?;
            println!("{}", e.report);
        }
        _ => bail!("usage: lm-retrieve route|retrieve|ask|eval … (see source header)"),
    }
    Ok(())
}
