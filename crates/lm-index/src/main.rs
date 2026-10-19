//! lm-index — build and fill the LocalMotive SQLite index.
//!
//! ```text
//! lm-index build [--corpus DIR] [--db PATH]   # chunks + FTS5 + empty vec table
//! lm-index embed [--db PATH] [--batch N]      # vectors via Ollama (qwen3-embedding:0.6b)
//! lm-index stats [--db PATH]
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let mut cmd = String::from("stats");
    let mut corpus = lm_index::default_corpus_dir();
    let mut db = lm_index::default_db_path();
    let mut batch = 16usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "build" | "embed" | "stats" => cmd = a,
            "--corpus" => corpus = PathBuf::from(it.next().context("--corpus needs a value")?),
            "--db" => db = PathBuf::from(it.next().context("--db needs a value")?),
            "--batch" => batch = it.next().context("--batch needs a value")?.parse()?,
            other => bail!("unknown argument {other:?}"),
        }
    }
    match cmd.as_str() {
        "build" => {
            let t = std::time::Instant::now();
            let s = lm_index::build(&corpus, &db)?;
            println!(
                "built {} in {:.1}s: {} html chunks + {} visible ocr chunks ({} hidden ocr chunks left out)",
                db.display(),
                t.elapsed().as_secs_f64(),
                s.html,
                s.ocr_visible,
                s.ocr_hidden
            );
            stats(&db)?;
        }
        "embed" => {
            let conn = lm_index::open(&db)?;
            let client = lm_ollama::Client::new()?;
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let n = rt.block_on(lm_index::embed_missing(&conn, &client, batch))?;
            println!(
                "embedded {n} chunks; vec_chunks now has {}",
                lm_index::vec_count(&conn)?
            );
        }
        _ => stats(&db)?,
    }
    Ok(())
}

fn stats(db: &std::path::Path) -> Result<()> {
    let conn = lm_index::open_readonly(db)?;
    let n: i64 = conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
    let codes: i64 = conn.query_row("SELECT count(DISTINCT code) FROM chunk_dtc", [], |r| {
        r.get(0)
    })?;
    let size = std::fs::metadata(db)?.len();
    println!(
        "{}: {n} chunks, {codes} distinct DTC codes, {} vectors, {:.1} MB",
        db.display(),
        lm_index::vec_count(&conn)?,
        size as f64 / 1e6
    );
    let mut st = conn.prepare(
        "SELECT kind, source, count(*) FROM chunks GROUP BY kind, source ORDER BY 3 DESC",
    )?;
    for row in st.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
        ))
    })? {
        let (k, s, c) = row?;
        println!("  {k:12} {s:4} {c:6}");
    }
    Ok(())
}
