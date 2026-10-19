//! lm-ingest — build LocalMotive's retrieval corpus from the offline manual.
//!
//! ```text
//! lm-ingest chunk   [--manual DIR] [--out DIR]
//! lm-ingest compare [--out DIR]
//! lm-ingest ocr     [--manual DIR] [--out DIR] [--workers N] [--page ID] [--limit N] [--dry-run] [--refix]
//! ```
//!
//! The manual folder is only ever read. Outputs go to `--out`
//! (default `~/localmotive/corpus`).

mod chunker;
mod compare;
mod ingest;
mod manual;
mod ocr;
mod ocrcheck;

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use lm_types::Chunk;

const MANUAL_DIR: &str = "localmotive/corpus/2006 Chevrolet Uplander FWD V6-3.5L VIN L";
const OUT_DIR: &str = "localmotive/corpus";

struct Args {
    cmd: String,
    manual: PathBuf,
    out: PathBuf,
    workers: usize,
    page: Option<u32>,
    limit: Option<usize>,
    dry_run: bool,
    refix: bool,
}

fn parse_args() -> Result<Args> {
    let home = PathBuf::from(std::env::var_os("HOME").context("HOME not set")?);
    let mut a = Args {
        cmd: "chunk".into(),
        manual: home.join(MANUAL_DIR),
        out: home.join(OUT_DIR),
        workers: std::thread::available_parallelism()
            .map_or(4, |n| n.get().saturating_sub(2).max(1)),
        page: None,
        limit: None,
        dry_run: false,
        refix: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "chunk" | "compare" | "ocr" => a.cmd = arg,
            "--manual" => a.manual = val("--manual")?.into(),
            "--out" => a.out = val("--out")?.into(),
            "--workers" => a.workers = val("--workers")?.parse()?,
            "--page" => a.page = Some(val("--page")?.parse()?),
            "--limit" => a.limit = Some(val("--limit")?.parse()?),
            "--dry-run" => a.dry_run = true,
            "--refix" => a.refix = true,
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("main.rs")
                        .lines()
                        .take(8)
                        .skip(2)
                        .map(|l| l.trim_start_matches("//! "))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?} (try --help)"),
        }
    }
    Ok(a)
}

fn write_jsonl(path: &Path, chunks: &[Chunk]) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    for c in chunks {
        serde_json::to_writer(&mut w, c)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

/// Keep the Python reference: on first run, move a Python-made chunks.jsonl
/// (its lines have no `source` field) to chunks.python.jsonl.
fn preserve_python_output(out: &Path) -> Result<()> {
    let cur = out.join("chunks.jsonl");
    let py = out.join("chunks.python.jsonl");
    if py.exists() || !cur.exists() {
        return Ok(());
    }
    let first = fs::read_to_string(&cur)?
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    if !first.contains("\"source\"") {
        fs::rename(&cur, &py).context("moving python chunks.jsonl aside")?;
        eprintln!("moved python output to {}", py.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    let a = parse_args()?;
    match a.cmd.as_str() {
        "chunk" => {
            let m = manual::Manual::open(&a.manual)?;
            let corpus = ingest::build_corpus(&m);
            let chunks: Vec<Chunk> = corpus.unique.iter().flat_map(ingest::chunks_for).collect();
            preserve_python_output(&a.out)?;
            write_jsonl(&a.out.join("chunks.jsonl"), &chunks)?;
            let report = ingest::report(&corpus, &chunks);
            fs::write(a.out.join("chunks_report.txt"), &report)?;
            println!("{}", report.lines().take(32).collect::<Vec<_>>().join("\n"));
        }
        "compare" => {
            let py = compare::load(&a.out.join("chunks.python.jsonl"))?;
            let rs = compare::load(&a.out.join("chunks.jsonl"))?;
            let report = compare::compare(&py, &rs);
            fs::write(a.out.join("compare_report.txt"), &report)?;
            println!("{report}");
        }
        "ocr" if a.refix => {
            // Re-apply fix, cross-check and visibility without re-running tesseract.
            let m = manual::Manual::open(&a.manual)?;
            let corpus = ingest::build_corpus(&m);
            let corroborate = ocr::html_corroborator(&corpus);
            let path = a.out.join("chunks.ocr.jsonl");
            let mut chunks = compare::load(&path)?;
            let (mut fixes, mut checks) = (ocr::FixCounts::default(), ocr::CheckCounts::default());
            for c in &mut chunks {
                let (n, k) = ocr::finish_chunk(c, &corroborate);
                fixes.add(n);
                checks.add(&k);
            }
            write_jsonl(&path, &chunks)?;
            let hidden = chunks.iter().filter(|c| c.hidden).count();
            println!(
                "refixed {} chunks; hidden {hidden}; fixes {fixes:?} (total {}); {checks:?}",
                chunks.len(),
                fixes.total()
            );
        }
        "ocr" => {
            let version = ocr::tesseract_version()
                .context("tesseract not found on PATH; install it (not done by this tool)")?;
            let m = manual::Manual::open(&a.manual)?;
            let corpus = ingest::build_corpus(&m);
            let mut sel = ocr::select(&corpus);
            if let Some(p) = a.page {
                sel.targets
                    .retain(|t| corpus.unique[t.unique_idx].members.contains(&p));
            }
            if let Some(n) = a.limit {
                sel.targets.truncate(n);
            }
            eprintln!(
                "{version}; {} images selected, {} workers",
                sel.targets.len(),
                a.workers
            );
            if a.dry_run {
                return Ok(());
            }
            let tmp = std::env::temp_dir().join(format!("lm-ocr-{}", std::process::id()));
            let prev = a.out.join("chunks.ocr.jsonl");
            let cache = if prev.exists() {
                ocr::load_cache(&compare::load(&prev)?)
            } else {
                ocr::OcrCache::new()
            };
            let run = ocr::run(&m.root, &corpus, &sel.targets, a.workers, &tmp, &cache)?;
            let _ = fs::remove_dir_all(&tmp);
            let full = a.page.is_none() && a.limit.is_none();
            let name = if full {
                "chunks.ocr.jsonl"
            } else {
                "chunks.ocr.sample.jsonl"
            };
            write_jsonl(&a.out.join(name), &run.chunks)?;
            let report = ocr::report(&sel, &run, &version);
            if full {
                fs::write(a.out.join("chunks_ocr_report.txt"), &report)?;
            }
            println!("{report}");
        }
        _ => unreachable!(),
    }
    Ok(())
}
