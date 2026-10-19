//! `lm-ingest compare`: check the Rust output against the Python reference.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use lm_types::{Chunk, Kind, approx_tokens};

pub fn load(path: &Path) -> Result<Vec<Chunk>> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1))
        })
        .collect()
}

fn pct(a: usize, b: usize) -> f64 {
    if a == 0 {
        if b == 0 { 0.0 } else { 100.0 }
    } else {
        (b as f64 - a as f64) * 100.0 / a as f64
    }
}

fn row(r: &mut String, name: &str, py: usize, rs: usize) {
    let d = pct(py, rs);
    let flag = if d.abs() > 2.0 { "  <-- >2%" } else { "" };
    let _ = writeln!(r, "  {name:34} {py:8} {rs:8} {d:+7.2}%{flag}");
}

fn by_page(c: &[Chunk]) -> BTreeMap<u32, Vec<&Chunk>> {
    let mut m: BTreeMap<u32, Vec<&Chunk>> = BTreeMap::new();
    for x in c {
        m.entry(x.page_id).or_default().push(x);
    }
    m
}

pub fn compare(py: &[Chunk], rs: &[Chunk]) -> String {
    let mut r = String::new();
    let _ = writeln!(
        r,
        "compare: python reference vs rust\n  {:34} {:>8} {:>8} {:>8}",
        "metric", "python", "rust", "diff"
    );
    let pages = |c: &[Chunk]| c.iter().map(|c| c.page_id).collect::<BTreeSet<_>>();
    let aliases = |c: &[Chunk]| {
        c.iter()
            .flat_map(|c| c.aliases.iter().cloned())
            .collect::<BTreeSet<_>>()
    };
    let (pp, rp) = (pages(py), pages(rs));
    let (pa, ra) = (aliases(py), aliases(rs));
    row(&mut r, "chunks", py.len(), rs.len());
    row(&mut r, "unique pages (page_id)", pp.len(), rp.len());
    row(&mut r, "paths covered (aliases)", pa.len(), ra.len());
    row(
        &mut r,
        "image-only chunks",
        py.iter().filter(|c| c.image_only).count(),
        rs.iter().filter(|c| c.image_only).count(),
    );
    row(
        &mut r,
        "chunks > 600 tokens",
        py.iter().filter(|c| approx_tokens(&c.text) > 600).count(),
        rs.iter().filter(|c| approx_tokens(&c.text) > 600).count(),
    );
    row(
        &mut r,
        "total tokens",
        py.iter().map(|c| approx_tokens(&c.text)).sum(),
        rs.iter().map(|c| approx_tokens(&c.text)).sum(),
    );
    for k in Kind::ALL {
        let n = |c: &[Chunk]| c.iter().filter(|c| c.kind == k).count();
        if n(py) + n(rs) > 0 {
            row(&mut r, &format!("kind {}", k.as_str()), n(py), n(rs));
        }
    }

    let only_py: Vec<_> = pp.difference(&rp).collect();
    let only_rs: Vec<_> = rp.difference(&pp).collect();
    let _ = writeln!(
        r,
        "\npages only in python: {} {:?}",
        only_py.len(),
        only_py.iter().take(20).collect::<Vec<_>>()
    );
    let _ = writeln!(
        r,
        "pages only in rust:   {} {:?}",
        only_rs.len(),
        only_rs.iter().take(20).collect::<Vec<_>>()
    );
    let _ = writeln!(
        r,
        "paths only in python: {}  only in rust: {}",
        pa.difference(&ra).count(),
        ra.difference(&pa).count()
    );
    for p in pa.difference(&ra).take(5) {
        let _ = writeln!(r, "    py-only path: {p}");
    }

    // per page chunk count and text agreement
    let (bp, br) = (by_page(py), by_page(rs));
    let (mut same_count, mut diff_count, mut same_text, mut kind_diff) = (0, 0, 0, 0);
    let mut examples = Vec::new();
    for (pid, pc) in &bp {
        let Some(rc) = br.get(pid) else { continue };
        if pc.len() == rc.len() {
            same_count += 1;
        } else {
            diff_count += 1;
            if examples.len() < 8 {
                examples.push(format!(
                    "page {pid} [{}] {}: python {} chunks, rust {}",
                    pc[0].kind.as_str(),
                    pc[0].title,
                    pc.len(),
                    rc.len()
                ));
            }
        }
        let norm = |v: &Vec<&Chunk>| {
            v.iter()
                .map(|c| c.text.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join(" ")
        };
        if norm(pc) == norm(rc) {
            same_text += 1;
        }
        if pc[0].kind != rc[0].kind {
            kind_diff += 1;
        }
    }
    let common = bp.keys().filter(|k| br.contains_key(k)).count();
    let _ = writeln!(r, "\ncommon pages: {common}");
    let _ = writeln!(
        r,
        "  same chunk count: {same_count}   different: {diff_count}"
    );
    let _ = writeln!(
        r,
        "  identical page text (whitespace-normalized, all chunks joined): {same_text}"
    );
    let _ = writeln!(r, "  different kind: {kind_diff}");
    for e in examples {
        let _ = writeln!(r, "    {e}");
    }
    r
}
