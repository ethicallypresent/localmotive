//! Turning the manual into deduplicated [`Chunk`]s plus a text report.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::LazyLock;

use lm_types::{Chunk, Kind, Source, approx_tokens};
use regex::Regex;
use sha2::{Digest, Sha256};

use crate::chunker::{self, OUTLIER_TOKENS, lines_text};
use crate::manual::{self, Manual, PARTS_ROOT, Page, REPAIR_ROOT};

pub const IMAGE_ONLY_CHARS: usize = 80;

static RE_DTC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[PBCU][0-9]{4}\b").unwrap());
static RE_DTC_TITLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[PBCU][0-9]{4}\b").unwrap());
/// Titles that make a page a spec page wherever it sits ("Torque" alone does not:
/// "Torque Converter Diagnosis" is a procedure).
pub static RE_SPEC_TITLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Specification|Tightening|Capacit|Gap|Firing Order").unwrap());
static RE_PROC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Service and Repair|Removal and Replacement|Testing and Inspection|Adjustments|Procedure|Replacement|Diagnos").unwrap()
});

pub fn hex16(input: &str) -> String {
    Sha256::digest(input.as_bytes())
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn sha256_hex(input: &str) -> String {
    Sha256::digest(input.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn dtc_codes(title: &str, text: &str) -> Vec<String> {
    let all = format!("{title}\n{text}");
    let set: BTreeSet<String> = RE_DTC
        .find_iter(&all)
        .map(|m| m.as_str().to_string())
        .collect();
    set.into_iter().collect()
}

/// Content type from the page's primary path, title and body text.
pub fn classify(path: &[String], title: &str, norm_text: &str) -> Kind {
    let p = path.join("/");
    let t = title;
    if p.contains("Technical Service Bulletins")
        || norm_text.trim_start().starts_with("Bulletin No.")
    {
        Kind::Tsb
    } else if t.contains("Labor Times") || p.contains("Labor Times") {
        Kind::Labor
    } else if RE_DTC_TITLE.is_match(t)
        || p.contains("Diagnostic Trouble Code")
        || p.contains("P Code Charts")
        || t.contains("DTC")
    {
        Kind::Dtc
    } else if path.iter().any(|s| s == "Specifications") || RE_SPEC_TITLE.is_match(t) {
        Kind::Spec
    } else if p.contains("Diagnostic Aids") {
        Kind::Procedure
    } else if p.starts_with("Diagrams")
        || p.starts_with("Locations")
        || p.contains("/Diagrams")
        || p.contains("/Locations")
        || p.contains("Connector View")
    {
        Kind::Diagram
    } else if p.contains("Description and Operation") || t.contains("Description and Operation") {
        Kind::Description
    } else if RE_PROC.is_match(&format!("{p}/{t}")) {
        Kind::Procedure
    } else {
        Kind::Other
    }
}

/// A unique page body and every page id/path it appears at.
pub struct UniquePage {
    pub rep: u32,
    /// Every page id with this body (rep first).
    pub members: Vec<u32>,
    pub page: Page,
    pub norm: String,
    pub primary: Vec<String>,
    pub aliases: Vec<String>,
    pub kind: Kind,
}

impl UniquePage {
    pub fn image_only(&self) -> bool {
        self.norm.chars().count() < IMAGE_ONLY_CHARS
    }
}

pub struct Corpus {
    pub unique: Vec<UniquePage>,
    pub content_pages: usize,
    pub fallback_pages: usize,
    pub repair_trees: usize,
    pub parts_trees: usize,
    pub parts_content: usize,
    pub parts_entries: usize,
    pub unreached_parts: usize,
}

/// Walk trees, parse every Repair & Diagnosis content page, dedupe by body.
pub fn build_corpus(m: &Manual) -> Corpus {
    let repair = manual::walk_tree(m, REPAIR_ROOT, Vec::new());
    let parts = manual::walk_tree(m, PARTS_ROOT, vec!["Parts and Labor".into()]);

    let mut page_paths: BTreeMap<u32, Vec<Vec<String>>> = BTreeMap::new();
    let (mut fallback, mut unreached_parts) = (0, 0);
    for id in m.ids() {
        if id == 1
            || id == REPAIR_ROOT
            || id == PARTS_ROOT
            || repair.tree_pages.contains(&id)
            || parts.tree_pages.contains(&id)
        {
            continue;
        }
        if let Some(p) = repair.paths.get(&id) {
            page_paths.insert(id, p.clone());
        } else if parts.paths.contains_key(&id) {
            continue;
        } else {
            let raw = m.raw(id).unwrap_or_default();
            if manual::is_tree_html(raw) {
                continue;
            }
            if manual::in_parts_breadcrumb(raw) {
                unreached_parts += 1;
                continue;
            }
            page_paths.insert(id, vec![manual::breadcrumb_path(raw)]);
            fallback += 1;
        }
    }

    // Dedupe by sha256 of normalized text + image list; representative = lowest id.
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<(u32, Page, String)>> = HashMap::new();
    for &id in page_paths.keys() {
        let page = manual::parse_page(m.raw(id).unwrap_or_default());
        let norm = page.norm_text();
        let key = sha256_hex(&format!("{norm}\0{}", page.images().join(",")));
        let g = groups.entry(key.clone()).or_default();
        if g.is_empty() {
            order.push(key);
        }
        g.push((id, page, norm));
    }

    let mut unique = Vec::with_capacity(order.len());
    for key in order {
        let mut members = groups.remove(&key).unwrap_or_default();
        let mut aliases: Vec<String> = Vec::new();
        for (id, _, _) in &members {
            for p in &page_paths[id] {
                let joined = p.join(" > ");
                if !aliases.contains(&joined) {
                    aliases.push(joined);
                }
            }
        }
        let ids: Vec<u32> = members.iter().map(|(id, _, _)| *id).collect();
        let (rep, page, norm) = members.swap_remove(0);
        let primary = page_paths[&rep][0].clone();
        let kind = classify(&primary, &page.title, &norm);
        unique.push(UniquePage {
            rep,
            members: ids,
            page,
            norm,
            primary,
            aliases,
            kind,
        });
    }

    Corpus {
        unique,
        content_pages: page_paths.len(),
        fallback_pages: fallback,
        repair_trees: repair.tree_pages.len(),
        parts_trees: parts.tree_pages.len(),
        parts_content: parts.paths.len(),
        parts_entries: parts.paths.values().map(Vec::len).sum(),
        unreached_parts,
    }
}

fn uniq<T: PartialEq + Clone>(it: impl Iterator<Item = T>) -> Vec<T> {
    let mut v: Vec<T> = Vec::new();
    for x in it {
        if !v.contains(&x) {
            v.push(x);
        }
    }
    v
}

pub fn chunks_for(u: &UniquePage) -> Vec<Chunk> {
    let pieces = if u.image_only() {
        vec![chunker::Piece {
            headings: Vec::new(),
            lines: u.page.lines.clone(),
        }]
    } else {
        chunker::chunk_page(&u.page.lines)
    };
    pieces
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let text = lines_text(&p.lines);
            Chunk {
                id: hex16(&format!("{}:{i}", u.rep)),
                page_id: u.rep,
                title: u.page.title.clone(),
                breadcrumb: u.primary.join(" > "),
                aliases: u.aliases.clone(),
                section_heading: p.headings.join(" | "),
                images: uniq(p.lines.iter().flat_map(|l| l.imgs.iter().cloned())),
                links: uniq(p.lines.iter().flat_map(|l| l.links.iter().copied())),
                kind: u.kind,
                dtc_codes: dtc_codes(&u.page.title, &text),
                image_only: u.image_only(),
                source: Source::Html,
                ocr_confidence: None,
                ocr_raw: None,
                ocr_checked_rows: Vec::new(),
                ocr_masked_values: 0,
                hidden: false,
                text,
            }
        })
        .collect()
}

/// Deterministic xorshift sampler so reports are stable between runs.
pub fn sample_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::new();
    while out.len() < k.min(n) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let i = (s % n as u64) as usize;
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

pub fn histogram(sizes: &[usize]) -> String {
    let bins = [0, 25, 50, 100, 200, 300, 450, 600, usize::MAX];
    let mut s = String::new();
    for w in bins.windows(2) {
        let n = sizes.iter().filter(|&&x| x >= w[0] && x < w[1]).count();
        let label = if w[1] == usize::MAX {
            format!("{}+", w[0])
        } else {
            format!("{}-{}", w[0], w[1] - 1)
        };
        let bar = if n == 0 {
            0
        } else {
            ((60 * n) as f64 / sizes.len() as f64).round().max(1.0) as usize
        };
        let _ = writeln!(s, "  {label:>9}: {n:6}  {}", "#".repeat(bar));
    }
    s
}

pub fn report(c: &Corpus, chunks: &[Chunk]) -> String {
    let mut r = String::new();
    let sizes: Vec<usize> = chunks.iter().map(|c| approx_tokens(&c.text)).collect();
    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
    let _ = writeln!(
        r,
        "LocalMotive chunk report (lm-ingest) — 2006 Chevrolet Uplander FWD V6-3.5L VIN L"
    );
    let _ = writeln!(
        r,
        "Repair & Diagnosis content pages (with paths): {}",
        c.content_pages
    );
    let _ = writeln!(
        r,
        "  of which path from breadcrumb fallback (not in any tree): {}",
        c.fallback_pages
    );
    let _ = writeln!(
        r,
        "  nav/tree pages walked: {} repair, {} parts&labor",
        c.repair_trees, c.parts_trees
    );
    let _ = writeln!(
        r,
        "Unique pages after dedupe: {}  (duplicates folded into aliases: {})",
        c.unique.len(),
        c.content_pages - c.unique.len()
    );
    let _ = writeln!(
        r,
        "Image-only unique pages: {}",
        c.unique.iter().filter(|u| u.image_only()).count()
    );
    let _ = writeln!(
        r,
        "Chunks: {}   (mean {mean:.0} tokens, max {})",
        chunks.len(),
        sizes.iter().max().unwrap_or(&0)
    );
    let _ = writeln!(
        r,
        "Parts and Labor (skipped): {} content pages in tree, {} tree entries; {} more pages with a Parts and Labor breadcrumb not in the tree\n",
        c.parts_content, c.parts_entries, c.unreached_parts
    );
    let _ = writeln!(
        r,
        "Token-size histogram (words*1.3):\n{}",
        histogram(&sizes)
    );
    let _ = writeln!(r, "Kind counts (chunks / unique pages):");
    for k in Kind::ALL {
        let nc = chunks.iter().filter(|c| c.kind == k).count();
        let np = c.unique.iter().filter(|u| u.kind == k).count();
        let _ = writeln!(r, "  {:12} {nc:6} / {np}", k.as_str());
    }
    let codes: BTreeSet<&String> = chunks.iter().flat_map(|c| &c.dtc_codes).collect();
    let _ = writeln!(
        r,
        "Chunks with dtc_codes: {}; distinct codes: {}\n",
        chunks.iter().filter(|c| !c.dtc_codes.is_empty()).count(),
        codes.len()
    );
    let _ = writeln!(r, "{}\n5 sample chunks (deterministic):", "=".repeat(78));
    for i in sample_indices(chunks.len(), 5, 7) {
        let _ = writeln!(
            r,
            "{}\n{}",
            "-".repeat(78),
            serde_json::to_string_pretty(&chunks[i]).unwrap_or_default()
        );
    }
    let big: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| approx_tokens(&c.text) > OUTLIER_TOKENS)
        .collect();
    let _ = writeln!(
        r,
        "\n{}\nChunks over {OUTLIER_TOKENS} tokens: {}",
        "=".repeat(78),
        big.len()
    );
    for c in big {
        let _ = writeln!(
            r,
            "  {:5}  {}  page {}  [{}]  {}  §{}",
            approx_tokens(&c.text),
            c.id,
            c.page_id,
            c.kind.as_str(),
            c.title,
            c.section_heading
        );
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Vec<String> {
        s.split('/').map(str::to_string).collect()
    }

    #[test]
    fn classify_examples() {
        assert_eq!(
            classify(
                &p(
                    "A L L  Diagnostic Trouble Codes ( DTC )/Testing and Inspection/P Code Charts/P0300"
                ),
                "P0300",
                ""
            ),
            Kind::Dtc
        );
        assert_eq!(
            classify(
                &p("Powertrain Management/Technical Service Bulletins/Customer Interest/X"),
                "Engine - DTC P0300",
                ""
            ),
            Kind::Tsb
        );
        assert_eq!(
            classify(
                &p("Specifications/Mechanical Specifications/Engine"),
                "Fastener Tightening Specifications",
                ""
            ),
            Kind::Spec
        );
        assert_eq!(
            classify(
                &p("Powertrain Management/Ignition System/Spark Plug/Service and Repair"),
                "Spark Plug: Service and Repair",
                ""
            ),
            Kind::Procedure
        );
        assert_eq!(
            classify(&p("Diagrams/Connector Views/C275"), "C275", ""),
            Kind::Diagram
        );
        assert_eq!(
            classify(
                &p(
                    "Transmission and Drivetrain/Automatic Transmission/Transaxle/Testing and Inspection/Component Tests and General Diagnostics/Torque Converter Diagnosis"
                ),
                "Torque Converter Diagnosis",
                ""
            ),
            Kind::Procedure
        );
        assert_eq!(
            classify(
                &p(
                    "Specifications/Mechanical Specifications/Powertrain Management/Ignition System/Spark Plug/Spark Plug Torque"
                ),
                "Spark Plug Torque",
                ""
            ),
            Kind::Spec
        );
        assert_eq!(
            classify(
                &p("Starting and Charging/Battery/Specifications/Battery Usage"),
                "Battery Usage",
                ""
            ),
            Kind::Spec
        );
    }

    #[test]
    fn hex16_matches_python() {
        // hashlib.sha256(b"15449:0").hexdigest()[:16]
        assert_eq!(hex16("15449:0").len(), 16);
        assert_eq!(hex16("abc"), "ba7816bf8f01cfea");
    }

    #[test]
    fn dtc_regex() {
        assert_eq!(
            dtc_codes("P0300", "see P0101, U1000 and xP01234"),
            vec!["P0101", "P0300", "U1000"]
        );
    }
}
