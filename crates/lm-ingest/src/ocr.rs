//! `lm-ingest ocr`: OCR the image-only spec/torque tables and the DTC
//! diagnostic step tables with the system `tesseract` CLI.
//!
//! Each image is converted to grayscale, table rules are erased, it is
//! upscaled 2× (Lanczos3), then
//! OCR'd with `--psm 6` and `--psm 4`; the run with the higher mean word
//! confidence wins. Output chunks have `source = "ocr"` and always carry the
//! image path so the UI can show the picture beside the recognised numbers.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use lm_types::{Chunk, Kind, Source};
use regex::Regex;

use crate::ingest::{Corpus, IMAGE_ONLY_CHARS, UniquePage, dtc_codes, hex16};
use crate::ocrcheck::{self, TableCheck};

static RE_STEP_HEAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^Step \d+").unwrap());

/// One image selected for OCR.
#[derive(Debug, Clone)]
pub struct Target {
    pub unique_idx: usize,
    pub image: String,
    pub kind: Kind,
    pub heading: String,
    pub links: Vec<u32>,
}

pub struct Selection {
    pub targets: Vec<Target>,
    pub spec_pages: usize,
    pub dtc_pages_with_steps: usize,
    pub other_pages_with_steps: usize,
    pub dtc_step_images: usize,
    pub other_step_images: usize,
    /// Images on DTC pages that are not step tables (wiring, connector views): not OCR'd.
    pub dtc_images_other: usize,
}

/// Spec/torque pages whose content is in images: fewer than
/// [`IMAGE_ONLY_CHARS`] characters of non-heading text (image captions such as
/// "Fastener Tightening Specifications:" are `<h3>` headings and don't count).
pub fn is_spec_candidate(u: &UniquePage) -> bool {
    let body: usize = u
        .page
        .lines
        .iter()
        .filter(|l| !l.heading)
        .map(|l| l.text.trim().chars().count())
        .sum();
    let has_images = u.page.lines.iter().any(|l| !l.imgs.is_empty());
    has_images && body < IMAGE_ONLY_CHARS && u.kind == Kind::Spec
}

/// (a) every image on image-only spec pages; (b) on any other page, images
/// whose nearest preceding heading is a "Step N - Step M" table heading
/// (kind `dtc_steps` on DTC pages, the page's own kind elsewhere).
pub fn select(corpus: &Corpus) -> Selection {
    let mut targets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut sel = Selection {
        targets: Vec::new(),
        spec_pages: 0,
        dtc_pages_with_steps: 0,
        other_pages_with_steps: 0,
        dtc_step_images: 0,
        other_step_images: 0,
        dtc_images_other: 0,
    };
    for (idx, u) in corpus.unique.iter().enumerate() {
        let spec = is_spec_candidate(u);
        sel.spec_pages += usize::from(spec);
        let mut heading = String::new();
        let mut page_has_steps = false;
        for l in &u.page.lines {
            if l.heading {
                heading = l.text.trim().trim_end_matches(':').to_string();
            }
            for img in &l.imgs {
                let step = !spec && RE_STEP_HEAD.is_match(&heading);
                if !spec && !step {
                    sel.dtc_images_other += usize::from(u.kind == Kind::Dtc);
                    continue;
                }
                if !seen.insert(img.clone()) {
                    continue; // same image already selected from another page
                }
                let kind = if spec {
                    Kind::Spec
                } else if u.kind == Kind::Dtc {
                    sel.dtc_step_images += 1;
                    Kind::DtcSteps
                } else {
                    sel.other_step_images += 1;
                    u.kind
                };
                page_has_steps |= step;
                targets.push(Target {
                    unique_idx: idx,
                    image: img.clone(),
                    kind,
                    heading: if heading.is_empty() {
                        u.page.title.clone()
                    } else {
                        heading.clone()
                    },
                    links: l.links.clone(),
                });
            }
        }
        if page_has_steps {
            if u.kind == Kind::Dtc {
                sel.dtc_pages_with_steps += 1;
            } else {
                sel.other_pages_with_steps += 1;
            }
        }
    }
    sel.targets = targets;
    sel
}

/// Recognised text and mean word confidence for one tesseract run.
#[derive(Debug, Clone, Default)]
pub struct OcrText {
    pub text: String,
    pub mean_conf: f32,
    pub words: usize,
    pub psm: u8,
}

/// Parse tesseract TSV: words (level 5) grouped into lines; blank line between blocks.
pub fn parse_tsv(tsv: &str, psm: u8) -> OcrText {
    let mut lines: BTreeMap<(u32, u32, u32), Vec<String>> = BTreeMap::new();
    let (mut sum, mut n) = (0f64, 0usize);
    for row in tsv.lines().skip(1) {
        let f: Vec<&str> = row.split('\t').collect();
        if f.len() < 12 || f[0] != "5" {
            continue;
        }
        let word = f[11].trim();
        let conf: f64 = f[10].parse().unwrap_or(-1.0);
        if word.is_empty() || conf < 0.0 {
            continue;
        }
        sum += conf;
        n += 1;
        let key = (
            f[2].parse().unwrap_or(0),
            f[3].parse().unwrap_or(0),
            f[4].parse().unwrap_or(0),
        );
        lines.entry(key).or_default().push(word.to_string());
    }
    let mut text = String::new();
    let mut prev_block = None;
    for ((block, _, _), words) in &lines {
        if !text.is_empty() {
            text.push_str(if prev_block == Some(*block) {
                "\n"
            } else {
                "\n\n"
            });
        }
        text.push_str(&words.join(" "));
        prev_block = Some(*block);
    }
    OcrText {
        text,
        mean_conf: if n == 0 { 0.0 } else { (sum / n as f64) as f32 },
        words: n,
        psm,
    }
}

/// Dark runs longer than this are table rules, not glyphs (glyphs here are ~15 px tall).
const MIN_RULE_RUN: u32 = 40;

/// Whiten long horizontal and vertical runs of dark pixels (table grid lines).
/// Tesseract drops words that touch cell borders, which loses whole columns.
pub fn remove_rules(img: &mut image::GrayImage) {
    let (w, h) = img.dimensions();
    let dark = |img: &image::GrayImage, x: u32, y: u32| img.get_pixel(x, y)[0] < 128;
    let mut erase = Vec::new();
    for y in 0..h {
        let mut run = 0;
        for x in 0..=w {
            if x < w && dark(img, x, y) {
                run += 1;
            } else {
                if run >= MIN_RULE_RUN {
                    erase.extend((x - run..x).map(|xx| (xx, y)));
                }
                run = 0;
            }
        }
    }
    for x in 0..w {
        let mut run = 0;
        for y in 0..=h {
            if y < h && dark(img, x, y) {
                run += 1;
            } else {
                if run >= MIN_RULE_RUN {
                    erase.extend((y - run..y).map(|yy| (x, yy)));
                }
                run = 0;
            }
        }
    }
    for (x, y) in erase {
        img.put_pixel(x, y, image::Luma([255]));
    }
}

fn preprocess(src: &Path, dst: &Path) -> Result<()> {
    let mut img = image::open(src)
        .with_context(|| format!("opening {}", src.display()))?
        .to_luma8();
    remove_rules(&mut img);
    let (w, h) = img.dimensions();
    let up = image::imageops::resize(&img, w * 2, h * 2, FilterType::Lanczos3);
    up.save(dst)
        .with_context(|| format!("writing {}", dst.display()))?;
    Ok(())
}

fn tesseract(img: &Path, psm: u8) -> Result<OcrText> {
    let out = Command::new("tesseract")
        .arg(img)
        .arg("stdout")
        .args(["--psm", &psm.to_string(), "-l", "eng", "tsv"])
        .env("OMP_THREAD_LIMIT", "1")
        .output()
        .context("running tesseract (is it installed?)")?;
    if !out.status.success() {
        bail!(
            "tesseract failed on {}: {}",
            img.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(parse_tsv(&String::from_utf8_lossy(&out.stdout), psm))
}

/// Preprocess and OCR one image with both page-segmentation modes.
pub fn ocr_image(src: &Path, tmp: &Path) -> Result<OcrText> {
    preprocess(src, tmp)?;
    let a = tesseract(tmp, 6)?;
    let b = tesseract(tmp, 4)?;
    let _ = std::fs::remove_file(tmp);
    Ok(if b.mean_conf > a.mean_conf { b } else { a })
}

pub fn tesseract_version() -> Option<String> {
    let out = Command::new("tesseract").arg("--version").output().ok()?;
    let s =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    s.lines().next().map(str::to_string)
}

pub struct OcrRun {
    pub chunks: Vec<Chunk>,
    pub fixes: FixCounts,
    pub checks: CheckCounts,
    pub cached: usize,
    pub failures: Vec<String>,
    pub psm_wins: BTreeMap<u8, usize>,
    pub seconds: f64,
}

/// Previous OCR results by image path: (raw text, mean confidence).
pub type OcrCache = HashMap<String, (String, f32)>;

pub fn load_cache(chunks: &[Chunk]) -> OcrCache {
    chunks
        .iter()
        .filter_map(|c| {
            let raw = c.ocr_raw.clone().unwrap_or_else(|| c.text.clone());
            Some((
                c.images.first()?.clone(),
                (raw, c.ocr_confidence.unwrap_or(0.0)),
            ))
        })
        .collect()
}

/// HTML corroboration for angle rows: some page's text names the component
/// and contains the exact value.
pub fn html_corroborator(corpus: &Corpus) -> impl Fn(&str, &str) -> bool + '_ {
    let texts: Vec<String> = corpus
        .unique
        .iter()
        .filter(|u| !u.norm.is_empty())
        .map(|u| u.norm.to_lowercase())
        .collect();
    move |label: &str, value: &str| {
        let label = label
            .trim_start_matches(['e', '*', '•', ' '])
            .trim()
            .to_lowercase();
        label.len() >= 4
            && texts
                .iter()
                .any(|t| t.contains(&label) && t.contains(&value.to_lowercase()))
    }
}

pub fn run(
    manual_root: &Path,
    corpus: &Corpus,
    targets: &[Target],
    workers: usize,
    tmp_dir: &Path,
    cache: &OcrCache,
) -> Result<OcrRun> {
    std::fs::create_dir_all(tmp_dir)?;
    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Result<OcrText>)>> = Mutex::new(
        targets
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                cache.get(&t.image).map(|(text, conf)| {
                    (
                        i,
                        Ok(OcrText {
                            text: text.clone(),
                            mean_conf: *conf,
                            words: 1,
                            psm: 0,
                        }),
                    )
                })
            })
            .collect(),
    );
    let cached = results.lock().unwrap().len();
    let todo: Vec<usize> = (0..targets.len())
        .filter(|i| !cache.contains_key(&targets[*i].image))
        .collect();
    eprintln!("ocr: {cached} cached, {} to run", todo.len());
    std::thread::scope(|s| {
        for w in 0..workers.max(1) {
            let (next, results, todo) = (&next, &results, &todo);
            s.spawn(move || {
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = todo.get(j) else { break };
                    let t = &targets[i];
                    let tmp: PathBuf = tmp_dir.join(format!("w{w}.png"));
                    let r = ocr_image(&manual_root.join(&t.image), &tmp);
                    let done = {
                        let mut g = results.lock().unwrap();
                        g.push((i, r));
                        g.len()
                    };
                    if done % 25 == 0 || done == targets.len() {
                        eprintln!("ocr: {done}/{}", targets.len());
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|(i, _)| *i);

    let mut run = OcrRun {
        chunks: Vec::new(),
        fixes: FixCounts::default(),
        checks: CheckCounts::default(),
        failures: Vec::new(),
        psm_wins: BTreeMap::new(),
        cached,
        seconds: 0.0,
    };
    for (i, r) in results {
        let t = &targets[i];
        match r {
            Ok(o) if o.words == 0 => run
                .failures
                .push(format!("{}: no words recognised", t.image)),
            Ok(o) => {
                if o.psm != 0 {
                    *run.psm_wins.entry(o.psm).or_default() += 1;
                }
                let u = &corpus.unique[t.unique_idx];
                run.chunks.push(Chunk {
                    id: hex16(&format!("ocr:{}", t.image)),
                    page_id: u.rep,
                    title: u.page.title.clone(),
                    breadcrumb: u.primary.join(" > "),
                    aliases: u.aliases.clone(),
                    section_heading: t.heading.clone(),
                    dtc_codes: dtc_codes(&u.page.title, &o.text),
                    text: o.text,
                    images: vec![t.image.clone()],
                    links: t.links.clone(),
                    kind: t.kind,
                    image_only: u.image_only(),
                    source: Source::Ocr,
                    ocr_confidence: Some((o.mean_conf * 10.0).round() / 10.0),
                    ocr_raw: None,
                    ocr_checked_rows: Vec::new(),
                    ocr_masked_values: 0,
                    hidden: false,
                });
            }
            Err(e) => run.failures.push(format!("{}: {e:#}", t.image)),
        }
    }
    let corroborate = html_corroborator(corpus);
    for c in &mut run.chunks {
        let (n, k) = finish_chunk(c, &corroborate);
        run.fixes.add(n);
        run.checks.add(&k);
    }
    run.seconds = start.elapsed().as_secs_f64();
    Ok(run)
}

pub fn report(sel: &Selection, run: &OcrRun, version: &str) -> String {
    let mut r = String::new();
    let _ = writeln!(r, "LocalMotive OCR report (lm-ingest ocr) — {version}");
    let _ = writeln!(
        r,
        "Spec/torque image-only candidate pages: {}",
        sel.spec_pages
    );
    let _ = writeln!(
        r,
        "Step tables: {} DTC pages / {} images, {} other pages / {} images; DTC-page images that are not step tables (not OCR'd): {}",
        sel.dtc_pages_with_steps,
        sel.dtc_step_images,
        sel.other_pages_with_steps,
        sel.other_step_images,
        sel.dtc_images_other
    );
    let _ = writeln!(
        r,
        "targets {}, reused from previous run {}",
        sel.targets.len(),
        run.cached
    );
    type Filter = fn(&Chunk) -> bool;
    let groups: [(&str, Filter); 3] = [
        ("spec", |c| c.kind == Kind::Spec),
        ("dtc_steps", |c| c.kind == Kind::DtcSteps),
        ("other step tables", |c| {
            c.kind != Kind::Spec && c.kind != Kind::DtcSteps
        }),
    ];
    for (name, f) in groups {
        let cs: Vec<&Chunk> = run.chunks.iter().filter(|c| f(c)).collect();
        let confs: Vec<f32> = cs.iter().filter_map(|c| c.ocr_confidence).collect();
        let mean = confs.iter().sum::<f32>() / confs.len().max(1) as f32;
        let _ = writeln!(
            r,
            "\n{name}: {} images OCR'd, mean confidence {mean:.1}",
            cs.len()
        );
        let bins = [0.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 101.0];
        for w in bins.windows(2) {
            let n = confs.iter().filter(|&&c| c >= w[0] && c < w[1]).count();
            let _ = writeln!(
                r,
                "  conf {:>3}-{:<3}: {n:5}",
                w[0],
                if w[1] > 100.0 { 100.0 } else { w[1] }
            );
        }
    }
    let hidden = run.chunks.iter().filter(|c| c.hidden).count();
    let _ = writeln!(
        r,
        "\nhidden (confidence < {HIDE_BELOW}): {hidden} of {}",
        run.chunks.len()
    );
    let f = run.fixes;
    let _ = writeln!(
        r,
        "numeric-context fixes: {} total — units {}, S→5 {}, Z→2 {}, O/o→0 {}, |→1 {}, l→1 {}, spacing-only {}",
        f.total(),
        f.units,
        f.s_to_5,
        f.z_to_2,
        f.o_to_0,
        f.bar_to_1,
        f.l_to_1,
        f.spacing
    );
    let k = run.checks;
    let _ = writeln!(
        r,
        "cross-check: spec rows checked {}, angle rows corroborated by HTML {}, spec rows masked {} ({} values); step-table unit values masked {}",
        k.spec_rows_checked,
        k.spec_rows_corroborated,
        k.spec_rows_masked,
        k.spec_values_masked,
        k.step_values_masked
    );
    let _ = writeln!(
        r,
        "chunks changed by the fix: {}",
        run.chunks
            .iter()
            .filter(|c| c.ocr_raw.as_deref() != Some(c.text.as_str()))
            .count()
    );
    let _ = writeln!(r, "psm wins: {:?}", run.psm_wins);
    let _ = writeln!(r, "failures: {}", run.failures.len());
    for f in run.failures.iter().take(20) {
        let _ = writeln!(r, "  {f}");
    }
    let _ = writeln!(r, "elapsed: {:.0}s", run.seconds);
    let mut low: Vec<&Chunk> = run.chunks.iter().collect();
    low.sort_by(|a, b| {
        a.ocr_confidence
            .partial_cmp(&b.ocr_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let _ = writeln!(r, "\nlowest-confidence 15:");
    for c in low.iter().take(15) {
        let _ = writeln!(
            r,
            "  {:5.1}  page {:5}  {}  {}",
            c.ocr_confidence.unwrap_or(0.0),
            c.page_id,
            c.images[0],
            c.title
        );
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_removed_glyph_kept() {
        let mut img = image::GrayImage::from_pixel(100, 100, image::Luma([255]));
        for x in 0..100 {
            img.put_pixel(x, 50, image::Luma([0])); // horizontal rule
        }
        for y in 0..100 {
            img.put_pixel(10, y, image::Luma([0])); // vertical rule
        }
        for y in 20..35 {
            img.put_pixel(60, y, image::Luma([0])); // 15 px glyph stroke
        }
        remove_rules(&mut img);
        assert_eq!(img.get_pixel(30, 50)[0], 255);
        assert_eq!(img.get_pixel(10, 80)[0], 255);
        assert_eq!(img.get_pixel(60, 25)[0], 0);
    }

    #[test]
    fn tsv_lines_and_confidence() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
            1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t\n\
            5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tCoolant\n\
            5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t80\tDrain\n\
            5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t70\t14\n\
            5\t1\t2\t1\t1\t1\t0\t0\t1\t1\t60\tlb\n\
            5\t1\t2\t1\t1\t2\t0\t0\t1\t1\t-1\t \n";
        let o = parse_tsv(tsv, 6);
        assert_eq!(o.text, "Coolant Drain\n14\n\nlb");
        assert_eq!(o.words, 4);
        assert!((o.mean_conf - 75.0).abs() < 1e-4);
    }
}

// ------------------------------------------------------------------ numeric-context fix

static RE_UNIT_FIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:\blb|\bIb|\|b)\s?(ft|tt|in)\b|\blbtt\b|\blbft\b|\blbin\b|\bIbin\b|\bIbft\b")
        .unwrap()
});
static RE_NUM_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(^|[\s(|=])([0-9SZOo|l][0-9SZOo|l.,]*?)[._]?\s?(N\.m|N·m|lb ft|lb in|degrees|mm|kPa|V\b)_?").unwrap()
});

/// Counts of substitutions made by [`fix_numeric`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FixCounts {
    pub units: usize,
    pub s_to_5: usize,
    pub z_to_2: usize,
    pub o_to_0: usize,
    pub bar_to_1: usize,
    pub l_to_1: usize,
    pub spacing: usize,
}

impl FixCounts {
    pub fn add(&mut self, o: FixCounts) {
        self.units += o.units;
        self.s_to_5 += o.s_to_5;
        self.z_to_2 += o.z_to_2;
        self.o_to_0 += o.o_to_0;
        self.bar_to_1 += o.bar_to_1;
        self.l_to_1 += o.l_to_1;
        self.spacing += o.spacing;
    }

    pub fn total(&self) -> usize {
        self.units
            + self.s_to_5
            + self.z_to_2
            + self.o_to_0
            + self.bar_to_1
            + self.l_to_1
            + self.spacing
    }
}

/// Deterministic repair of OCR confusions, applied only to the value right
/// before a unit: unit spellings (`lbtt`, `lb tt`, `|b ft`, `Ib in` → `lb ft`/`lb in`),
/// then in the value S→5, Z→2, O/o→0, |→1, l→1 between digits, and a single
/// space between value and unit (`10N.m_` → `10 N.m`, as the manual prints it).
/// Values with no digit after the fix are left alone.
pub fn fix_numeric(raw: &str) -> (String, FixCounts) {
    let mut n = FixCounts::default();
    let units = RE_UNIT_FIX.replace_all(raw, |c: &regex::Captures| {
        let m = c.get(0).unwrap().as_str();
        let unit = if m.ends_with("in") { "lb in" } else { "lb ft" };
        if m != unit {
            n.units += 1;
        }
        unit.to_string()
    });
    let out = RE_NUM_UNIT.replace_all(&units, |c: &regex::Captures| {
        let (lead, val, unit, whole) = (&c[1], &c[2], &c[3], c.get(0).unwrap().as_str());
        let chars: Vec<char> = val.chars().collect();
        let mut fixed = String::new();
        let mut local = FixCounts::default();
        for (i, &ch) in chars.iter().enumerate() {
            let digit_at = |j: Option<usize>| {
                j.and_then(|j| chars.get(j))
                    .is_some_and(|c| c.is_ascii_digit())
            };
            let r = match ch {
                'S' => {
                    local.s_to_5 += 1;
                    '5'
                }
                'Z' => {
                    local.z_to_2 += 1;
                    '2'
                }
                'O' | 'o' => {
                    local.o_to_0 += 1;
                    '0'
                }
                '|' => {
                    local.bar_to_1 += 1;
                    '1'
                }
                'l' if digit_at(i.checked_sub(1)) && digit_at(Some(i + 1)) => {
                    local.l_to_1 += 1;
                    '1'
                }
                c => c,
            };
            fixed.push(r);
        }
        let fixed = fixed.trim_end_matches(['.', ',']).to_string();
        if !fixed
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
            || !fixed.chars().any(|c| c.is_ascii_digit())
        {
            return whole.to_string();
        }
        let rebuilt = format!("{lead}{fixed} {unit}");
        if rebuilt != whole {
            if local.total() == 0 {
                local.spacing += 1;
            }
            n.add(local);
        }
        rebuilt
    });
    (out.into_owned(), n)
}

pub const HIDE_BELOW: f32 = 60.0;

/// Totals of the cross-check over many chunks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CheckCounts {
    pub spec_rows_checked: usize,
    pub spec_rows_corroborated: usize,
    pub spec_rows_masked: usize,
    pub spec_values_masked: usize,
    pub step_values_masked: usize,
}

impl CheckCounts {
    pub fn add(&mut self, k: &(Kind, TableCheck)) {
        let (kind, t) = k;
        if *kind == Kind::Spec {
            self.spec_rows_checked += t.checked_rows.len() - t.corroborated_rows as usize;
            self.spec_rows_corroborated += t.corroborated_rows as usize;
            self.spec_rows_masked += t.masked_rows as usize;
            self.spec_values_masked += t.masked_values as usize;
        } else {
            self.step_values_masked += t.masked_values as usize;
        }
    }
}

/// Apply the numeric fix, the metric/English cross-check (spec tables) or unit
/// masking (step tables), and the visibility rule. The raw text is kept.
pub fn finish_chunk(
    c: &mut Chunk,
    corroborate: &dyn Fn(&str, &str) -> bool,
) -> (FixCounts, (Kind, TableCheck)) {
    let raw = c.ocr_raw.clone().unwrap_or_else(|| c.text.clone());
    let (fixed, n) = fix_numeric(&raw);
    let check = if c.kind == Kind::Spec {
        ocrcheck::check_spec_table(&fixed, corroborate)
    } else {
        ocrcheck::check_step_table(&fixed)
    };
    c.dtc_codes = dtc_codes(&c.title, &check.text);
    c.text = check.text.clone();
    c.ocr_raw = Some(raw);
    c.ocr_checked_rows = check.checked_rows.clone();
    c.ocr_masked_values = check.masked_values;
    c.hidden = c.ocr_confidence.is_some_and(|x| x < HIDE_BELOW);
    (n, (c.kind, check))
}

#[cfg(test)]
mod fix_tests {
    use super::*;

    #[test]
    fn fixes_values_next_to_units() {
        let (t, n) = fix_numeric(
            "e First Pass SON.m 37 |b ft\nCylinder Head Plug 2ZON.m_ 15 lbtt\nValve 10N.m_ 89 Ibin\nKnock Sensor ZON.m 18 |b ft",
        );
        assert_eq!(
            t,
            "e First Pass 50 N.m 37 lb ft\nCylinder Head Plug 220 N.m 15 lb ft\nValve 10 N.m 89 lb in\nKnock Sensor 20 N.m 18 lb ft"
        );
        assert_eq!(n.units, 4);
        assert_eq!(n.s_to_5, 1);
        assert_eq!(n.z_to_2, 2);
        assert_eq!(n.o_to_0, 3);
    }

    #[test]
    fn leaves_words_and_unrelated_text_alone() {
        let (t, n) = fix_numeric("ALSO V belt SO the Oil Filter Bolt and Step 1 - Step 8");
        assert_eq!(t, "ALSO V belt SO the Oil Filter Bolt and Step 1 - Step 8");
        assert_eq!(n.total(), 0);
        let (t, _) = fix_numeric("Spark Plug 15 N.m 11 lb ft");
        assert_eq!(t, "Spark Plug 15 N.m 11 lb ft");
        let (t, _) = fix_numeric("1l5 mm and 3.5l mm");
        assert_eq!(t, "115 mm and 3.5l mm");
    }
}
