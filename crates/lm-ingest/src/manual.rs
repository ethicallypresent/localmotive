//! Reading the LEMON offline manual: navigation trees, page bodies and lines.
//!
//! The manual is opened read-only. Every page under `pages/` is loaded into
//! memory once (≈130 MB) since the tree walk touches most of them anyway.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use percent_encoding::percent_decode_str;
use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};
use std::sync::LazyLock;

pub const REPAIR_ROOT: u32 = 2;
pub const PARTS_ROOT: u32 = 3;

static RE_PAGE_HREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\.\./pages/(\d+)\.html").unwrap());
static RE_STEP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\t*\d+\.\s").unwrap());
static SEL_MAIN: LazyLock<Selector> = LazyLock::new(|| Selector::parse("div.main").unwrap());
static SEL_TREE_A: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("div.main a[name]").unwrap());
static SEL_CRUMB: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("div.header a.breadcrumb-part").unwrap());
static SEL_TITLE: LazyLock<Selector> = LazyLock::new(|| Selector::parse("title").unwrap());
static SEL_TR: LazyLock<Selector> = LazyLock::new(|| Selector::parse("tr").unwrap());
static SEL_A_HREF: LazyLock<Selector> = LazyLock::new(|| Selector::parse("a[href]").unwrap());
static SEL_TD: LazyLock<Selector> = LazyLock::new(|| Selector::parse("td").unwrap());

/// The whole manual, held in memory as raw HTML.
pub struct Manual {
    pub root: PathBuf,
    pub pages: HashMap<u32, String>,
}

impl Manual {
    pub fn open(root: &Path) -> Result<Self> {
        let dir = root.join("pages");
        let mut pages = HashMap::new();
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let path = entry?.path();
            let Some(id) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".html"))
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            pages.insert(
                id,
                fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?,
            );
        }
        Ok(Self {
            root: root.to_path_buf(),
            pages,
        })
    }

    pub fn raw(&self, id: u32) -> Option<&str> {
        self.pages.get(&id).map(String::as_str)
    }

    pub fn is_tree(&self, id: u32) -> bool {
        self.raw(id).is_some_and(is_tree_html)
    }

    /// Ids in ascending order.
    pub fn ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.pages.keys().copied().collect();
        v.sort_unstable();
        v
    }
}

/// Nav/index pages carry the "Expand All" button.
pub fn is_tree_html(raw: &str) -> bool {
    raw.contains("id='expand-all'")
}

pub fn page_href_id(href: &str) -> Option<u32> {
    RE_PAGE_HREF.captures(href).and_then(|c| c[1].parse().ok())
}

/// Result of walking one navigation tree and its sub-tree pages.
#[derive(Default)]
pub struct TreeWalk {
    /// Content page id → every path it appears at, in discovery order.
    pub paths: HashMap<u32, Vec<Vec<String>>>,
    pub tree_pages: HashSet<u32>,
}

/// Breadth-first walk from a tree page. Sub-tree pages (DTC index, TSB lists…)
/// name their entries relative to themselves, so their path is the prefix.
pub fn walk_tree(manual: &Manual, root: u32, root_path: Vec<String>) -> TreeWalk {
    let mut walk = TreeWalk::default();
    let mut queue = VecDeque::from([(root, root_path)]);
    while let Some((tid, prefix)) = queue.pop_front() {
        if !walk.tree_pages.insert(tid) {
            continue;
        }
        let Some(raw) = manual.raw(tid) else { continue };
        let doc = Html::parse_document(raw);
        for a in doc.select(&SEL_TREE_A) {
            let Some(href) = a.value().attr("href").and_then(page_href_id) else {
                continue; // folder node
            };
            let name = a.value().attr("name").unwrap_or_default();
            let decoded = percent_decode_str(name).decode_utf8_lossy();
            let mut path = prefix.clone();
            path.extend(
                decoded
                    .split('/')
                    .filter(|p| !p.is_empty())
                    .map(str::to_string),
            );
            if manual.is_tree(href) {
                queue.push_back((href, path));
            } else {
                let entry = walk.paths.entry(href).or_default();
                if !entry.contains(&path) {
                    entry.push(path);
                }
            }
        }
    }
    walk
}

/// Header breadcrumb below Home/Make/Year/Model/section root.
pub fn breadcrumb_path(raw: &str) -> Vec<String> {
    let doc = Html::parse_document(raw);
    doc.select(&SEL_CRUMB)
        .skip(5)
        .map(|a| a.text().collect::<String>())
        .collect()
}

/// Whether the header breadcrumb goes through Parts and Labor (pages/3.html).
pub fn in_parts_breadcrumb(raw: &str) -> bool {
    raw.split("<div class=\"main\">")
        .next()
        .is_some_and(|h| h.contains("href=\"../pages/3.html\""))
}

/// One rendered line of a page body.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Line {
    pub text: String,
    pub imgs: Vec<String>,
    pub links: Vec<u32>,
    pub heading: bool,
    pub table: bool,
    pub step: bool,
    pub level: usize,
    /// Joins to the previous line with a space instead of a newline (sentence split).
    pub cont: bool,
}

/// A parsed content page.
pub struct Page {
    pub title: String,
    pub lines: Vec<Line>,
}

impl Page {
    /// Whitespace-collapsed body text, used for dedupe and the image-only test.
    pub fn norm_text(&self) -> String {
        let joined: Vec<&str> = self.lines.iter().map(|l| l.text.as_str()).collect();
        joined
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn images(&self) -> Vec<&str> {
        self.lines
            .iter()
            .flat_map(|l| l.imgs.iter().map(String::as_str))
            .collect()
    }
}

pub fn parse_page(raw: &str) -> Page {
    let doc = Html::parse_document(raw);
    let main = doc.select(&SEL_MAIN).next();
    let mut title = main
        .and_then(|m| {
            m.children()
                .filter_map(ElementRef::wrap)
                .find(|e| e.value().name() == "h1")
        })
        .map(|h| h.text().collect::<String>())
        .unwrap_or_default();
    if title.is_empty() {
        title = doc
            .select(&SEL_TITLE)
            .next()
            .map(|t| t.text().collect::<String>())
            .and_then(|t| t.split(" — ").next().map(str::to_string))
            .unwrap_or_default();
    }
    let mut w = Walker::default();
    if let Some(m) = main {
        w.children(m);
    }
    w.finish_line();
    Page {
        title,
        lines: w.lines,
    }
}

/// Builds [`Line`]s from the DOM the same way the reference chunker did from
/// raw HTML: `<br>` ends a line, a leading `span.indent-N` becomes tabs,
/// later indent spans become a space, `<h3>` and `<table>` get their own lines.
#[derive(Default)]
struct Walker {
    lines: Vec<Line>,
    segs: Vec<(String, bool)>,
    lead: Option<usize>,
    imgs: Vec<String>,
    links: Vec<u32>,
    bold: usize,
    force_heading: bool,
}

impl Walker {
    fn has_content(&self) -> bool {
        self.segs.iter().any(|(s, _)| !s.trim().is_empty())
    }

    fn children(&mut self, el: ElementRef) {
        for child in el.children() {
            match child.value() {
                Node::Text(t) => self.segs.push((t.to_string(), self.bold > 0)),
                Node::Element(_) => {
                    if let Some(e) = ElementRef::wrap(child) {
                        self.element(e);
                    }
                }
                _ => {}
            }
        }
    }

    fn element(&mut self, e: ElementRef) {
        let v = e.value();
        match v.name() {
            "br" => self.finish_line(),
            "h1" | "button" | "script" => {}
            "h3" => {
                self.finish_line();
                self.bold += 1;
                self.children(e);
                self.bold -= 1;
                self.force_heading = true;
                self.finish_line();
            }
            "span" if v.classes().any(|c| c.starts_with("indent-")) => {
                let n = v
                    .classes()
                    .find_map(|c| c.strip_prefix("indent-"))
                    .and_then(|n| n.parse::<usize>().ok());
                match n {
                    Some(n)
                        if !self.has_content() && self.lead.is_none() && self.imgs.is_empty() =>
                    {
                        self.lead = Some(n)
                    }
                    _ => self.segs.push((" ".into(), self.bold > 0)),
                }
            }
            "b" | "strong" => {
                self.bold += 1;
                self.children(e);
                self.bold -= 1;
            }
            "img" => {
                if let Some(src) = v.attr("src") {
                    self.imgs.push(src.trim_start_matches("../").to_string());
                }
            }
            "a" => {
                if let Some(id) = v.attr("href").and_then(page_href_id) {
                    self.links.push(id);
                }
                self.children(e);
            }
            "table" => {
                self.finish_line();
                let mut rows = Vec::new();
                for tr in e.select(&SEL_TR) {
                    let cells: Vec<String> = tr
                        .select(&SEL_TD)
                        .map(|td| td.text().collect::<String>().trim().to_string())
                        .collect();
                    rows.push(cells.join(" | "));
                }
                for a in e.select(&SEL_A_HREF) {
                    if let Some(id) = a.value().attr("href").and_then(page_href_id) {
                        self.links.push(id);
                    }
                }
                self.lines.push(Line {
                    text: rows.join("\n"),
                    imgs: std::mem::take(&mut self.imgs),
                    links: std::mem::take(&mut self.links),
                    table: true,
                    ..Line::default()
                });
            }
            _ => self.children(e),
        }
    }

    fn finish_line(&mut self) {
        let segs = std::mem::take(&mut self.segs);
        let lead = self.lead.take();
        let force_heading = std::mem::take(&mut self.force_heading);
        let joined: String = segs.iter().map(|(s, _)| s.as_str()).collect();
        let body = collapse_spaces(&joined.replace('\t', " "));
        let text = if body.is_empty() {
            String::new()
        } else {
            let tabs = match lead {
                Some(0) | None => 0,
                Some(n) => ((n + 1) / 3).max(1),
            };
            format!("{}{}", "\t".repeat(tabs), body)
        };
        let heading = !text.is_empty() && (force_heading || is_bold_upper(&segs, &body));
        let level = text.len() - text.trim_start_matches('\t').len();
        self.lines.push(Line {
            step: RE_STEP.is_match(&text),
            heading,
            level,
            text,
            imgs: std::mem::take(&mut self.imgs),
            links: std::mem::take(&mut self.links),
            table: false,
            cont: false,
        });
    }
}

fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch == ' ' {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim_matches(' ').to_string()
}

/// Callout labels are bold and uppercase but belong to the following step.
const CALLOUTS: [&str; 5] = ["NOTE", "CAUTION", "WARNING", "IMPORTANT", "NOTICE"];

/// A line is an uppercase bold heading when all of its visible text is bold
/// and it has letters, all uppercase, and it is not a bare callout label.
fn is_bold_upper(segs: &[(String, bool)], text: &str) -> bool {
    if CALLOUTS.contains(&text.trim().trim_end_matches(':')) {
        return false;
    }
    let all_bold = segs
        .iter()
        .filter(|(s, _)| !s.trim().is_empty())
        .all(|(_, b)| *b);
    let has_letters = text.chars().any(|c| c.is_ascii_alphabetic());
    all_bold && has_letters && text == text.to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(body: &str) -> Page {
        parse_page(&format!(
            "<html><head><title>T — x</title></head><body><div class=\"main\"><h1>Title</h1>{body}</div><div class=\"theme-colors footer\">f</div></body></html>"
        ))
    }

    #[test]
    fn steps_indents_and_headings() {
        let p = page(
            "<br><b><a href='../pages/9453.html'>SPARK PLUG</a> REPLACEMENT</b><br><span class='indent-2'>&#09;</span>1.<span class='indent-5'>&#09;</span>Remove the <a href='../pages/9453.html'>plugs</a>.<br><span class='indent-2'>&#09;</span><b>NOTE:</b>  Observe<br>",
        );
        assert_eq!(p.title, "Title");
        let l: Vec<&Line> = p.lines.iter().filter(|l| !l.text.is_empty()).collect();
        assert_eq!(l[0].text, "SPARK PLUG REPLACEMENT");
        assert!(l[0].heading);
        assert_eq!(l[0].links, vec![9453]);
        assert_eq!(l[1].text, "\t1. Remove the plugs.");
        assert!(l[1].step && !l[1].heading);
        assert_eq!(l[1].level, 1);
        assert_eq!(l[2].text, "\tNOTE: Observe");
        assert!(!l[2].heading);
    }

    #[test]
    fn callout_label_is_not_heading() {
        let p = page(
            "<span class='indent-2'>&#09;</span><b>CAUTION:<br><span class='indent-2'>&#09;</span>- Refer to X</b>.<br>",
        );
        assert!(p.lines.iter().all(|l| !l.heading));
    }

    #[test]
    fn spec_line_and_images() {
        let p = page(
            "<br><a href='../pages/9453.html'>Spark Plug</a> Gap <span class='indent-right-align'>&#09;</span>  1.52 mm (0.060 in.)<div class='oxe-image'><a class='oxe-region' href='../pages/9528.html'></a><img class='big-img' src='../images/DM13Q313/gm60/1.png'></div>",
        );
        let l = p.lines.last().unwrap();
        assert_eq!(l.text, "Spark Plug Gap 1.52 mm (0.060 in.)");
        assert_eq!(l.imgs, vec!["images/DM13Q313/gm60/1.png"]);
        assert_eq!(l.links, vec![9453, 9528]);
    }

    #[test]
    fn h3_is_heading_line() {
        let p = page(
            "<b>TEST DESCRIPTION</b><br><br><h3>Step 1 - Step 8:</h3><div class='oxe-image'><img src='../images/a.png'></div><br>",
        );
        let h3 = p
            .lines
            .iter()
            .find(|l| l.text.starts_with("Step 1"))
            .unwrap();
        assert!(h3.heading);
        assert!(
            p.lines
                .iter()
                .any(|l| l.imgs == vec!["images/a.png"] && l.text.is_empty())
        );
    }
}
