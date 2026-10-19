//! Chunking a page's lines on the manual's own boundaries.
//!
//! Sections start at uppercase bold heads (and `<h3>`); within a section,
//! numbered steps and tables are atomic blocks; blocks are packed greedily up
//! to [`TARGET_TOKENS`]; tiny neighbours are merged. Anything still over
//! [`OUTLIER_TOKENS`] is split at paragraph boundaries, then lines, then
//! sentences.

use lm_types::approx_tokens;

use crate::manual::Line;

pub const TARGET_TOKENS: usize = 450;
pub const OUTLIER_TOKENS: usize = 600;
/// Pieces smaller than this get merged with a neighbour when the result fits.
pub const TINY_TOKENS: usize = 120;

/// A chunk before ids and metadata are attached.
#[derive(Debug, Clone)]
pub struct Piece {
    pub headings: Vec<String>,
    pub lines: Vec<Line>,
}

pub fn lines_text(lines: &[Line]) -> String {
    let mut s = String::new();
    for (i, l) in lines.iter().enumerate() {
        if i > 0 {
            s.push(if l.cont { ' ' } else { '\n' });
        }
        s.push_str(&l.text);
    }
    collapse_blank_runs(&s).trim_matches('\n').to_string()
}

fn collapse_blank_runs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut nl = 0;
    for ch in s.chars() {
        if ch == '\n' {
            nl += 1;
            if nl <= 2 {
                out.push(ch);
            }
        } else {
            nl = 0;
            out.push(ch);
        }
    }
    out
}

fn toks(lines: &[Line]) -> usize {
    approx_tokens(&lines_text(lines))
}

fn substantive(l: &Line) -> bool {
    !l.text.is_empty() || !l.imgs.is_empty()
}

/// Group lines into (heading path, lines). Consecutive heads nest.
pub fn sections(lines: &[Line]) -> Vec<(Vec<String>, Vec<Line>)> {
    let mut out = Vec::new();
    let mut heading: Vec<String> = Vec::new();
    let mut cur: Vec<Line> = Vec::new();
    let mut prev_heading = false;
    for ln in lines {
        if ln.heading {
            if cur.iter().any(|l| !l.heading && substantive(l)) {
                out.push((std::mem::take(&mut heading), std::mem::take(&mut cur)));
            } else if !prev_heading {
                heading.clear();
            }
            heading.push(ln.text.trim().trim_end_matches(':').to_string());
            cur.push(ln.clone());
            prev_heading = true;
            continue;
        }
        if substantive(ln) {
            prev_heading = false;
        }
        cur.push(ln.clone());
    }
    if !cur.is_empty() {
        out.push((heading, cur));
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BlockKind {
    Head,
    Table,
    Step,
    Para,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub lines: Vec<Line>,
    pub atomic: bool,
}

fn step_level(lines: &[Line]) -> usize {
    lines.iter().find(|l| l.step).map_or(0, |l| l.level)
}

/// Atomic blocks: heads, numbered steps with their sub-lines, tables,
/// paragraphs. Image-only lines attach to the following block.
pub fn blocks(sec: &[Line]) -> Vec<Block> {
    let mut out = Vec::new();
    let mut cur: Vec<Line> = Vec::new();
    let mut kind: Option<BlockKind> = None;
    let mut pending: Vec<Line> = Vec::new();
    let mut after_blank = false;

    let flush = |cur: &mut Vec<Line>, kind: &mut Option<BlockKind>, out: &mut Vec<Block>| {
        if !cur.is_empty() {
            out.push(Block {
                lines: std::mem::take(cur),
                atomic: matches!(kind, Some(BlockKind::Step | BlockKind::Table)),
            });
        }
        *kind = None;
    };

    for ln in sec {
        if !substantive(ln) {
            after_blank = true;
            if !cur.is_empty() {
                cur.push(ln.clone());
            }
            continue;
        }
        if ln.text.is_empty() && !ln.table {
            if kind == Some(BlockKind::Step) && !after_blank {
                cur.push(ln.clone());
            } else {
                pending.push(ln.clone());
            }
            continue;
        }
        let start_new = |k: BlockKind,
                         cur: &mut Vec<Line>,
                         kind: &mut Option<BlockKind>,
                         out: &mut Vec<Block>,
                         pending: &mut Vec<Line>| {
            flush(cur, kind, out);
            cur.append(pending);
            cur.push(ln.clone());
            *kind = Some(k);
        };
        if ln.heading {
            start_new(BlockKind::Head, &mut cur, &mut kind, &mut out, &mut pending);
            flush(&mut cur, &mut kind, &mut out);
        } else if ln.table {
            start_new(
                BlockKind::Table,
                &mut cur,
                &mut kind,
                &mut out,
                &mut pending,
            );
            flush(&mut cur, &mut kind, &mut out);
        } else if ln.step {
            start_new(BlockKind::Step, &mut cur, &mut kind, &mut out, &mut pending);
        } else if (kind == Some(BlockKind::Step) && (!after_blank || ln.level > step_level(&cur)))
            || (kind == Some(BlockKind::Para) && !after_blank)
        {
            cur.append(&mut pending);
            cur.push(ln.clone());
        } else {
            start_new(BlockKind::Para, &mut cur, &mut kind, &mut out, &mut pending);
        }
        after_blank = false;
    }
    if !pending.is_empty() {
        if cur.is_empty() {
            kind = Some(BlockKind::Para);
        }
        cur.append(&mut pending);
    }
    flush(&mut cur, &mut kind, &mut out);
    out
}

/// Greedy-pack line runs into groups of at most `limit` tokens.
fn greedy(units: Vec<Vec<Line>>, limit: usize) -> Vec<Vec<Line>> {
    let mut groups = Vec::new();
    let mut cur: Vec<Line> = Vec::new();
    for u in units {
        if !cur.is_empty() {
            let mut trial = cur.clone();
            trial.extend(u.iter().cloned());
            if toks(&trial) > limit {
                groups.push(std::mem::take(&mut cur));
            }
        }
        cur.extend(u);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

fn split_by_lines(lines: &[Line], limit: usize) -> Vec<Vec<Line>> {
    greedy(lines.iter().map(|l| vec![l.clone()]).collect(), limit)
}

/// Pack a section's blocks; only non-atomic paragraphs are split to fit.
pub fn pack(blocks: Vec<Block>) -> Vec<Vec<Line>> {
    let mut units = Vec::new();
    for b in blocks {
        if b.atomic || toks(&b.lines) <= TARGET_TOKENS {
            units.push(b.lines);
        } else {
            units.extend(split_by_lines(&b.lines, TARGET_TOKENS));
        }
    }
    greedy(units, TARGET_TOKENS)
}

/// Split an outlier group (> OUTLIER_TOKENS): paragraphs (blank-line runs),
/// then lines, then sentences within a single over-long line.
pub fn split_outlier(lines: &[Line]) -> Vec<Vec<Line>> {
    let mut paras: Vec<Vec<Line>> = vec![Vec::new()];
    for l in lines {
        if !substantive(l) && paras.last().is_some_and(|p| !p.is_empty()) {
            paras.last_mut().unwrap().push(l.clone());
            paras.push(Vec::new());
        } else {
            paras.last_mut().unwrap().push(l.clone());
        }
    }
    let mut units = Vec::new();
    for p in paras.into_iter().filter(|p| !p.is_empty()) {
        if toks(&p) <= TARGET_TOKENS {
            units.push(p);
            continue;
        }
        for l in p {
            if approx_tokens(&l.text) > TARGET_TOKENS {
                units.extend(split_sentences(&l).into_iter().map(|s| vec![s]));
            } else {
                units.push(vec![l]);
            }
        }
    }
    greedy(units, TARGET_TOKENS)
}

/// Split one line at sentence ends (". " followed by an uppercase letter).
/// Pieces after the first are marked `cont` so the text rejoins with a space.
fn split_sentences(l: &Line) -> Vec<Line> {
    let text = &l.text;
    let bytes = text.as_bytes();
    let mut cuts = Vec::new();
    for i in 1..bytes.len().saturating_sub(2) {
        if matches!(bytes[i], b'.' | b'?' | b'!')
            && bytes[i + 1] == b' '
            && bytes[i + 2].is_ascii_uppercase()
        {
            cuts.push(i + 1);
        }
    }
    let mut out = Vec::new();
    let mut start = 0;
    for c in cuts.into_iter().chain(std::iter::once(text.len())) {
        let piece = text[start..c].trim_start_matches(' ');
        if !piece.is_empty() {
            let first = out.is_empty();
            out.push(Line {
                text: piece.to_string(),
                imgs: if first { l.imgs.clone() } else { Vec::new() },
                links: if first { l.links.clone() } else { Vec::new() },
                cont: !first,
                ..l.clone()
            });
        }
        start = c;
    }
    out
}

/// Chunk a whole (non-image-only) page.
pub fn chunk_page(lines: &[Line]) -> Vec<Piece> {
    let mut pieces = Vec::new();
    for (heading, sec) in sections(lines) {
        let headings = if heading.is_empty() {
            Vec::new()
        } else {
            vec![heading.join(" > ")]
        };
        for g in pack(blocks(&sec)) {
            let groups = if toks(&g) > OUTLIER_TOKENS {
                split_outlier(&g)
            } else {
                vec![g]
            };
            for g in groups {
                if !lines_text(&g).is_empty() || g.iter().any(|l| !l.imgs.is_empty()) {
                    pieces.push(Piece {
                        headings: headings.clone(),
                        lines: g,
                    });
                }
            }
        }
    }
    let mut merged: Vec<Piece> = Vec::new();
    for p in pieces {
        if let Some(last) = merged.last_mut() {
            let mut both = last.lines.clone();
            both.extend(p.lines.iter().cloned());
            let tiny = toks(&last.lines) < TINY_TOKENS || toks(&p.lines) < TINY_TOKENS;
            if tiny && toks(&both) <= TARGET_TOKENS {
                last.lines = both;
                for h in p.headings {
                    if !last.headings.contains(&h) {
                        last.headings.push(h);
                    }
                }
                continue;
            }
        }
        merged.push(p);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> Line {
        let level = text.len() - text.trim_start_matches('\t').len();
        Line {
            text: text.into(),
            step: text
                .trim_start_matches('\t')
                .split_once(". ")
                .is_some_and(|(n, _)| n.parse::<u32>().is_ok()),
            heading: !text.is_empty()
                && text == text.to_uppercase()
                && text.chars().any(|c| c.is_alphabetic()),
            level,
            ..Line::default()
        }
    }

    #[test]
    fn steps_are_never_split() {
        let long = "word ".repeat(200);
        let lines = vec![
            line("REMOVAL PROCEDURE"),
            line(&format!("\t1. {long}")),
            line(&format!("\t\t- {long}")),
            line("\t2. Short step."),
        ];
        let pieces = chunk_page(&lines);
        // step 1 with its sub-bullet stays together even though it exceeds the target
        let step1 = pieces
            .iter()
            .find(|p| lines_text(&p.lines).contains("1. word"))
            .unwrap();
        assert!(lines_text(&step1.lines).contains("- word"));
    }

    #[test]
    fn outlier_single_line_splits_at_sentences_and_rejoins_exactly() {
        let sentence = "The module monitors the circuit voltage continuously. ";
        let text = sentence.repeat(80);
        let text = text.trim_end().to_string();
        let lines = vec![line(&text)];
        let pieces = chunk_page(&lines);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|p| toks(&p.lines) <= TARGET_TOKENS));
        let rejoined: Vec<String> = pieces.iter().map(|p| lines_text(&p.lines)).collect();
        assert_eq!(rejoined.join(" "), text);
    }

    #[test]
    fn tiny_sections_merge() {
        let lines = vec![
            line("DTC DESCRIPTOR"),
            line("\tDTC P0300 Engine Misfire Detected"),
            line("CONDITIONS"),
            line("\tEngine running."),
        ];
        let pieces = chunk_page(&lines);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].headings, vec!["DTC DESCRIPTOR", "CONDITIONS"]);
    }
}
