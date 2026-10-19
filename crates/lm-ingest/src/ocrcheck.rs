//! Safety cross-check for OCR'd tables.
//!
//! OCR can turn a printed 25 N.m into a plausible 20 N.m. The verifier only
//! checks that a number is *in* the chunk text, so a misread would pass. To
//! prevent that, every row of an OCR spec table is checked deterministically:
//! the row must hold exactly one metric and one English value of a known pair
//! (N.m↔lb ft, N.m↔lb in, L↔qt, mm↔in, kPa↔psi, °C↔°F) that agree by unit
//! conversion within ±1 unit of the printed precision or ±5 %.
//!
//! * Rows that agree → their line index goes in `ocr_checked_rows`.
//! * Rows that disagree, or carry a single unit value → every number on the
//!   row is replaced by [`MASK`], so no claim can be verified from it and the
//!   answer has to point at the image (the raw text stays in `ocr_raw`).
//! * Angles (`95 degrees`) have no pair: masked unless the same value appears
//!   in HTML text that also names the component.
//! * Step tables (diagnostic procedures) have no metric/English pairs: values
//!   with a unit are masked, bare numbers (step numbers) are kept.

use std::sync::LazyLock;

use regex::Regex;

pub const MASK: &str = "[see image]";

/// Number with an optional unit. Units are matched longest-first.
static RE_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+(?:[.,]\d+)?(?:\s?[-–]\s?\d+(?:[.,]\d+)?)?)(?:\s?(N\.m|N·m|Nm|lb ft|lb in|kPa|psi|qt|mm|in\.|in\b|L\b|°C|°F|degrees|V\b|%|ohms|Ω|RPM|ms|Hz|A\b|km/h|mph|°))?").unwrap()
});

#[derive(Debug, Clone, Copy, PartialEq)]
enum Unit {
    Nm,
    LbFt,
    LbIn,
    Litre,
    Quart,
    Mm,
    Inch,
    Kpa,
    Psi,
    Celsius,
    Fahrenheit,
    Degrees,
    Other,
}

fn unit_of(s: &str) -> Unit {
    match s {
        "N.m" | "N·m" | "Nm" => Unit::Nm,
        "lb ft" => Unit::LbFt,
        "lb in" => Unit::LbIn,
        "L" => Unit::Litre,
        "qt" => Unit::Quart,
        "mm" => Unit::Mm,
        "in." | "in" => Unit::Inch,
        "kPa" => Unit::Kpa,
        "psi" => Unit::Psi,
        "°C" => Unit::Celsius,
        "°F" => Unit::Fahrenheit,
        "degrees" | "°" => Unit::Degrees,
        _ => Unit::Other,
    }
}

#[derive(Debug, Clone)]
struct Value {
    start: usize,
    end: usize,
    num: f64,
    decimals: i32,
    unit: Option<Unit>,
}

fn values(line: &str) -> Vec<Value> {
    RE_VALUE
        .captures_iter(line)
        .filter_map(|c| {
            let m = c.get(0)?;
            // skip digits glued to letters on the left (2WD, C275, Step2)
            if line[..m.start()]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_alphabetic())
            {
                return None;
            }
            // ... or on the right without forming a unit (2WD, 4T65), or part of a fraction in a label (1/4 inch)
            let next = line[m.end()..].chars().next();
            let prev = line[..m.start()].chars().next_back();
            if c.get(2).is_none()
                && (next.is_some_and(|ch| ch.is_alphanumeric() || ch == '/') || prev == Some('/'))
            {
                return None;
            }
            let first = c[1].split(['-', '–']).next().unwrap_or_default().trim();
            let n = first.replace(',', ".");
            let decimals = n.split_once('.').map_or(0, |(_, d)| d.len() as i32);
            Some(Value {
                start: m.start(),
                end: m.end(),
                num: n.parse().ok()?,
                decimals,
                unit: c.get(2).map(|u| unit_of(u.as_str())),
            })
        })
        .collect()
}

/// metric → English conversion for a known pair.
fn convert(metric: Unit, english: Unit, x: f64) -> Option<f64> {
    Some(match (metric, english) {
        (Unit::Nm, Unit::LbFt) => x * 0.737_562,
        (Unit::Nm, Unit::LbIn) => x * 8.850_75,
        (Unit::Litre, Unit::Quart) => x * 1.056_69,
        (Unit::Mm, Unit::Inch) => x / 25.4,
        (Unit::Kpa, Unit::Psi) => x * 0.145_038,
        (Unit::Celsius, Unit::Fahrenheit) => x * 9.0 / 5.0 + 32.0,
        _ => return None,
    })
}

fn is_metric(u: Unit) -> bool {
    matches!(
        u,
        Unit::Nm | Unit::Litre | Unit::Mm | Unit::Kpa | Unit::Celsius
    )
}

/// True when the printed English value matches the converted metric value
/// within ±1 unit of its printed precision or ±5 %.
pub fn agrees(expected: f64, printed: f64, decimals: i32) -> bool {
    let diff = (expected - printed).abs();
    diff <= 10f64.powi(-decimals) || diff <= expected.abs() * 0.05
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOutcome {
    /// No numbers on the row.
    NoValues,
    /// Metric and English agree.
    Checked,
    /// Angle corroborated by HTML text naming the component.
    Corroborated,
    /// Numbers replaced by the mask.
    Masked,
}

fn mask_values(line: &str, vals: &[&Value]) -> String {
    let mut out = line.to_string();
    for v in vals.iter().rev() {
        out.replace_range(v.start..v.end, MASK);
    }
    out
}

/// Check one table row. `corroborate(label, value_text)` reports whether HTML
/// text names the component together with the exact angle value.
pub fn check_spec_row(
    line: &str,
    label: &str,
    corroborate: &dyn Fn(&str, &str) -> bool,
) -> (String, RowOutcome) {
    let vals = values(line);
    if vals.is_empty() {
        return (line.to_string(), RowOutcome::NoValues);
    }
    let with_unit: Vec<&Value> = vals.iter().filter(|v| v.unit.is_some()).collect();
    let metric: Vec<&&Value> = with_unit
        .iter()
        .filter(|v| v.unit.is_some_and(is_metric))
        .collect();
    let english: Vec<&&Value> = with_unit
        .iter()
        .filter(|v| {
            v.unit
                .is_some_and(|u| !is_metric(u) && u != Unit::Degrees && u != Unit::Other)
        })
        .collect();
    if vals.len() == 2 && metric.len() == 1 && english.len() == 1 {
        let (m, e) = (metric[0], english[0]);
        if let Some(exp) = convert(m.unit.unwrap(), e.unit.unwrap(), m.num)
            && agrees(exp, e.num, e.decimals)
        {
            return (line.to_string(), RowOutcome::Checked);
        }
    }
    if vals.len() == 1
        && vals[0].unit == Some(Unit::Degrees)
        && corroborate(label, &line[vals[0].start..vals[0].end])
    {
        return (line.to_string(), RowOutcome::Corroborated);
    }
    let all: Vec<&Value> = vals.iter().collect();
    (mask_values(line, &all), RowOutcome::Masked)
}

/// Step tables: mask every value that carries a unit; keep bare numbers.
pub fn check_step_line(line: &str) -> (String, bool) {
    let vals = values(line);
    let unit_vals: Vec<&Value> = vals.iter().filter(|v| v.unit.is_some()).collect();
    if unit_vals.is_empty() {
        (line.to_string(), false)
    } else {
        (mask_values(line, &unit_vals), true)
    }
}

/// Text of a row without its values, used as the component label.
pub fn row_label(line: &str) -> String {
    let vals = values(line);
    let mut s = line.to_string();
    for v in vals.iter().rev() {
        s.replace_range(v.start..v.end, "");
    }
    s.trim_matches(|c: char| !c.is_alphanumeric())
        .trim()
        .to_string()
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct TableCheck {
    pub text: String,
    pub checked_rows: Vec<u32>,
    pub masked_rows: u32,
    pub masked_values: u32,
    pub corroborated_rows: u32,
}

const PASS_WORDS: [&str; 4] = ["First Pass", "Final Pass", "Second Pass", "Third Pass"];

/// Cross-check a whole OCR spec table.
pub fn check_spec_table(text: &str, corroborate: &dyn Fn(&str, &str) -> bool) -> TableCheck {
    let mut out = TableCheck::default();
    let mut lines = Vec::new();
    let mut parent = String::new();
    for (i, line) in text.lines().enumerate() {
        let label = row_label(line);
        let is_pass = PASS_WORDS.iter().any(|p| label.contains(p)) || label.is_empty();
        let comp = if is_pass {
            parent.clone()
        } else {
            label.clone()
        };
        let (fixed, outcome) = check_spec_row(line, &comp, corroborate);
        match outcome {
            RowOutcome::Checked => out.checked_rows.push(i as u32),
            RowOutcome::Corroborated => {
                out.checked_rows.push(i as u32);
                out.corroborated_rows += 1;
            }
            RowOutcome::Masked => {
                out.masked_rows += 1;
                out.masked_values += fixed.matches(MASK).count() as u32;
            }
            RowOutcome::NoValues => {}
        }
        if values(line).is_empty() && !label.is_empty() && !is_pass {
            parent = label; // header row such as "Cylinder Head Bolt"
        }
        lines.push(fixed);
    }
    out.text = lines.join("\n");
    out
}

/// Mask unit values in a step table.
pub fn check_step_table(text: &str) -> TableCheck {
    let mut out = TableCheck::default();
    let lines: Vec<String> = text
        .lines()
        .map(|l| {
            let (fixed, masked) = check_step_line(l);
            if masked {
                out.masked_rows += 1;
                out.masked_values += fixed.matches(MASK).count() as u32;
            }
            fixed
        })
        .collect();
    out.text = lines.join("\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never(_: &str, _: &str) -> bool {
        false
    }

    #[test]
    fn real_rows_from_page_2536() {
        // correct reads pass
        for ok in [
            "Camshaft Sprocket Bolt 140 N.m 103 lb ft",
            "Camshaft Position Sensor Bolt 10 N.m 89 lb in",
            "Engine Wiring Harness Bracket Bolt 13 N.m 115 lb in",
            "e Brake Caliper Bracket Bolt 185 Nm 137 lb ft",
            "Coolant Drain Plug 19 N.m 14 lb ft",
        ] {
            assert_eq!(
                check_spec_row(ok, "", &never).1,
                RowOutcome::Checked,
                "{ok}"
            );
        }
        // the plausible misreads are masked
        for (bad, why) in [
            ("Knock Sensor 20 N.m 18 lb ft", "25 read as 20"),
            ("Oil Filter 500 N.m 22 lb ft", "30 read as 500"),
            ("Heater Inlet Pipe Nut 209 N.m 18 lb ft", "25 read as 209"),
            (
                "Lower Intake Manifold Bolt — Corner 25 N.m 186 lb ft",
                "18 read as 186",
            ),
            ("Throttle Body Stud 6 N.m 63 lb in", "53 read as 63"),
            ("Cylinder Head Plug 220 N.m 15 lb ft", "20 read as 220"),
        ] {
            let (t, o) = check_spec_row(bad, "", &never);
            assert_eq!(o, RowOutcome::Masked, "{why}");
            assert!(!t.chars().any(|c| c.is_ascii_digit()), "{t}");
            assert!(t.contains(MASK));
        }
    }

    #[test]
    fn single_unit_rows_masked_capacity_pairs_checked() {
        assert_eq!(
            check_spec_row("Throttle Body Stud 6 N.m", "", &never).1,
            RowOutcome::Masked
        );
        assert_eq!(
            check_spec_row("Bottom Pan Removal (2WD) 7.0L 7.4 qt", "", &never).1,
            RowOutcome::Checked
        );
        assert_eq!(
            check_spec_row("Complete Overhaul (2VWWD) 9 5L 10.0 af", "", &never).1,
            RowOutcome::Masked
        );
        assert_eq!(
            check_spec_row("Application Metric English", "", &never).1,
            RowOutcome::NoValues
        );
        assert_eq!(
            check_spec_row(
                "Rear Oil Gallery Plug — 1/4 inch 19 N.m 14 lb ft",
                "",
                &never
            )
            .1,
            RowOutcome::Checked
        );
    }

    #[test]
    fn angles_need_html_corroboration() {
        let html = |label: &str, value: &str| {
            label.contains("Cylinder Head Bolt") && value == "95 degrees"
        };
        let t = check_spec_table(
            "Cylinder Head Bolt\nFirst Pass 60 N.m 44 lb ft\nFinal Pass 95 degrees\nConnecting Rod Bearing Cap Bolt\ne Final Pass 110 degrees",
            &html,
        );
        assert_eq!(t.checked_rows, vec![1, 2]);
        assert_eq!(t.corroborated_rows, 1);
        assert!(t.text.ends_with(&format!("e Final Pass {MASK}")));
        assert_eq!(t.masked_rows, 1);
    }

    #[test]
    fn step_tables_keep_step_numbers() {
        let t =
            check_step_table("Go to Step 3 Aids\n5 Measure the voltage. 4.5-5.2 V Go to Step 6");
        assert_eq!(
            t.text,
            format!("Go to Step 3 Aids\n5 Measure the voltage. {MASK} Go to Step 6")
        );
        assert_eq!(t.masked_values, 1);
    }

    #[test]
    fn tolerance() {
        assert!(agrees(18.44, 18.0, 0)); // 25 N.m -> 18 lb ft
        assert!(!agrees(14.75, 18.0, 0)); // 20 N.m printed with 18 lb ft
        assert!(agrees(7.397, 7.4, 1));
    }
}
