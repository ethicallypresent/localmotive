//! Verifier tests against chunks shaped like the Uplander manual
//! (fixtures keep the real chunk ids, titles and spec values; the prose is paraphrased).

use lm_types::{
    Chunk, Claim, DraftFile, NextStep, RetrievedChunk, RetrievedFile, SCHEMA_VERSION, Source,
    Vehicle, Verdict, VerifiedFile,
};
use lm_verify::{claims_from_inline, verify};

const GAP: &str = "4184c22de9df6a3b"; // Spark Plug Gap (page 8668, alias of 15449)
const INSTALL: &str = "f3598b8bea83afe7"; // Spark Plug: Service and Repair, installation (page 15448)
const REMOVAL: &str = "3b5f9e721f88ebc9"; // Spark Plug: Service and Repair, removal (page 15448)
const P0300: &str = "7b32f4441dbbaacd"; // P0300 conditions for running (page 21499)
const OCR_GAP: &str = "00000000000000aa"; // synthetic OCR copy of the gap chunk

fn fixtures() -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> =
        serde_json::from_str(include_str!("fixtures/chunks.json")).unwrap();
    let mut ocr = chunks.iter().find(|c| c.id == GAP).unwrap().clone();
    ocr.id = OCR_GAP.into();
    ocr.source = Source::Ocr;
    ocr.ocr_confidence = Some(84.0);
    chunks.push(ocr);
    chunks
}

fn vehicle() -> Vehicle {
    serde_json::from_str(include_str!("fixtures/vehicle.json")).unwrap()
}

fn retrieved() -> RetrievedFile {
    RetrievedFile {
        schema_version: SCHEMA_VERSION,
        query: "q".into(),
        chunks: fixtures()
            .into_iter()
            .map(|chunk| RetrievedChunk { score: 1.0, chunk })
            .collect(),
    }
}

fn verify_claims(answer: Vec<Claim>, next_step: Option<NextStep>) -> VerifiedFile {
    let draft = DraftFile {
        schema_version: SCHEMA_VERSION,
        answer,
        next_step,
    };
    let v = vehicle();
    verify(&draft, &retrieved(), Some(&v))
}

/// Tests are written with inline `[id]` markers for brevity and converted to claims.
fn run(answer: &str) -> VerifiedFile {
    verify_claims(claims_from_inline(answer), None)
}

fn claim(text: &str, cites: &[&str]) -> Claim {
    Claim {
        text: text.into(),
        cites: cites.iter().map(|s| s.to_string()).collect(),
    }
}

fn verdicts(v: &VerifiedFile) -> Vec<Verdict> {
    v.claims.iter().map(|s| s.verdict).collect()
}

#[test]
fn fixture_ids_are_real() {
    let ids: Vec<String> = fixtures().into_iter().map(|c| c.id).collect();
    for id in [GAP, INSTALL, REMOVAL, P0300] {
        assert!(ids.contains(&id.to_string()), "{id} missing from fixtures");
    }
}

#[test]
fn spark_plug_gap_exact_passes() {
    let v = run(&format!(
        "Gap the spark plugs to 1.52 mm (0.060 in.) [{GAP}]."
    ));
    assert!(v.ok, "{v:#?}");
    assert_eq!(v.claims[0].numbers.len(), 2);
    assert_eq!(v.claims[0].numbers[0].found_in.as_deref(), Some(GAP));
    assert!(!v.claims[0].needs_image);
}

#[test]
fn spark_plug_gap_near_misses_fail() {
    for wrong in [
        "1.5 mm", "1.52 in.", "0.06 in.", "1.25 mm", "52 mm", "1.520 mm",
    ] {
        let v = run(&format!("Gap the spark plugs to {wrong} [{GAP}]."));
        assert!(!v.ok, "{wrong} should fail: {v:#?}");
        assert!(
            v.claims[0].reasons[0].contains(wrong),
            "{:?}",
            v.claims[0].reasons
        );
    }
}

#[test]
fn p0300_rpm_range() {
    assert!(
        run(&format!(
            "The misfire test runs when engine speed is between 525-5,700 RPM [{P0300}]."
        ))
        .ok
    );
    for wrong in [
        "525-5700 RPM",
        "5700 RPM",
        "500-5,700 RPM",
        "525-5,700 rpm",
        "525 - 5,700 RPM",
    ] {
        let v = run(&format!(
            "The misfire test runs when engine speed is between {wrong} [{P0300}]."
        ));
        assert!(!v.ok, "{wrong} should fail");
    }
}

#[test]
fn p0300_temperatures_and_voltage() {
    assert!(
        run(&format!(
            "The ignition voltage must be between 9-18 volts [{P0300}]."
        ))
        .ok
    );
    assert!(run(&format!("Below -7°C (+19°F) at start-up the test is disabled until the ECT is more than 21°C (69°F) [{P0300}].")).ok);
    assert!(
        !run(&format!(
            "Below -8°C at start-up the test is disabled [{P0300}]."
        ))
        .ok
    );
    assert!(
        !run(&format!(
            "The ignition voltage must be between 9-16 volts [{P0300}]."
        ))
        .ok
    );
}

#[test]
fn dtc_code_in_title_counts() {
    // "P0300" is only in this chunk's title; the title is part of the verifiable context.
    assert!(
        run(&format!(
            "P0300 runs when engine speed is between 525-5,700 RPM [{P0300}]."
        ))
        .ok
    );
    assert!(
        run(&format!(
            "The test does not run while P0101 or P0335 is set [{P0300}]."
        ))
        .ok
    );
    assert!(
        !run(&format!(
            "P0301 runs when engine speed is between 525-5,700 RPM [{P0300}]."
        ))
        .ok
    );
}

#[test]
fn vehicle_pseudo_source() {
    assert!(
        verify_claims(
            vec![claim(
                "Your 2006 Uplander has the 3.5L V6 (LX9).",
                &["vehicle"]
            )],
            None
        )
        .ok
    );
    assert!(
        verify_claims(
            vec![claim(
                "VIN 1GNDV23L00D000000 is a VIN code L engine.",
                &["vehicle"]
            )],
            None
        )
        .ok
    );
    // numbers not in vehicle.json fail, and the vehicle can't vouch for specs
    assert!(
        !verify_claims(
            vec![claim("Your 2007 Uplander has the 3.9L V6.", &["vehicle"])],
            None
        )
        .ok
    );
    assert!(!verify_claims(vec![claim("Gap the plugs to 1.52 mm.", &["vehicle"])], None).ok);
    // vehicle numbers without citing the vehicle fail
    assert!(!verify_claims(vec![claim("Your 2006 Uplander.", &[])], None).ok);
    // without a vehicle file, "vehicle" is an unknown source
    let draft = DraftFile {
        schema_version: SCHEMA_VERSION,
        answer: vec![claim("A 2006 van.", &["vehicle"])],
        next_step: None,
    };
    assert!(!verify(&draft, &retrieved(), None).ok);
}

#[test]
fn next_step_checked() {
    let ok = verify_claims(
        vec![claim("Start with the removal procedure.", &[REMOVAL])],
        Some(NextStep {
            chunk_id: INSTALL.into(),
            step_label: "Step 3".into(),
        }),
    );
    assert!(ok.ok, "{ok:#?}");
    let bad_step = verify_claims(
        vec![],
        Some(NextStep {
            chunk_id: INSTALL.into(),
            step_label: "Step 9".into(),
        }),
    );
    assert!(!bad_step.ok);
    let bad_chunk = verify_claims(
        vec![],
        Some(NextStep {
            chunk_id: "0123456789abcdef".into(),
            step_label: "Step 1".into(),
        }),
    );
    assert!(!bad_chunk.ok);
}

#[test]
fn torque_from_installation_step() {
    let ok = run(&format!(
        "Tighten the spark plugs to 15 N.m (11 lb ft) [{INSTALL}]."
    ));
    assert!(ok.ok, "{ok:#?}");
    assert!(
        !run(&format!(
            "Tighten the spark plugs to 15 N·m (11 lb ft) [{INSTALL}]."
        ))
        .ok,
        "N·m vs N.m is not normalized"
    );
    assert!(
        !run(&format!(
            "Tighten the spark plugs to 15 N.m (11 lb-ft) [{INSTALL}]."
        ))
        .ok
    );
    assert!(!run(&format!("Tighten the spark plugs to 16 N.m [{INSTALL}].")).ok);
}

#[test]
fn numbers_need_a_citation() {
    let v = run("Gap the spark plugs to 1.52 mm (0.060 in.).");
    assert_eq!(verdicts(&v), vec![Verdict::Fail]);
    assert!(v.claims[0].reasons[0].contains("cites no source"));
    // a sentence without numbers needs no citation
    assert!(run("Let the engine cool before you start.").ok);
}

#[test]
fn number_must_be_in_a_cited_chunk_not_just_retrieved() {
    // 1.52 mm is retrieved (GAP) but this sentence only cites the removal chunk
    let v = run(&format!("Gap the spark plugs to 1.52 mm [{REMOVAL}]."));
    assert!(!v.ok);
    // citing both is fine
    assert!(
        run(&format!(
            "Gap the spark plugs to 1.52 mm [{REMOVAL}][{GAP}]."
        ))
        .ok
    );
}

#[test]
fn unknown_citation_fails() {
    let v = run("Allow the engine to cool [0123456789abcdef].");
    assert!(!v.ok);
    assert!(v.claims[0].reasons[0].contains("unknown source"));
}

#[test]
fn ocr_citation_passes_but_needs_image() {
    let v = run(&format!("Gap the spark plugs to 1.52 mm [{OCR_GAP}]."));
    assert!(v.ok);
    assert!(v.claims[0].needs_image);
    // OCR text is still checked
    assert!(!run(&format!("Gap the spark plugs to 1.6 mm [{OCR_GAP}].")).ok);
}

#[test]
fn per_sentence_verdicts() {
    let v = run(&format!(
        "Let the engine cool first. Gap the spark plugs to 1.52 mm (0.060 in.) [{GAP}]. Then torque them to 25 N.m [{INSTALL}].\nThe misfire test needs 525-5,700 RPM [{P0300}]."
    ));
    assert_eq!(
        verdicts(&v),
        vec![Verdict::Pass, Verdict::Pass, Verdict::Fail, Verdict::Pass],
        "{v:#?}"
    );
    assert!(!v.ok);
}

#[test]
fn verified_file_round_trips() {
    let v = run(&format!("Gap the spark plugs to 1.52 mm [{GAP}]."));
    let s = serde_json::to_string(&v).unwrap();
    let back: VerifiedFile = serde_json::from_str(&s).unwrap();
    assert_eq!(back, v);
    assert!(s.contains(r#""verdict":"pass""#));
}
