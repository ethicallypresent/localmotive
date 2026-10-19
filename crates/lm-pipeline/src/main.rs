//! lm — run LocalMotive turns from the command line.
//!
//! ```text
//! lm ask "<question>" [--session ID] [--preset a|b|c] [--model M] [--num-thread N] [--bm25-only]
//! lm eval-e2e [golden.yaml ...] [--limit N] [--preset ...] [--model M] [--num-thread N] [--bm25-only]
//! ```

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use lm_pipeline::{
    Config, Ctx, Progress, answer_markdown, new_session_id, run_turn_with, sessions_root,
};
use lm_types::Verdict;

fn take(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    let v = args.get(i + 1).cloned();
    args.drain(i..(i + 2).min(args.len()));
    v
}

fn avg(v: &[u64]) -> f64 {
    v.iter().sum::<u64>() as f64 / v.len().max(1) as f64
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let preset = take(&mut args, "--preset").unwrap_or_else(|| "b".into());
    let mut cfg =
        Config::preset(&preset).with_context(|| format!("unknown preset {preset:?} (a, b, c)"))?;
    if let Some(m) = take(&mut args, "--model") {
        cfg.model = m;
        cfg.name.push_str("+model");
    }
    if let Some(n) = take(&mut args, "--num-thread") {
        cfg.num_thread = Some(n.parse()?);
        cfg.name.push_str(&format!("+t{n}"));
    }
    let session = take(&mut args, "--session");
    let limit: Option<usize> = take(&mut args, "--limit").map(|n| n.parse()).transpose()?;
    let bm25_only = args.iter().any(|a| a == "--bm25-only");
    args.retain(|a| a != "--bm25-only");

    let model = cfg.model.clone();
    let mut ctx = Ctx::open(cfg)?;
    ctx.use_vectors = !bm25_only;
    if !ctx.client.has_model(&model).await? {
        bail!("answer model {model} is not pulled; run: ollama pull {model}");
    }
    let quiet = |_: Progress| {};
    match args.first().map(String::as_str) {
        Some("ask") => {
            let q = args.get(1).cloned().unwrap_or_default();
            let dir = sessions_root().join(session.unwrap_or_else(new_session_id));
            let show = |p: Progress| match p {
                Progress::StageStart { stage } => eprint!("[{stage}…]"),
                Progress::StageEnd { stage, ms, detail } => eprintln!(" {stage} {ms} ms: {detail}"),
                Progress::Verdicts { passed, failed, .. } => {
                    eprintln!("  verifier: {passed} pass, {failed} fail")
                }
                Progress::Retry { .. } => eprintln!("  retrying with verifier reasons"),
                _ => {}
            };
            let turn = run_turn_with(&ctx, &dir, &q, &show).await?;
            println!("{}", answer_markdown(&turn.answer));
            let t = &turn.timings;
            eprintln!(
                "files: {}\nconfig {}: route {} ms, retrieve {} ms, draft {:?} ms (first token {:?}), verify {:?} ms, total {} ms; tokens in {:?} out {:?}; ollama peak RSS {:.0} MiB",
                turn.dir.display(),
                t.config,
                t.route_ms,
                t.retrieve_ms,
                t.draft_ms,
                t.first_token_ms,
                t.verify_ms,
                t.total_ms,
                t.prompt_tokens,
                t.output_tokens,
                t.ollama_peak_rss_mib
            );
        }
        Some("reeval") => {
            // Recompute the shown answer (order, best match, weak flag) from saved turn files.
            let dir = PathBuf::from(args.get(1).context("reeval needs a session dir")?);
            let mut turns: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.join("verified.json").exists())
                .collect();
            turns.sort_by_key(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.strip_prefix("turn-"))
                    .and_then(|n| n.parse::<u32>().ok())
                    .unwrap_or(0)
            });
            let mut weak = 0;
            for t in &turns {
                let r: lm_types::RetrievedFile =
                    serde_json::from_str(&std::fs::read_to_string(t.join("retrieved.json"))?)?;
                let v: lm_types::VerifiedFile =
                    serde_json::from_str(&std::fs::read_to_string(t.join("verified.json"))?)?;
                let a = lm_pipeline::shown_answer(&r.query, &r, &v, false);
                weak += usize::from(a.weak);
                let b = a
                    .best_match
                    .as_ref()
                    .map_or(String::new(), |b| format!("{} {}", b.page_id, b.title));
                println!(
                    "{:<8} weak={:<5} best={:<50} {}",
                    t.file_name().unwrap_or_default().to_string_lossy(),
                    a.weak,
                    b,
                    r.query
                );
                if a.claims.len() >= 3 {
                    for c in &a.claims {
                        println!(
                            "           · {}",
                            c.text.chars().take(70).collect::<String>()
                        );
                    }
                }
            }
            println!("{weak} of {} flagged weak", turns.len());
            return Ok(());
        }
        Some("eval-e2e") => {
            let files: Vec<PathBuf> = if args.len() > 1 {
                args[1..].iter().map(PathBuf::from).collect()
            } else {
                vec!["tests/golden.yaml".into()]
            };
            let dir = sessions_root().join(format!("eval-{}-{}", ctx.cfg.name, new_session_id()));
            let mut report = format!("config {:?}\n\n", ctx.cfg);
            for f in files {
                let mut golden = lm_retrieve::golden::load(&f)?;
                if let Some(n) = limit {
                    golden.truncate(n);
                }
                let (
                    mut first_ok,
                    mut final_ok,
                    mut claims,
                    mut passed,
                    mut cited_expected,
                    mut empty,
                ) = (0, 0, 0, 0, 0, 0);
                let (mut draft_ms, mut total_ms, mut first_tok, mut prompt_tok, mut peak) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0f64);
                for g in &golden {
                    let turn = match run_turn_with(&ctx, &dir, &g.query, &quiet).await {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = writeln!(report, "ERR  {:<48} {e:#}", format!("{:?}", g.query));
                            continue;
                        }
                    };
                    let v = &turn.verified;
                    let n_pass = v
                        .claims
                        .iter()
                        .filter(|c| c.verdict == Verdict::Pass)
                        .count();
                    first_ok += usize::from(!turn.answer.retried);
                    final_ok += usize::from(v.ok);
                    claims += v.claims.len();
                    passed += n_pass;
                    empty += usize::from(n_pass == 0);
                    let cites_expected =
                        v.claims
                            .iter()
                            .filter(|c| c.verdict == Verdict::Pass)
                            .flat_map(|c| &c.cites)
                            .any(|id| {
                                turn.retrieved.chunks.iter().any(|r| {
                                    &r.chunk.id == id && g.expect.contains(&r.chunk.page_id)
                                })
                            });
                    cited_expected += usize::from(cites_expected);
                    draft_ms.extend(turn.timings.draft_ms.iter().copied());
                    first_tok.extend(turn.timings.first_token_ms.iter().copied());
                    prompt_tok.extend(turn.timings.prompt_tokens.iter().map(|&x| x as u64));
                    total_ms.push(turn.timings.total_ms);
                    peak = peak.max(turn.timings.ollama_peak_rss_mib);
                    let _ = writeln!(
                        report,
                        "{} {:<48} claims {n_pass}/{} pass, retried {}, cites expected page {cites_expected}, prompt {:?} tok, {:.1}s  [{}]",
                        if v.ok { "PASS" } else { "FAIL" },
                        format!("{:?}", g.query),
                        v.claims.len(),
                        turn.answer.retried,
                        turn.timings.prompt_tokens,
                        turn.timings.total_ms as f64 / 1000.0,
                        turn.dir.file_name().unwrap_or_default().to_string_lossy()
                    );
                    eprintln!("{}", report.lines().last().unwrap_or_default());
                }
                let n = golden.len().max(1) as f64;
                let _ = writeln!(
                    report,
                    "\n{}: {} questions — all claims verified {:.0}% (without retry {:.0}%), claim pass {:.0}% ({passed}/{claims}), no verified claim {empty}, cites an expected page {:.0}%\n  avg prompt {:.0} tok, avg first token {:.1}s, avg draft {:.1}s, avg turn {:.1}s, max turn {:.1}s, ollama peak RSS {:.0} MiB\n",
                    f.display(),
                    golden.len(),
                    100.0 * final_ok as f64 / n,
                    100.0 * first_ok as f64 / n,
                    100.0 * passed as f64 / claims.max(1) as f64,
                    100.0 * cited_expected as f64 / n,
                    avg(&prompt_tok),
                    avg(&first_tok) / 1000.0,
                    avg(&draft_ms) / 1000.0,
                    avg(&total_ms) / 1000.0,
                    total_ms.iter().max().copied().unwrap_or(0) as f64 / 1000.0,
                    peak
                );
            }
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join("e2e_report.txt"), &report)?;
            println!("{report}\nturn files under {}", dir.display());
        }
        _ => bail!(
            "usage: lm ask \"<question>\" | lm eval-e2e [golden.yaml ...] (see source header)"
        ),
    }
    Ok(())
}
