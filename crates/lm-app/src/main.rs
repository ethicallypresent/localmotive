//! LocalMotive desktop app (Tauri v2).
//!
//! Boot: config → corpus DB → vehicle → Ollama (use running, or start and
//! supervise) → models present → pipeline → answer-model warm-up → telemetry.
//! Each check drives the boot screen. Then the dashboard: fake telemetry at
//! 5 Hz, DTC cards, procedure walker, event log, and the query bar that runs
//! one lm-pipeline turn and returns only verified claims.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod supervisor;
mod telemetry;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lm_index::CHUNK_COLUMNS;
use lm_pipeline::{Answer, Config as PipelineConfig, Ctx, Progress, Timings, run_turn_with};
use lm_types::{Chunk, Source, VerifiedFile};
use serde::Serialize;
use tauri::http::{Response, StatusCode};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

#[derive(Debug, Clone, Serialize)]
struct BootStep {
    id: String,
    label: String,
    /// run | ok | warn | fail
    status: String,
    detail: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct BootState {
    steps: Vec<BootStep>,
    done: bool,
    ok: bool,
    model: String,
    vin: String,
    chunks: i64,
    vectors: i64,
}

struct AppState {
    boot: Mutex<BootState>,
    ui_db: Mutex<Option<rusqlite::Connection>>,
    ctx: Mutex<Option<Ctx>>,
    ollama: Mutex<Option<supervisor::Ollama>>,
    session: PathBuf,
}

type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

// ------------------------------------------------------------------ boot

fn boot_step(app: &AppHandle, id: &str, label: &str, status: &str, detail: &str) {
    let step = BootStep {
        id: id.into(),
        label: label.into(),
        status: status.into(),
        detail: detail.into(),
    };
    let state = app.state::<AppState>();
    let mut b = state.boot.lock().unwrap();
    if let Some(s) = b.steps.iter_mut().find(|s| s.id == id) {
        *s = step.clone();
    } else {
        b.steps.push(step.clone());
    }
    drop(b);
    let _ = app.emit("boot", step);
}

fn boot_fail(app: &AppHandle, id: &str, label: &str, detail: String) {
    boot_step(app, id, label, "fail", &detail);
    let state = app.state::<AppState>();
    let mut b = state.boot.lock().unwrap();
    b.done = true;
    b.ok = false;
    let snapshot = b.clone();
    drop(b);
    let _ = app.emit("boot-done", snapshot);
}

async fn boot(app: AppHandle, cfg: config::Config, cfg_warning: Option<String>) {
    let state = app.state::<AppState>();

    boot_step(&app, "config", "CONFIG", "run", "");
    match &cfg_warning {
        Some(w) => boot_step(&app, "config", "CONFIG", "warn", w),
        None => boot_step(
            &app,
            "config",
            "CONFIG",
            "ok",
            &config::config_path().display().to_string(),
        ),
    }

    boot_step(&app, "db", "CORPUS INDEX", "run", "");
    let db_path = lm_index::default_db_path();
    let conn = match lm_index::open_readonly(&db_path) {
        Ok(c) => c,
        Err(e) => return boot_fail(&app, "db", "CORPUS INDEX", format!("{e:#}")),
    };
    let chunks: i64 = conn
        .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
        .unwrap_or(0);
    let vectors = lm_index::vec_count(&conn).unwrap_or(0);
    boot_step(
        &app,
        "db",
        "CORPUS INDEX",
        "ok",
        &format!("{chunks} chunks, {vectors} vectors"),
    );
    *state.ui_db.lock().unwrap() = Some(conn);

    boot_step(&app, "vehicle", "VEHICLE", "run", "");
    let vehicle: lm_types::Vehicle = match std::fs::read_to_string(lm_pipeline::vehicle_path())
        .map_err(err)
        .and_then(|s| serde_json::from_str(&s).map_err(err))
    {
        Ok(v) => v,
        Err(e) => {
            return boot_fail(
                &app,
                "vehicle",
                "VEHICLE",
                format!("~/localmotive/vehicle.json: {e}"),
            );
        }
    };
    boot_step(
        &app,
        "vehicle",
        "VEHICLE",
        "ok",
        &format!(
            "{} {} {} {} · VIN {}",
            vehicle.year, vehicle.make, vehicle.model, vehicle.engine, vehicle.vin
        ),
    );

    boot_step(&app, "ollama", "OLLAMA", "run", "checking 127.0.0.1:11434");
    let client = match lm_ollama::Client::new() {
        Ok(c) => c,
        Err(e) => return boot_fail(&app, "ollama", "OLLAMA", e.to_string()),
    };
    let progress_app = app.clone();
    let say = move |m: &str| boot_step(&progress_app, "ollama", "OLLAMA", "run", m);
    let ollama = match supervisor::ensure(&client, &say).await {
        Ok(o) => o,
        Err(e) => return boot_fail(&app, "ollama", "OLLAMA", format!("{e:#}")),
    };
    let how = if ollama.owned() {
        "started by LocalMotive"
    } else {
        "system service"
    };
    boot_step(
        &app,
        "ollama",
        "OLLAMA",
        "ok",
        &format!("v{} ({how})", ollama.version()),
    );
    *state.ollama.lock().unwrap() = Some(ollama);

    boot_step(&app, "models", "MODELS", "run", "");
    match supervisor::check_models(&client).await {
        Ok(m) => boot_step(&app, "models", "MODELS", "ok", &m.join(" · ")),
        Err(e) => return boot_fail(&app, "models", "MODELS", format!("{e:#}")),
    }

    boot_step(&app, "pipeline", "PIPELINE", "run", "");
    let preset = PipelineConfig::preset(&cfg.answer.preset)
        .unwrap_or_else(|| PipelineConfig::preset("b").expect("preset b"));
    let model = preset.model.clone();
    let ctx = match Ctx::open(preset) {
        Ok(c) => c,
        Err(e) => return boot_fail(&app, "pipeline", "PIPELINE", format!("{e:#}")),
    };
    boot_step(
        &app,
        "pipeline",
        "PIPELINE",
        "ok",
        &format!("preset {} · {model}", ctx.cfg.name),
    );

    // Load the answer model and cache the fixed system prefix.
    boot_step(&app, "warmup", "ANSWER MODEL", "run", "loading into memory");
    let mut req = lm_ollama::ChatRequest::new(
        &model,
        vec![
            lm_ollama::Message::system(lm_pipeline::system_prompt(&ctx.vehicle)),
            lm_ollama::Message::user("ready"),
        ],
    );
    req.options = Some(lm_ollama::Options {
        num_ctx: Some(ctx.cfg.num_ctx),
        num_predict: Some(1),
        ..Default::default()
    });
    req.keep_alive = Some(ctx.cfg.keep_alive.clone());
    let t = std::time::Instant::now();
    match client.chat(&req).await {
        Ok(_) => boot_step(
            &app,
            "warmup",
            "ANSWER MODEL",
            "ok",
            &format!("loaded, prefix cached in {:.1}s", t.elapsed().as_secs_f64()),
        ),
        Err(e) => boot_step(
            &app,
            "warmup",
            "ANSWER MODEL",
            "warn",
            &format!("warm-up failed ({e}); first answer will be slower"),
        ),
    }
    *state.ctx.lock().unwrap() = Some(ctx);

    boot_step(&app, "telemetry", "TELEMETRY", "run", "");
    telemetry::spawn(app.clone());
    boot_step(
        &app,
        "telemetry",
        "TELEMETRY",
        "warn",
        "SIMULATED data (no OBD-II adapter yet)",
    );

    let mut b = state.boot.lock().unwrap();
    b.done = true;
    b.ok = true;
    b.model = model;
    b.vin = vehicle.vin.clone();
    b.chunks = chunks;
    b.vectors = vectors;
    let snapshot = b.clone();
    drop(b);
    let _ = app.emit("boot-done", snapshot);
}

// ------------------------------------------------------------------ commands

#[tauri::command]
fn boot_state(state: State<'_, AppState>) -> BootState {
    state.boot.lock().unwrap().clone()
}

#[derive(Serialize)]
struct SourceView {
    id: String,
    page_id: u32,
    title: String,
    breadcrumb: String,
    section_heading: String,
    text: String,
    images: Vec<String>,
    ocr: bool,
}

impl From<&Chunk> for SourceView {
    fn from(c: &Chunk) -> Self {
        Self {
            id: c.id.clone(),
            page_id: c.page_id,
            title: c.title.clone(),
            breadcrumb: c.breadcrumb.clone(),
            section_heading: c.section_heading.clone(),
            text: c.text.clone(),
            images: c.images.clone(),
            ocr: c.source == Source::Ocr,
        }
    }
}

#[derive(Serialize)]
struct AskResult {
    answer: Answer,
    verified: VerifiedFile,
    sources: Vec<SourceView>,
    timings: Timings,
    dir: String,
}

#[tauri::command]
async fn ask(app: AppHandle, question: String) -> CmdResult<AskResult> {
    let question = question.trim().to_string();
    if question.is_empty() {
        return Err("empty question".into());
    }
    tauri::async_runtime::spawn_blocking(move || -> CmdResult<AskResult> {
        let state = app.state::<AppState>();
        let guard = state.ctx.lock().map_err(err)?;
        let ctx = guard.as_ref().ok_or("still starting up")?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(err)?;
        let emitter = app.clone();
        let emit = move |p: Progress| {
            let _ = emitter.emit("progress", &p);
        };
        let turn = rt
            .block_on(run_turn_with(ctx, &state.session, &question, &emit))
            .map_err(|e| format!("{e:#}"))?;
        Ok(AskResult {
            sources: turn
                .retrieved
                .chunks
                .iter()
                .map(|r| SourceView::from(&r.chunk))
                .collect(),
            answer: turn.answer,
            verified: turn.verified,
            timings: turn.timings,
            dir: turn.dir.display().to_string(),
        })
    })
    .await
    .map_err(err)?
}

#[derive(Serialize)]
struct DtcCard {
    code: String,
    status: String,
    description: String,
    page_id: Option<u32>,
    fake: bool,
}

fn with_db<T>(
    state: &AppState,
    f: impl FnOnce(&rusqlite::Connection) -> anyhow::Result<T>,
) -> CmdResult<T> {
    let g = state.ui_db.lock().map_err(err)?;
    let conn = g.as_ref().ok_or("index not open yet")?;
    f(conn).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn dtcs(state: State<'_, AppState>) -> CmdResult<Vec<DtcCard>> {
    with_db(&state, |conn| {
        let mut out = Vec::new();
        for d in telemetry::FAKE_DTCS {
            let row: Option<(u32, String)> = conn
                .query_row(
                    "SELECT page_id, text FROM chunks WHERE title = ?1 AND kind = 'dtc' AND source = 'html' ORDER BY rid LIMIT 1",
                    [d.code],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ok();
            let description = row
                .as_ref()
                .and_then(|(_, text)| {
                    let needle = format!("DTC {} ", d.code);
                    text.lines()
                        .map(str::trim)
                        .find(|l| l.starts_with(&needle) && l.len() > needle.len() + 3)
                        .map(|l| l[needle.len()..].to_string())
                })
                .unwrap_or_default();
            out.push(DtcCard {
                code: d.code.into(),
                status: d.status.into(),
                description,
                page_id: row.map(|r| r.0),
                fake: d.fake,
            });
        }
        Ok(out)
    })
}

#[derive(Serialize)]
struct Procedure {
    code: String,
    title: String,
    breadcrumb: String,
    page_id: u32,
    sections: Vec<SourceView>,
}

#[tauri::command]
fn procedure(state: State<'_, AppState>, code: String) -> CmdResult<Procedure> {
    with_db(&state, |conn| {
        let page: u32 = conn.query_row(
            "SELECT page_id FROM chunks WHERE title = ?1 AND kind = 'dtc' ORDER BY rid LIMIT 1",
            [&code],
            |r| r.get(0),
        )?;
        let mut st = conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE page_id = ?1 AND hidden = 0 ORDER BY rid"
        ))?;
        let chunks: Vec<Chunk> = st
            .query_map([page], lm_index::row_to_chunk)?
            .map(|r| r.map(|(_, c)| c))
            .collect::<Result<_, _>>()?;
        let first = chunks
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no chunks for page {page}"))?;
        Ok(Procedure {
            code,
            title: first.title.clone(),
            breadcrumb: first.breadcrumb.clone(),
            page_id: page,
            sections: chunks.iter().map(SourceView::from).collect(),
        })
    })
}

#[tauri::command]
fn source(state: State<'_, AppState>, id: String) -> CmdResult<SourceView> {
    with_db(&state, |conn| {
        let c = lm_index::chunk_by_id(conn, &id)?
            .ok_or_else(|| anyhow::anyhow!("unknown chunk {id}"))?;
        Ok(SourceView::from(&c))
    })
}

#[tauri::command]
fn set_fullscreen(app: AppHandle, on: bool) -> CmdResult<()> {
    let w = app.get_webview_window("main").ok_or("no main window")?;
    w.set_fullscreen(on).map_err(err)
}

// ------------------------------------------------------------------ images

/// `lmimg://localhost/images/...` → a file under the manual's images dir only.
fn serve_image(images_root: &Path, uri_path: &str) -> Response<Vec<u8>> {
    let not_found = |code: StatusCode| Response::builder().status(code).body(Vec::new()).unwrap();
    let decoded = percent_encoding::percent_decode_str(uri_path.trim_start_matches('/'))
        .decode_utf8_lossy()
        .to_string();
    let Some(rel) = decoded.strip_prefix("images/") else {
        return not_found(StatusCode::FORBIDDEN);
    };
    if rel.split('/').any(|p| p == ".." || p.is_empty()) {
        return not_found(StatusCode::FORBIDDEN);
    }
    let Ok(path) = images_root.join(rel).canonicalize() else {
        return not_found(StatusCode::NOT_FOUND);
    };
    if !path.starts_with(images_root) {
        return not_found(StatusCode::FORBIDDEN);
    }
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        _ => return not_found(StatusCode::FORBIDDEN),
    };
    match std::fs::read(&path) {
        Ok(bytes) => Response::builder()
            .header("Content-Type", mime)
            .header("Cache-Control", "max-age=86400")
            .header("X-Content-Type-Options", "nosniff")
            .body(bytes)
            .unwrap(),
        Err(_) => not_found(StatusCode::NOT_FOUND),
    }
}

fn main() {
    let (cfg, cfg_warning) = config::load();
    let images_root = config::manual_root()
        .join("images")
        .canonicalize()
        .unwrap_or_else(|_| config::manual_root().join("images"));
    let session =
        lm_pipeline::sessions_root().join(format!("app-{}", lm_pipeline::new_session_id()));
    let state = AppState {
        boot: Mutex::new(BootState::default()),
        ui_db: Mutex::new(None),
        ctx: Mutex::new(None),
        ollama: Mutex::new(None),
        session,
    };
    let win_cfg = cfg.window.clone();
    let app = tauri::Builder::default()
        .manage(state)
        .register_uri_scheme_protocol("lmimg", move |_ctx, request| {
            serve_image(&images_root, request.uri().path())
        })
        .invoke_handler(tauri::generate_handler![
            boot_state,
            ask,
            dtcs,
            procedure,
            source,
            set_fullscreen
        ])
        .setup(move |app| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.set_size(tauri::LogicalSize::new(win_cfg.width, win_cfg.height));
                let _ = w.set_always_on_top(win_cfg.always_on_top);
                let _ = w.set_fullscreen(win_cfg.fullscreen);
            }
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(boot(handle, cfg, cfg_warning));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("building LocalMotive");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            let state = handle.state::<AppState>();
            if let Some(mut o) = state.ollama.lock().unwrap().take() {
                o.shutdown();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_protocol_is_confined_to_images_dir() {
        let root = config::manual_root().join("images");
        let Ok(root) = root.canonicalize() else {
            return;
        }; // manual not present on this machine
        let ok = serve_image(&root, "/images/DM13Q313/gm60/323435938.png");
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(ok.headers()["Content-Type"], "image/png");
        for bad in [
            "/images/../pages/1.html",
            "/pages/1.html",
            "/images/%2e%2e/README.txt",
            "/images//etc/passwd",
            "/images/DM13Q313/../../index.html",
        ] {
            assert_ne!(serve_image(&root, bad).status(), StatusCode::OK, "{bad}");
        }
    }
}
