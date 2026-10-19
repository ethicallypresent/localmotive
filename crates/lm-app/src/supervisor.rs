//! Ollama supervisor: use a running server, or start `ollama serve` bound to
//! loopback with conservative settings, and kill it only if we started it.

use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

pub const HOST: &str = "127.0.0.1:11434";
pub const REQUIRED_MODELS: [&str; 3] = [
    lm_pipeline::ANSWER_MODEL,
    lm_pipeline::SMALL_MODEL,
    lm_index::EMBED_MODEL,
];

/// How Ollama is being provided.
pub enum Ollama {
    /// Already answering (e.g. the system ollama.service).
    External { version: String },
    /// Started by us; killed on exit (and by PR_SET_PDEATHSIG if we crash).
    Owned { version: String, child: Child },
}

impl Ollama {
    pub fn version(&self) -> &str {
        match self {
            Ollama::External { version } | Ollama::Owned { version, .. } => version,
        }
    }

    pub fn owned(&self) -> bool {
        matches!(self, Ollama::Owned { .. })
    }

    /// Kill the server only if this process started it.
    pub fn shutdown(&mut self) {
        if let Ollama::Owned { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn spawn_serve() -> Result<Child> {
    let log_dir = crate::config::home().join("localmotive/logs");
    std::fs::create_dir_all(&log_dir)?;
    let log = std::fs::File::create(log_dir.join("ollama-serve.log"))?;
    let mut cmd = Command::new("ollama");
    cmd.arg("serve")
        .env("OLLAMA_HOST", HOST)
        .env("OLLAMA_MAX_LOADED_MODELS", "1")
        .env("OLLAMA_NUM_PARALLEL", "1")
        .env("OLLAMA_FLASH_ATTENTION", "1")
        .env("OLLAMA_KV_CACHE_TYPE", "q8_0")
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // SAFETY: prctl is async-signal-safe and runs in the forked child before exec.
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    cmd.spawn()
        .context("starting `ollama serve` (is ollama on PATH?)")
}

/// Use or start Ollama. `step` reports human-readable progress.
pub async fn ensure(client: &lm_ollama::Client, step: &(dyn Fn(&str) + Sync)) -> Result<Ollama> {
    if let Ok(version) = client.version().await {
        step(&format!("Ollama {version} already running on {HOST}"));
        return Ok(Ollama::External { version });
    }
    step(&format!("no Ollama on {HOST}; starting `ollama serve`"));
    let mut child = spawn_serve()?;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(15) {
        if let Some(status) = child.try_wait()? {
            bail!(
                "`ollama serve` exited early ({status}); see ~/localmotive/logs/ollama-serve.log"
            );
        }
        if let Ok(version) = client.version().await {
            step(&format!(
                "Ollama {version} up after {:.1}s",
                start.elapsed().as_secs_f64()
            ));
            return Ok(Ollama::Owned { version, child });
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let _ = child.kill();
    bail!("Ollama did not answer within 15 s; see ~/localmotive/logs/ollama-serve.log")
}

/// Confirm the required models are pulled (never pulls).
pub async fn check_models(client: &lm_ollama::Client) -> Result<Vec<String>> {
    let tags: Vec<String> = client.tags().await?.into_iter().map(|m| m.name).collect();
    let missing: Vec<&str> = REQUIRED_MODELS
        .iter()
        .copied()
        .filter(|m| !tags.iter().any(|t| t == m))
        .collect();
    if !missing.is_empty() {
        bail!(
            "missing models: {}. Pull them with: {}",
            missing.join(", "),
            missing
                .iter()
                .map(|m| format!("ollama pull {m}"))
                .collect::<Vec<_>>()
                .join(" && ")
        );
    }
    Ok(REQUIRED_MODELS.iter().map(|s| s.to_string()).collect())
}
