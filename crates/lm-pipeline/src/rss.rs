//! Peak resident memory of processes by name, sampled from /proc (Linux).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// Process names that make up a running Ollama: the server and its model runner.
pub const OLLAMA_PROCS: &[&str] = &["ollama", "llama-server"];

/// Current total RSS in KiB of processes whose `comm` starts with any of `names`.
pub fn total_rss_kib(names: &[&str]) -> u64 {
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return 0;
    };
    let mut total = 0;
    for e in dir.flatten() {
        let p = e.path();
        if !p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        let Ok(comm) = std::fs::read_to_string(p.join("comm")) else {
            continue;
        };
        if !names.iter().any(|n| comm.trim().starts_with(n)) {
            continue;
        }
        let Ok(status) = std::fs::read_to_string(p.join("status")) else {
            continue;
        };
        if let Some(kb) = status
            .lines()
            .find_map(|l| l.strip_prefix("VmRSS:"))
            .and_then(|v| v.split_whitespace().next())
            .and_then(|v| v.parse::<u64>().ok())
        {
            total += kb;
        }
    }
    total
}

/// Samples every 100 ms on a background thread until stopped.
pub struct Sampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl Sampler {
    pub fn start(names: &'static [&'static str]) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(total_rss_kib(names)));
        let (s, p) = (stop.clone(), peak.clone());
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(total_rss_kib(names), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    /// Stop sampling; returns the peak in MiB.
    pub fn stop(mut self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peak.load(Ordering::Relaxed) as f64 / 1024.0
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sees_own_process() {
        // the test binary's comm is a prefix of its name
        let comm = std::fs::read_to_string("/proc/self/comm").unwrap();
        assert!(super::total_rss_kib(&[comm.trim()]) > 0);
    }
}
