//! `~/localmotive/config.toml` — written with defaults on first start.

use std::path::PathBuf;

use serde::Deserialize;

pub const DEFAULT_CONFIG: &str = r#"# LocalMotive configuration
[window]
# Kiosk mode for the in-car screen. Esc leaves fullscreen.
fullscreen = false
always_on_top = false
width = 1280
height = 800

[answer]
# Pipeline preset: "b" (4B, 4 trimmed chunks) or "c" (1.7B, same context).
preset = "b"
"#;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WindowCfg {
    pub fullscreen: bool,
    pub always_on_top: bool,
    pub width: f64,
    pub height: f64,
}

impl Default for WindowCfg {
    fn default() -> Self {
        Self {
            fullscreen: false,
            always_on_top: false,
            width: 1280.0,
            height: 800.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AnswerCfg {
    pub preset: String,
}

impl Default for AnswerCfg {
    fn default() -> Self {
        Self { preset: "b".into() }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub window: WindowCfg,
    pub answer: AnswerCfg,
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
}

pub fn config_path() -> PathBuf {
    home().join("localmotive/config.toml")
}

pub fn manual_root() -> PathBuf {
    home().join("localmotive/corpus/2006 Chevrolet Uplander FWD V6-3.5L VIN L")
}

/// Load the config, writing the default file if there is none. A broken
/// file falls back to defaults (and says so) rather than blocking startup.
pub fn load() -> (Config, Option<String>) {
    let path = config_path();
    if !path.exists() {
        let _ = std::fs::write(&path, DEFAULT_CONFIG);
    }
    match std::fs::read_to_string(&path).map(|s| toml::from_str::<Config>(&s)) {
        Ok(Ok(c)) => (c, None),
        Ok(Err(e)) => (
            Config::default(),
            Some(format!("{}: {e}; using defaults", path.display())),
        ),
        Err(e) => (
            Config::default(),
            Some(format!("{}: {e}; using defaults", path.display())),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_text_parses() {
        let c: Config = toml::from_str(DEFAULT_CONFIG).unwrap();
        assert!(!c.window.fullscreen);
        assert_eq!(c.answer.preset, "b");
        let partial: Config = toml::from_str("[window]\nfullscreen = true\n").unwrap();
        assert!(partial.window.fullscreen);
        assert_eq!(partial.window.width, 1280.0);
    }
}
