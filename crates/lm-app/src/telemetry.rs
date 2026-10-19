//! FAKE telemetry until the OBD-II reader exists: 5 Hz samples shaped like a
//! warm-up with a lean condition (high LTFT) and occasional misfire stumbles,
//! so the dashboard and the fake DTCs (P0300, P0171) tell one story.

use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub t_ms: u64,
    pub rpm: f64,
    pub coolant_f: f64,
    pub voltage: f64,
    pub stft: f64,
    pub ltft: f64,
    pub fake: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FakeDtc {
    pub code: &'static str,
    pub status: &'static str,
    pub fake: bool,
}

pub const FAKE_DTCS: [FakeDtc; 3] = [
    FakeDtc {
        code: "P0300",
        status: "active",
        fake: true,
    },
    FakeDtc {
        code: "P0171",
        status: "active",
        fake: true,
    },
    FakeDtc {
        code: "P0442",
        status: "history",
        fake: true,
    },
];

pub struct Rng(pub u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn noise(&mut self, amp: f64) -> f64 {
        (self.next() * 2.0 - 1.0) * amp
    }
}

pub fn sample(t: f64, rng: &mut Rng) -> Sample {
    let warm = 1.0 - (-t / 90.0).exp(); // ~3 min warm-up
    let coolant_c = 20.0 + 72.0 * warm + rng.noise(0.3);
    let stumble = if rng.next() < 0.02 {
        -120.0 - rng.next() * 80.0
    } else {
        0.0
    };
    let rpm = 1150.0 - 430.0 * warm + 25.0 * (t * 1.7).sin() + rng.noise(18.0) + stumble;
    Sample {
        t_ms: (t * 1000.0) as u64,
        rpm: rpm.max(0.0),
        coolant_f: coolant_c * 9.0 / 5.0 + 32.0,
        voltage: 14.1 + 0.08 * (t * 0.3).sin() + rng.noise(0.04),
        stft: 3.0 * (t * 0.9).sin() + rng.noise(2.0),
        ltft: 11.5 + 0.8 * (t * 0.05).sin(),
        fake: true,
    }
}

/// Emit `telemetry` events at 5 Hz forever.
pub fn spawn(app: AppHandle) {
    std::thread::spawn(move || {
        let start = Instant::now();
        let mut rng = Rng(0x5eed_1234_abcd);
        loop {
            let s = sample(start.elapsed().as_secs_f64(), &mut rng);
            let _ = app.emit("telemetry", s);
            std::thread::sleep(Duration::from_millis(200));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warms_up_and_stays_lean() {
        let mut rng = Rng(1);
        let cold = sample(0.0, &mut rng);
        let warm = sample(600.0, &mut rng);
        assert!(cold.coolant_f < 80.0 && warm.coolant_f > 190.0);
        assert!(warm.ltft > 10.0);
        assert!(warm.rpm > 500.0 && warm.rpm < 900.0);
    }
}
