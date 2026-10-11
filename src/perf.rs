// perf.rs
// Performance measurement without screenshots: a recording of every frame's CPU frame time and GPU time
// per pass (gpu_timer.rs) over a few seconds, summarised as one JSON line appended to perf.log
// (`/perf <seconds> [label]`, and `--bench`, which records from a fixed, frozen pose and quits). Median,
// mean, 95th percentile and maximum per series, so hitches show instead of vanishing into an average.

use std::path::PathBuf;
use std::time::Instant;

pub const LOG_PATH: &str = "perf.log";
pub const MAX_SECONDS: f32 = 120.0;
pub const PERF_USAGE: &str = "Usage: /perf <seconds 1-120> [label]";

// the GPU series, in gpu_timer::PARTS order, plus their sum
const GPU_KEYS: [&str; 5] = ["geometry", "rays", "blur", "lighting", "text"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub median: f32,
    pub mean: f32,
    pub p95: f32,
    pub max: f32,
}

// median, mean, 95th percentile (nearest rank) and maximum; None for no samples
pub fn stats(samples: &[f32]) -> Option<Stats> {
    if samples.is_empty() {
        return None;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let median = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    };
    let rank = ((0.95 * n as f32).ceil() as usize).clamp(1, n);
    Some(Stats {
        median,
        mean: v.iter().sum::<f32>() / n as f32,
        p95: v[rank - 1],
        max: v[n - 1],
    })
}

// what a report says about the run besides the numbers
#[derive(Clone, Debug, Default)]
pub struct PerfContext {
    pub width: u32,
    pub height: u32,
    pub mode: String,
    pub settings: Vec<(String, String)>, // key, value (already formatted, written as JSON strings)
}

pub struct PerfRecorder {
    pub label: String,
    seconds: f32,
    start: Instant,
    last_frame: Option<Instant>,
    frame_ms: Vec<f32>,
    gpu_ms: [Vec<f32>; 5],
    gpu_total_ms: Vec<f32>,
}

impl PerfRecorder {
    pub fn new(seconds: f32, label: &str) -> Self {
        Self {
            label: label.to_string(),
            seconds: seconds.clamp(0.1, MAX_SECONDS),
            start: Instant::now(),
            last_frame: None,
            frame_ms: Vec::new(),
            gpu_ms: Default::default(),
            gpu_total_ms: Vec::new(),
        }
    }

    // once per rendered frame, with that frame's GPU times if a readback arrived; true when done
    pub fn frame(&mut self, gpu: Option<[f32; 5]>) -> bool {
        let now = Instant::now();
        if let Some(last) = self.last_frame {
            self.frame_ms.push((now - last).as_secs_f32() * 1000.0);
        }
        self.last_frame = Some(now);
        if let Some(g) = gpu {
            for (series, ms) in self.gpu_ms.iter_mut().zip(g) {
                series.push(ms);
            }
            self.gpu_total_ms.push(g.iter().sum());
        }
        self.start.elapsed().as_secs_f32() >= self.seconds
    }

    // the report as one JSON line (no trailing newline)
    pub fn report(&self, ctx: &PerfContext, time_unix: u64, commit: &str) -> String {
        let mut out = format!(
            "{{\"label\":{},\"time\":{time_unix},\"commit\":{},\"width\":{},\"height\":{},\"mode\":{},\"seconds\":{:.2},\"frames\":{}",
            json_str(&self.label),
            json_str(commit),
            ctx.width,
            ctx.height,
            json_str(&ctx.mode),
            self.start.elapsed().as_secs_f32(),
            self.frame_ms.len()
        );
        out.push_str(",\"settings\":{");
        out.push_str(
            &ctx.settings
                .iter()
                .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
                .collect::<Vec<_>>()
                .join(","),
        );
        out.push('}');
        let series = std::iter::once(("frame", &self.frame_ms))
            .chain(GPU_KEYS.iter().copied().zip(self.gpu_ms.iter()))
            .chain(std::iter::once(("gpu_total", &self.gpu_total_ms)));
        for (key, samples) in series {
            match stats(samples) {
                Some(s) => out.push_str(&format!(
                    ",\"{key}\":{{\"median\":{:.3},\"mean\":{:.3},\"p95\":{:.3},\"max\":{:.3}}}",
                    s.median, s.mean, s.p95, s.max
                )),
                None => out.push_str(&format!(",\"{key}\":null")),
            }
        }
        out.push('}');
        out
    }

    // a one-line console summary
    pub fn summary(&self) -> String {
        let f = stats(&self.frame_ms);
        let g = stats(&self.gpu_total_ms);
        format!(
            "Perf '{}': frame {} ms, GPU {} ms (median, {} frames) -> {LOG_PATH}",
            self.label,
            f.map_or("-".into(), |s| format!("{:.2}", s.median)),
            g.map_or("-".into(), |s| format!("{:.2}", s.median)),
            self.frame_ms.len()
        )
    }
}

// a JSON string literal
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// the current commit, or "unknown" outside a git checkout
pub fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

// appends one line to perf.log
pub fn append_log(line: &str) -> std::io::Result<PathBuf> {
    use std::io::Write;
    let path = PathBuf::from(LOG_PATH);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(f, "{line}")?;
    Ok(path)
}

// `/perf <seconds> [label...]`
pub fn parse_command(args: &[&str]) -> Result<(f32, String), &'static str> {
    let (secs, label) = args.split_first().ok_or(PERF_USAGE)?;
    let secs: f32 = secs.parse().map_err(|_| PERF_USAGE)?;
    if !(1.0..=MAX_SECONDS).contains(&secs) {
        return Err(PERF_USAGE);
    }
    let label = if label.is_empty() {
        "perf".to_string()
    } else {
        label.join(" ")
    };
    Ok((secs, label))
}

// --bench <seconds> [--label <text>] [--exec "<cmd>;<cmd>"]
#[derive(Clone, Debug, PartialEq)]
pub struct BenchArgs {
    pub seconds: f32,
    pub label: String,
    pub exec: Vec<String>,
}

pub fn parse_bench_args(args: &[String]) -> Option<BenchArgs> {
    let value = |flag: &str| args.iter().skip_while(|a| *a != flag).nth(1).cloned();
    let seconds: f32 = value("--bench")?.parse().ok()?;
    Some(BenchArgs {
        seconds: seconds.clamp(1.0, MAX_SECONDS),
        label: value("--label").unwrap_or_else(|| "bench".into()),
        exec: value("--exec")
            .map(|e| {
                e.split(';')
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_are_robust() {
        assert_eq!(stats(&[]), None);
        let s = stats(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!((s.median, s.mean, s.max), (2.0, 2.0, 3.0));
        assert_eq!(stats(&[1.0, 2.0, 3.0, 4.0]).unwrap().median, 2.5);
        // 100 samples 1..=100: p95 is the 95th smallest
        let v: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        assert_eq!(stats(&v).unwrap().p95, 95.0);
        // a single hitch shows in max, not in the median
        let mut v = vec![10.0; 99];
        v.push(80.0);
        let s = stats(&v).unwrap();
        assert_eq!((s.median, s.max), (10.0, 80.0));
    }

    #[test]
    fn the_report_is_one_json_line() {
        let mut r = PerfRecorder::new(10.0, "storm \"A\"");
        r.frame(Some([1.0, 2.0, 0.5, 1.5, 0.1]));
        r.frame(Some([1.2, 2.0, 0.5, 1.5, 0.1]));
        r.frame(None);
        let ctx = PerfContext {
            width: 1920,
            height: 1200,
            mode: "planet".into(),
            settings: vec![("weather".into(), "storm".into())],
        };
        let line = r.report(&ctx, 1_700_000_000, "abc1234");
        assert!(!line.contains('\n'));
        assert!(line.starts_with(
            "{\"label\":\"storm \\\"A\\\"\",\"time\":1700000000,\"commit\":\"abc1234\""
        ));
        assert!(line.contains("\"width\":1920,\"height\":1200,\"mode\":\"planet\""));
        assert!(line.contains("\"settings\":{\"weather\":\"storm\"}"));
        assert!(line.contains("\"frames\":2"));
        assert!(line.contains("\"geometry\":{\"median\":1.100,"));
        assert!(line.contains("\"gpu_total\":{\"median\":5.200,"));
        assert!(line.ends_with('}'));
        // braces balance
        let open = line.matches('{').count();
        assert_eq!(open, line.matches('}').count());
    }

    #[test]
    fn commands_and_flags_parse() {
        assert_eq!(parse_command(&["10"]), Ok((10.0, "perf".into())));
        assert_eq!(
            parse_command(&["5", "rain", "on"]),
            Ok((5.0, "rain on".into()))
        );
        assert_eq!(parse_command(&[]), Err(PERF_USAGE));
        assert_eq!(parse_command(&["0.5"]), Err(PERF_USAGE));
        assert_eq!(parse_command(&["x"]), Err(PERF_USAGE));
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_bench_args(&a(&["voxanet", "--windowed"])), None);
        assert_eq!(
            parse_bench_args(&a(&[
                "voxanet",
                "--bench",
                "8",
                "--label",
                "off",
                "--exec",
                "/weather off; /shadow off"
            ])),
            Some(BenchArgs {
                seconds: 8.0,
                label: "off".into(),
                exec: vec!["/weather off".into(), "/shadow off".into()]
            })
        );
    }
}
