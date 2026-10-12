// lightning.rs
// Lightning (spec docs/superpowers/specs/2026-10-12-lightning-design.md), the CPU side: a deterministic
// strike schedule – a strike is a hash of (spatial cell, time slot) gated by the storm value and the
// planet type, a pure function of the planet-frame position and the shared clock – the strikes active
// near a camera, their flash envelope, and the values for the GPU. atmosphere.wgsl adds the cloud glow
// and the flash; bolt.rs builds the bolt geometry.

use glam::Vec3;

use super::{smoothstep, storm, WeatherForce, WeatherKind};

pub const SLOT_SECS: f32 = 1.5; // 3600 / 1.5 = 2400 slots: the clock wrap falls on a slot edge
pub const CELL_SIZE: f32 = 400.0;
pub const RANGE: f32 = 3000.0; // the voxel frame's strike distance
pub const MAX_STRIKES: usize = 6;
pub const LIFE: f32 = 0.65; // seconds a strike lasts (main flash plus re-strikes)
pub const FLASH_RANGE: f32 = 600.0; // distance at which a strike's ambient flash is a quarter
const RATE_FULL: f32 = 0.25; // strikes per cell-slot at storm 1 (about one per 6 s per cell)
const STORM_MIN: f32 = 0.5;
const RISE: f32 = 0.03;
const DECAY: f32 = 0.07;
const WRAP: f32 = 3600.0;

// the glow and flash constants shared with atmosphere.wgsl (name there, value)
pub const GLOW_RADIUS: f32 = 0.35; // cloud glow radius, in planet radii at the cloud shell
pub const CONSTANTS: [(&str, f32); 1] = [("LIGHTNING_GLOW_RADIUS", GLOW_RADIUS)];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strike {
    pub pos: Vec3,      // on the planet's surface sphere, planet frame
    pub age: f32,       // seconds since it started, 0..=LIFE
    pub intensity: f32, // the envelope now, 0..1
    pub seed: u32,      // the cell/slot hash: seeds the bolt's shape
}

pub const USAGE: &str = "Usage: /weather lightning [on|off|rate <0-3>]";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightningSettings {
    pub on: bool,
    pub rate: f32,
}

impl Default for LightningSettings {
    fn default() -> Self {
        Self {
            on: true,
            rate: 1.0,
        }
    }
}

// how often a planet type's storms strike, relative to rain storms
pub fn kind_rate(kind: WeatherKind) -> f32 {
    match kind {
        WeatherKind::Rain | WeatherKind::AcidRain => 1.0,
        WeatherKind::Ash => 0.4,
        WeatherKind::Dust => 0.1,
        WeatherKind::Snow | WeatherKind::None => 0.0,
    }
}

fn pcg(mut h: u32) -> u32 {
    h = h.wrapping_mul(747796405).wrapping_add(2891336453);
    h = ((h >> ((h >> 28) + 4)) ^ h).wrapping_mul(277803737);
    (h >> 22) ^ h
}

fn hash(ix: i32, iy: i32, iz: i32, slot: u32, salt: u32) -> u32 {
    let mut h = pcg(ix as u32 ^ 0x68bc21eb);
    h = pcg(h ^ iy as u32);
    h = pcg(h ^ iz as u32);
    h = pcg(h ^ slot);
    pcg(h ^ salt)
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

fn pulse(age: f32) -> f32 {
    if age < 0.0 {
        0.0
    } else if age < RISE {
        age / RISE
    } else {
        (-(age - RISE) / DECAY).exp()
    }
}

// the flash over a strike's life: the main stroke, then a dip and one or two weaker re-strikes, the timing
// from the seed
pub fn envelope(age: f32, seed: u32) -> f32 {
    let r1 = 0.14 + 0.08 * unit(pcg(seed ^ 0x9e37_79b9));
    let r2 = r1 + 0.08 + 0.06 * unit(pcg(seed ^ 0x85eb_ca6b));
    (pulse(age) + 0.5 * pulse(age - r1) + 0.3 * pulse(age - r2)).min(1.0)
}

// seconds from `start` to `t` across the clock wrap
fn age_since(t: f32, start: f32) -> f32 {
    let a = t - start;
    if a < -(SLOT_SECS * 2.0) {
        a + WRAP
    } else {
        a
    }
}

// The strikes alive at clock `t` (wrapped at 3600) seen from `cam` (planet frame): all cells within `range`
// of the camera (`f32::MAX`: the whole planet), strongest first, at most MAX_STRIKES. `rate` is the
// /weather lightning setting.
pub fn active_strikes(
    kind: WeatherKind,
    rate: f32,
    planet_radius: f32,
    cam: Vec3,
    range: f32,
    t: f32,
    force: &WeatherForce,
) -> Vec<Strike> {
    let per_slot = RATE_FULL * kind_rate(kind) * rate;
    if per_slot <= 0.0 || planet_radius <= 0.0 {
        return vec![];
    }
    let t = t.rem_euclid(WRAP);
    let slot = (t / SLOT_SECS).floor() as u32;
    let slot_start = slot as f32 * SLOT_SECS;
    // the cells that can hold surface points: the planet's cube, cut by the camera's range
    let reach = planet_radius + CELL_SIZE;
    let lo = (cam - Vec3::splat(range)).max(Vec3::splat(-reach));
    let hi = (cam + Vec3::splat(range)).min(Vec3::splat(reach));
    if lo.cmpgt(hi).any() {
        return vec![];
    }
    let cell = |v: f32| (v / CELL_SIZE).floor() as i32;
    let mut out = Vec::new();
    for ix in cell(lo.x)..=cell(hi.x) {
        for iy in cell(lo.y)..=cell(hi.y) {
            for iz in cell(lo.z)..=cell(hi.z) {
                // the cell's jittered point, independent of the slot; only cells whose point lies in the
                // shell around the surface hold strikes
                let j = Vec3::new(
                    unit(hash(ix, iy, iz, 0, 1)),
                    unit(hash(ix, iy, iz, 0, 2)),
                    unit(hash(ix, iy, iz, 0, 3)),
                );
                let p = (Vec3::new(ix as f32, iy as f32, iz as f32) + j) * CELL_SIZE;
                if (p.length() - planet_radius).abs() > CELL_SIZE * 0.5 || p.length() < 1e-3 {
                    continue;
                }
                let dir = p.normalize();
                let s = storm(dir, slot_start, force);
                let chance = per_slot * smoothstep(STORM_MIN, 1.0, s);
                let h = hash(ix, iy, iz, slot, 4);
                if unit(h) >= chance {
                    continue;
                }
                let start = slot_start + unit(pcg(h ^ 5)) * (SLOT_SECS - LIFE);
                let age = age_since(t, start);
                if !(0.0..=LIFE).contains(&age) {
                    continue;
                }
                let seed = pcg(h ^ 6);
                let intensity = envelope(age, seed);
                if intensity < 0.01 {
                    continue;
                }
                let pos = dir * planet_radius;
                if pos.distance(cam) > range && range < f32::MAX {
                    continue;
                }
                out.push(Strike {
                    pos,
                    age,
                    intensity,
                    seed,
                });
            }
        }
    }
    out.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    out.truncate(MAX_STRIKES);
    out
}

// the scene flash at `cam`: each strike's intensity, falling off with distance
pub fn flash_at(strikes: &[Strike], cam: Vec3) -> f32 {
    strikes
        .iter()
        .map(|s| {
            let d = s.pos.distance(cam) / FLASH_RANGE;
            s.intensity / (1.0 + d * d)
        })
        .sum::<f32>()
        .min(1.0)
}

// Global.strikes / PlanetUniform.strikes: xyz position, w intensity
pub fn gpu_strikes(strikes: &[Strike]) -> [[f32; 4]; MAX_STRIKES] {
    let mut out = [[0.0; 4]; MAX_STRIKES];
    for (o, s) in out.iter_mut().zip(strikes) {
        *o = [s.pos.x, s.pos.y, s.pos.z, s.intensity];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biome::PlanetType;
    use crate::weather::{WeatherForce, WeatherSettings};

    const R: f32 = 250.0;

    fn kind(t: PlanetType) -> WeatherKind {
        WeatherKind::for_planet(t)
    }

    // a force that makes the whole planet one storm of value 1
    fn full_storm() -> WeatherForce {
        let mut s = WeatherSettings::default();
        s.apply(
            crate::weather::WeatherCommand::Force(crate::weather::Force::Storm),
            WeatherKind::Rain,
            Vec3::Y,
            0.0,
        );
        s.tick(1000.0);
        s.force_uniform()
    }

    fn clear_sky() -> WeatherForce {
        let mut s = WeatherSettings::default();
        s.apply(
            crate::weather::WeatherCommand::Force(crate::weather::Force::Clear),
            WeatherKind::Rain,
            Vec3::Y,
            0.0,
        );
        s.force_uniform()
    }

    // every strike over an hour of slots, as (slot time, strike) – used to count and compare
    fn hour(kind: WeatherKind, force: &WeatherForce) -> Vec<(f32, Strike)> {
        let mut all = vec![];
        let mut t = 0.0;
        while t < 3600.0 {
            // sample mid-life of the slot's strikes: ask for all cells at several ages
            for s in active_strikes(kind, 1.0, R, Vec3::new(0.0, R, 0.0), f32::MAX, t, force) {
                all.push((t, s));
            }
            t += 0.25;
        }
        all
    }

    #[test]
    fn wgsl_constants_match_the_rust_mirror() {
        let w = include_str!("../render/atmosphere.wgsl");
        for (name, v) in CONSTANTS {
            let line = w
                .lines()
                .find(|l| l.starts_with(&format!("const {name}: f32 = ")))
                .unwrap_or_else(|| panic!("{name} missing"));
            let value: f32 = line
                .split('=')
                .nth(1)
                .unwrap()
                .split(';')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            assert_eq!(value, v, "{name}");
        }
        assert!(
            w.contains("array<vec4<f32>, 6>"),
            "MAX_STRIKES is 6 in the WGSL arrays"
        );
        assert_eq!(MAX_STRIKES, 6);
    }

    #[test]
    fn the_schedule_is_deterministic() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let a = active_strikes(k, 1.0, R, Vec3::new(0.0, R, 0.0), f32::MAX, 123.4, &f);
        let b = active_strikes(k, 1.0, R, Vec3::new(0.0, R, 0.0), f32::MAX, 123.4, &f);
        assert_eq!(a, b);
    }

    #[test]
    fn a_full_storm_strikes_at_roughly_the_planned_rate() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        // distinct strikes over the hour, counted by (position, seed)
        let mut seen = std::collections::HashSet::new();
        for (t, s) in hour(k, &f) {
            seen.insert((s.seed, (t - s.age).round() as i64));
        }
        // ~ cells in the shell × 2400 slots × 0.25; allow a wide band, it only guards the order of magnitude
        assert!(seen.len() > 1000 && seen.len() < 200_000, "{}", seen.len());
    }

    #[test]
    fn no_storm_no_lightning() {
        let k = kind(PlanetType::EarthLike);
        // the force covers a region of about 60 degrees around +Y (elsewhere the natural storms go on), so look
        // only at the neighbourhood of the camera there, and after the force has faded in
        let mut t = 10.0;
        while t < 600.0 {
            assert!(active_strikes(k, 1.0, R, Vec3::Y * R, 300.0, t, &clear_sky()).is_empty());
            t += 0.5;
        }
    }

    #[test]
    fn weather_off_means_no_lightning() {
        let mut s = WeatherSettings::default();
        s.on = false;
        let f = s.force_uniform();
        let k = kind(PlanetType::EarthLike);
        assert!(hour(k, &f).is_empty());
    }

    #[test]
    fn ice_and_shattered_never_strike_desert_rarely() {
        let f = full_storm();
        assert!(hour(kind(PlanetType::Ice), &f).is_empty());
        assert!(hour(kind(PlanetType::Shattered), &f).is_empty());
        let wet = hour(kind(PlanetType::EarthLike), &f).len();
        let dry = hour(kind(PlanetType::Desert), &f).len();
        assert!(dry > 0, "a forced storm still thunders on a desert");
        assert!(dry * 4 < wet, "{dry} vs {wet}");
        assert_eq!(kind_rate(kind(PlanetType::Ice)), 0.0);
    }

    #[test]
    fn the_rate_setting_scales_the_strikes() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let count = |rate: f32| {
            let mut n = 0;
            let mut t = 0.0;
            while t < 1200.0 {
                n += active_strikes(k, rate, R, Vec3::Y * R, f32::MAX, t, &f).len();
                t += 0.25;
            }
            n
        };
        assert_eq!(count(0.0), 0);
        assert!(count(2.0) > count(1.0));
    }

    #[test]
    fn strikes_lie_on_the_planet_surface_and_age_within_a_life() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        for (_, s) in hour(k, &f).into_iter().take(500) {
            assert!((s.pos.length() - R).abs() < 1e-2, "{}", s.pos.length());
            assert!((0.0..=LIFE).contains(&s.age), "{}", s.age);
            assert!(s.intensity > 0.0 && s.intensity <= 1.0);
        }
    }

    #[test]
    fn the_active_set_is_capped_and_strongest_first() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let mut t = 0.0;
        while t < 300.0 {
            let v = active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, t, &f);
            assert!(v.len() <= MAX_STRIKES);
            let score = |s: &Strike| s.intensity;
            for w in v.windows(2) {
                assert!(score(&w[0]) >= score(&w[1]) - 1e-6);
            }
            t += 0.25;
        }
    }

    #[test]
    fn the_range_limits_the_strikes_to_the_neighbourhood() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let cam = Vec3::new(0.0, R, 0.0);
        let mut t = 0.0;
        while t < 300.0 {
            for s in active_strikes(k, 3.0, R, cam, 800.0, t, &f) {
                assert!(
                    s.pos.distance(cam) <= 800.0 + CELL_SIZE,
                    "{}",
                    s.pos.distance(cam)
                );
            }
            t += 0.25;
        }
    }

    #[test]
    fn the_schedule_is_continuous_across_the_clock_wrap() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        // a strike that starts shortly before the wrap is still alive just after it, at the right age
        let before = active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, 3599.8, &f);
        let after = active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, 0.1, &f);
        for b in &before {
            if let Some(a) = after.iter().find(|a| a.seed == b.seed) {
                assert!(
                    (a.age - (b.age + 0.3)).abs() < 1e-3,
                    "{} vs {}",
                    a.age,
                    b.age
                );
            }
        }
        // strikes never outlive their slot, so across the wrap itself there are no pairs by construction;
        // the same age arithmetic is checked within slots (10.0 and 10.1 share one), and the wrap maps t
        // = 3600 + x onto x
        let mut pairs = 0;
        for i in 0..200 {
            let t0 = 10.0 + i as f32 * 1.5;
            let a = active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, t0, &f);
            let b = active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, t0 + 0.1, &f);
            for x in &a {
                if let Some(y) = b.iter().find(|y| y.seed == x.seed) {
                    pairs += 1;
                    assert!(
                        (y.age - (x.age + 0.1)).abs() < 1e-3,
                        "{} vs {}",
                        y.age,
                        x.age
                    );
                }
            }
        }
        assert!(pairs > 0);
        assert_eq!(
            active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, 3600.1, &f),
            active_strikes(k, 3.0, R, Vec3::Y * R, f32::MAX, 0.1, &f)
        );
    }

    #[test]
    fn the_envelope_flashes_dips_and_dies() {
        let seed = 12345;
        assert_eq!(envelope(-0.1, seed), 0.0);
        let peak = (0..60)
            .map(|i| envelope(i as f32 * 0.005, seed))
            .fold(0.0, f32::max);
        assert!(peak > 0.9 && peak <= 1.0, "{peak}");
        assert!(envelope(LIFE, seed) < 0.02);
        // dips between the main flash and a re-strike
        let dip = (0..40)
            .map(|i| envelope(0.1 + i as f32 * 0.003, seed))
            .fold(1.0, f32::min);
        assert!(dip < 0.5 * peak);
    }

    #[test]
    fn flash_falls_off_with_distance() {
        let at = |d: f32| {
            let s = Strike {
                pos: Vec3::new(d, 0.0, 0.0),
                age: 0.05,
                intensity: 1.0,
                seed: 1,
            };
            flash_at(&[s], Vec3::ZERO)
        };
        assert!(at(10.0) > at(600.0));
        assert!(at(600.0) > at(5000.0));
        assert!(at(0.0) <= 1.0 && at(0.0) > 0.9);
        assert_eq!(flash_at(&[], Vec3::ZERO), 0.0);
    }

    #[test]
    fn gpu_packing_pads_with_zeros() {
        let s = Strike {
            pos: Vec3::new(1.0, 2.0, 3.0),
            age: 0.0,
            intensity: 0.5,
            seed: 0,
        };
        let g = gpu_strikes(&[s]);
        assert_eq!(g[0], [1.0, 2.0, 3.0, 0.5]);
        assert_eq!(g[1], [0.0; 4]);
    }

    #[test]
    fn a_camera_at_the_strike_or_the_centre_gives_finite_values() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        for cam in [Vec3::ZERO, Vec3::Y * R] {
            for s in active_strikes(k, 3.0, R, cam, f32::MAX, 10.0, &f) {
                assert!(s.pos.is_finite() && s.intensity.is_finite());
            }
            assert!(flash_at(&active_strikes(k, 3.0, R, cam, f32::MAX, 10.0, &f), cam).is_finite());
        }
    }
}
