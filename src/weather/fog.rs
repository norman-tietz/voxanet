// fog.rs
// Storm and mist fog (spec docs/superpowers/specs/2026-10-12-fog-storms-design.md), the CPU side: the
// per-type storm extinction and tint, the ground mist field, the wind offset the structure scrolls with,
// the /weather fog settings, and the uniform values for the GPU. The fog maths themselves are in
// atmosphere.wgsl (atmo_air_fog); CONSTANTS and the closed forms here mirror it.

use glam::Vec3;

use super::{noise, rotate, smoothstep, WeatherKind};

#[cfg(test)]
pub const STORM_LO: f32 = 0.25; // storm value below which the fog does not start (FOG_STORM_LO)
#[cfg(test)]
pub const STRUCTURE: f32 = 0.6; // the storm density swings by this share around its mean (FOG_STRUCTURE)
#[cfg(test)]
pub const STRUCT_SCALE: f32 = 0.02; // structure noise cells per unit, ~50 units each (FOG_STRUCT_SCALE)
#[cfg(test)]
pub const STRUCT_MAX_DIST: f32 = 300.0; // the structure taps lie within this of the camera (FOG_STRUCT_MAX_DIST)
#[cfg(test)]
pub const MIST_HEIGHT: f32 = 10.0; // e-folding height of the mist, blocks (MIST_HEIGHT)
#[cfg(test)]
pub const MIST_EXTINCTION: f32 = 0.02; // per unit at mist 1, at the surface (MIST_EXTINCTION)

#[cfg(test)]
pub const STORM_HEIGHT: f32 = 30.0; // e-folding height of the storm haze above the highest peak, blocks (STORM_HEIGHT)
#[cfg(test)]
pub const MIST_REGION_HEIGHT: f32 = 150.0; // the camera's mist fades out with its height over this (MIST_REGION_HEIGHT)

// (WGSL name in atmosphere.wgsl, value)
#[cfg(test)]
pub const CONSTANTS: [(&str, f32); 8] = [
    ("FOG_STORM_LO", STORM_LO),
    ("FOG_STRUCTURE", STRUCTURE),
    ("FOG_STRUCT_SCALE", STRUCT_SCALE),
    ("FOG_STRUCT_MAX_DIST", STRUCT_MAX_DIST),
    ("MIST_HEIGHT", MIST_HEIGHT),
    ("MIST_EXTINCTION", MIST_EXTINCTION),
    ("STORM_HEIGHT", STORM_HEIGHT),
    ("MIST_REGION_HEIGHT", MIST_REGION_HEIGHT),
];

#[cfg(test)]
pub fn storm_weight(storm: f32) -> f32 {
    smoothstep(STORM_LO, 1.0, storm)
}

// what a storm does to the air on a planet type
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StormFog {
    pub peak: f32,      // extinction per unit at storm 1 (visibility ≈ 3 / peak)
    pub tint: [f32; 3], // the haze's linear colour at full daylight
}

pub fn storm_fog(kind: WeatherKind) -> StormFog {
    let (peak, tint) = match kind {
        WeatherKind::Dust => (0.07, [0.45, 0.28, 0.12]), // sandstorm
        WeatherKind::Ash => (0.06, [0.03, 0.028, 0.028]), // ash cloud
        WeatherKind::Snow => (0.08, [0.80, 0.82, 0.88]), // blizzard whiteout
        WeatherKind::AcidRain => (0.012, [0.40, 0.45, 0.12]), // acid haze
        WeatherKind::Rain => (0.004, [0.50, 0.52, 0.56]), // rain haze
        WeatherKind::None => (0.0, [0.0; 3]),
    };
    StormFog { peak, tint }
}

// does this planet type get ground mist
fn has_mist(kind: WeatherKind) -> bool {
    matches!(kind, WeatherKind::Rain | WeatherKind::AcidRain)
}

// The ground mist at a direction (planet frame) and the shared clock (wrapped at 3600 s) with the sun at
// `sun_elevation` (sine of its elevation at the camera): a regional field of its own (the weather noise
// seen through a turned frame, so it is periodic over the wrap too) that is strongest while the sun is
// near the horizon, dawn and dusk.
pub fn mist_strength(kind: WeatherKind, dir: Vec3, t: f32, sun_elevation: f32) -> f32 {
    if !has_mist(kind) {
        return 0.0;
    }
    let d = rotate(dir, Vec3::new(0.8, 0.0, 0.6), 2.1);
    let field = smoothstep(0.42, 0.62, noise(d, t));
    let low_sun =
        smoothstep(-0.12, 0.0, sun_elevation) * (1.0 - smoothstep(0.1, 0.45, sun_elevation));
    field * low_sun
}

// atmo_mist_optical_depth: the mist's optical depth along a path of length `dist` whose ends are `h0`
// and `h1` blocks above the sea surface. The density mist · MIST_EXTINCTION · exp(-h / MIST_HEIGHT) along
// a straight climb integrates exactly: the mean of exp(-h / H) is H (e0 − e1) / (h1 − h0).
#[cfg(test)]
pub fn mist_optical_depth(mist: f32, h0: f32, h1: f32, dist: f32) -> f32 {
    if mist <= 0.0 {
        return 0.0;
    }
    mist * MIST_EXTINCTION * mean_falloff(h0, h1, MIST_HEIGHT) * dist
}

// mean of exp(-h / H) along a straight climb from height h0 to h1 (both clamped to >= 0)
#[cfg(test)]
fn mean_falloff(h0: f32, h1: f32, scale: f32) -> f32 {
    let (h0, h1) = (h0.max(0.0), h1.max(0.0));
    let (e0, e1) = ((-h0 / scale).exp(), (-h1 / scale).exp());
    let dh = h1 - h0;
    if dh.abs() > 1e-3 {
        scale * (e0 - e1) / dh
    } else {
        e0
    }
}

// atmo_storm_optical_depth: the haze's optical depth over `dist`. Heights are measured above `storm_base`
// (the radius of the planet's highest peak), so on the ground the factor is exactly 1 and the haze keeps its
// extinction there, and falls off with STORM_HEIGHT above it; from altitude only the storm at the shaded
// point counts (the camera's own weight fades with its height).
#[cfg(test)]
pub fn storm_optical_depth(
    peak: f32,
    storm_cam: f32,
    storm_there: f32,
    structure: f32,
    h0: f32,
    h1: f32,
    dist: f32,
) -> f32 {
    let w = storm_weight(camera_storm_mix(storm_cam, storm_there, h0));
    if w * peak <= 0.0 || dist <= 0.0 {
        return 0.0;
    }
    peak * w * structure * mean_falloff(h0, h1, STORM_HEIGHT) * dist
}

// the storm value the haze uses: the mean of the camera's and the shaded point's near the ground, the
// shaded point's alone from altitude
#[cfg(test)]
pub fn camera_storm_mix(storm_cam: f32, storm_there: f32, h0: f32) -> f32 {
    let k = (-h0.max(0.0) / STORM_HEIGHT).exp();
    storm_there + (0.5 * (storm_cam + storm_there) - storm_there) * k
}

// the camera's mist fades out with its height above the mist base (MIST_REGION_HEIGHT)
#[cfg(test)]
pub fn mist_region_weight(h0: f32) -> f32 {
    (-h0.max(0.0) / MIST_REGION_HEIGHT).exp()
}

// The structure's scroll offset (planet frame, units): integrated here, not time × wind, so gusts do
// not jump when the wind changes.
pub fn advance_wind(wind: Vec3, dir: Vec3, storm: f32, dt: f32) -> Vec3 {
    wind + super::precipitation::wind_at(dir, storm) * dt
}

pub const FOG_USAGE: &str = "Usage: /weather fog [on|off|density <0-3>|mist <0-3>]";

// /weather fog
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogSettings {
    pub on: bool,
    pub density: f32, // multiplies the storm extinction
    pub mist: f32,    // multiplies the mist strength
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            on: true,
            density: 1.0,
            mist: 1.0,
        }
    }
}

// the fog values at the camera, for the GPU
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogParams {
    pub storm: f32,      // the storm over the camera
    pub peak: f32,       // the type's extinction × density; 0 = off
    pub mist: f32,       // 0..1 (× the setting)
    pub mist_base: f32,  // radius the mist's height is measured from (the sea surface)
    pub storm_base: f32, // radius of the highest natural peak: the storm haze thins above it
    pub tint: [f32; 3],
    pub wind: Vec3,
}

// the sine of the sun's elevation over the camera's local horizon (`cam` from the planet centre)
pub fn sun_elevation(cam: Vec3, sun_dir: Vec3) -> f32 {
    cam.try_normalize()
        .map_or(0.0, |up| up.dot(sun_dir.normalize_or_zero()))
}

impl FogParams {
    #[allow(clippy::too_many_arguments)]
    pub fn at_camera(
        kind: WeatherKind,
        s: &FogSettings,
        storm_here: f32,
        dir: Vec3,
        t: f32,
        sun_elevation: f32,
        mist_base: f32,
        storm_base: f32,
        wind: Vec3,
    ) -> Self {
        let sf = storm_fog(kind);
        Self {
            storm: storm_here,
            peak: if s.on { sf.peak * s.density } else { 0.0 },
            mist: if s.on {
                (mist_strength(kind, dir, t, sun_elevation) * s.mist).min(1.0)
            } else {
                0.0
            },
            mist_base,
            storm_base,
            tint: sf.tint,
            wind,
        }
    }

    // Global.fog / fog_tint / fog_wind (shader.wgsl)
    pub fn uniform(&self) -> [[f32; 4]; 3] {
        [
            [self.storm, self.peak, self.mist, self.mist_base],
            [self.tint[0], self.tint[1], self.tint[2], 0.0],
            [self.wind.x, self.wind.y, self.wind.z, self.storm_base],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biome::PlanetType;
    use glam::Vec3;

    #[test]
    fn sun_elevation_is_the_sine_of_the_sun_above_the_local_horizon() {
        let up = Vec3::new(0.0, 1.0, 0.0);
        assert!((sun_elevation(up * 100.0, up) - 1.0).abs() < 1e-6);
        assert!(sun_elevation(up * 100.0, Vec3::X).abs() < 1e-6);
        assert!((sun_elevation(up * 100.0, -up) + 1.0).abs() < 1e-6);
        assert_eq!(
            sun_elevation(Vec3::ZERO, up),
            0.0,
            "no direction at the centre"
        );
    }

    #[test]
    fn wgsl_constants_match_the_rust_mirror() {
        let w = include_str!("../render/atmosphere.wgsl");
        crate::weather::tests::assert_wgsl_constants(w, &CONSTANTS);
    }

    fn kind(t: PlanetType) -> WeatherKind {
        WeatherKind::for_planet(t)
    }

    #[test]
    fn storm_weight_is_zero_in_calm_and_one_at_full_storm() {
        assert_eq!(storm_weight(0.0), 0.0);
        assert_eq!(storm_weight(STORM_LO), 0.0);
        assert!((storm_weight(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_thick_storm_types_drop_visibility_to_30_60_blocks() {
        for t in [PlanetType::Desert, PlanetType::Volcanic, PlanetType::Ice] {
            let peak = storm_fog(kind(t)).peak;
            let visibility = 3.0 / peak; // optical depth 3 is practically opaque
            assert!((30.0..=60.0).contains(&visibility), "{t:?}: {visibility}");
        }
    }

    #[test]
    fn storm_extinction_is_ordered_by_type() {
        let p = |t| storm_fog(kind(t)).peak;
        assert!(p(PlanetType::Ice) > p(PlanetType::Toxic));
        assert!(p(PlanetType::Toxic) > p(PlanetType::EarthLike));
        assert_eq!(p(PlanetType::EarthLike), p(PlanetType::Ocean));
        assert!(p(PlanetType::EarthLike) > 0.0);
        assert_eq!(p(PlanetType::Shattered), 0.0);
        assert_eq!(p(PlanetType::Cratered), 0.0, "no air, no haze");
    }

    #[test]
    fn only_the_wet_types_get_mist() {
        let dir = Vec3::new(0.3, 0.5, 0.8).normalize();
        // find a time at which the field is on, then check each type at it
        let t = (0..3600)
            .step_by(7)
            .map(|s| s as f32)
            .find(|&t| mist_strength(kind(PlanetType::EarthLike), dir, t, 0.0) > 0.0)
            .expect("the field is on somewhere in an hour");
        assert!(mist_strength(kind(PlanetType::Ocean), dir, t, 0.0) > 0.0);
        assert!(mist_strength(kind(PlanetType::Toxic), dir, t, 0.0) > 0.0);
        for ty in [
            PlanetType::Desert,
            PlanetType::Volcanic,
            PlanetType::Ice,
            PlanetType::Shattered,
            PlanetType::Cratered,
        ] {
            assert_eq!(mist_strength(kind(ty), dir, t, 0.0), 0.0, "{ty:?}");
        }
    }

    #[test]
    fn mist_is_strongest_with_the_sun_at_the_horizon() {
        let dir = Vec3::new(0.3, 0.5, 0.8).normalize();
        let k = kind(PlanetType::EarthLike);
        let t = (0..3600)
            .step_by(7)
            .map(|s| s as f32)
            .find(|&t| mist_strength(k, dir, t, 0.0) > 0.0)
            .unwrap();
        let dawn = mist_strength(k, dir, t, 0.0);
        assert!(dawn > mist_strength(k, dir, t, 0.6), "noon is thin");
        assert_eq!(mist_strength(k, dir, t, -0.5), 0.0, "night has none");
        assert!(dawn <= 1.0);
    }

    #[test]
    fn the_mist_field_is_continuous_across_the_clock_wrap() {
        let dir = Vec3::new(0.2, 0.9, 0.1).normalize();
        let k = kind(PlanetType::Toxic);
        let a = mist_strength(k, dir, 3599.99, 0.0);
        let b = mist_strength(k, dir, 0.0, 0.0);
        assert!((a - b).abs() < 0.02, "{a} vs {b}");
    }

    #[test]
    fn mist_optical_depth_is_exact_for_flat_and_sloped_paths() {
        // flat path at height 0: density mist * 1 over 100 units
        let flat = mist_optical_depth(1.0, 0.0, 0.0, 100.0);
        assert!((flat - MIST_EXTINCTION * 100.0).abs() < 1e-5);
        // sloped path: matches a numeric integral
        let (h0, h1, d) = (0.0_f32, 30.0_f32, 200.0_f32);
        let n = 20000;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let s = (i as f64 + 0.5) / n as f64;
            let h = h0 as f64 + (h1 - h0) as f64 * s;
            sum += (-h / MIST_HEIGHT as f64).exp() * d as f64 / n as f64;
        }
        let want = MIST_EXTINCTION as f64 * sum;
        let got = mist_optical_depth(1.0, h0, h1, d) as f64;
        assert!((got - want).abs() / want < 1e-3, "{got} vs {want}");
        // no mist, no depth; above the layer nearly nothing
        assert_eq!(mist_optical_depth(0.0, 0.0, 0.0, 100.0), 0.0);
        assert!(mist_optical_depth(1.0, 200.0, 200.0, 100.0) < 1e-6);
    }

    #[test]
    fn wind_advances_along_the_tangent_and_stays_finite() {
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let w = advance_wind(Vec3::ZERO, dir, 1.0, 2.0);
        assert!((w.length() - 2.0 * super::super::precipitation::WIND_MAX).abs() < 1e-3);
        assert!(w.dot(dir).abs() < 1e-4, "tangent to the sphere");
        let mut w = Vec3::ZERO;
        for _ in 0..200_000 {
            w = advance_wind(w, dir, 1.0, 0.05);
        }
        assert!(w.is_finite() && w.length() < 1e6);
        assert_eq!(
            advance_wind(Vec3::ONE, dir, 0.0, 1.0),
            Vec3::ONE,
            "no storm, no wind"
        );
    }

    #[test]
    fn off_settings_give_zero_fog() {
        let s = FogSettings {
            on: false,
            ..FogSettings::default()
        };
        let p = FogParams::at_camera(
            kind(PlanetType::Desert),
            &s,
            1.0,
            Vec3::Y,
            10.0,
            0.0,
            100.0,
            120.0,
            Vec3::ZERO,
        );
        let u = p.uniform();
        assert_eq!(u[0][1], 0.0, "peak");
        assert_eq!(u[0][2], 0.0, "mist");
    }

    #[test]
    fn a_calm_dry_planet_has_no_fog_uniform_values() {
        let p = FogParams::at_camera(
            kind(PlanetType::Shattered),
            &FogSettings::default(),
            1.0,
            Vec3::Y,
            10.0,
            0.0,
            100.0,
            120.0,
            Vec3::ZERO,
        );
        let u = p.uniform();
        assert_eq!((u[0][1], u[0][2]), (0.0, 0.0));
    }

    #[test]
    fn the_fog_uniform_carries_storm_tint_and_wind() {
        let wind = Vec3::new(1.0, 2.0, 3.0);
        let p = FogParams::at_camera(
            kind(PlanetType::Desert),
            &FogSettings::default(),
            0.8,
            Vec3::Y,
            10.0,
            0.0,
            100.0,
            120.0,
            wind,
        );
        let u = p.uniform();
        assert_eq!(u[0][0], 0.8);
        assert_eq!(u[0][1], storm_fog(kind(PlanetType::Desert)).peak);
        assert_eq!(u[0][3], 100.0);
        assert_eq!(&u[1][..3], &storm_fog(kind(PlanetType::Desert)).tint);
        assert_eq!(&u[2][..3], &[1.0, 2.0, 3.0]);
        assert_eq!(u[2][3], 120.0);
    }

    #[test]
    fn storm_haze_from_orbit_stays_thin() {
        // straight down from 3 planet radii through the atmosphere, structure 1
        let r = 250.0_f32;
        let base = 1.05 * r;
        let dist = r * 1.2 * 1.6 - r; // the chord inside the atmosphere sphere, an upper bound
        for t in PlanetType::ALL {
            let sf = storm_fog(kind(t));
            let tau = storm_optical_depth(sf.peak, 1.0, 1.0, 1.0, 3.0 * r - base, 0.0, dist);
            assert!(tau < 2.5, "{t:?}: {tau}");
        }
    }

    #[test]
    fn storm_haze_on_the_ground_is_unchanged() {
        let (peak, dist) = (0.07, 120.0);
        let tau = storm_optical_depth(peak, 1.0, 1.0, 1.3, 0.0, 0.0, dist);
        assert_eq!(tau, peak * storm_weight(1.0) * 1.3 * dist);
        assert_eq!(
            storm_optical_depth(peak, 0.0, 0.0, 1.0, 0.0, 0.0, dist),
            0.0
        );
    }

    #[test]
    fn the_camera_storm_weight_fades_with_height() {
        assert_eq!(camera_storm_mix(1.0, 0.0, 0.0), 0.5);
        assert!((camera_storm_mix(1.0, 0.0, 1000.0) - 0.0).abs() < 1e-6);
        // in between: from the mean (0.5) toward the shaded point's 0.7 with height
        let m = camera_storm_mix(0.3, 0.7, 50.0);
        assert!(m > 0.5 && m < 0.7, "{m}");
    }

    #[test]
    fn the_camera_mist_fades_with_height() {
        assert_eq!(mist_region_weight(0.0), 1.0);
        assert!((mist_region_weight(MIST_REGION_HEIGHT) - (-1.0_f32).exp()).abs() < 1e-6);
        assert!(mist_region_weight(600.0) < 0.02);
    }
}
