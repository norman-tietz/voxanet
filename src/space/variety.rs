// solar-system variety (spec 2026-10-12-system-variety-design.md): pure, seeded functions for every
// system but the start one — how many planets, how hot each orbit is, which type it gets, its traits
use crate::planet::biome::PlanetType;
use crate::planet::traits::PlanetTraits;
use crate::space::solar_system::{seeded_unit, StarType, MAX_PLANET_RADIUS, MIN_PLANET_RADIUS};

// salts for seeded_unit (orbit_for uses 1–4)
const SALT_COUNT_A: u64 = 20;
const SALT_COUNT_B: u64 = 21;
const SALT_TYPE: u64 = 10;
const SALT_DAY: u64 = 11;
const SALT_RELIEF: u64 = 12;
const SALT_LAND: u64 = 13;
const SALT_HUE: u64 = 14;
const SALT_STORM: u64 = 15;
const SALT_DENSITY: u64 = 16;

// the habitable distance around a yellow star: the 4th orbit (INNER_ORBIT_RADIUS × 1.5³)
pub const HABITABLE_ORBIT: f64 = 27_000.0 * crate::space::solar_system::SYSTEM_SCALE;

// 3–10 planets: 3 + a + b, a in 0..=3, b in 0..=4 (6–7 most common)
pub fn planet_count(seed: u64) -> usize {
    let a = (seeded_unit(seed, 0, SALT_COUNT_A) * 4.0) as usize;
    let b = (seeded_unit(seed, 0, SALT_COUNT_B) * 5.0) as usize;
    3 + a.min(3) + b.min(4)
}

pub fn habitable_orbit(star: StarType) -> f64 {
    HABITABLE_ORBIT * star.def().luminosity.sqrt()
}

// 0 at the habitable distance, negative inward (hot), positive outward (cold), one unit per doubling
pub fn temperature_index(orbit_radius: f64, star: StarType) -> f64 {
    (orbit_radius / habitable_orbit(star)).log2()
}

fn bump(tau: f64, centre: f64, width: f64) -> f64 {
    let x = (tau - centre) / width;
    (-x * x).exp()
}

// each type's weight at temperature index tau, in PlanetType::ALL order
pub fn type_weights(tau: f64) -> [f64; PlanetType::ALL.len()] {
    PlanetType::ALL.map(|t| match t {
        PlanetType::Volcanic => {
            if tau <= -1.6 {
                1.0
            } else {
                bump(tau, -1.6, 0.6)
            }
        }
        PlanetType::Desert => bump(tau, -0.8, 0.5),
        PlanetType::Toxic => 0.8 * bump(tau, -0.6, 0.6),
        PlanetType::EarthLike => bump(tau, 0.0, 0.35),
        PlanetType::Ocean => 0.9 * bump(tau, 0.3, 0.45),
        PlanetType::Ice => {
            if tau >= 1.4 {
                1.0
            } else {
                bump(tau, 1.4, 0.7)
            }
        }
        PlanetType::Shattered => 0.08,
        // airless rock: present everywhere, somewhat likelier on the cold outer orbits
        PlanetType::Cratered => 0.12 + 0.25 * bump(tau, 1.8, 0.8),
    })
}

// a seeded pick among the types by their weights at tau
pub fn pick_type(seed: u64, index: usize, tau: f64) -> PlanetType {
    let w = type_weights(tau);
    let total: f64 = w.iter().sum();
    let mut x = seeded_unit(seed, index, SALT_TYPE) * total;
    for (k, &wk) in w.iter().enumerate() {
        if x < wk {
            return PlanetType::ALL[k];
        }
        x -= wk;
    }
    *PlanetType::ALL.last().expect("types exist")
}

// a planet's own variation; gravity grows with its radius (160 → ×0.6, 1000 → ×1.3) times a density jitter
pub fn traits_for(seed: u64, index: usize, planet_type: PlanetType, radius: f32) -> PlanetTraits {
    let u = |salt| seeded_unit(seed, index, salt) as f32;
    let size =
        ((radius - MIN_PLANET_RADIUS) / (MAX_PLANET_RADIUS - MIN_PLANET_RADIUS)).clamp(0.0, 1.0);
    PlanetTraits {
        day_scale: 0.6 + 1.0 * u(SALT_DAY),
        relief: 0.7 + 0.7 * u(SALT_RELIEF),
        land_shift: if planet_type.def().liquid.is_some() {
            -0.15 + 0.3 * u(SALT_LAND)
        } else {
            0.0
        },
        sky_hue: -0.25 + 0.5 * u(SALT_HUE),
        storminess: 0.5 + 1.0 * u(SALT_STORM),
        gravity: (0.6 + 0.7 * size) * (0.9 + 0.2 * u(SALT_DENSITY)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::solar_system::SolarSystem;

    // the scale moves the habitable distance with the orbits: every orbit's temperature, so every type, is as before
    #[test]
    fn the_habitable_orbit_scales_with_the_system() {
        assert_eq!(
            HABITABLE_ORBIT,
            27_000.0 * crate::space::solar_system::SYSTEM_SCALE
        );
    }

    #[test]
    fn planet_count_spans_3_to_10_with_the_middle_likelier() {
        let mut hist = [0usize; 11];
        for seed in 0..2000u64 {
            let n = planet_count(seed);
            assert!((3..=10).contains(&n));
            hist[n] += 1;
        }
        assert!(hist[3..=10].iter().all(|&c| c > 0), "{hist:?}");
        let mid = hist[6].max(hist[7]);
        assert!(mid > hist[3] && mid > hist[10], "{hist:?}");
    }

    #[test]
    fn habitable_distance_grows_with_luminosity() {
        let d: Vec<f64> = StarType::ALL.iter().map(|&s| habitable_orbit(s)).collect();
        let lum: Vec<f64> = StarType::ALL.iter().map(|s| s.def().luminosity).collect();
        for i in 0..d.len() {
            for j in 0..d.len() {
                if lum[i] < lum[j] {
                    assert!(d[i] < d[j]);
                }
            }
        }
        assert!(
            temperature_index(habitable_orbit(StarType::Yellow), StarType::Yellow).abs() < 1e-12
        );
    }

    // over many systems: hot orbits rocky, cold ones icy, the habitable band Earth-like/Ocean, Shattered rare
    #[test]
    fn types_follow_the_temperature() {
        let (mut hot, mut hot_n, mut cold, mut cold_n, mut mild, mut mild_n) = (0, 0, 0, 0, 0, 0);
        let (mut shattered, mut total) = (0, 0);
        let mut cold_icy = 0;
        let (mut duplicate, mut no_earth) = (false, false);
        for seed in 1000..3000u64 {
            let s = SolarSystem::generate_varied(seed);
            let mut seen = Vec::new();
            for p in &s.planets {
                let tau = temperature_index(p.orbit_radius, s.star.star_type);
                total += 1;
                if p.planet_type == PlanetType::Shattered {
                    shattered += 1;
                }
                if tau < -1.2 {
                    hot_n += 1;
                    if matches!(
                        p.planet_type,
                        PlanetType::Volcanic | PlanetType::Desert | PlanetType::Toxic
                    ) {
                        hot += 1;
                    }
                } else if tau > 1.2 {
                    cold_n += 1;
                    if p.planet_type == PlanetType::Ice {
                        cold += 1;
                    }
                    if matches!(p.planet_type, PlanetType::Ice | PlanetType::Cratered) {
                        cold_icy += 1;
                    }
                } else if tau.abs() < 0.25 {
                    mild_n += 1;
                    if matches!(p.planet_type, PlanetType::EarthLike | PlanetType::Ocean) {
                        mild += 1;
                    }
                }
                duplicate |= seen.contains(&p.planet_type);
                seen.push(p.planet_type);
            }
            no_earth |= !seen.contains(&PlanetType::EarthLike);
        }
        let share = |a: usize, n: usize| a as f64 / n.max(1) as f64;
        assert!(share(hot, hot_n) > 0.85, "hot {}", share(hot, hot_n));
        // 0.65 (was 0.8): the 8th type, Cratered, takes up to 0.37 of the weight on cold orbits
        assert!(share(cold, cold_n) > 0.65, "cold {}", share(cold, cold_n));
        assert!(
            share(cold_icy, cold_n) > 0.85,
            "cold ice or cratered {}",
            share(cold_icy, cold_n)
        );
        assert!(share(mild, mild_n) > 0.6, "mild {}", share(mild, mild_n));
        let s = share(shattered, total);
        assert!(s > 0.0 && s < 0.1, "shattered {s}");
        assert!(duplicate && no_earth);
    }

    #[test]
    fn traits_stay_in_their_ranges() {
        for seed in 0..500u64 {
            for i in 0..10 {
                for &t in &PlanetType::ALL {
                    for r in [160.0f32, 580.0, 1000.0] {
                        let x = traits_for(seed, i, t, r);
                        assert!((0.6..=1.6).contains(&x.day_scale));
                        assert!((0.7..=1.4).contains(&x.relief));
                        assert!((-0.15..=0.15).contains(&x.land_shift));
                        if t.def().liquid.is_none() {
                            assert_eq!(x.land_shift, 0.0);
                        }
                        assert!((-0.25..=0.25).contains(&x.sky_hue));
                        assert!((0.5..=1.5).contains(&x.storminess));
                        assert!((0.54..=1.43).contains(&x.gravity));
                    }
                }
            }
        }
    }

    #[test]
    fn gravity_grows_with_the_radius() {
        for seed in 0..200u64 {
            // the same density jitter (same seed, index): only the radius differs
            let small = traits_for(seed, 3, PlanetType::Ice, 160.0).gravity;
            let big = traits_for(seed, 3, PlanetType::Ice, 1000.0).gravity;
            assert!(big > small * 2.0);
        }
    }
}
