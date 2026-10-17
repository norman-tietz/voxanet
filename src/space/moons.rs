// moons.rs
// Moon generation (spec docs/superpowers/specs/2026-10-12-moons-design.md): the Hill sphere decides where
// stable orbits exist, the doc's rules cap how many there are, everything is hashed from (system seed,
// parent, moon index), so a system always regenerates the same.

use crate::planet::biome::PlanetType;
use crate::planet::traits::PlanetTraits;
use crate::space::solar_system::{gravity_for_radius, noise_seed_for, seeded_unit, Planet, Star};

pub const STAR_DENSITY_RATIO: f64 = 0.25; // the star's density as a share of a planet's
pub const MOON_MIN_ORBIT_RADII: f64 = 2.5; // the Roche limit, in parent radii
pub const MOON_STABLE_FRACTION: f64 = 0.5; // of the Hill radius
pub const MOON_ORBIT_STEP: f64 = 1.4; // minimum ratio between consecutive orbits
pub const MOON_MAX_TILT: f64 = 5.0 * std::f64::consts::PI / 180.0;
pub const MOON_MIN_RADIUS: f32 = 40.0;
pub const MOON_MAX_RADIUS_SHARE: f32 = 0.25;
pub const MOON_REF_PERIOD_SECS: f64 = 150.0; // at three parent radii

// the Hill radius in parent radii: r_H / R_p = (a / R_*) · (ρ_p / (3 ρ_*))^(1/3) (mass ∝ ρ R³)
pub fn hill_radii(parent_orbit: f64, star_radius: f64) -> f64 {
    (parent_orbit / star_radius) * (1.0 / (3.0 * STAR_DENSITY_RATIO)).powf(1.0 / 3.0)
}

// orbit radii (parent radii) between `room_min` and `room_max`, each at least MOON_ORBIT_STEP times the
// last; `u` supplies 0..1 values for the jitter
pub fn moon_orbit_radii(room_min: f64, room_max: f64, u: &mut dyn FnMut() -> f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut r = room_min * (1.0 + 0.15 * u());
    while r <= room_max {
        out.push(r);
        r *= MOON_ORBIT_STEP + 0.2 * u();
    }
    out
}

// period 150 s at three parent radii, Kepler beyond (equal densities: independent of the parent's size)
pub fn moon_orbit_speed(orbit_in_parent_radii: f64) -> f64 {
    std::f64::consts::TAU / MOON_REF_PERIOD_SECS * (3.0 / orbit_in_parent_radii).powf(1.5)
}

// a moon's traits: neutral but for gravity (its locked spin sets its day; it shares the weather like every planet)
pub fn moon_traits(radius: f32) -> PlanetTraits {
    PlanetTraits {
        gravity: gravity_for_radius(radius),
        ..PlanetTraits::NEUTRAL
    }
}

fn cap_for(t: PlanetType) -> usize {
    match t {
        PlanetType::Ice => 6,
        PlanetType::Shattered | PlanetType::Cratered => 1,
        _ => 2,
    }
}

// the moon's own unit value: from (seed, parent, moon index, salt), not from list positions
fn unit(seed: u64, parent: usize, k: usize, salt: u64) -> f64 {
    seeded_unit(
        seed.wrapping_add(0x1000 * (parent as u64 + 1)),
        k,
        0x100 + salt,
    )
}

fn moon_type(u: f64, radius: f32) -> PlanetType {
    let big = radius >= 100.0;
    if u < 0.70 {
        PlanetType::Cratered
    } else if u < 0.82 {
        PlanetType::Ice
    } else if u < 0.91 && big {
        PlanetType::Toxic
    } else if big {
        PlanetType::Desert
    } else {
        PlanetType::Cratered
    }
}

pub fn moons_for(seed: u64, parent_index: usize, parent: &Planet, star: &Star) -> Vec<Planet> {
    let room_max = MOON_STABLE_FRACTION * hill_radii(parent.orbit_radius, star.radius);
    let mut k = 0usize;
    let mut jitter = || {
        k += 1;
        unit(seed, parent_index, 0, 50 + k as u64)
    };
    let orbits = moon_orbit_radii(MOON_MIN_ORBIT_RADII, room_max, &mut jitter);
    let cap = cap_for(parent.planet_type);
    let lo = usize::from(parent.planet_type == PlanetType::Ice);
    let want = lo + (unit(seed, parent_index, 0, 1) * (cap - lo + 1) as f64) as usize;
    let n = want.min(cap).min(orbits.len());
    let max_r = (MOON_MAX_RADIUS_SHARE * parent.radius).max(MOON_MIN_RADIUS);
    (0..n)
        .map(|k| {
            let u = |salt| unit(seed, parent_index, k, salt);
            let radius = MOON_MIN_RADIUS + (max_r - MOON_MIN_RADIUS) * u(2).powf(1.5) as f32;
            let orbit_in_radii = orbits[k];
            Planet {
                orbit_radius: orbit_in_radii * parent.radius as f64,
                orbit_speed: moon_orbit_speed(orbit_in_radii),
                orbit_phase: std::f64::consts::TAU * u(3),
                orbit_tilt: MOON_MAX_TILT * u(4),
                orbit_node: std::f64::consts::TAU * u(5),
                radius,
                planet_type: moon_type(u(6), radius),
                noise_seed: noise_seed_for(seed ^ 0x6D6F_6F6E, parent_index * 16 + k),
                traits: moon_traits(radius),
                parent: Some(parent_index),
                moon_index: k as u8,
                parent_orbit: Some(parent.own_orbit()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::solar_system::{body_label, SolarSystem};

    #[test]
    fn hill_radius_in_parent_radii_grows_with_the_orbit_and_shrinks_with_the_star() {
        let a = hill_radii(32_000.0, 7_200.0);
        assert!((a - 1.1006 * 32_000.0 / 7_200.0).abs() < 1e-2, "{a}");
        assert!(hill_radii(364_000.0, 7_200.0) > 10.0 * a);
        assert!(hill_radii(32_000.0, 14_400.0) < a);
    }

    #[test]
    fn orbits_are_packed_inside_the_stable_zone_with_the_minimum_spacing() {
        let mut k = 0.0;
        let mut u = || {
            k += 0.37;
            k % 1.0
        };
        let orbits = moon_orbit_radii(2.5, 20.0, &mut u);
        assert!(!orbits.is_empty());
        assert!(orbits[0] >= 2.5 && *orbits.last().unwrap() <= 20.0);
        for w in orbits.windows(2) {
            assert!(w[1] / w[0] >= MOON_ORBIT_STEP - 1e-9, "{w:?}");
        }
        assert!(moon_orbit_radii(2.5, 2.4, &mut u).is_empty(), "no room");
        assert!(moon_orbit_radii(2.5, 3.0, &mut u).len() <= 1);
    }

    #[test]
    fn every_generated_moon_obeys_the_rules() {
        let systems = (1..40u64).flat_map(|seed| {
            [
                SolarSystem::generate(seed),
                SolarSystem::generate_varied(seed),
            ]
        });
        for s in systems {
            let seed = s.seed;
            for (pi, p) in s
                .planets
                .iter()
                .enumerate()
                .filter(|(_, p)| p.parent.is_none())
            {
                let hill = hill_radii(p.orbit_radius, s.star.radius);
                let moons: Vec<_> = s.moons_of(pi).map(|i| &s.planets[i]).collect();
                let cap = match p.planet_type.def().name {
                    "Ice" => 6,
                    "Shattered" | "Cratered" => 1,
                    _ => 2,
                };
                assert!(
                    moons.len() <= cap,
                    "seed {seed} planet {pi}: {}",
                    moons.len()
                );
                let mut last = 0.0_f64;
                for (k, m) in moons.iter().enumerate() {
                    let r = m.orbit_radius / p.radius as f64;
                    assert!(r >= MOON_MIN_ORBIT_RADII - 1e-9, "{r}");
                    assert!(
                        r <= MOON_STABLE_FRACTION * hill + 1e-9,
                        "{r} vs hill {hill}"
                    );
                    assert!(r / last.max(1e-9) >= MOON_ORBIT_STEP - 1e-9 || k == 0);
                    last = r;
                    assert!(
                        m.radius >= 40.0 && m.radius <= 0.25 * p.radius + 1e-3,
                        "{}",
                        m.radius
                    );
                    assert_eq!(m.parent, Some(pi));
                    assert_eq!(m.moon_index as usize, k);
                    assert!(m.parent_orbit == Some(p.own_orbit()));
                    assert!(m.orbit_tilt <= MOON_MAX_TILT + 1e-12);
                    // Kepler: 150 s at 3 parent radii
                    let t = std::f64::consts::TAU / m.orbit_speed;
                    assert!((t - 150.0 * (r / 3.0).powf(1.5)).abs() < 1e-6 * t, "{t}");
                }
            }
        }
    }

    #[test]
    fn generation_is_deterministic_and_independent_of_list_position() {
        let a = SolarSystem::generate(7);
        let b = SolarSystem::generate(7);
        assert_eq!(a.planets, b.planets);
        let p = a.planets[2];
        assert_eq!(moons_for(7, 2, &p, &a.star), moons_for(7, 2, &p, &a.star));
        assert_eq!(
            SolarSystem::generate_varied(7).planets,
            SolarSystem::generate_varied(7).planets
        );
        // planets come first, unchanged by their moons: planet #k is still planets[k - 1]
        let n = a.planet_count();
        assert!(
            a.planets[..n].iter().all(|p| !p.is_moon())
                && a.planets[n..].iter().all(|p| p.is_moon())
        );
    }

    // a 10-planet system of icy outer planets could hold 70 bodies: generation stops at MAX_PLANETS
    #[test]
    fn generated_systems_stay_within_the_body_cap() {
        use crate::space::solar_system::MAX_PLANETS;
        let mut most = 0;
        for seed in 0..2000u64 {
            for s in [
                SolarSystem::generate(seed),
                SolarSystem::generate_varied(seed),
            ] {
                assert!(
                    s.planets.len() <= MAX_PLANETS,
                    "seed {seed}: {}",
                    s.planets.len()
                );
                most = most.max(s.planets.len());
            }
        }
        assert!(most > 12, "some systems have moons: {most}");
    }

    // moons are light and otherwise neutral (tidal lock decides their day, they share the weather)
    #[test]
    fn moon_traits_are_neutral() {
        let t = moon_traits(60.0);
        assert_eq!(PlanetTraits { gravity: 1.0, ..t }, PlanetTraits::NEUTRAL);
    }

    #[test]
    fn inner_planets_of_big_stars_have_none_and_outer_planets_have_some() {
        // over many seeds: no moons where the stable zone is smaller than the Roche limit, some where it is wide
        let (mut none_ok, mut some_ok) = (false, false);
        for seed in 1..200u64 {
            let s = SolarSystem::generate_varied(seed);
            for (pi, p) in s
                .planets
                .iter()
                .enumerate()
                .filter(|(_, p)| p.parent.is_none())
            {
                let room = MOON_STABLE_FRACTION * hill_radii(p.orbit_radius, s.star.radius);
                let n = s.moons_of(pi).count();
                if room < MOON_MIN_ORBIT_RADII {
                    assert_eq!(n, 0, "seed {seed} planet {pi}: room {room}");
                    none_ok = true;
                }
                // (a varied system near the body cap may have dropped the outer planets' moons: skip those)
                if s.planets.len() >= crate::space::solar_system::MAX_PLANETS {
                    continue;
                }
                if room > 8.0 && n > 0 {
                    some_ok = true;
                }
            }
        }
        assert!(none_ok && some_ok);
    }

    #[test]
    fn small_moons_are_cratered_and_only_big_ones_get_atmospheres() {
        use crate::planet::biome::PlanetType;
        let (mut cratered, mut total) = (0, 0);
        for seed in 1..200u64 {
            let s = SolarSystem::generate_varied(seed);
            for m in s.planets.iter().filter(|p| p.is_moon()) {
                total += 1;
                if m.planet_type == PlanetType::Cratered {
                    cratered += 1;
                }
                if matches!(m.planet_type, PlanetType::Toxic | PlanetType::Desert) {
                    assert!(
                        m.radius >= 100.0,
                        "{:?} moon of radius {}",
                        m.planet_type,
                        m.radius
                    );
                }
            }
        }
        assert!(total > 50, "{total}");
        assert!(cratered as f64 / total as f64 > 0.6, "{cratered}/{total}");
    }

    #[test]
    fn add_moon_picks_a_free_orbit_inside_the_stable_zone() {
        let mut s = SolarSystem::generate(1);
        // the planet with the widest stable zone has room
        let parent = (0..s.planet_count())
            .max_by(|&a, &b| {
                let room = |i: usize| hill_radii(s.planets[i].orbit_radius, s.star.radius);
                room(a).total_cmp(&room(b))
            })
            .unwrap();
        let before = s.moons_of(parent).count();
        // radius 40 fits every planet (a quarter of the smallest, 160)
        let i = s
            .add_moon(parent, crate::planet::biome::PlanetType::Cratered, 40.0)
            .unwrap();
        assert_eq!(s.planets[i].parent, Some(parent));
        assert_eq!(s.moons_of(parent).count(), before + 1);
        assert!(
            s.add_moon(0, crate::planet::biome::PlanetType::Cratered, 10.0)
                .is_err(),
            "below 40"
        );
        assert!(s
            .add_moon(99, crate::planet::biome::PlanetType::Cratered, 60.0)
            .is_err());
        assert!(
            s.add_moon(i, crate::planet::biome::PlanetType::Cratered, 60.0)
                .is_err(),
            "no moons of moons"
        );
        assert_eq!(s.planets[i].traits, moon_traits(40.0));
        // an added planet after the moons still gets the next planet number's orbit
        let n = s.planet_count();
        let j = s
            .add_planet(crate::planet::biome::PlanetType::Ice, 300.0)
            .unwrap();
        assert_eq!(body_label(&s, j), (n + 1).to_string());
    }
}
