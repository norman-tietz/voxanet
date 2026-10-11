#![allow(dead_code)] // wired up by the galaxy map and jump tasks (Tasks 2-3)
                     // galaxy.rs
                     // The galaxy (spec docs/superpowers/specs/2026-10-12-galaxy-design.md §1): a seeded catalogue of solar
                     // systems in a two-armed spiral with a core — position (light-years), system seed, star type and name per
                     // entry, computed up front (tiny). A system's planets are only generated when you jump there
                     // (universe.rs); entry 0, the start system, has the galaxy seed itself, so a seed's start system is the
                     // solar system the game always had for it.

use glam::DVec3;

use crate::space::solar_system::{star_type_for, StarType};

pub const GALAXY_SYSTEMS: usize = 300;
pub const GALAXY_RADIUS_LY: f64 = 50.0;
pub const CORE_RADIUS_LY: f64 = 8.0;
pub const MIN_SPACING_LY: f64 = 1.5;
const CORE_SHARE: f64 = 0.25;
const ARMS: usize = 2;
const ARM_TWIST: f64 = 2.6; // radians of turn from the core to the rim
const ARM_SPREAD_LY: f64 = 3.5; // scatter around an arm
const EDGE_THICKNESS_LY: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct SystemEntry {
    pub position: DVec3, // light-years from the galaxy centre
    pub seed: u64,
    pub star_type: StarType,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct Galaxy {
    pub seed: u64,
    pub systems: Vec<SystemEntry>,
}

// splitmix64: a value in 0..1 for (seed, k, salt)
fn unit(seed: u64, k: u64, salt: u64) -> f64 {
    let mut x =
        seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn system_seed(galaxy_seed: u64, index: usize) -> u64 {
    if index == 0 {
        return galaxy_seed; // the start system: today's solar system for this seed
    }
    let mut x = galaxy_seed.wrapping_add((index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    x = (x ^ (x >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x = (x ^ (x >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^ (x >> 33)
}

const SYLLABLES: [&str; 24] = [
    "ka", "ri", "on", "the", "mar", "vel", "is", "do", "ra", "len", "su", "tor", "ae", "qui",
    "nor", "fa", "zen", "lo", "ith", "ser", "um", "bra", "co", "vys",
];

fn name_for(seed: u64) -> String {
    let n = 2 + (unit(seed, 0, 91) * 2.0) as usize; // 2-3 syllables
    let mut s: String = (0..n)
        .map(|k| {
            SYLLABLES
                [(unit(seed, k as u64 + 1, 92) * SYLLABLES.len() as f64) as usize % SYLLABLES.len()]
        })
        .collect();
    if let Some(first) = s.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    s
}

impl Galaxy {
    pub fn generate(seed: u64) -> Self {
        let mut positions: Vec<DVec3> = Vec::with_capacity(GALAXY_SYSTEMS);
        // the start system first: on arm 0, ~2/3 of the way out
        let r0 = GALAXY_RADIUS_LY * 0.66;
        let a0 = ARM_TWIST * (r0 / GALAXY_RADIUS_LY);
        positions.push(DVec3::new(r0 * a0.cos(), 0.0, r0 * a0.sin()));
        let mut attempt = 0u64;
        while positions.len() < GALAXY_SYSTEMS {
            attempt += 1;
            let k = attempt;
            let p = if unit(seed, k, 1) < CORE_SHARE {
                // the core: a flattened ball
                let r = CORE_RADIUS_LY * unit(seed, k, 2).cbrt();
                let theta = std::f64::consts::TAU * unit(seed, k, 3);
                let y = (unit(seed, k, 4) * 2.0 - 1.0) * CORE_RADIUS_LY * 0.35;
                DVec3::new(r * theta.cos(), y, r * theta.sin())
            } else {
                // an arm: a logarithmic-looking spiral, scattered around it, thinner toward the rim
                let arm = (unit(seed, k, 5) * ARMS as f64) as usize % ARMS;
                let t = CORE_RADIUS_LY / GALAXY_RADIUS_LY
                    + (1.0 - CORE_RADIUS_LY / GALAXY_RADIUS_LY) * unit(seed, k, 6).sqrt();
                let r = GALAXY_RADIUS_LY * t;
                let angle = ARM_TWIST * t + std::f64::consts::PI * arm as f64;
                let spread = ARM_SPREAD_LY * (unit(seed, k, 7) * 2.0 - 1.0);
                let along = ARM_SPREAD_LY * (unit(seed, k, 8) * 2.0 - 1.0);
                let dir = DVec3::new(angle.cos(), 0.0, angle.sin());
                let side = DVec3::new(-angle.sin(), 0.0, angle.cos());
                let y =
                    (unit(seed, k, 9) * 2.0 - 1.0) * EDGE_THICKNESS_LY * (1.0 + (1.0 - t) * 2.0);
                (dir * (r + along) + side * spread).with_y(y)
            };
            if p.length() <= GALAXY_RADIUS_LY * 1.1
                && positions
                    .iter()
                    .all(|q| (*q - p).length() >= MIN_SPACING_LY)
            {
                positions.push(p);
            }
            assert!(
                attempt < 1_000_000,
                "galaxy generation can't place its systems"
            );
        }
        let systems = positions
            .into_iter()
            .enumerate()
            .map(|(i, position)| {
                let s = system_seed(seed, i);
                SystemEntry {
                    position,
                    seed: s,
                    star_type: star_type_for(s),
                    name: name_for(s),
                }
            })
            .collect();
        Galaxy { seed, systems }
    }

    pub fn distance_ly(&self, a: usize, b: usize) -> f64 {
        (self.systems[a].position - self.systems[b].position).length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::solar_system::{star_type_for, SolarSystem};

    #[test]
    fn galaxy_is_deterministic_and_seeded() {
        assert_eq!(Galaxy::generate(1).systems, Galaxy::generate(1).systems);
        assert_ne!(Galaxy::generate(1).systems, Galaxy::generate(2).systems);
        assert_eq!(Galaxy::generate(7).systems.len(), GALAXY_SYSTEMS);
    }

    #[test]
    fn systems_keep_their_distance() {
        let g = Galaxy::generate(3);
        for (i, a) in g.systems.iter().enumerate() {
            for b in &g.systems[i + 1..] {
                assert!((a.position - b.position).length() >= MIN_SPACING_LY - 1e-9);
            }
        }
    }

    #[test]
    fn the_galaxy_is_a_spiral_with_a_core() {
        let g = Galaxy::generate(5);
        let core = g
            .systems
            .iter()
            .filter(|s| s.position.length() <= CORE_RADIUS_LY * 1.2)
            .count();
        let share = core as f64 / GALAXY_SYSTEMS as f64;
        assert!((0.18..0.34).contains(&share), "core share {share}");
        assert!(g
            .systems
            .iter()
            .all(|s| s.position.length() <= GALAXY_RADIUS_LY * 1.15));
        // flat: thin compared to its radius
        let max_y = g
            .systems
            .iter()
            .map(|s| s.position.y.abs())
            .fold(0.0, f64::max);
        assert!(max_y < GALAXY_RADIUS_LY * 0.1);
    }

    // Review Focus 5
    #[test]
    fn start_system_is_todays_solar_system() {
        let g = Galaxy::generate(1);
        let start = &g.systems[0];
        assert_eq!(start.seed, 1);
        assert_eq!(start.star_type, star_type_for(1));
        let r = start.position.length();
        assert!(
            r > GALAXY_RADIUS_LY * 0.5 && r < GALAXY_RADIUS_LY * 0.85,
            "start ~2/3 out: {r}"
        );
        let s = SolarSystem::generate(start.seed);
        assert_eq!(s.planets.len(), SolarSystem::generate(1).planets.len());
    }

    #[test]
    fn entries_carry_their_star_type_and_a_name() {
        let g = Galaxy::generate(9);
        for s in &g.systems {
            assert_eq!(s.star_type, star_type_for(s.seed));
            assert!(
                s.name.len() >= 4 && s.name.chars().next().unwrap().is_uppercase(),
                "{}",
                s.name
            );
        }
        let names: std::collections::HashSet<_> =
            g.systems.iter().map(|s| s.name.clone()).collect();
        assert!(
            names.len() > GALAXY_SYSTEMS * 9 / 10,
            "names mostly distinct"
        );
    }
}
