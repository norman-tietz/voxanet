// galaxy.rs
// Sub-project #1 (galaxy foundation) of the planned multi-planet feature: a seeded single-star
// solar system of orbiting placeholder bodies, deliberately separate from the existing
// single-planet engine (PlanetData/CoordSystem/Player/Physics) — no voxel terrain, no collision,
// and no connection yet to the player's actual planet (that's a later sub-project, landing/liftoff).

use crate::biome::PlanetType;
use glam::DVec3;

pub struct Star {
    pub radius: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GalaxyPlanet {
    pub orbit_radius: f64,
    pub orbit_speed: f64, // radians/sec; positive = counterclockwise looking down +Y
    pub orbit_phase: f64, // starting angle, radians
    pub radius: f32,
    pub planet_type: PlanetType,
}

pub struct Galaxy {
    pub star: Star,
    pub planets: Vec<GalaxyPlanet>,
}

const PLANET_COUNT: usize = 7;
const INNER_ORBIT_RADIUS: f64 = 8_000.0;
const ORBIT_RADIUS_GROWTH: f64 = 1.5; // each planet this many times farther out than the last
const INNER_ORBIT_PERIOD_SECS: f64 = 180.0;
const STAR_RADIUS: f64 = 3_000.0;
const MIN_PLANET_RADIUS: f32 = 40.0;
const MAX_PLANET_RADIUS: f32 = 250.0;

impl Galaxy {
    pub fn generate(seed: u64) -> Self {
        let planets = (0..PLANET_COUNT)
            .map(|i| {
                let orbit_radius = INNER_ORBIT_RADIUS * ORBIT_RADIUS_GROWTH.powi(i as i32);
                GalaxyPlanet {
                    orbit_radius,
                    orbit_speed: orbit_speed_for(orbit_radius),
                    orbit_phase: std::f64::consts::TAU * (i as f64) / (PLANET_COUNT as f64),
                    radius: planet_radius_for(seed, i),
                    planet_type: PlanetType::ALL[i % PlanetType::ALL.len()],
                }
            })
            .collect();
        Self {
            star: Star {
                radius: STAR_RADIUS,
            },
            planets,
        }
    }
}

fn orbit_speed_for(orbit_radius: f64) -> f64 {
    let c = INNER_ORBIT_RADIUS.sqrt() * (std::f64::consts::TAU / INNER_ORBIT_PERIOD_SECS);
    c / orbit_radius.sqrt()
}

// deterministic pseudo-random value in [MIN_PLANET_RADIUS, MAX_PLANET_RADIUS] from seed + index —
// no true randomness anywhere in generation, so a given seed always reproduces the same system
fn planet_radius_for(seed: u64, index: usize) -> f32 {
    let mut h = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(index as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    let t = (h >> 11) as f64 / (1u64 << 53) as f64; // 0..1, using the top 53 bits
    MIN_PLANET_RADIUS + (t as f32) * (MAX_PLANET_RADIUS - MIN_PLANET_RADIUS)
}

impl GalaxyPlanet {
    pub fn position_at(&self, t: f64) -> DVec3 {
        let angle = self.orbit_phase + self.orbit_speed * t;
        DVec3::new(
            self.orbit_radius * angle.cos(),
            0.0,
            self.orbit_radius * angle.sin(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_produces_seven_planets() {
        let g = Galaxy::generate(1);
        assert_eq!(g.planets.len(), 7);
    }

    #[test]
    fn generate_orbit_radii_strictly_increase() {
        let g = Galaxy::generate(1);
        for pair in g.planets.windows(2) {
            assert!(
                pair[1].orbit_radius > pair[0].orbit_radius,
                "{} should be > {}",
                pair[1].orbit_radius,
                pair[0].orbit_radius
            );
        }
    }

    #[test]
    fn generate_is_deterministic_for_a_fixed_seed() {
        let a = Galaxy::generate(42);
        let b = Galaxy::generate(42);
        assert_eq!(a.planets.len(), b.planets.len());
        for (pa, pb) in a.planets.iter().zip(b.planets.iter()) {
            assert_eq!(pa.orbit_radius, pb.orbit_radius);
            assert_eq!(pa.orbit_speed, pb.orbit_speed);
            assert_eq!(pa.orbit_phase, pb.orbit_phase);
            assert_eq!(pa.radius, pb.radius);
            assert_eq!(pa.planet_type, pb.planet_type);
        }
    }

    #[test]
    fn generate_is_different_for_a_different_seed() {
        let a = Galaxy::generate(1);
        let b = Galaxy::generate(2);
        // at least one planet's radius differs between seeds (radius is the only
        // seed-dependent-per-planet field; orbit layout is fixed by index, not seed)
        assert!(a
            .planets
            .iter()
            .zip(b.planets.iter())
            .any(|(pa, pb)| pa.radius != pb.radius));
    }

    #[test]
    fn generate_planet_radii_are_in_range() {
        let g = Galaxy::generate(7);
        for p in &g.planets {
            assert!(
                (40.0..=250.0).contains(&p.radius),
                "radius {} out of range",
                p.radius
            );
        }
    }

    #[test]
    fn position_at_traces_a_circle_of_the_right_radius() {
        let g = Galaxy::generate(1);
        let p = &g.planets[0];
        for t in [0.0, 10.0, 123.4, 10_000.0] {
            let pos = p.position_at(t);
            assert!(
                (pos.length() - p.orbit_radius).abs() < 1e-6,
                "at t={t}, length {} != orbit_radius {}",
                pos.length(),
                p.orbit_radius
            );
            assert_eq!(
                pos.y, 0.0,
                "orbits are coplanar in the XZ plane this milestone"
            );
        }
    }

    #[test]
    fn position_at_advances_in_the_direction_orbit_speed_implies() {
        let g = Galaxy::generate(1);
        let p = &g.planets[0];
        assert!(
            p.orbit_speed > 0.0,
            "sanity check: generated planets orbit forward"
        );
        let angle_at = |t: f64| {
            let pos = p.position_at(t);
            pos.z.atan2(pos.x)
        };
        let dt = 0.01; // small enough that neither angle is near a +/-PI wraparound for t near 0
        let a0 = angle_at(0.0);
        let a1 = angle_at(dt);
        assert!(
            a1 > a0,
            "angle should increase with positive orbit_speed: {a0} -> {a1}"
        );
    }

    #[test]
    fn inner_planet_period_is_about_three_minutes() {
        let g = Galaxy::generate(1);
        let inner = &g.planets[0];
        let period = std::f64::consts::TAU / inner.orbit_speed;
        assert!(
            (period - 180.0).abs() < 1.0,
            "period was {period}s, expected ~180s"
        );
    }
}
