// galaxy.rs
// Sub-project #1 (galaxy foundation) of the planned multi-planet feature: a seeded single-star
// solar system of orbiting placeholder bodies, deliberately separate from the existing
// single-planet engine (PlanetData/CoordSystem/Player/Physics) — no voxel terrain, no collision,
// and no connection yet to the player's actual planet (that's a later sub-project, landing/liftoff).

use crate::biome::PlanetType;
use glam::{DVec3, Quat, Vec3};

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
    pub noise_seed: u32, // seeds this planet's TerrainShape/NoiseGenerator (src/galaxy_terrain.rs);
                         // independent of planet_radius_for's hash so radius and terrain shape don't correlate
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
                    noise_seed: noise_seed_for(seed, i),
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

// deterministic pseudo-random seed per (galaxy seed, index), independent of planet_radius_for's
// hash (different multiplier constant) so a planet's size and its terrain shape don't correlate
fn noise_seed_for(seed: u64, index: usize) -> u32 {
    let mut h = seed
        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
        .wrapping_add(index as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h & 0xFFFF_FFFF) as u32
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

const GALAXY_CRUISE_SPEED: f32 = 500.0;
const GALAXY_BOOST_SPEED: f32 = 2000.0;
const GALAXY_ACCEL: f32 = 800.0; // units/s^2

pub struct GalaxyFlight {
    pub position: DVec3,
    pub velocity: DVec3,
    pub rotation: Quat,
    mouse_sens: f32,
}

impl GalaxyFlight {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            mouse_sens: 0.002,
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        input: Vec3,
        jump: bool,
        down: bool,
        mouse_delta: (f32, f32),
        sprint: bool,
    ) {
        let yaw_delta = -mouse_delta.0 * self.mouse_sens;
        if yaw_delta.abs() > 1e-6 {
            self.rotation = Quat::from_axis_angle(Vec3::Y, yaw_delta) * self.rotation;
        }
        let pitch_delta = -mouse_delta.1 * self.mouse_sens;
        if pitch_delta.abs() > 1e-6 {
            self.rotation *= Quat::from_axis_angle(Vec3::X, pitch_delta);
        }

        let forward = if input.length() > 0.01 {
            self.rotation * input.normalize()
        } else {
            Vec3::ZERO
        };
        let dir = compose_fly_direction(forward, jump, down);

        let target_speed = if sprint {
            GALAXY_BOOST_SPEED
        } else {
            GALAXY_CRUISE_SPEED
        };
        let target_velocity = dir * target_speed;

        let vel_f32 = Vec3::new(
            self.velocity.x as f32,
            self.velocity.y as f32,
            self.velocity.z as f32,
        );
        let new_vel = accelerate_toward(vel_f32, target_velocity, GALAXY_ACCEL * dt);
        self.velocity = DVec3::new(new_vel.x as f64, new_vel.y as f64, new_vel.z as f64);

        self.position += self.velocity * dt as f64;
    }
}

// Pure: blends current velocity toward target, clamped by max acceleration this tick. Factored out
// so the clamp is regression-tested without mouse/keyboard state.
fn accelerate_toward(current: Vec3, target: Vec3, max_delta: f32) -> Vec3 {
    current + (target - current).clamp_length_max(max_delta)
}

// Pure: composes the desired flight direction from forward-thrust (already rotated into world
// space by the caller) and vertical climb/descend keys, normalized. Factored out for testability.
fn compose_fly_direction(forward: Vec3, climb: bool, descend: bool) -> Vec3 {
    let mut dir = forward;
    if climb {
        dir += Vec3::Y;
    }
    if descend {
        dir -= Vec3::Y;
    }
    if dir.length() > 0.01 {
        dir.normalize()
    } else {
        Vec3::ZERO
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
            assert_eq!(pa.noise_seed, pb.noise_seed);
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

    #[test]
    fn accelerate_toward_is_clamped_by_max_delta() {
        let result = accelerate_toward(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 5.0);
        assert_eq!(result, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn accelerate_toward_reaches_target_when_under_max_delta() {
        let result = accelerate_toward(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 50.0);
        assert_eq!(result, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn accelerate_toward_does_not_move_when_already_at_target() {
        let target = Vec3::new(3.0, 4.0, 0.0);
        let result = accelerate_toward(target, target, 10.0);
        assert_eq!(result, target);
    }

    #[test]
    fn compose_fly_direction_climb_only_is_straight_up() {
        assert_eq!(compose_fly_direction(Vec3::ZERO, true, false), Vec3::Y);
    }

    #[test]
    fn compose_fly_direction_descend_only_is_straight_down() {
        assert_eq!(compose_fly_direction(Vec3::ZERO, false, true), -Vec3::Y);
    }

    #[test]
    fn compose_fly_direction_climb_and_descend_cancel_to_zero() {
        assert_eq!(compose_fly_direction(Vec3::ZERO, true, true), Vec3::ZERO);
    }

    #[test]
    fn compose_fly_direction_forward_only_is_normalized() {
        let result = compose_fly_direction(Vec3::new(2.0, 0.0, 0.0), false, false);
        assert!((result.length() - 1.0).abs() < 1e-6);
        assert_eq!(result, Vec3::X);
    }

    #[test]
    fn galaxy_flight_update_moves_position_in_input_direction() {
        let mut flight = GalaxyFlight::new(DVec3::ZERO);
        for _ in 0..120 {
            flight.update(
                1.0 / 60.0,
                Vec3::new(0.0, 0.0, -1.0),
                false,
                false,
                (0.0, 0.0),
                false,
            );
        }
        // forward is -Z at identity rotation (matches Player's convention); after 2s should have
        // moved forward a meaningful distance, and not drifted sideways or vertically
        assert!(
            flight.position.z < -10.0,
            "expected forward movement, got {:?}",
            flight.position
        );
        assert!((flight.position.x).abs() < 1e-6);
        assert!((flight.position.y).abs() < 1e-6);
    }

    #[test]
    fn noise_seeds_are_distinct_across_all_planets() {
        let g = Galaxy::generate(1);
        let mut seeds: Vec<u32> = g.planets.iter().map(|p| p.noise_seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(
            seeds.len(),
            g.planets.len(),
            "duplicate noise seeds would give identical terrain on different planets"
        );
    }

    #[test]
    fn noise_seed_is_deterministic_for_a_fixed_seed() {
        let a = Galaxy::generate(42);
        let b = Galaxy::generate(42);
        for (pa, pb) in a.planets.iter().zip(b.planets.iter()) {
            assert_eq!(pa.noise_seed, pb.noise_seed);
        }
    }

    #[test]
    fn noise_seed_differs_from_planet_radius_hash() {
        // sanity check that the two per-planet hashes aren't accidentally the same function
        // reused (which would correlate radius and terrain shape in a confusing way)
        let g = Galaxy::generate(1);
        let p = &g.planets[0];
        assert_ne!(
            p.noise_seed as f32, p.radius,
            "noise_seed and radius should come from independent hashes"
        );
    }
}
