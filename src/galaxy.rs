// galaxy.rs
// Sub-project #1 (galaxy foundation) of the planned multi-planet feature: a seeded single-star
// solar system of orbiting placeholder bodies, deliberately separate from the existing
// single-planet engine (PlanetData/CoordSystem/Player/Physics) — no voxel terrain, no collision,
// and no connection yet to the player's actual planet (that's a later sub-project, landing/liftoff).

use crate::biome::PlanetType;
use glam::{DQuat, DVec3, Quat, Vec3};

pub struct Star {
    pub radius: f64,
}

impl Star {
    // the star sits at the origin of galaxy space; kept as a method so callers don't hardcode it
    pub fn position(&self) -> DVec3 {
        DVec3::ZERO
    }
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
    // full MurmurHash3 fmix64 (two multiply-xorshift rounds): the single-round version this used
    // to be doesn't avalanche small sequential `index` values enough — an entire galaxy's planets
    // came out clustered within a few units of each other instead of spanning the intended range
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
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
    // full MurmurHash3 fmix64 — see planet_radius_for's comment; same weak-avalanche issue would
    // otherwise apply here too
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
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

    // The planet frame: origin at the planet centre, turned by the spin angle about +Y, the axis
    // the voxel engine already uses for day/night. Positions stored in this frame stay put on the
    // planet's surface while it orbits and spins. The spin adds the orbit rate on top of one turn per
    // day_length_secs, so the *solar* day (sun back at the same sky direction) is exactly the planet
    // type's day length — without it the orbit would stretch or shrink every planet's day.
    pub fn spin_angle(&self, t: f64) -> f64 {
        let day = self.planet_type.def().day_length_secs as f64;
        t * (std::f64::consts::TAU / day + self.orbit_speed)
    }

    fn spin(&self, t: f64) -> DQuat {
        DQuat::from_axis_angle(DVec3::Y, self.spin_angle(t) % std::f64::consts::TAU)
    }

    // f32 version for orientations (Quat); the angle is reduced first so f32 keeps its precision
    fn spin_f32(&self, t: f64) -> Quat {
        Quat::from_axis_angle(Vec3::Y, (self.spin_angle(t) % std::f64::consts::TAU) as f32)
    }

    pub fn to_planet_frame(&self, abs: DVec3, t: f64) -> DVec3 {
        self.spin(t) * (abs - self.position_at(t))
    }

    pub fn from_planet_frame(&self, local: DVec3, t: f64) -> DVec3 {
        self.position_at(t) + self.spin(t).inverse() * local
    }

    pub fn rotation_to_planet_frame(&self, abs_rot: Quat, t: f64) -> Quat {
        self.spin_f32(t) * abs_rot
    }

    pub fn rotation_from_planet_frame(&self, local_rot: Quat, t: f64) -> Quat {
        self.spin_f32(t).inverse() * local_rot
    }

    // planet frame -> galaxy space rotation: how the planet (and so its impostor mesh) is turned at t
    pub fn orientation(&self, t: f64) -> Quat {
        self.spin_f32(t).inverse()
    }

    // direction toward the star as seen in this planet's frame: what the voxel engine uses as its
    // sun on this planet. Turns about +Y once per day_length_secs (see spin_angle), the same rotation
    // sense as home_sun_dir; always in the planet's equatorial plane (spin axis ⟂ orbit plane).
    pub fn sun_dir_in_planet_frame(&self, star_pos: DVec3, t: f64) -> Vec3 {
        self.to_planet_frame(star_pos, t).normalize().as_vec3()
    }

    // voxel resolution whose planet radius (resolution / 2) matches this planet's radius
    pub fn voxel_resolution(&self) -> u32 {
        ((self.radius as f64 * 2.0).round() as u32).max(8)
    }

    // the real voxel planet for this galaxy planet: own seed, matching size, own planet type
    pub fn bake(&self) -> crate::common::PlanetData {
        let mut data =
            crate::common::PlanetData::new_seeded(self.voxel_resolution(), self.noise_seed);
        data.switch_planet_type(self.planet_type);
        data
    }
}

// how close flight must come to a planet's centre, in that planet's radii, to be captured into its
// frame (and how far it must leave again to be released)
pub const CAPTURE_RADII: f64 = 10.0;

// which frame galaxy flight lives in: galaxy space, or a planet's own frame (spec §1 "captured")
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlightFrame {
    Free,
    Captured(usize), // index into Galaxy::planets
}

// the frame flight should be in, given its position `pos` in its current frame (galaxy space when
// Free, the planet frame when Captured)
pub fn next_flight_frame(galaxy: &Galaxy, frame: FlightFrame, pos: DVec3, t: f64) -> FlightFrame {
    match frame {
        FlightFrame::Captured(i) => {
            if pos.length() > CAPTURE_RADII * galaxy.planets[i].radius as f64 {
                FlightFrame::Free
            } else {
                frame
            }
        }
        FlightFrame::Free => galaxy
            .planets
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (pos - p.position_at(t)).length() / p.radius as f64))
            .filter(|&(_, radii)| radii < CAPTURE_RADII)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(FlightFrame::Free, |(i, _)| FlightFrame::Captured(i)),
    }
}

// The home planet's sun (it isn't part of the galaxy, so there's no real star): a fixed direction
// rotated about +Y once per day. Moved here unchanged from Renderer::render, including the hour wrap.
pub fn home_sun_dir(t: f64, day_length_secs: f32) -> Vec3 {
    let time = (t % 3600.0) as f32;
    let spin_angle = (time / day_length_secs) * std::f32::consts::TAU;
    Quat::from_axis_angle(Vec3::Y, spin_angle) * Vec3::new(0.5, 0.2, 0.4).normalize()
}

// home planet frame -> sky frame: undoes the same daily turn home_sun_dir applies, so the sun (and
// with it the starfield backdrop) holds still in the sky's frame while the planet turns under it
pub fn home_sky_rotation(t: f64, day_length_secs: f32) -> Quat {
    let time = (t % 3600.0) as f32;
    let spin_angle = (time / day_length_secs) * std::f32::consts::TAU;
    Quat::from_axis_angle(Vec3::Y, -spin_angle)
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

    // re-expresses position, orientation and velocity in another frame, so capture and release
    // don't move the view. Velocity is only rotated: the planet's own orbital and spin motion is
    // neither added nor removed, so on capture the player simply starts co-moving with the planet.
    pub fn change_frame(&mut self, from: FlightFrame, to: FlightFrame, galaxy: &Galaxy, t: f64) {
        if from == to {
            return;
        }
        if let FlightFrame::Captured(i) = from {
            let p = &galaxy.planets[i];
            self.position = p.from_planet_frame(self.position, t);
            self.rotation = p.rotation_from_planet_frame(self.rotation, t);
            // a direction, not a point: rotate without the planet's offset
            self.velocity = p.from_planet_frame(self.velocity, t) - p.position_at(t);
        }
        if let FlightFrame::Captured(i) = to {
            let p = &galaxy.planets[i];
            self.position = p.to_planet_frame(self.position, t);
            self.rotation = p.rotation_to_planet_frame(self.rotation, t);
            self.velocity = p.to_planet_frame(self.velocity + p.position_at(t), t);
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

// how fast roll eases toward level while captured by a planet, radians per second
pub const ROLL_LEVEL_RATE: f32 = 0.8;

// turns `rotation` about its own view axis by at most `max_angle`, toward the orientation whose
// right axis is horizontal for the given `up` (the planet camera has no roll). Leaves it alone when
// looking straight along `up`, where roll is undefined.
pub fn level_roll(rotation: Quat, up: Vec3, max_angle: f32) -> Quat {
    let forward = rotation * Vec3::NEG_Z;
    let level_right = forward.cross(up);
    if level_right.length_squared() < 1e-6 {
        return rotation;
    }
    let level_right = level_right.normalize();
    let right = rotation * Vec3::X;
    let angle = right
        .cross(level_right)
        .dot(forward)
        .atan2(right.dot(level_right));
    let step = angle.clamp(-max_angle, max_angle);
    (Quat::from_axis_angle(forward, step) * rotation).normalize()
}

// Where a target sits relative to the camera, for the galaxy-mode compass overlay: yaw is the
// left/right angle from the view direction (positive = right, ±180 = straight behind), pitch the
// up/down angle (positive = above). Both in degrees, camera-relative, so they follow mouse-look.
#[derive(Clone, Copy, Debug)]
pub struct CompassBearing {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

pub fn compass_bearing(rotation: Quat, from: DVec3, to: DVec3) -> CompassBearing {
    let world = (to - from).as_vec3();
    let local = rotation.inverse() * world; // camera looks down -Z
    let horizontal = (local.x * local.x + local.z * local.z).sqrt();
    CompassBearing {
        yaw_deg: local.x.atan2(-local.z).to_degrees(),
        pitch_deg: local.y.atan2(horizontal).to_degrees(),
    }
}

// Compact, non-technical distance label for the compass: "850", "12.3k", "250k", "3.4M".
pub fn format_distance(d: f64) -> String {
    let d = d.max(0.0);
    let short = |v: f64, suffix: &str| {
        if v < 100.0 {
            format!("{v:.1}{suffix}")
        } else {
            format!("{v:.0}{suffix}")
        }
    };
    if d < 1_000.0 {
        format!("{d:.0}")
    } else if d < 1_000_000.0 {
        short(d / 1_000.0, "k")
    } else {
        short(d / 1_000_000.0, "M")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roll_of(rotation: Quat, up: Vec3) -> f32 {
        // how far the camera's right axis tips out of the horizontal plane
        (rotation * Vec3::X).dot(up).asin().abs()
    }

    #[test]
    fn level_roll_converges_to_level_and_keeps_the_view_direction() {
        let up = Vec3::Y;
        let mut rot =
            Quat::from_axis_angle(Vec3::NEG_Z, 0.5) * Quat::from_axis_angle(Vec3::X, -0.2);
        let forward = rot * Vec3::NEG_Z;
        assert!(roll_of(rot, up) > 0.3);
        for _ in 0..10 {
            rot = level_roll(rot, up, 0.1);
        }
        assert!(
            roll_of(rot, up) < 1e-3,
            "still rolled: {}",
            roll_of(rot, up)
        );
        assert!(
            ((rot * Vec3::NEG_Z) - forward).length() < 1e-4,
            "view direction moved"
        );
    }

    #[test]
    fn level_roll_turns_at_most_max_angle_per_step() {
        let up = Vec3::Y;
        let rot = Quat::from_axis_angle(Vec3::NEG_Z, 0.5);
        let stepped = level_roll(rot, up, 0.1);
        assert!((rot.angle_between(stepped) - 0.1).abs() < 1e-4);
    }

    // looking straight along `up` (at the planet's centre, or straight away) roll is undefined:
    // leave it alone rather than spin
    #[test]
    fn level_roll_leaves_a_straight_up_or_down_view_alone() {
        let up = Vec3::Y;
        for rot in [
            Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::Y),
            Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::NEG_Y)
                * Quat::from_axis_angle(Vec3::NEG_Z, 0.7),
        ] {
            assert!(level_roll(rot, up, 0.1).dot(rot).abs() > 1.0 - 1e-6);
        }
    }

    #[test]
    fn capture_inside_ten_radii_and_release_outside() {
        let g = Galaxy::generate(1);
        let p = &g.planets[0];
        let t = 12.0;
        let near = p.position_at(t) + DVec3::new(9.0 * p.radius as f64, 0.0, 0.0);
        let far = p.position_at(t) + DVec3::new(11.0 * p.radius as f64, 0.0, 0.0);
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Free, near, t),
            FlightFrame::Captured(0)
        );
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Free, far, t),
            FlightFrame::Free
        );
        // captured: the position is planet-frame, so its length is the distance to the centre
        let inside = DVec3::new(0.0, 9.5 * p.radius as f64, 0.0);
        let outside = DVec3::new(0.0, 10.5 * p.radius as f64, 0.0);
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Captured(0), inside, t),
            FlightFrame::Captured(0)
        );
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Captured(0), outside, t),
            FlightFrame::Free
        );
    }

    #[test]
    fn capture_picks_the_planet_you_are_near() {
        let g = Galaxy::generate(1);
        let t = 300.0;
        let p = &g.planets[4];
        let pos = p.position_at(t) + DVec3::new(0.0, 3.0 * p.radius as f64, 0.0);
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Free, pos, t),
            FlightFrame::Captured(4)
        );
    }

    // capture/release must not move the view: the same galaxy-space position and orientation before
    // and after, whichever way round (the flicker case at exactly 10 radii relies on this)
    #[test]
    fn change_frame_keeps_the_absolute_view() {
        let g = Galaxy::generate(1);
        let p = &g.planets[0];
        let t = 77.0;
        let mut flight = GalaxyFlight::new(p.position_at(t) + DVec3::new(500.0, 200.0, -300.0));
        flight.rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.4, -0.2, 0.3);
        flight.velocity = DVec3::new(10.0, -5.0, 2.0);
        let (abs_pos, abs_rot, abs_vel) = (flight.position, flight.rotation, flight.velocity);

        flight.change_frame(FlightFrame::Free, FlightFrame::Captured(0), &g, t);
        assert!((p.from_planet_frame(flight.position, t) - abs_pos).length() < 1e-6);
        assert!(
            p.rotation_from_planet_frame(flight.rotation, t)
                .dot(abs_rot)
                .abs()
                > 1.0 - 1e-5
        );
        assert!(
            (flight.velocity.length() - abs_vel.length()).abs() < 1e-9,
            "velocity only rotated"
        );

        flight.change_frame(FlightFrame::Captured(0), FlightFrame::Free, &g, t);
        assert!((flight.position - abs_pos).length() < 1e-6);
        assert!(flight.rotation.dot(abs_rot).abs() > 1.0 - 1e-5);
        assert!((flight.velocity - abs_vel).length() < 1e-9);
    }

    #[test]
    fn orientation_maps_the_planet_frame_into_galaxy_space() {
        let p = Galaxy::generate(1).planets[2];
        let local = Vec3::new(30.0, -40.0, 120.0);
        for t in [0.0, 55.0, 9_000.0] {
            let rotated = (p.orientation(t) * local).as_dvec3();
            let expected = p.from_planet_frame(local.as_dvec3(), t) - p.position_at(t);
            assert!(
                (rotated - expected).length() < 1e-3,
                "t={t}: {rotated:?} vs {expected:?}"
            );
        }
    }

    // the home planet's sky must turn with its sun: the sun, taken from the planet frame into the
    // sky's frame, stays put — so stars keep their place relative to the sun through the day
    #[test]
    fn home_sky_rotation_holds_the_sun_still() {
        let day = 120.0;
        let base = Vec3::new(0.5, 0.2, 0.4).normalize();
        for t in [0.0, 17.0, 61.5, 3_600.0 + 17.0] {
            let sun_in_sky = home_sky_rotation(t, day) * home_sun_dir(t, day);
            assert!((sun_in_sky - base).length() < 1e-5, "t={t}: {sun_in_sky:?}");
        }
    }

    fn inner_planet() -> GalaxyPlanet {
        Galaxy::generate(1).planets[0]
    }

    #[test]
    fn planet_frame_round_trips_positions() {
        let p = inner_planet();
        let abs = DVec3::new(9_000.0, 120.0, -300.0);
        for t in [0.0, 13.7, 1_234.5, 100_000.0] {
            let back = p.from_planet_frame(p.to_planet_frame(abs, t), t);
            assert!((back - abs).length() < 1e-6, "t={t}: {back:?} != {abs:?}");
        }
    }

    #[test]
    fn planet_centre_is_the_planet_frame_origin() {
        let p = inner_planet();
        for t in [0.0, 42.0, 9_999.0] {
            assert!(p.to_planet_frame(p.position_at(t), t).length() < 1e-6);
        }
    }

    // something parked in the planet frame (a player hovering over the surface) is carried along by
    // both the orbit and the spin, always at the same distance from the moving planet centre
    #[test]
    fn a_point_at_rest_in_the_planet_frame_follows_orbit_and_spin() {
        let p = inner_planet();
        let local = DVec3::new(500.0, 0.0, 0.0);
        let a = p.from_planet_frame(local, 10.0);
        let b = p.from_planet_frame(local, 40.0);
        assert!(((a - p.position_at(10.0)).length() - 500.0).abs() < 1e-6);
        assert!(((b - p.position_at(40.0)).length() - 500.0).abs() < 1e-6);
        // the offset from the centre has turned (spin), not just been translated (orbit)
        let off_a = (a - p.position_at(10.0)).normalize();
        let off_b = (b - p.position_at(40.0)).normalize();
        assert!(
            off_a.dot(off_b) < 0.99,
            "offset didn't rotate: {off_a:?} vs {off_b:?}"
        );
    }

    #[test]
    fn planet_frame_round_trips_orientations() {
        let p = inner_planet();
        let rot = Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.3, 0.1);
        for t in [0.0, 77.0, 50_000.0] {
            let back = p.rotation_from_planet_frame(p.rotation_to_planet_frame(rot, t), t);
            assert!(
                back.dot(rot).abs() > 1.0 - 1e-5,
                "t={t}: {back:?} != {rot:?}"
            );
        }
    }

    #[test]
    fn sun_points_at_the_star() {
        let g = Galaxy::generate(1);
        for p in &g.planets {
            for t in [0.0, 33.0, 4_321.0] {
                let sun = p.sun_dir_in_planet_frame(g.star.position(), t);
                let abs_dir = p.from_planet_frame(sun.as_dvec3(), t) - p.position_at(t);
                let to_star = (g.star.position() - p.position_at(t)).normalize();
                assert!(
                    (abs_dir.normalize() - to_star).length() < 1e-5,
                    "t={t}: sun {abs_dir:?} vs star {to_star:?}"
                );
            }
        }
    }

    // the spin includes the orbit rate, so a day lasts exactly the planet type's day length (the
    // same as on the home planet) despite the planet also moving around the star
    #[test]
    fn solar_day_equals_the_planet_types_day_length() {
        let g = Galaxy::generate(1);
        for p in &g.planets {
            let day = p.planet_type.def().day_length_secs as f64;
            let morning = p.sun_dir_in_planet_frame(g.star.position(), 5.0);
            let next_morning = p.sun_dir_in_planet_frame(g.star.position(), 5.0 + day);
            let evening = p.sun_dir_in_planet_frame(g.star.position(), 5.0 + day / 2.0);
            assert!(
                (morning - next_morning).length() < 1e-3,
                "{morning:?} vs {next_morning:?}"
            );
            assert!(
                morning.dot(evening) < -0.99,
                "half a day later the sun should be opposite"
            );
        }
    }

    #[test]
    fn voxel_resolution_is_twice_the_radius() {
        let mut p = inner_planet();
        p.radius = 168.48;
        assert_eq!(p.voxel_resolution(), 337);
        p.radius = 40.0;
        assert_eq!(p.voxel_resolution(), 80);
        p.radius = 2.0;
        assert_eq!(p.voxel_resolution(), 8); // PlanetData::resize's floor
    }

    #[test]
    fn bake_uses_the_planets_own_seed_resolution_and_type() {
        let mut p = Galaxy::generate(1).planets[1]; // Volcanic
        p.radius = 40.0; // smallest size: keeps the bake quick in a debug test build
        let data = p.bake();
        assert_eq!(data.resolution, 80);
        assert_eq!(data.seed, p.noise_seed);
        assert_eq!(data.planet_type, p.planet_type);
    }

    #[test]
    fn home_sun_starts_at_the_original_direction_and_turns_about_y() {
        let day = 120.0;
        let start = home_sun_dir(0.0, day);
        assert!((start - Vec3::new(0.5, 0.2, 0.4).normalize()).length() < 1e-6);
        let quarter = home_sun_dir(30.0, day);
        assert!(
            (quarter.y - start.y).abs() < 1e-6,
            "elevation must not change"
        );
        let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z).normalize();
        assert!(
            flat(start).dot(flat(quarter)).abs() < 1e-5,
            "a quarter day turns 90 degrees"
        );
    }

    #[test]
    fn home_sun_repeats_after_the_hour_wrap() {
        for day in [60.0, 120.0, 240.0] {
            let a = home_sun_dir(17.3, day);
            let b = home_sun_dir(3_600.0 + 17.3, day);
            assert!((a - b).length() < 1e-4, "day {day}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn compass_bearing_straight_ahead_is_zero() {
        let b = compass_bearing(Quat::IDENTITY, DVec3::ZERO, DVec3::new(0.0, 0.0, -100.0));
        assert!(b.yaw_deg.abs() < 1e-3 && b.pitch_deg.abs() < 1e-3, "{b:?}");
    }

    #[test]
    fn compass_bearing_right_left_behind_above() {
        let at = |to: DVec3| compass_bearing(Quat::IDENTITY, DVec3::ZERO, to);
        assert!((at(DVec3::new(100.0, 0.0, 0.0)).yaw_deg - 90.0).abs() < 1e-3);
        assert!((at(DVec3::new(-100.0, 0.0, 0.0)).yaw_deg + 90.0).abs() < 1e-3);
        assert!((at(DVec3::new(0.0, 0.0, 100.0)).yaw_deg.abs() - 180.0).abs() < 1e-3);
        assert!((at(DVec3::new(0.0, 100.0, -100.0)).pitch_deg - 45.0).abs() < 1e-3);
    }

    #[test]
    fn compass_bearing_follows_camera_rotation() {
        // turned 90 degrees left (yaw about +Y): a target straight ahead in world space is now to the right
        let rot = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let b = compass_bearing(rot, DVec3::ZERO, DVec3::new(0.0, 0.0, -100.0));
        assert!((b.yaw_deg - 90.0).abs() < 1e-3, "{b:?}");
    }

    #[test]
    fn format_distance_is_short_and_readable() {
        assert_eq!(format_distance(0.0), "0");
        assert_eq!(format_distance(850.4), "850");
        assert_eq!(format_distance(12_345.0), "12.3k");
        assert_eq!(format_distance(250_000.0), "250k");
        assert_eq!(format_distance(3_400_000.0), "3.4M");
    }

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

    // regression test for a real bug: the hash's finalizer was missing a round and didn't
    // avalanche small sequential `index` values enough, so an entire galaxy's planets came out
    // clustered within a few units of each other (observed: 160-164 across all 7 planets) instead
    // of spanning anywhere near the intended 40-250 range. "in range" alone doesn't catch this —
    // a tightly clustered set of values is still technically in range.
    #[test]
    fn generate_planet_radii_are_well_spread_not_clustered() {
        for seed in [1, 2, 7, 42] {
            let g = Galaxy::generate(seed);
            let radii: Vec<f32> = g.planets.iter().map(|p| p.radius).collect();
            let min = radii.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = radii.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max - min > 50.0,
                "seed {seed}: radii spread only {:.1} ({radii:?}) — looks clustered, not well distributed",
                max - min
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
