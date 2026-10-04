// galaxy.rs
// Sub-project #1 (galaxy foundation) of the planned multi-planet feature: a seeded single-star
// solar system of orbiting placeholder bodies, deliberately separate from the existing
// single-planet engine (PlanetData/CoordSystem/Player/Physics) — no voxel terrain, no collision,
// and no connection yet to the player's actual planet (that's a later sub-project, landing/liftoff).

use crate::biome::PlanetType;
use glam::{DQuat, DVec3, Quat, Vec3};

pub struct Star {
    pub radius: f64,
    pub star_type: StarType,
}

// the galaxy's star: one of five types, picked from the galaxy seed (star_type_for); sets the star's
// look and size and the sunlight colour on its planets (spec 2026-10-04 star rendering)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StarType {
    RedDwarf,
    Orange,
    Yellow,
    White,
    BlueGiant,
}

pub struct StarTypeDef {
    pub name: &'static str,
    pub surface_color: [f32; 3], // disc centre, linear, before tone mapping
    pub limb_color: [f32; 3],    // toward the edge (limb darkening shifts warmer)
    pub radius: f64,
    pub granulation: f32,   // surface noise contrast 0..1
    pub granule_scale: f32, // noise frequency over the unit sphere
    pub sunspots: f32,      // share of the surface covered by spots 0..1
    pub corona_color: [f32; 3],
    pub corona_size: f32,   // glow falloff length in star radii
    pub sunlight: [f32; 3], // the planets' sunlight (Atmosphere.sun_color)
}

const STAR_TYPES: [StarTypeDef; 5] = [
    StarTypeDef {
        name: "Red dwarf",
        surface_color: [1.10, 0.45, 0.23],
        limb_color: [0.83, 0.21, 0.07],
        radius: 1_800.0,
        granulation: 0.35,
        granule_scale: 14.0,
        sunspots: 0.12,
        corona_color: [1.0, 0.35, 0.15],
        corona_size: 0.35,
        sunlight: [1.72, 1.45, 1.12],
    },
    StarTypeDef {
        name: "Orange",
        surface_color: [1.10, 0.68, 0.35],
        limb_color: [0.83, 0.33, 0.12],
        radius: 2_400.0,
        granulation: 0.3,
        granule_scale: 16.0,
        sunspots: 0.08,
        corona_color: [1.0, 0.55, 0.25],
        corona_size: 0.4,
        sunlight: [1.68, 1.48, 1.18],
    },
    StarTypeDef {
        name: "Yellow",
        surface_color: [1.10, 0.96, 0.69],
        limb_color: [0.83, 0.52, 0.25],
        radius: 3_000.0,
        granulation: 0.25,
        granule_scale: 18.0,
        sunspots: 0.05,
        corona_color: [1.0, 0.8, 0.5],
        corona_size: 0.45,
        sunlight: [1.6, 1.5, 1.3],
    },
    StarTypeDef {
        name: "White",
        surface_color: [1.07, 1.07, 1.10],
        limb_color: [0.82, 0.76, 0.67],
        radius: 3_300.0,
        granulation: 0.2,
        granule_scale: 20.0,
        sunspots: 0.03,
        corona_color: [0.85, 0.9, 1.0],
        corona_size: 0.5,
        sunlight: [1.52, 1.52, 1.48],
    },
    StarTypeDef {
        name: "Blue giant",
        surface_color: [0.72, 0.84, 1.10],
        limb_color: [0.52, 0.59, 0.83],
        radius: 3_600.0,
        granulation: 0.15,
        granule_scale: 22.0,
        sunspots: 0.0,
        corona_color: [0.55, 0.7, 1.0],
        corona_size: 0.6,
        sunlight: [1.38, 1.5, 1.68],
    },
];

impl StarType {
    pub const ALL: [StarType; 5] = [
        StarType::RedDwarf,
        StarType::Orange,
        StarType::Yellow,
        StarType::White,
        StarType::BlueGiant,
    ];

    pub fn def(self) -> &'static StarTypeDef {
        &STAR_TYPES[self as usize]
    }
}

// the star type of the galaxy with this seed: a bit-mixing hash (fmix64, as planet_radius_for) of the
// seed alone, independent of the planets' hashes
pub fn star_type_for(seed: u64) -> StarType {
    let mut h = seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ 0x5DEE_CE66_D1CE_4E5B;
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    StarType::ALL[(h % StarType::ALL.len() as u64) as usize]
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
    pub orbit_tilt: f64, // inclination out of the common orbital plane, radians (0..MAX_ORBIT_TILT)
    pub orbit_node: f64, // direction of the tilt axis within the plane, radians
    pub radius: f32,
    pub planet_type: PlanetType,
    pub noise_seed: u32, // seeds this planet's TerrainShape/NoiseGenerator (src/galaxy_terrain.rs);
                         // independent of planet_radius_for's hash so radius and terrain shape don't correlate
}

pub struct Galaxy {
    pub star: Star,
    pub planets: Vec<GalaxyPlanet>,
    pub seed: u64, // generation seed: planets added later (/galaxy add) are seeded from it too
}

const PLANET_COUNT: usize = 7;
const INNER_ORBIT_RADIUS: f64 = 8_000.0;
const ORBIT_RADIUS_GROWTH: f64 = 1.5; // each planet this many times farther out than the last
const INNER_ORBIT_PERIOD_SECS: f64 = 180.0; // at INNER_ORBIT_RADIUS; Kepler beyond (orbit_speed_for)
                                            // seeded orbit variety (orbit_for): each radius x0.95..x1.10 around the spacing (neighbours keep a gap of
                                            // at least x1.29, the innermost stays at INNER_ORBIT_RADIUS or more, outside every star's heat zone),
                                            // a random starting angle and a small tilt out of the common plane
const ORBIT_JITTER_MIN: f64 = 0.95;
const ORBIT_JITTER_MAX: f64 = 1.10;
const MAX_ORBIT_TILT: f64 = 4.0 * std::f64::consts::PI / 180.0;
const MIN_PLANET_RADIUS: f32 = 40.0;
const MAX_PLANET_RADIUS: f32 = 250.0;
// most planets a galaxy can hold (generated + /galaxy add): the galaxy renderer sizes its per-planet
// uniform buffer for this many
pub const MAX_PLANETS: usize = 15;
// /galaxy add accepts radii in this range (bake time grows with the square of the radius)
pub const MIN_ADDED_RADIUS: f32 = 20.0;
pub const MAX_ADDED_RADIUS: f32 = 500.0;

impl Galaxy {
    pub fn generate(seed: u64) -> Self {
        let planets = (0..PLANET_COUNT)
            .map(|i| {
                let (orbit_radius, orbit_phase, orbit_tilt, orbit_node) = orbit_for(seed, i);
                GalaxyPlanet {
                    orbit_radius,
                    orbit_speed: orbit_speed_for(orbit_radius),
                    orbit_phase,
                    orbit_tilt,
                    orbit_node,
                    radius: planet_radius_for(seed, i),
                    planet_type: PlanetType::ALL[i % PlanetType::ALL.len()],
                    noise_seed: noise_seed_for(seed, i),
                }
            })
            .collect();
        let star_type = star_type_for(seed);
        Self {
            star: Star {
                radius: star_type.def().radius,
                star_type,
            },
            planets,
            seed,
        }
    }

    // the planet a new game starts on: the first one of the preferred type (--biome), else #1
    pub fn start_planet_index(&self, preferred: PlanetType) -> usize {
        self.planets
            .iter()
            .position(|p| p.planet_type == preferred)
            .unwrap_or(0)
    }

    // debug (/galaxy add): appends a planet of the given type and radius on the next orbit outward,
    // shaped like the generated ones (same orbit progression, phase spread and seeded terrain);
    // returns its index
    pub fn add_planet(
        &mut self,
        planet_type: PlanetType,
        radius: f32,
    ) -> Result<usize, &'static str> {
        if self.planets.len() >= MAX_PLANETS {
            return Err("The galaxy is full (15 planets).");
        }
        if !(MIN_ADDED_RADIUS..=MAX_ADDED_RADIUS).contains(&radius) {
            return Err("Planet radius must be 20-500.");
        }
        let i = self.planets.len();
        let (orbit_radius, orbit_phase, orbit_tilt, orbit_node) = orbit_for(self.seed, i);
        self.planets.push(GalaxyPlanet {
            orbit_radius,
            orbit_speed: orbit_speed_for(orbit_radius),
            orbit_phase,
            orbit_tilt,
            orbit_node,
            radius,
            planet_type,
            noise_seed: noise_seed_for(self.seed, i),
        });
        Ok(i)
    }
}

// angular speed (radians/s) on a circular orbit: Kepler's third law (period ~ radius^1.5), one orbit in
// INNER_ORBIT_PERIOD_SECS at INNER_ORBIT_RADIUS
fn orbit_speed_for(orbit_radius: f64) -> f64 {
    std::f64::consts::TAU / INNER_ORBIT_PERIOD_SECS * (INNER_ORBIT_RADIUS / orbit_radius).powf(1.5)
}

// a deterministic value in 0..1 for (galaxy seed, planet index, `salt` naming what it's for): fmix64 as
// planet_radius_for, with a salt per property so they don't correlate
fn seeded_unit(seed: u64, index: usize, salt: u64) -> f64 {
    let mut h = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(index as u64)
        ^ salt.wrapping_mul(0xA24B_AED4_963E_E407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

// planet `index`'s orbit in the galaxy with `seed`: (radius, starting angle, tilt, tilt axis direction)
fn orbit_for(seed: u64, index: usize) -> (f64, f64, f64, f64) {
    let base = INNER_ORBIT_RADIUS * ORBIT_RADIUS_GROWTH.powi(index as i32);
    let jitter =
        ORBIT_JITTER_MIN + (ORBIT_JITTER_MAX - ORBIT_JITTER_MIN) * seeded_unit(seed, index, 1);
    let mut radius = base * jitter;
    if index == 0 {
        radius = radius.max(INNER_ORBIT_RADIUS);
    }
    let tau = std::f64::consts::TAU;
    (
        radius,
        tau * seeded_unit(seed, index, 2),
        MAX_ORBIT_TILT * seeded_unit(seed, index, 3),
        tau * seeded_unit(seed, index, 4),
    )
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
    // the star's angular radius seen from this planet (radians): its apparent size in the planet's sky
    pub fn star_angular_radius(&self, star: &Star) -> f32 {
        (star.radius / self.orbit_radius).min(1.0).asin() as f32
    }

    // on a circle of orbit_radius around the star, tilted by orbit_tilt about a horizontal axis
    // (direction orbit_node) out of the common orbital plane (XZ)
    pub fn position_at(&self, t: f64) -> DVec3 {
        let angle = self.orbit_phase + self.orbit_speed * t;
        let flat = DVec3::new(
            self.orbit_radius * angle.cos(),
            0.0,
            self.orbit_radius * angle.sin(),
        );
        let axis = DVec3::new(self.orbit_node.cos(), 0.0, self.orbit_node.sin());
        DQuat::from_axis_angle(axis, self.orbit_tilt) * flat
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

    // bake() plus the edits kept from an earlier visit (refused if they belong to another terrain)
    pub fn bake_with_edits(
        &self,
        edits: Option<crate::common::PlanetEdits>,
    ) -> crate::common::PlanetData {
        let mut data = self.bake();
        if let Some(edits) = edits {
            data.restore_edits(edits);
        }
        data
    }
}

// how close flight must come to a planet's centre, in that planet's radii, to be captured into its
// frame, and how far it must leave again to be released — the gap is hysteresis, so hovering at the
// boundary can't flip-flop between co-moving with the planet and not
pub const CAPTURE_RADII: f64 = 10.0;
pub const RELEASE_RADII: f64 = 11.0;

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
            if pos.length() > RELEASE_RADII * galaxy.planets[i].radius as f64 {
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

pub const GALAXY_CRUISE_SPEED: f32 = 500.0;
pub const GALAXY_BOOST_SPEED: f32 = 2000.0;

pub struct GalaxyFlight {
    pub position: DVec3,
    pub velocity: DVec3,
    pub rotation: Quat,
    pub roll_rate: f32, // eased roll rate (A/D), rad/s, positive rolls left
    pub turn_rate: f32, // eased turn rate (Q/E), rad/s, positive turns left
    mouse_sens: f32,
}

impl GalaxyFlight {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            roll_rate: 0.0,
            turn_rate: 0.0,
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

    // `up`: the local radial up while captured by a planet (turning left/right and climbing with
    // Space / descending with Shift follow it, like the planet camera), None in free flight, where
    // space has no up and both follow the camera's own up instead
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        input: Vec3,
        jump: bool,
        down: bool,
        mouse_delta: (f32, f32),
        sprint: bool,
        keys: crate::controller::FlightKeys, // A/D roll, Q/E turn: eased rates (smoothing.rs), pure roll
        up: Option<Vec3>,
    ) {
        use crate::smoothing::{approach, FLIGHT_TURN_SPEED, ROLL_ACCEL, ROLL_SPEED, TURN_ACCEL};
        self.roll_rate = approach(self.roll_rate, keys.roll * ROLL_SPEED, ROLL_ACCEL * dt);
        self.turn_rate = approach(
            self.turn_rate,
            keys.turn * FLIGHT_TURN_SPEED,
            TURN_ACCEL * dt,
        );
        if self.roll_rate != 0.0 {
            let roll = Quat::from_axis_angle(Vec3::Z, self.roll_rate * dt);
            self.rotation = (self.rotation * roll).normalize();
        }
        let yaw_delta = -mouse_delta.0 * self.mouse_sens + self.turn_rate * dt;
        if yaw_delta.abs() > 1e-6 {
            self.rotation =
                Quat::from_axis_angle(yaw_axis(self.rotation, up), yaw_delta) * self.rotation;
        }
        let up = up.unwrap_or(self.rotation * Vec3::Y);
        let pitch_delta = -mouse_delta.1 * self.mouse_sens;
        if pitch_delta.abs() > 1e-6 {
            self.rotation *= Quat::from_axis_angle(Vec3::X, pitch_delta);
        }

        let forward = if input.length() > 0.01 {
            self.rotation * input.normalize()
        } else {
            Vec3::ZERO
        };
        let dir = compose_fly_direction(forward, jump, down, up);

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
        // eased like planet fly mode, so cruise and boost respond alike (no long glide from boost)
        let new_vel = crate::smoothing::ease_toward(
            vel_f32,
            target_velocity,
            dt,
            crate::smoothing::FLIGHT_EASE_SECONDS,
        );
        self.velocity = DVec3::new(new_vel.x as f64, new_vel.y as f64, new_vel.z as f64);

        self.position += self.velocity * dt as f64;
    }
}

// the axis sideways mouse turns the view about. It must agree with the camera's own up, or the turn
// comes out as a pitch (camera on its side) or mirrored (upside down). Free flight: the camera's up.
// Captured: the planet's local up, like the planet camera, once roll is about level; while it's
// rolled (Q/E, or captured with any roll) it blends toward the camera's
// up, and on the side of the local up the camera's up is on, so a turn is never inverted.
fn yaw_axis(rotation: Quat, up: Option<Vec3>) -> Vec3 {
    let camera_up = rotation * Vec3::Y;
    let Some(up) = up else {
        return camera_up;
    };
    let horizon_up = if camera_up.dot(up) >= 0.0 { up } else { -up };
    // how far the camera's right axis tips out of the horizontal plane: 0 level, 1 on its side
    let tilt = (rotation * Vec3::X).dot(up).abs();
    let t = ((tilt - 0.3) / 0.5).clamp(0.0, 1.0);
    let rolled = t * t * (3.0 - 2.0 * t); // smoothstep
    horizon_up.lerp(camera_up, rolled).normalize()
}

// Pure: composes the desired flight direction from forward-thrust (already rotated into world
// space by the caller) and vertical climb/descend keys, normalized. Factored out for testability.
fn compose_fly_direction(forward: Vec3, climb: bool, descend: bool, up: Vec3) -> Vec3 {
    let mut dir = forward;
    if climb {
        dir += up;
    }
    if descend {
        dir -= up;
    }
    if dir.length() > 0.01 {
        dir.normalize()
    } else {
        Vec3::ZERO
    }
}

// the star's heat zone: from HEAT_ZONE_RADII star radii inward the heat rises (screen glow, HUD
// warning, an outward push), full at HEAT_LIMIT_RADII, which flight can't pass. The innermost planet
// orbits outside the zone.
pub const HEAT_ZONE_RADII: f64 = 2.0;
pub const HEAT_LIMIT_RADII: f64 = 1.15;
// outward push at full heat: eased flight pulls toward boost speed at up to ~8000 units/s^2
// (GALAXY_BOOST_SPEED / FLIGHT_EASE_SECONDS), so this always wins near the limit
const HEAT_PUSH_ACCEL: f64 = 12_000.0;

// heat 0..1 at `distance` from the star's centre (smooth: 0 at the zone's edge, 1 at the limit)
pub fn star_heat(distance: f64, star_radius: f64) -> f32 {
    let t = ((HEAT_ZONE_RADII * star_radius - distance)
        / ((HEAT_ZONE_RADII - HEAT_LIMIT_RADII) * star_radius))
        .clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)) as f32
}

// the velocity after this step's outward heat push (galaxy space, the star at the origin)
pub fn star_heat_push(velocity: DVec3, position: DVec3, star_radius: f64, dt: f32) -> DVec3 {
    let heat = star_heat(position.length(), star_radius) as f64;
    if heat <= 0.0 {
        return velocity;
    }
    velocity + position.normalize_or_zero() * heat * HEAT_PUSH_ACCEL * dt as f64
}

// flight can't get closer to the star's centre than HEAT_LIMIT_RADII star radii
pub fn keep_off_star(position: DVec3, star_radius: f64) -> DVec3 {
    let min = HEAT_LIMIT_RADII * star_radius;
    let dist = position.length();
    if dist < min && dist > 1e-9 {
        position * (min / dist)
    } else {
        position
    }
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
    use crate::controller::FlightKeys;

    // a bake with stored edits contains them; one without doesn't
    #[test]
    fn bake_with_edits_restores_them() {
        let galaxy = Galaxy::generate(1);
        let planet = &galaxy.planets[0];
        let mut first = planet.bake();
        let (face, u, v) = (0u8, 40u32, 40u32);
        let top = first.terrain.get_height(face, u, v) + 1; // one block on the surface: under the ceiling
        let block = crate::common::BlockId {
            face,
            layer: top,
            u,
            v,
        };
        first
            .add_block(block, crate::material::BlockType::Stone)
            .unwrap();
        let edits = first.take_edits();
        assert!(planet.bake_with_edits(Some(edits)).exists(block));
        assert!(!planet.bake_with_edits(None).exists(block));
    }

    #[test]
    fn start_planet_is_the_first_of_the_preferred_type() {
        let mut g = Galaxy::generate(1);
        assert_eq!(g.start_planet_index(PlanetType::EarthLike), 0);
        assert_eq!(g.start_planet_index(PlanetType::Volcanic), 1);
        assert_eq!(g.start_planet_index(PlanetType::Ice), 2);
        g.planets.retain(|p| p.planet_type != PlanetType::Ice);
        assert_eq!(
            g.start_planet_index(PlanetType::Ice),
            0,
            "no planet of that type: #1"
        );
    }

    #[test]
    fn added_planet_continues_the_orbit_progression_and_is_deterministic() {
        let mut a = Galaxy::generate(1);
        let i = a.add_planet(PlanetType::Ice, 120.0).unwrap();
        assert_eq!(i, 7);
        let p = a.planets[7];
        let base = INNER_ORBIT_RADIUS * ORBIT_RADIUS_GROWTH.powi(7);
        assert!(p.orbit_radius >= 0.9 * base && p.orbit_radius <= 1.15 * base);
        assert!(
            p.orbit_radius > a.planets[6].orbit_radius * 1.25,
            "inside the last orbit"
        );
        assert_eq!(p.planet_type, PlanetType::Ice);
        assert_eq!(p.radius, 120.0);
        assert_eq!(p.noise_seed, noise_seed_for(1, 7));
        let mut b = Galaxy::generate(1);
        b.add_planet(PlanetType::Ice, 120.0).unwrap();
        assert_eq!(a.planets[7], b.planets[7]);
    }

    #[test]
    fn adding_beyond_the_cap_or_out_of_range_fails() {
        let mut g = Galaxy::generate(1);
        assert!(g.add_planet(PlanetType::Ice, 10.0).is_err());
        assert!(g.add_planet(PlanetType::Ice, 600.0).is_err());
        while g.planets.len() < MAX_PLANETS {
            g.add_planet(PlanetType::EarthLike, 50.0).unwrap();
        }
        assert!(
            g.add_planet(PlanetType::EarthLike, 50.0).is_err(),
            "full at MAX_PLANETS"
        );
        assert_eq!(g.planets.len(), MAX_PLANETS);
    }

    // captured over a planet's equator (local up = +X here), levelled, looking along the horizon:
    // sideways mouse must turn the view around the local up — before, it turned around the spin
    // axis (Y), which there pitched the view instead, so the player couldn't turn left or right
    #[test]
    fn captured_yaw_turns_about_the_local_up() {
        let up = Vec3::X;
        // camera right = -Y, camera up = +X (radial), looking along -Z
        let rot = Quat::from_mat3(&glam::Mat3::from_cols(Vec3::NEG_Y, Vec3::X, Vec3::Z));
        let mut flight = GalaxyFlight::new(DVec3::new(600.0, 0.0, 0.0));
        flight.rotation = rot;
        let before = flight.rotation * Vec3::NEG_Z;
        flight.update(
            1.0 / 60.0,
            Vec3::ZERO,
            false,
            false,
            (80.0, 0.0),
            false,
            FlightKeys::default(),
            Some(up),
        );
        let after = flight.rotation * Vec3::NEG_Z;
        assert!(
            after.dot(up).abs() < 1e-4,
            "view tipped off the horizon: {after:?}"
        );
        assert!(before.dot(after) < 0.999, "view didn't turn");
    }

    // after mouse-right, how far the view turned toward where the camera's right was, and how far it
    // tipped toward the camera's up (a pure turn: first > 0, second ~ 0)
    fn turn_of(flight: &mut GalaxyFlight, up: Option<Vec3>) -> (f32, f32) {
        let (before, right, cam_up) = (
            flight.rotation * Vec3::NEG_Z,
            flight.rotation * Vec3::X,
            flight.rotation * Vec3::Y,
        );
        flight.update(
            1.0 / 60.0,
            Vec3::ZERO,
            false,
            false,
            (80.0, 0.0),
            false,
            FlightKeys::default(),
            up,
        );
        let after = flight.rotation * Vec3::NEG_Z;
        ((after - before).dot(right), (after - before).dot(cam_up))
    }

    // free flight has no up: whatever the roll after leaving orbit (level, on its side, upside down
    // relative to galaxy +Y), sideways mouse turns the view sideways — before, it turned about +Y,
    // which pitched the view or mirrored the turn
    #[test]
    fn free_flight_mouse_turns_sideways_at_any_roll() {
        for roll in [0.0f32, 1.0, 1.5708, 2.5, 3.14159] {
            let mut flight = GalaxyFlight::new(DVec3::ZERO);
            flight.rotation = Quat::from_rotation_y(0.4) * Quat::from_rotation_z(roll);
            let (sideways, tipped) = turn_of(&mut flight, None);
            assert!(
                sideways > 0.1,
                "roll {roll}: turned {sideways} toward the right"
            );
            assert!(tipped.abs() < 1e-3, "roll {roll}: view tipped by {tipped}");
        }
    }

    // just captured with the camera upside down or on its side relative to the planet's local up
    // (or rolled with Q/E): mouse-right still turns right, never mirrored
    #[test]
    fn captured_yaw_is_never_inverted_while_roll_levels() {
        let up = Some(Vec3::Y);
        for roll in [1.5708f32, 2.5, 3.14159, -2.0] {
            let mut flight = GalaxyFlight::new(DVec3::new(0.0, 600.0, 0.0));
            flight.rotation = Quat::from_rotation_z(roll);
            let (sideways, _) = turn_of(&mut flight, up);
            assert!(
                sideways > 0.1,
                "roll {roll}: turned {sideways} toward the right"
            );
        }
    }

    // A/D: roll about the view direction (A left: the camera's right side comes up), ramping up while
    // held and coasting after release; the view direction itself doesn't move
    #[test]
    fn a_d_roll_about_the_view_direction_with_a_ramp() {
        let roll_left = FlightKeys {
            roll: 1.0,
            turn: 0.0,
        };
        for up in [None, Some(Vec3::Y)] {
            let mut flight = GalaxyFlight::new(DVec3::new(0.0, 600.0, 0.0));
            flight.rotation = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.2);
            let (forward, camera_up) = (flight.rotation * Vec3::NEG_Z, flight.rotation * Vec3::Y);
            let step = |flight: &mut GalaxyFlight, keys| {
                flight.update(
                    1.0 / 60.0,
                    Vec3::ZERO,
                    false,
                    false,
                    (0.0, 0.0),
                    false,
                    keys,
                    up,
                )
            };
            step(&mut flight, roll_left);
            assert!(
                flight.roll_rate < 0.5 * crate::smoothing::ROLL_SPEED,
                "{up:?}: no ramp"
            );
            for _ in 0..20 {
                step(&mut flight, roll_left);
            }
            assert!(
                (flight.rotation * Vec3::NEG_Z).dot(forward) > 1.0 - 1e-4,
                "{up:?}"
            );
            assert!(
                (flight.rotation * Vec3::X).dot(camera_up) > 0.1,
                "{up:?}: A didn't raise the right side"
            );
            step(&mut flight, FlightKeys::default());
            assert!(
                flight.roll_rate > 0.5 * crate::smoothing::ROLL_SPEED,
                "{up:?}: stopped dead"
            );
            for _ in 0..20 {
                step(&mut flight, FlightKeys::default());
            }
            assert_eq!(flight.roll_rate, 0.0, "{up:?}: still rolling");
        }
    }

    // Q/E turn (Q left) in galaxy flight, without rolling
    #[test]
    fn q_e_turn_in_galaxy_flight() {
        let mut flight = GalaxyFlight::new(DVec3::ZERO);
        let left = -(flight.rotation * Vec3::X);
        let forward = flight.rotation * Vec3::NEG_Z;
        for _ in 0..30 {
            flight.update(
                1.0 / 60.0,
                Vec3::ZERO,
                false,
                false,
                (0.0, 0.0),
                false,
                FlightKeys {
                    roll: 0.0,
                    turn: 1.0,
                },
                None,
            );
        }
        assert!(((flight.rotation * Vec3::NEG_Z) - forward).dot(left) > 0.1);
        assert!(
            (flight.rotation * Vec3::Y).dot(Vec3::Y) > 1.0 - 1e-4,
            "turning rolled the view"
        );
    }

    // free flight: Space/Shift climb and descend along the camera's own up
    #[test]
    fn free_flight_climbs_along_the_camera_up() {
        let mut flight = GalaxyFlight::new(DVec3::ZERO);
        flight.rotation = Quat::from_rotation_z(3.14159); // upside down relative to galaxy +Y
        flight.update(
            1.0,
            Vec3::ZERO,
            true,
            false,
            (0.0, 0.0),
            false,
            FlightKeys::default(),
            None,
        );
        assert!(flight.velocity.y < -1.0, "{:?}", flight.velocity);
    }

    #[test]
    fn climbing_follows_the_given_up() {
        assert_eq!(
            compose_fly_direction(Vec3::ZERO, true, false, Vec3::X),
            Vec3::X
        );
        assert_eq!(
            compose_fly_direction(Vec3::ZERO, false, true, Vec3::Z),
            -Vec3::Z
        );
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
        let outside = DVec3::new(0.0, 11.5 * p.radius as f64, 0.0);
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Captured(0), inside, t),
            FlightFrame::Captured(0)
        );
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Captured(0), outside, t),
            FlightFrame::Free
        );
        // hysteresis: between capture (10) and release (11) radii, the current frame stays, so
        // hovering at the boundary can't flip-flop between co-moving and not
        let between = DVec3::new(0.0, 10.5 * p.radius as f64, 0.0);
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Captured(0), between, t),
            FlightFrame::Captured(0)
        );
        let between_abs = p.position_at(t) + between;
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Free, between_abs, t),
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
        // exact for an orbit in the common plane; on a tilted orbit (spin axis still +Y) the sun's height
        // swings by up to +-tilt over the year (small seasons), so within one day it can shift by up to twice
        // the tilt's sine (inner planets cover much of their orbit in one of their days)
        let flat = GalaxyPlanet {
            orbit_tilt: 0.0,
            ..g.planets[0]
        };
        for p in g.planets.iter().chain([&flat]) {
            let tolerance = 2.0 * p.orbit_tilt.sin() as f32 + 1e-3;
            let day = p.planet_type.def().day_length_secs as f64;
            let morning = p.sun_dir_in_planet_frame(g.star.position(), 5.0);
            let next_morning = p.sun_dir_in_planet_frame(g.star.position(), 5.0 + day);
            let evening = p.sun_dir_in_planet_frame(g.star.position(), 5.0 + day / 2.0);
            assert!(
                (morning - next_morning).length() < tolerance,
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
            assert_eq!(pa.orbit_tilt, pb.orbit_tilt);
            assert_eq!(pa.orbit_node, pb.orbit_node);
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
            assert!(
                pos.y.abs() <= p.orbit_radius * p.orbit_tilt.sin() + 1e-6,
                "out of the orbit's tilt"
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

    // Kepler: the period grows with the orbit radius ^ 1.5, 180 s at the base inner orbit
    #[test]
    fn orbit_periods_follow_kepler() {
        let period = |r: f64| std::f64::consts::TAU / orbit_speed_for(r);
        assert!((period(INNER_ORBIT_RADIUS) - 180.0).abs() < 1e-6);
        let g = Galaxy::generate(1);
        for p in &g.planets {
            let expected = 180.0 * (p.orbit_radius / INNER_ORBIT_RADIUS).powf(1.5);
            assert!((std::f64::consts::TAU / p.orbit_speed - expected).abs() < 1e-6 * expected);
        }
    }

    // starting angles are seeded, not the old even spread, and differ between systems
    #[test]
    fn starting_angles_are_seeded() {
        let (a, b) = (Galaxy::generate(1), Galaxy::generate(2));
        let even = |i: usize| std::f64::consts::TAU * i as f64 / PLANET_COUNT as f64;
        assert!(a
            .planets
            .iter()
            .enumerate()
            .any(|(i, p)| (p.orbit_phase - even(i)).abs() > 0.2));
        assert!(a
            .planets
            .iter()
            .zip(&b.planets)
            .any(|(p, q)| (p.orbit_phase - q.orbit_phase).abs() > 0.2));
        for p in &a.planets {
            assert!((0.0..std::f64::consts::TAU).contains(&p.orbit_phase));
        }
    }

    // orbit radii vary around the spacing but never cross, and the inner orbit stays clear of every
    // star's heat zone
    #[test]
    fn orbit_radii_vary_without_crossing() {
        for seed in 0..50u64 {
            let g = Galaxy::generate(seed);
            for (i, p) in g.planets.iter().enumerate() {
                let base = INNER_ORBIT_RADIUS * ORBIT_RADIUS_GROWTH.powi(i as i32);
                assert!(
                    p.orbit_radius >= 0.9 * base && p.orbit_radius <= 1.15 * base,
                    "seed {seed} #{i}"
                );
                if i > 0 {
                    assert!(
                        p.orbit_radius >= 1.25 * g.planets[i - 1].orbit_radius,
                        "seed {seed} #{i}"
                    );
                }
            }
            assert!(
                g.planets[0].orbit_radius >= INNER_ORBIT_RADIUS,
                "seed {seed}"
            );
        }
        let varied = (0..20u64).map(|s| Galaxy::generate(s).planets[3].orbit_radius);
        let (lo, hi) = varied.fold((f64::MAX, 0.0f64), |(l, h), r| (l.min(r), h.max(r)));
        assert!(hi - lo > 0.05 * lo, "radii don't vary between seeds");
    }

    // orbits are tilted a little (up to MAX_ORBIT_TILT) out of the common plane, staying circles
    #[test]
    fn orbits_are_slightly_tilted() {
        let g = Galaxy::generate(1);
        let mut out_of_plane = false;
        for p in &g.planets {
            assert!(p.orbit_tilt >= 0.0 && p.orbit_tilt <= MAX_ORBIT_TILT);
            for k in 0..8 {
                let pos = p.position_at(k as f64 * 37.0);
                assert!((pos.length() - p.orbit_radius).abs() < 1e-6 * p.orbit_radius);
                assert!(pos.y.abs() <= p.orbit_radius * p.orbit_tilt.sin() + 1e-6);
                out_of_plane |= pos.y.abs() > 1.0;
            }
        }
        assert!(out_of_plane, "every planet still in one plane");
    }

    #[test]
    fn compose_fly_direction_climb_only_is_straight_up() {
        assert_eq!(
            compose_fly_direction(Vec3::ZERO, true, false, Vec3::Y),
            Vec3::Y
        );
    }

    #[test]
    fn compose_fly_direction_descend_only_is_straight_down() {
        assert_eq!(
            compose_fly_direction(Vec3::ZERO, false, true, Vec3::Y),
            -Vec3::Y
        );
    }

    #[test]
    fn compose_fly_direction_climb_and_descend_cancel_to_zero() {
        assert_eq!(
            compose_fly_direction(Vec3::ZERO, true, true, Vec3::Y),
            Vec3::ZERO
        );
    }

    #[test]
    fn compose_fly_direction_forward_only_is_normalized() {
        let result = compose_fly_direction(Vec3::new(2.0, 0.0, 0.0), false, false, Vec3::Y);
        assert!((result.length() - 1.0).abs() < 1e-6);
        assert_eq!(result, Vec3::X);
    }

    #[test]
    fn star_angular_radius_follows_size_and_distance() {
        let g = Galaxy::generate(1);
        let (inner, outer) = (&g.planets[0], &g.planets[6]);
        let a = inner.star_angular_radius(&g.star);
        assert!((a - (g.star.radius / inner.orbit_radius).asin() as f32).abs() < 1e-6);
        assert!(outer.star_angular_radius(&g.star) < a);
    }

    #[test]
    fn star_type_is_seeded() {
        assert_eq!(star_type_for(7), star_type_for(7));
        let seen: std::collections::HashSet<_> = (0..200u64).map(star_type_for).collect();
        assert_eq!(seen.len(), StarType::ALL.len(), "not every type appears");
        let g = Galaxy::generate(9);
        assert_eq!(g.star.star_type, star_type_for(9));
        assert_eq!(g.star.radius, star_type_for(9).def().radius);
    }

    #[test]
    fn every_star_keeps_its_heat_zone_off_the_orbits() {
        for t in StarType::ALL {
            let limit = INNER_ORBIT_RADIUS - MAX_PLANET_RADIUS as f64;
            assert!(HEAT_ZONE_RADII * t.def().radius < limit, "{}", t.def().name);
        }
    }

    #[test]
    fn sunlight_is_bright_and_only_moderately_tinted() {
        let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let today = lum([1.6, 1.5, 1.3]);
        for t in StarType::ALL {
            let s = t.def().sunlight;
            assert!(
                (lum(s) - today).abs() < 0.05 * today,
                "{}: {s:?}",
                t.def().name
            );
            let (lo, hi) = (s[0].min(s[1]).min(s[2]), s[0].max(s[1]).max(s[2]));
            assert!(lo >= 0.6 * hi, "{}: {s:?}", t.def().name);
        }
        assert_eq!(StarType::Yellow.def().sunlight, [1.6, 1.5, 1.3]);
    }

    // heat: none outside HEAT_ZONE_RADII (the innermost planet's orbit included), rising toward the
    // star, full at HEAT_LIMIT_RADII
    #[test]
    fn star_heat_rises_toward_the_star() {
        let galaxy = Galaxy::generate(1);
        let r = galaxy.star.radius;
        assert_eq!(star_heat(HEAT_ZONE_RADII * r, r), 0.0);
        assert_eq!(star_heat(5.0 * r, r), 0.0);
        let inner_orbit = galaxy
            .planets
            .iter()
            .map(|p| p.orbit_radius - p.radius as f64)
            .fold(f64::MAX, f64::min);
        assert_eq!(
            star_heat(inner_orbit, r),
            0.0,
            "the innermost planet sits in the heat zone"
        );
        assert_eq!(star_heat(HEAT_LIMIT_RADII * r, r), 1.0);
        assert_eq!(star_heat(r, r), 1.0);
        let (a, b) = (star_heat(1.8 * r, r), star_heat(1.4 * r, r));
        assert!(0.0 < a && a < b && b < 1.0, "{a} {b}");
    }

    // boosting straight at the star never gets inside the limit, and is pushed back out from it
    #[test]
    fn boosting_at_the_star_is_pushed_back() {
        let galaxy = Galaxy::generate(1);
        let r = galaxy.star.radius;
        let mut flight = GalaxyFlight::new(DVec3::new(0.0, 0.0, 1.9 * r));
        let dt = 1.0 / 60.0;
        let mut closest = f64::MAX;
        for _ in 0..600 {
            // W + boost, looking straight at the star (-Z)
            flight.update(
                dt,
                Vec3::new(0.0, 0.0, -1.0),
                false,
                false,
                (0.0, 0.0),
                true,
                FlightKeys::default(),
                None,
            );
            flight.velocity = star_heat_push(flight.velocity, flight.position, r, dt);
            flight.position = keep_off_star(flight.position, r);
            closest = closest.min(flight.position.length());
        }
        assert!(closest >= HEAT_LIMIT_RADII * r - 1e-6, "got to {closest}");
        // right at the limit at full boost: pushed outward
        let mut at_limit = GalaxyFlight::new(DVec3::new(0.0, 0.0, HEAT_LIMIT_RADII * r));
        at_limit.velocity = DVec3::new(0.0, 0.0, -(GALAXY_BOOST_SPEED as f64));
        for _ in 0..120 {
            at_limit.update(
                dt,
                Vec3::new(0.0, 0.0, -1.0),
                false,
                false,
                (0.0, 0.0),
                true,
                FlightKeys::default(),
                None,
            );
            at_limit.velocity = star_heat_push(at_limit.velocity, at_limit.position, r, dt);
            at_limit.position = keep_off_star(at_limit.position, r);
        }
        assert!(
            at_limit.velocity.z > 0.0,
            "not pushed out: {:?}",
            at_limit.velocity
        );
    }

    // flying past the star (not at it) isn't held back outside the zone
    #[test]
    fn no_push_outside_the_heat_zone() {
        let v = DVec3::new(0.0, 0.0, -500.0);
        assert_eq!(
            star_heat_push(v, DVec3::new(0.0, 0.0, 3.0 * 3000.0), 3000.0, 0.1),
            v
        );
        assert_eq!(
            keep_off_star(DVec3::new(0.0, 0.0, 9000.0), 3000.0),
            DVec3::new(0.0, 0.0, 9000.0)
        );
    }

    // braking from boost speed takes well under a second (it used to glide for 2.5 s)
    #[test]
    fn galaxy_flight_brakes_from_boost_within_a_second() {
        let mut flight = GalaxyFlight::new(DVec3::ZERO);
        flight.velocity = DVec3::new(0.0, 0.0, -GALAXY_BOOST_SPEED as f64);
        for _ in 0..60 {
            flight.update(
                1.0 / 60.0,
                Vec3::ZERO,
                false,
                false,
                (0.0, 0.0),
                false,
                FlightKeys::default(),
                None,
            );
        }
        assert!(
            flight.velocity.length() < 50.0,
            "{}",
            flight.velocity.length()
        );
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
                FlightKeys::default(),
                None,
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
