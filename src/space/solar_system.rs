// solar_system.rs
// Sub-project #1 (system foundation) of the planned multi-planet feature: a seeded single-star
// solar system of orbiting placeholder bodies, deliberately separate from the existing
// single-planet engine (PlanetData/CoordSystem/Player/Physics) — no voxel terrain, no collision,
// and no connection yet to the player's actual planet (that's a later sub-project, landing/liftoff).

use crate::planet::biome::PlanetType;
use crate::space::moons::{
    hill_radii, moon_orbit_speed, moon_traits, MOON_MAX_RADIUS_SHARE, MOON_MAX_TILT,
    MOON_MIN_ORBIT_RADII, MOON_MIN_RADIUS, MOON_ORBIT_STEP, MOON_STABLE_FRACTION,
};
use glam::{DQuat, DVec3, Quat, Vec3};

#[derive(Debug)]
pub struct Star {
    pub radius: f64,
    pub star_type: StarType,
}

// the solar system's star: one of five types, picked from the solar system seed (star_type_for); sets the star's
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
    pub luminosity: f64,    // relative to the yellow star; sets the habitable distance (variety.rs)
}

const STAR_TYPES: [StarTypeDef; 5] = [
    StarTypeDef {
        name: "Red dwarf",
        surface_color: [1.10, 0.45, 0.23],
        limb_color: [0.83, 0.21, 0.07],
        radius: 7_200.0,
        granulation: 0.35,
        granule_scale: 14.0,
        sunspots: 0.12,
        corona_color: [1.0, 0.35, 0.15],
        corona_size: 0.35,
        sunlight: [1.72, 1.45, 1.12],
        luminosity: 0.3,
    },
    StarTypeDef {
        name: "Orange",
        surface_color: [1.10, 0.68, 0.35],
        limb_color: [0.83, 0.33, 0.12],
        radius: 9_600.0,
        granulation: 0.3,
        granule_scale: 16.0,
        sunspots: 0.08,
        corona_color: [1.0, 0.55, 0.25],
        corona_size: 0.4,
        sunlight: [1.68, 1.48, 1.18],
        luminosity: 0.6,
    },
    StarTypeDef {
        name: "Yellow",
        surface_color: [1.10, 0.96, 0.69],
        limb_color: [0.83, 0.52, 0.25],
        radius: 12_000.0,
        granulation: 0.25,
        granule_scale: 18.0,
        sunspots: 0.05,
        corona_color: [1.0, 0.8, 0.5],
        corona_size: 0.45,
        sunlight: [1.6, 1.5, 1.3],
        luminosity: 1.0,
    },
    StarTypeDef {
        name: "White",
        surface_color: [1.07, 1.07, 1.10],
        limb_color: [0.82, 0.76, 0.67],
        radius: 13_200.0,
        granulation: 0.2,
        granule_scale: 20.0,
        sunspots: 0.03,
        corona_color: [0.85, 0.9, 1.0],
        corona_size: 0.5,
        sunlight: [1.52, 1.52, 1.48],
        luminosity: 2.0,
    },
    StarTypeDef {
        name: "Blue giant",
        surface_color: [0.72, 0.84, 1.10],
        limb_color: [0.52, 0.59, 0.83],
        radius: 14_400.0,
        granulation: 0.15,
        granule_scale: 22.0,
        sunspots: 0.0,
        corona_color: [0.55, 0.7, 1.0],
        corona_size: 0.6,
        sunlight: [1.38, 1.5, 1.68],
        luminosity: 4.0,
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

// the star type of the solar system with this seed: a bit-mixing hash (fmix64, as planet_radius_for) of the
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
    // the star sits at the origin of space; kept as a method so callers don't hardcode it
    pub fn position(&self) -> DVec3 {
        DVec3::ZERO
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Planet {
    pub orbit_radius: f64,
    pub orbit_speed: f64, // radians/sec; positive = counterclockwise looking down +Y
    pub orbit_phase: f64, // starting angle, radians
    pub orbit_tilt: f64, // inclination out of the common orbital plane, radians (0..MAX_ORBIT_TILT)
    pub orbit_node: f64, // direction of the tilt axis within the plane, radians
    pub radius: f32,
    pub planet_type: PlanetType,
    // seeds this planet's TerrainShape/NoiseGenerator (src/impostor.rs); independent of
    // planet_radius_for's hash so radius and terrain shape don't correlate
    pub noise_seed: u32,
    pub traits: crate::planet::traits::PlanetTraits, // per-planet variation (NEUTRAL = the type exactly)
    // moons: the index of the planet this one orbits (always smaller than its own), its letter (0 = a),
    // and the parent's orbit around the star, so position_at stays self-contained. None / 0 / None for planets
    pub parent: Option<usize>,
    pub moon_index: u8,
    pub parent_orbit: Option<OrbitParams>,
}

// a circular orbit: around the star for a planet, around the parent for a moon
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitParams {
    pub radius: f64,
    pub speed: f64, // radians/s, positive counterclockwise looking down +Y
    pub phase: f64,
    pub tilt: f64,
    pub node: f64,
}

impl OrbitParams {
    // on a circle of `radius`, tilted by `tilt` about a horizontal axis (direction `node`) out of XZ
    pub fn position_at(&self, t: f64) -> DVec3 {
        let angle = self.phase + self.speed * t;
        let flat = DVec3::new(self.radius * angle.cos(), 0.0, self.radius * angle.sin());
        let axis = DVec3::new(self.node.cos(), 0.0, self.node.sin());
        DQuat::from_axis_angle(axis, self.tilt) * flat
    }
}

#[derive(Debug)]
pub struct SolarSystem {
    pub star: Star,
    pub planets: Vec<Planet>,
    pub seed: u64, // generation seed: planets added later (/system add) are seeded from it too
}

const PLANET_COUNT: usize = 8;
/// Every absolute distance, speed and acceleration of the solar system is the original value times this;
/// every angle, period and share is unchanged (the start system is the old one x4). Reviewed and kept:
/// map units (`D_MAX`, `STAR_MAP_RADIUS`, `LY_TO_VIEW`); radii-relative constants (`CAPTURE_RADII`/`RELEASE_RADII`,
/// `CO_ROTATION_*_RADII`, `HEAT_ZONE_RADII`/`HEAT_LIMIT_RADII`, `landing.rs`'s `*_HANDOVER_RADII`,
/// `MIN_CAPTURED_RADII`, `travel::ORBIT_RADII`, `universe::ARRIVAL_ORBITS`, the LOD multipliers,
/// `ATMOSPHERE_SHELL_RADII`, the impostors' 60-radii lightning cut-off); block-scale surface constants (fog,
/// lightning, precipitation, `FLY_*`, glide and landing rates); `travel::climb_speed` (altitude-proportional);
/// `impostor::storm_base` (from the radius).
pub const SYSTEM_SCALE: f64 = 4.0;
const INNER_ORBIT_RADIUS: f64 = 8_000.0 * SYSTEM_SCALE;
const ORBIT_RADIUS_GROWTH: f64 = 1.5; // each planet this many times farther out than the last
const INNER_ORBIT_PERIOD_SECS: f64 = 180.0; // at INNER_ORBIT_RADIUS; Kepler beyond (orbit_speed_for)
                                            // seeded orbit variety (orbit_for): each radius x0.95..x1.10 around the spacing (neighbours keep a gap of
                                            // at least x1.29, the innermost stays at INNER_ORBIT_RADIUS or more, outside every star's heat zone),
                                            // a random starting angle and a small tilt out of the common plane
const ORBIT_JITTER_MIN: f64 = 0.95;
const ORBIT_JITTER_MAX: f64 = 1.10;
const MAX_ORBIT_TILT: f64 = 4.0 * std::f64::consts::PI / 180.0;
pub(crate) const MIN_PLANET_RADIUS: f32 = 40.0 * SYSTEM_SCALE as f32;
pub(crate) const MAX_PLANET_RADIUS: f32 = 250.0 * SYSTEM_SCALE as f32;
// most bodies (planets + moons) a system can hold: the space renderer sizes its per-body uniform buffer,
// ring budget and mesh cache for this many
pub const MAX_PLANETS: usize = 48;
// most planets (bodies without a parent) a system can hold; /system add refuses beyond it (the 48th planet
// would sit past f32 depth precision)
pub const MAX_PLANET_COUNT: usize = 15;
// /system add accepts radii in this range (bake time grows with the square of the radius)
pub const MIN_ADDED_RADIUS: f32 = 20.0;
pub const MAX_ADDED_RADIUS: f32 = 1000.0;

impl SolarSystem {
    pub fn generate(seed: u64) -> Self {
        let planets = (0..PLANET_COUNT)
            .map(|i| {
                let (orbit_radius, orbit_phase, orbit_tilt, orbit_node) = orbit_for(seed, i);
                Planet {
                    orbit_radius,
                    orbit_speed: orbit_speed_for(orbit_radius),
                    orbit_phase,
                    orbit_tilt,
                    orbit_node,
                    radius: planet_radius_for(seed, i),
                    planet_type: PlanetType::ALL[i % PlanetType::ALL.len()],
                    noise_seed: noise_seed_for(seed, i),
                    traits: crate::planet::traits::PlanetTraits::NEUTRAL,
                    parent: None,
                    moon_index: 0,
                    parent_orbit: None,
                }
            })
            .collect();
        let star_type = star_type_for(seed);
        let mut system = Self {
            star: Star {
                radius: star_type.def().radius,
                star_type,
            },
            planets,
            seed,
        };
        system.append_moons();
        system
    }

    // every system but the start one (Universe): 3-10 planets on the usual orbits, each type picked by
    // its orbit's temperature around this star, each planet with its own traits (space/variety.rs)
    pub fn generate_varied(seed: u64) -> Self {
        let star_type = star_type_for(seed);
        let planets = (0..crate::space::variety::planet_count(seed))
            .map(|i| {
                let (orbit_radius, orbit_phase, orbit_tilt, orbit_node) = orbit_for(seed, i);
                let radius = planet_radius_for(seed, i);
                let tau = crate::space::variety::temperature_index(orbit_radius, star_type);
                let planet_type = crate::space::variety::pick_type(seed, i, tau);
                Planet {
                    orbit_radius,
                    orbit_speed: orbit_speed_for(orbit_radius),
                    orbit_phase,
                    orbit_tilt,
                    orbit_node,
                    radius,
                    planet_type,
                    noise_seed: noise_seed_for(seed, i),
                    traits: crate::space::variety::traits_for(seed, i, planet_type, radius),
                    parent: None,
                    moon_index: 0,
                    parent_orbit: None,
                }
            })
            .collect();
        let mut system = Self {
            star: Star {
                radius: star_type.def().radius,
                star_type,
            },
            planets,
            seed,
        };
        system.append_moons();
        system
    }

    // the moons of every planet (moons.rs), appended after all planets so planet #k stays planets[k - 1];
    // stops at MAX_PLANETS bodies (inner planets first keep theirs)
    fn append_moons(&mut self) {
        let n = self.planets.len();
        for i in 0..n {
            let p = self.planets[i];
            for m in crate::space::moons::moons_for(self.seed, i, &p, &self.star) {
                if self.planets.len() >= MAX_PLANETS {
                    return;
                }
                self.planets.push(m);
            }
        }
    }

    // the planet a new game starts on: the first one of the preferred type (--biome), else #1
    pub fn start_planet_index(&self, preferred: PlanetType) -> usize {
        self.planets
            .iter()
            .position(|p| !p.is_moon() && p.planet_type == preferred)
            .unwrap_or(0)
    }

    // the largest orbit radius among the planets (moons orbit their parents, not the star)
    pub fn outermost_planet_orbit(&self) -> f64 {
        self.planets
            .iter()
            .filter(|p| !p.is_moon())
            .map(|p| p.orbit_radius)
            .fold(0.0, f64::max)
    }

    // bodies without a parent
    pub fn planet_count(&self) -> usize {
        self.planets.iter().filter(|p| p.parent.is_none()).count()
    }

    // the moons orbiting planet `parent`
    pub fn moons_of(&self, parent: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.planets.len()).filter(move |&i| self.planets[i].parent == Some(parent))
    }

    // debug (/system add moon): a moon of planet `parent` on the next free orbit inside its stable zone
    pub fn add_moon(
        &mut self,
        parent: usize,
        planet_type: PlanetType,
        radius: f32,
    ) -> Result<usize, &'static str> {
        if self.planets.len() >= MAX_PLANETS {
            return Err("The system is full.");
        }
        let Some(p) = self.planets.get(parent).copied() else {
            return Err("No such planet.");
        };
        if p.is_moon() {
            return Err("Moons have no moons.");
        }
        if !(MOON_MIN_RADIUS..=MOON_MAX_RADIUS_SHARE * p.radius).contains(&radius) {
            return Err("Moon radius must be 40 to a quarter of its planet's.");
        }
        let hill = hill_radii(p.orbit_radius, self.star.radius);
        let existing: Vec<f64> = self
            .moons_of(parent)
            .map(|m| self.planets[m].orbit_radius / p.radius as f64)
            .collect();
        let next = existing.iter().cloned().fold(0.0_f64, f64::max);
        let orbit = if next == 0.0 {
            MOON_MIN_ORBIT_RADII
        } else {
            next * (MOON_ORBIT_STEP + 0.1)
        };
        if orbit > MOON_STABLE_FRACTION * hill {
            return Err("No stable orbit left around that planet.");
        }
        let i = self.planets.len();
        let k = existing.len();
        if k >= 16 {
            return Err("That planet has enough moons.");
        }
        self.planets.push(Planet {
            orbit_radius: orbit * p.radius as f64,
            orbit_speed: moon_orbit_speed(orbit),
            // salts 31-33: unused elsewhere (orbit_for 1-4, variety 10-16 and 20-21)
            orbit_phase: std::f64::consts::TAU * seeded_unit(self.seed, i, 31),
            orbit_tilt: MOON_MAX_TILT * seeded_unit(self.seed, i, 32),
            orbit_node: std::f64::consts::TAU * seeded_unit(self.seed, i, 33),
            radius,
            planet_type,
            // as generated moons: never the seed of a planet added later (add_planet uses the ordinal)
            noise_seed: noise_seed_for(self.seed ^ 0x6D6F_6F6E, parent * 16 + k),
            traits: moon_traits(radius),
            parent: Some(parent),
            moon_index: k as u8,
            parent_orbit: Some(p.own_orbit()),
        });
        Ok(i)
    }

    // debug (/system add): appends a planet of the given type and radius on the next orbit outward,
    // shaped like the generated ones (same orbit progression, phase spread and seeded terrain);
    // returns its index
    pub fn add_planet(
        &mut self,
        planet_type: PlanetType,
        radius: f32,
    ) -> Result<usize, &'static str> {
        if self.planets.len() >= MAX_PLANETS {
            return Err("The system is full (48 bodies).");
        }
        if self.planet_count() >= MAX_PLANET_COUNT {
            return Err("The system is full (15 planets).");
        }
        if !(MIN_ADDED_RADIUS..=MAX_ADDED_RADIUS).contains(&radius) {
            return Err("Planet radius must be 20-1000.");
        }
        let i = self.planets.len();
        // orbit and terrain seed follow the planet's ordinal, not the list index (moons sit in the list)
        let n = self.planet_count();
        let (orbit_radius, orbit_phase, orbit_tilt, orbit_node) = orbit_for(self.seed, n);
        self.planets.push(Planet {
            orbit_radius,
            orbit_speed: orbit_speed_for(orbit_radius),
            orbit_phase,
            orbit_tilt,
            orbit_node,
            radius,
            planet_type,
            noise_seed: noise_seed_for(self.seed, n),
            traits: crate::planet::traits::PlanetTraits::NEUTRAL,
            parent: None,
            moon_index: 0,
            parent_orbit: None,
        });
        Ok(i)
    }
}

// gravity multiplier by body radius, piecewise linear through (40, 0.15), (160, 0.6), (1000, 1.3), clamped to
// [40, 1000]: the one formula for moons and the base of varied planets' gravity (variety::traits_for)
pub fn gravity_for_radius(radius: f32) -> f32 {
    let r = radius.clamp(40.0, 1000.0);
    if r <= 160.0 {
        0.15 + (r - 40.0) / 120.0 * 0.45
    } else {
        0.6 + (r - 160.0) / 840.0 * 0.7
    }
}

// angular speed (radians/s) on a circular orbit: Kepler's third law (period ~ radius^1.5), one orbit in
// INNER_ORBIT_PERIOD_SECS at INNER_ORBIT_RADIUS
fn orbit_speed_for(orbit_radius: f64) -> f64 {
    std::f64::consts::TAU / INNER_ORBIT_PERIOD_SECS * (INNER_ORBIT_RADIUS / orbit_radius).powf(1.5)
}

// a deterministic value in 0..1 for (system seed, planet index, `salt` naming what it's for): fmix64 as
// planet_radius_for, with a salt per property so they don't correlate
pub(crate) fn seeded_unit(seed: u64, index: usize, salt: u64) -> f64 {
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

// planet `index`'s orbit in the solar system with `seed`: (radius, starting angle, tilt, tilt axis direction)
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
    // to be doesn't avalanche small sequential `index` values enough — an entire system's planets
    // came out clustered within a few units of each other instead of spanning the intended range
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    let t = (h >> 11) as f64 / (1u64 << 53) as f64; // 0..1, using the top 53 bits
    MIN_PLANET_RADIUS + (t as f32) * (MAX_PLANET_RADIUS - MIN_PLANET_RADIUS)
}

// deterministic pseudo-random seed per (system seed, index), independent of planet_radius_for's
// hash (different multiplier constant) so a planet's size and its terrain shape don't correlate
pub(crate) fn noise_seed_for(seed: u64, index: usize) -> u32 {
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

// "3" for planet #3, "3b" for its second moon
pub fn body_label(system: &SolarSystem, i: usize) -> String {
    let p = &system.planets[i];
    match p.parent {
        None => (system.planets[..i]
            .iter()
            .filter(|q| q.parent.is_none())
            .count()
            + 1)
        .to_string(),
        Some(parent) => format!(
            "{}{}",
            body_label(system, parent),
            (b'a' + p.moon_index.min(25)) as char
        ),
    }
}

// the body a label names ("3", "3b", case-insensitive), None if there is no such body
pub fn parse_body(system: &SolarSystem, s: &str) -> Option<usize> {
    let s = s.trim().to_ascii_lowercase();
    let digits = s.trim_end_matches(|c: char| c.is_ascii_lowercase());
    let letters = &s[digits.len()..];
    let n: usize = digits.parse().ok().filter(|&n| n >= 1)?;
    let planet = system
        .planets
        .iter()
        .enumerate()
        .filter(|(_, p)| p.parent.is_none())
        .nth(n - 1)?
        .0;
    match letters.len() {
        0 => Some(planet),
        1 => {
            let k = letters.as_bytes()[0] - b'a';
            system
                .moons_of(planet)
                .find(|&m| system.planets[m].moon_index == k)
        }
        _ => None,
    }
}

impl Planet {
    // the star's angular radius seen from this planet (radians): its apparent size in the planet's sky
    pub fn star_angular_radius(&self, star: &Star) -> f32 {
        // a moon sees the star from its parent's distance, not from its own tiny orbit
        let a = self.parent_orbit.map_or(self.orbit_radius, |o| o.radius);
        (star.radius / a).min(1.0).asin() as f32
    }

    pub fn own_orbit(&self) -> OrbitParams {
        OrbitParams {
            radius: self.orbit_radius,
            speed: self.orbit_speed,
            phase: self.orbit_phase,
            tilt: self.orbit_tilt,
            node: self.orbit_node,
        }
    }

    pub fn is_moon(&self) -> bool {
        self.parent.is_some()
    }

    // around the star for a planet; for a moon the parent's position plus its own orbit around the parent
    pub fn position_at(&self, t: f64) -> DVec3 {
        let own = self.own_orbit().position_at(t);
        match &self.parent_orbit {
            Some(o) => o.position_at(t) + own,
            None => own,
        }
    }

    // The planet frame: origin at the planet centre, turned by the spin angle about +Y, the axis
    // the voxel engine already uses for day/night. Positions stored in this frame stay put on the
    // planet's surface while it orbits and spins. The spin adds the orbit rate on top of one turn per
    // day_length_secs, so the *solar* day (sun back at the same sky direction) is exactly the planet
    // type's day length — without it the orbit would stretch or shrink every planet's day.
    pub fn spin_angle(&self, t: f64) -> f64 {
        t * self.spin_rate()
    }

    // how fast the planet's frame turns about +Y, rad/s (spin_angle)
    // A moon is tidally locked: its frame turns once per orbit, so the parent stays put in its sky
    // (day_length for a moon is derived from this rate, not the other way round)
    pub fn spin_rate(&self) -> f64 {
        if self.is_moon() {
            return self.orbit_speed;
        }
        std::f64::consts::TAU / self.day_length() + self.orbit_speed
    }

    // seconds per solar day: a planet's type day length scaled by its trait; a moon's is the synodic period
    // of its locked spin against the star (about one orbit: the parent barely moves around the star meanwhile)
    pub fn day_length(&self) -> f64 {
        if let Some(po) = self.parent_orbit {
            let rel = (self.orbit_speed - po.speed).abs().max(1e-9);
            return std::f64::consts::TAU / rel;
        }
        self.planet_type.def().day_length_secs as f64 * self.traits.day_scale as f64
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

    // planet frame -> space rotation: how the planet (and so its impostor mesh) is turned at t
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

    // the real voxel planet for this planet: own seed, matching size, own planet type
    pub fn bake(&self) -> crate::planet::data::PlanetData {
        crate::planet::data::PlanetData::new_for_planet(
            self.voxel_resolution(),
            self.noise_seed,
            self.planet_type,
            self.traits,
        )
    }

    // bake() plus the edits kept from an earlier visit (refused if they belong to another terrain)
    pub fn bake_with_edits(
        &self,
        edits: Option<crate::planet::data::PlanetEdits>,
    ) -> crate::planet::data::PlanetData {
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

// which frame space flight lives in: space, or a planet's own frame (spec §1 "captured")
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlightFrame {
    Free,
    Captured(usize), // index into SolarSystem::planets
}

// a moon of body `i` that captures `abs_pos` before `i` does: within CAPTURE_RADII of its own radii and
// nearer in its own radii than `i` (the nearest such moon); None for a moon `i` or when there is none
pub fn moon_takes_over(system: &SolarSystem, i: usize, abs_pos: DVec3, t: f64) -> Option<usize> {
    let parent = &system.planets[i];
    let own = (abs_pos - parent.position_at(t)).length() / parent.radius as f64;
    system
        .moons_of(i)
        .map(|m| {
            let p = &system.planets[m];
            (m, (abs_pos - p.position_at(t)).length() / p.radius as f64)
        })
        .filter(|&(_, radii)| radii < CAPTURE_RADII && radii < own)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(m, _)| m)
}

// the frame flight should be in, given its position `pos` in its current frame (space when
// Free, the planet frame when Captured)
pub fn next_flight_frame(
    system: &SolarSystem,
    frame: FlightFrame,
    pos: DVec3,
    t: f64,
) -> FlightFrame {
    match frame {
        FlightFrame::Captured(i) => {
            let abs = system.planets[i].from_planet_frame(pos, t);
            if pos.length() > RELEASE_RADII * system.planets[i].radius as f64
                || moon_takes_over(system, i, abs, t).is_some()
            {
                FlightFrame::Free
            } else {
                frame
            }
        }
        FlightFrame::Free => system
            .planets
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (pos - p.position_at(t)).length() / p.radius as f64))
            .filter(|&(_, radii)| radii < CAPTURE_RADII)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(FlightFrame::Free, |(i, _)| FlightFrame::Captured(i)),
    }
}

pub const SPACE_CRUISE_SPEED: f32 = 2000.0;
pub const SPACE_BOOST_SPEED: f32 = 8000.0;

pub struct SpaceFlight {
    pub position: DVec3,
    pub velocity: DVec3,
    pub rotation: Quat,
    pub roll_rate: f32, // eased roll rate (A/D), rad/s, positive rolls left
    pub turn_rate: f32, // eased turn rate (Q/E), rad/s, positive turns left
    mouse_sens: f32,
}

impl SpaceFlight {
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
    pub fn change_frame(
        &mut self,
        from: FlightFrame,
        to: FlightFrame,
        system: &SolarSystem,
        t: f64,
    ) {
        if from == to {
            return;
        }
        if let FlightFrame::Captured(i) = from {
            let p = &system.planets[i];
            self.position = p.from_planet_frame(self.position, t);
            self.rotation = p.rotation_from_planet_frame(self.rotation, t);
            // a direction, not a point: rotate without the planet's offset
            self.velocity = p.from_planet_frame(self.velocity, t) - p.position_at(t);
        }
        if let FlightFrame::Captured(i) = to {
            let p = &system.planets[i];
            self.position = p.to_planet_frame(self.position, t);
            self.rotation = p.rotation_to_planet_frame(self.rotation, t);
            self.velocity = p.to_planet_frame(self.velocity + p.position_at(t), t);
        }
    }

    // `up`: the local radial up while captured by a planet (climbing with Space / descending with
    // Shift follow it, like the planet camera), None in free flight, where space has no up and they
    // follow the camera's own up instead; turning is always about the ship's own axes
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        input: Vec3,
        jump: bool,
        down: bool,
        mouse_delta: (f32, f32),
        sprint: bool,
        keys: crate::player::controller::FlightKeys, // A/D roll, Q/E turn: eased rates (smoothing.rs), pure roll
        up: Option<Vec3>,
    ) {
        use crate::player::smoothing::{
            approach, FLIGHT_TURN_SPEED, ROLL_ACCEL, ROLL_SPEED, TURN_ACCEL,
        };
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
            // ship axes: about the ship's own up, like the pitch about its own right, so turns follow
            // the screen at any attitude (in orbit too: the local up only steers Space/Shift)
            self.rotation *= Quat::from_axis_angle(Vec3::Y, yaw_delta);
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
            SPACE_BOOST_SPEED
        } else {
            SPACE_CRUISE_SPEED
        };
        let target_velocity = dir * target_speed;

        let vel_f32 = Vec3::new(
            self.velocity.x as f32,
            self.velocity.y as f32,
            self.velocity.z as f32,
        );
        // eased like planet fly mode, so cruise and boost respond alike (no long glide from boost)
        let new_vel = crate::player::smoothing::ease_toward(
            vel_f32,
            target_velocity,
            dt,
            crate::player::smoothing::FLIGHT_EASE_SECONDS,
        );
        self.velocity = DVec3::new(new_vel.x as f64, new_vel.y as f64, new_vel.z as f64);

        self.position += self.velocity * dt as f64;
    }
}

// Flight near a planet turns with it (the planet frame spins about +Y at spin_rate), like an
// aircraft with its air: the ground holds still and the sky turns, 1.5–6°/s with these day lengths.
// Out in space that read as the ship being pulled round with the stars sweeping past, so the
// co-rotation fades out with distance: full below CO_ROTATION_INNER_RADII (the glide ceiling, about
// where the sky turns from opaque to black), none from CO_ROTATION_OUTER_RADII, where position,
// attitude and velocity keep still against the stars (and the planet turns below). Planet fly mode
// and captured flight use the same function of distance, so the handovers don't change it.
pub const CO_ROTATION_INNER_RADII: f32 = 1.64;
pub const CO_ROTATION_OUTER_RADII: f32 = 3.0;

// the share of the planet's spin a flight at `distance_radii` from its centre turns with
pub fn co_rotation(distance_radii: f32) -> f32 {
    let t = ((distance_radii - CO_ROTATION_INNER_RADII)
        / (CO_ROTATION_OUTER_RADII - CO_ROTATION_INNER_RADII))
        .clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

// the rotation (planet frame, about +Y) to apply this step to a flight's position, velocity and
// attitude so it turns only with co_rotation's share of the planet's spin (`spin_rate`, rad/s): the
// rest is undone, the planet frame turning the other way under it
pub fn counter_spin(spin_rate: f32, distance_radii: f32, dt: f32) -> Quat {
    Quat::from_axis_angle(
        Vec3::Y,
        spin_rate * (1.0 - co_rotation(distance_radii)) * dt,
    )
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
// outward push at full heat: eased flight pulls toward boost speed at up to ~32,000 units/s^2
// (SPACE_BOOST_SPEED / FLIGHT_EASE_SECONDS), so this always wins near the limit
const HEAT_PUSH_ACCEL: f64 = 48_000.0;

// heat 0..1 at `distance` from the star's centre (smooth: 0 at the zone's edge, 1 at the limit)
pub fn star_heat(distance: f64, star_radius: f64) -> f32 {
    let t = ((HEAT_ZONE_RADII * star_radius - distance)
        / ((HEAT_ZONE_RADII - HEAT_LIMIT_RADII) * star_radius))
        .clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)) as f32
}

// the velocity after this step's outward heat push (space, the star at the origin)
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

// Where a target sits relative to the camera, for the solar system-mode compass overlay: yaw is the
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
pub(crate) mod tests {
    use super::*;

    // the start system without its generated moons (Task 3 adds them) plus one test moon of planet #3
    #[test]
    fn gravity_hits_its_anchors_and_rises_with_the_radius() {
        assert!((gravity_for_radius(40.0) - 0.15).abs() < 1e-6);
        assert!((gravity_for_radius(160.0) - 0.6).abs() < 1e-6);
        assert!((gravity_for_radius(1000.0) - 1.3).abs() < 1e-6);
        let mut last = 0.0;
        for r in (20..=1200).step_by(10) {
            let g = gravity_for_radius(r as f32);
            assert!(g >= last, "{r}");
            last = g;
        }
        assert!(
            gravity_for_radius(5.0) >= 0.1 && gravity_for_radius(5000.0) <= 1.3 + 1e-6,
            "clamped"
        );
        // on the planets' range it is the variety base, 0.6 + 0.7 * size
        for r in [160.0f32, 400.0, 777.0, 1000.0] {
            let size = (r - MIN_PLANET_RADIUS) / (MAX_PLANET_RADIUS - MIN_PLANET_RADIUS);
            assert!(
                (gravity_for_radius(r) - (0.6 + 0.7 * size)).abs() < 1e-5,
                "{r}"
            );
        }
    }

    #[test]
    fn a_baked_moon_carries_its_gravity() {
        let mut s = with_moon();
        let m = s.planets.len() - 1;
        s.planets[m].traits = crate::space::moons::moon_traits(s.planets[m].radius);
        let data = s.planets[m].bake();
        assert!((data.traits.gravity - gravity_for_radius(40.0)).abs() < 1e-6);
        assert!(data.traits.gravity < 0.2);
    }

    pub(crate) fn with_moon() -> SolarSystem {
        let mut s = SolarSystem::generate(1);
        s.planets.retain(|p| !p.is_moon());
        let parent = 2;
        let p = s.planets[parent];
        s.planets.push(Planet {
            orbit_radius: 3.0 * p.radius as f64,
            orbit_speed: std::f64::consts::TAU / 150.0,
            orbit_phase: 0.7,
            orbit_tilt: 0.05,
            orbit_node: 1.1,
            radius: 40.0,
            planet_type: PlanetType::Cratered,
            noise_seed: 99,
            traits: crate::planet::traits::PlanetTraits::NEUTRAL,
            parent: Some(parent),
            moon_index: 0,
            parent_orbit: Some(p.own_orbit()),
        });
        s
    }

    // inside a parent's capture sphere a moon wins once you are closer to it in its own radii; leaving the
    // moon's sphere hands over to the parent without ping-pong
    #[test]
    fn capture_prefers_the_moon_near_it_and_hands_back_to_the_parent() {
        let s = with_moon();
        let m = s.planets.len() - 1;
        let (mp, pp) = (&s.planets[m], &s.planets[2]);
        let t = 123.0;
        let moon = mp.position_at(t);
        let near = moon + DVec3::new(2.0 * mp.radius as f64, 0.0, 0.0);
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Free, near, t),
            FlightFrame::Captured(m)
        );
        let near_local = pp.to_planet_frame(near, t);
        assert_eq!(moon_takes_over(&s, 2, near, t), Some(m));
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Captured(2), near_local, t),
            FlightFrame::Free
        );
        assert_eq!(moon_takes_over(&s, m, near, t), None, "moons have no moons");
        // away from the parent's side, 9 moon radii out is inside the moon's sphere but the parent is
        // nearer in its radii (a moon of 40 at 3 parent radii: the point is ~3 parent radii from it)
        let far = moon + (moon - pp.position_at(t)).normalize() * 9.0 * mp.radius as f64;
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Free, far, t),
            FlightFrame::Captured(2)
        );
        let far_local = pp.to_planet_frame(far, t);
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Captured(2), far_local, t),
            FlightFrame::Captured(2)
        );
        let in_moon = DVec3::new(10.9 * mp.radius as f64, 0.0, 0.0);
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Captured(m), in_moon, t),
            FlightFrame::Captured(m)
        );
        let out = DVec3::new(11.1 * mp.radius as f64, 0.0, 0.0);
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Captured(m), out, t),
            FlightFrame::Free
        );
        // from there the parent captures and does not hand straight back to the moon: no ping-pong
        let abs = mp.from_planet_frame(out, t);
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Free, abs, t),
            FlightFrame::Captured(2)
        );
        let back = pp.to_planet_frame(abs, t);
        assert_eq!(
            next_flight_frame(&s, FlightFrame::Captured(2), back, t),
            FlightFrame::Captured(2)
        );
    }

    #[test]
    fn the_outermost_planet_orbit_ignores_moons() {
        let s = with_moon();
        let expect = s
            .planets
            .iter()
            .filter(|p| !p.is_moon())
            .map(|p| p.orbit_radius)
            .fold(0.0, f64::max);
        assert_eq!(s.outermost_planet_orbit(), expect);
    }

    #[test]
    fn a_moon_orbits_its_parent_at_its_radius() {
        let s = with_moon();
        let (m, p) = (&s.planets[s.planets.len() - 1], &s.planets[2]);
        for t in [0.0, 17.0, 400.0, 1234.5] {
            let d = (m.position_at(t) - p.position_at(t)).length();
            assert!((d - m.orbit_radius).abs() < 1e-6, "{t}: {d}");
        }
        assert!(m.is_moon() && !p.is_moon());
    }

    #[test]
    fn a_moon_is_tidally_locked_to_its_parent() {
        let s = with_moon();
        let m = &s.planets[s.planets.len() - 1];
        // the parent's direction in the moon's frame is the same at every time (up to the tilt's libration)
        let dir = |t: f64| {
            m.to_planet_frame(s.planets[2].position_at(t), t)
                .normalize()
        };
        let d0 = dir(0.0);
        for t in [10.0, 55.5, 149.0, 600.0] {
            assert!(dir(t).dot(d0) > 0.99, "{t}: {}", dir(t).dot(d0));
        }
        assert!((m.spin_rate() - m.orbit_speed).abs() < 1e-12);
        // its solar day is about one orbit (the parent barely moves around the star meanwhile)
        let day = m.day_length();
        let orbit = std::f64::consts::TAU / m.orbit_speed;
        // (planet #3 moves round the star at about a quarter of the moon's rate here, so the synodic day is ~35 % longer)
        let parent_speed = s.planets[2].orbit_speed;
        assert!((day - std::f64::consts::TAU / (m.orbit_speed - parent_speed)).abs() < 1e-6);
        assert!((day - orbit).abs() < 0.5 * orbit, "{day} vs {orbit}");
    }

    #[test]
    fn a_moon_sees_the_star_at_its_parents_distance() {
        let s = with_moon();
        let (m, p) = (&s.planets[s.planets.len() - 1], &s.planets[2]);
        assert_eq!(
            m.star_angular_radius(&s.star),
            p.star_angular_radius(&s.star)
        );
    }

    // /system add after moons exist puts the new planet on the next planet orbit, not on 1.5^(list length)
    #[test]
    fn add_planet_orbits_by_planet_count() {
        let mut s = with_moon();
        let n = s.planet_count();
        assert_eq!(n, 8);
        let i = s.add_planet(PlanetType::Desert, 200.0).unwrap();
        assert_eq!(i, s.planets.len() - 1);
        let expected = orbit_for(s.seed, n).0;
        assert!((s.planets[i].orbit_radius - expected).abs() < 1e-6);
        assert_eq!(
            s.planets[i].noise_seed,
            noise_seed_for(s.seed, n),
            "seeded by the planet ordinal"
        );
        assert_eq!(s.planet_count(), 9);
        assert!(s.planets[i].parent.is_none());
        assert_eq!(body_label(&s, i), "9");
        // the planet cap counts planets, not bodies
        while s.planet_count() < MAX_PLANET_COUNT {
            s.add_planet(PlanetType::Ice, 100.0).unwrap();
        }
        assert!(s.add_planet(PlanetType::Ice, 100.0).is_err());
    }

    #[test]
    fn labels_number_planets_and_letter_moons() {
        let s = with_moon();
        let m = s.planets.len() - 1;
        assert_eq!(body_label(&s, 0), "1");
        assert_eq!(body_label(&s, 2), "3");
        assert_eq!(body_label(&s, m), "3a");
        assert_eq!(parse_body(&s, "3"), Some(2));
        assert_eq!(parse_body(&s, "3a"), Some(m));
        assert_eq!(parse_body(&s, "3A"), Some(m));
        assert_eq!(parse_body(&s, "3b"), None, "only one moon");
        assert_eq!(parse_body(&s, "9"), None);
        assert_eq!(parse_body(&s, "0"), None);
        assert_eq!(parse_body(&s, "a"), None);
        assert_eq!(parse_body(&s, ""), None);
        // a moon moves with its parent: the parent's lookups are unchanged
        assert_eq!(s.moons_of(2).collect::<Vec<_>>(), vec![m]);
    }

    #[test]
    fn generate_varied_is_deterministic() {
        for seed in [2u64, 77, 4096] {
            let (a, b) = (
                SolarSystem::generate_varied(seed),
                SolarSystem::generate_varied(seed),
            );
            assert_eq!(a.planets, b.planets);
            assert_eq!(a.star.star_type, b.star.star_type);
        }
    }

    #[test]
    fn add_planet_extends_a_varied_system() {
        let mut s = (0..2000u64)
            .map(SolarSystem::generate_varied)
            .find(|s| s.planet_count() == 3)
            .expect("a 3-planet system");
        let last = s.planets[2].orbit_radius;
        let i = s.add_planet(PlanetType::Ice, 100.0).unwrap();
        assert_eq!(i, s.planets.len() - 1);
        assert_eq!(body_label(&s, i), "4");
        assert!(s.planets[i].orbit_radius > last);
        assert_eq!(
            s.planets[i].traits,
            crate::planet::traits::PlanetTraits::NEUTRAL
        );
        while s.planet_count() < MAX_PLANET_COUNT {
            s.add_planet(PlanetType::Ice, 100.0).unwrap();
        }
        assert_eq!(s.planet_count(), MAX_PLANET_COUNT);
        assert!(s.add_planet(PlanetType::Ice, 100.0).is_err());
    }

    use crate::player::controller::FlightKeys;

    #[test]
    fn the_start_system_has_eight_planets_with_cratered_last() {
        let s = SolarSystem::generate(1);
        assert_eq!(s.planet_count(), 8);
        for (i, p) in s.planets[..s.planet_count()].iter().enumerate() {
            assert_eq!(p.planet_type, PlanetType::ALL[i]);
            assert_eq!(p.traits, crate::planet::traits::PlanetTraits::NEUTRAL);
        }
        assert_eq!(s.planets[7].planet_type, PlanetType::Cratered);
    }

    // the new outermost planet leaves #1-#7 as they were: all of a planet's data comes from (seed, index)
    #[test]
    fn the_first_seven_planets_are_unchanged() {
        // literal values for seed 1, recorded before the Cratered planet and the 4x scale; scaled here
        const RADII: [f32; 7] = [
            168.48273, 228.66467, 244.23021, 118.63764, 229.99347, 128.8151, 65.814064,
        ];
        const ORBIT_RADII: [f64; 7] = [
            8265.632315539202,
            12703.207739244883,
            17481.896074639295,
            28813.253273767295,
            43261.14888753833,
            65883.98870669717,
            92760.43226435978,
        ];
        let s = SolarSystem::generate(1);
        for i in 0..7 {
            let (r, phase, tilt, node) = orbit_for(1, i);
            let p = &s.planets[i];
            assert!(
                (p.radius - RADII[i] * SYSTEM_SCALE as f32).abs() < 1e-3,
                "planet {i} radius"
            );
            assert!(
                (p.orbit_radius - ORBIT_RADII[i] * SYSTEM_SCALE).abs() < 1e-3,
                "planet {i} orbit"
            );
            assert_eq!(p.orbit_radius, r);
            assert_eq!(p.orbit_phase, phase);
            assert_eq!(p.orbit_tilt, tilt);
            assert_eq!(p.orbit_node, node);
            assert_eq!(p.noise_seed, noise_seed_for(1, i));
        }
        // the outermost keeps the neighbour gap and the system stays inside the planet cap
        assert!(s.planets[7].orbit_radius >= s.planets[6].orbit_radius * 1.29);
        assert!(s.planets.len() <= MAX_PLANETS);
    }

    // the 4x scale changes no angle: the star's apparent size and every planet's share of its orbit
    #[test]
    fn the_scale_keeps_every_angle() {
        // the seed-1 values before SYSTEM_SCALE (the_first_seven_planets_are_unchanged's literals) and its
        // Yellow star's old radius
        const OLD_STAR: f64 = 3_000.0;
        const RADII: [f32; 7] = [
            168.48273, 228.66467, 244.23021, 118.63764, 229.99347, 128.8151, 65.814064,
        ];
        const ORBIT_RADII: [f64; 7] = [
            8265.632315539202,
            12703.207739244883,
            17481.896074639295,
            28813.253273767295,
            43261.14888753833,
            65883.98870669717,
            92760.43226435978,
        ];
        let s = SolarSystem::generate(1);
        assert!((s.star.radius - OLD_STAR * SYSTEM_SCALE).abs() < 1e-6);
        for i in 0..7 {
            let p = &s.planets[i];
            let old_angle = (OLD_STAR / ORBIT_RADII[i]).min(1.0).asin() as f32;
            assert!(
                (p.star_angular_radius(&s.star) - old_angle).abs() < 1e-6,
                "{i}"
            );
            let share = p.radius as f64 / p.orbit_radius;
            assert!(
                (share - RADII[i] as f64 / ORBIT_RADII[i]).abs() < 1e-8,
                "{i}"
            );
        }
        const _: () = assert!(MIN_PLANET_RADIUS == 160.0 && MAX_PLANET_RADIUS == 1000.0);
    }

    #[test]
    fn speeds_scale_with_the_distances_and_periods_do_not() {
        let s = SolarSystem::generate(1);
        // Kepler's reference is unchanged: 180 s at INNER_ORBIT_RADIUS (now 32,000)
        assert!(
            (orbit_speed_for(INNER_ORBIT_RADIUS) - std::f64::consts::TAU / INNER_ORBIT_PERIOD_SECS)
                .abs()
                < 1e-12
        );
        assert_eq!(INNER_ORBIT_RADIUS, 8_000.0 * SYSTEM_SCALE);
        // crossing the gap between the two innermost orbits at cruise speed takes as long as before (500 u/s)
        let gap = s.planets[1].orbit_radius - s.planets[0].orbit_radius;
        assert!((gap / SPACE_CRUISE_SPEED as f64 - (gap / SYSTEM_SCALE) / 500.0).abs() < 1e-9);
        assert_eq!(SPACE_BOOST_SPEED as f64, 2_000.0 * SYSTEM_SCALE);
        // trips take as long as before
        assert!(
            (crate::space::travel::duration(40_000.0 * SYSTEM_SCALE) - (6.0 + 1.5 * 40f64.log10()))
                .abs()
                < 1e-9
        );
    }

    // a bake with stored edits contains them; one without doesn't
    #[test]
    fn bake_with_edits_restores_them() {
        let system = SolarSystem::generate(1);
        let mut planet = system.planets[0];
        planet.radius = 40.0; // a small bake keeps the debug test quick (planet #1 is res 1348 at the 4x scale)
        let mut first = planet.bake();
        let (face, u, v) = (0u8, 40u32, 40u32);
        let top = first.terrain.get_height(face, u, v) + 1; // one block on the surface: under the ceiling
        let block = crate::planet::data::BlockId {
            face,
            layer: top,
            u,
            v,
        };
        first
            .add_block(block, crate::planet::material::BlockType::Stone)
            .unwrap();
        let edits = first.take_edits();
        assert!(planet.bake_with_edits(Some(edits)).exists(block));
        assert!(!planet.bake_with_edits(None).exists(block));
    }

    #[test]
    fn start_planet_is_the_first_of_the_preferred_type() {
        let mut g = SolarSystem::generate(1);
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
        let mut a = SolarSystem::generate(1);
        let i = a.add_planet(PlanetType::Ice, 120.0).unwrap();
        assert_eq!(i, a.planets.len() - 1, "after the moons");
        let p = a.planets[i];
        let base = INNER_ORBIT_RADIUS * ORBIT_RADIUS_GROWTH.powi(PLANET_COUNT as i32);
        assert!(p.orbit_radius >= 0.9 * base && p.orbit_radius <= 1.15 * base);
        assert!(
            p.orbit_radius > a.planets[PLANET_COUNT - 1].orbit_radius * 1.25,
            "inside the last orbit"
        );
        assert_eq!(p.planet_type, PlanetType::Ice);
        assert_eq!(p.radius, 120.0);
        assert_eq!(p.noise_seed, noise_seed_for(1, PLANET_COUNT));
        let mut b = SolarSystem::generate(1);
        b.add_planet(PlanetType::Ice, 120.0).unwrap();
        assert_eq!(a.planets[i], b.planets[i]);
    }

    #[test]
    fn adding_beyond_the_cap_or_out_of_range_fails() {
        let mut g = SolarSystem::generate(1);
        assert!(g.add_planet(PlanetType::Ice, 10.0).is_err());
        assert!(g
            .add_planet(PlanetType::Ice, MAX_ADDED_RADIUS + 100.0)
            .is_err());
        while g.planet_count() < MAX_PLANET_COUNT {
            g.add_planet(PlanetType::EarthLike, 50.0).unwrap();
        }
        assert!(
            g.add_planet(PlanetType::EarthLike, 50.0).is_err(),
            "full at MAX_PLANET_COUNT"
        );
        assert_eq!(g.planet_count(), MAX_PLANET_COUNT);
    }

    // captured over a planet's equator (local up = +X here), levelled, looking along the horizon:
    // sideways mouse must turn the view around the local up — before, it turned around the spin
    // axis (Y), which there pitched the view instead, so the player couldn't turn left or right
    #[test]
    fn captured_yaw_turns_about_the_local_up() {
        let up = Vec3::X;
        // camera right = -Y, camera up = +X (radial), looking along -Z
        let rot = Quat::from_mat3(&glam::Mat3::from_cols(Vec3::NEG_Y, Vec3::X, Vec3::Z));
        let mut flight = SpaceFlight::new(DVec3::new(600.0, 0.0, 0.0));
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
    fn turn_of(flight: &mut SpaceFlight, up: Option<Vec3>) -> (f32, f32) {
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
    // relative to system +Y), sideways mouse turns the view sideways — before, it turned about +Y,
    // which pitched the view or mirrored the turn
    #[test]
    fn free_flight_mouse_turns_sideways_at_any_roll() {
        for roll in [0.0f32, 1.0, 1.5708, 2.5, 3.14159] {
            let mut flight = SpaceFlight::new(DVec3::ZERO);
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
            let mut flight = SpaceFlight::new(DVec3::new(0.0, 600.0, 0.0));
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
            let mut flight = SpaceFlight::new(DVec3::new(0.0, 600.0, 0.0));
            flight.rotation = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.2);
            let (forward, camera_up) = (flight.rotation * Vec3::NEG_Z, flight.rotation * Vec3::Y);
            let step = |flight: &mut SpaceFlight, keys| {
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
                flight.roll_rate < 0.5 * crate::player::smoothing::ROLL_SPEED,
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
                flight.roll_rate > 0.5 * crate::player::smoothing::ROLL_SPEED,
                "{up:?}: stopped dead"
            );
            for _ in 0..20 {
                step(&mut flight, FlightKeys::default());
            }
            assert_eq!(flight.roll_rate, 0.0, "{up:?}: still rolling");
        }
    }

    // Q/E turn (Q left) in space flight, without rolling
    #[test]
    fn q_e_turn_in_space_flight() {
        let mut flight = SpaceFlight::new(DVec3::ZERO);
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

    // far from the planet (from CO_ROTATION_OUTER_RADII) captured flight keeps its attitude against
    // the stars: undoing the planet frame's spin each step leaves the space orientation
    // unchanged; close to it (below CO_ROTATION_INNER_RADII) it turns with the planet as before
    #[test]
    fn far_out_captured_flight_keeps_still_against_the_stars() {
        let system = SolarSystem::generate(1);
        let p = &system.planets[0];
        let start = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.3);
        for (distance, inertial) in [(5.0f32, true), (1.2, false)] {
            let (mut local, mut t, dt) = (start, 0.0f64, 1.0 / 60.0);
            for _ in 0..600 {
                t += dt;
                local = crate::space::solar_system::counter_spin(
                    p.spin_rate() as f32,
                    distance,
                    dt as f32,
                ) * local;
            }
            let before = p.rotation_from_planet_frame(start, 0.0) * Vec3::NEG_Z;
            let after = p.rotation_from_planet_frame(local, t) * Vec3::NEG_Z;
            if inertial {
                assert!(after.dot(before) > 1.0 - 1e-4, "turned with the planet");
            } else {
                assert_eq!(local, start, "didn't turn with the planet close to it");
            }
        }
        assert_eq!(co_rotation(CO_ROTATION_INNER_RADII), 1.0);
        assert_eq!(co_rotation(CO_ROTATION_OUTER_RADII), 0.0);
    }

    // free flight: Space/Shift climb and descend along the camera's own up
    #[test]
    fn free_flight_climbs_along_the_camera_up() {
        let mut flight = SpaceFlight::new(DVec3::ZERO);
        flight.rotation = Quat::from_rotation_z(3.14159); // upside down relative to system +Y
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
        let g = SolarSystem::generate(1);
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
        let g = SolarSystem::generate(1);
        let t = 300.0;
        let p = &g.planets[4];
        let pos = p.position_at(t) + DVec3::new(0.0, 3.0 * p.radius as f64, 0.0);
        assert_eq!(
            next_flight_frame(&g, FlightFrame::Free, pos, t),
            FlightFrame::Captured(4)
        );
    }

    // capture/release must not move the view: the same space position and orientation before
    // and after, whichever way round (the flicker case at exactly 10 radii relies on this)
    #[test]
    fn change_frame_keeps_the_absolute_view() {
        let g = SolarSystem::generate(1);
        let p = &g.planets[0];
        let t = 77.0;
        let mut flight = SpaceFlight::new(p.position_at(t) + DVec3::new(500.0, 200.0, -300.0));
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
    fn orientation_maps_the_planet_frame_into_space() {
        let p = SolarSystem::generate(1).planets[2];
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

    fn inner_planet() -> Planet {
        SolarSystem::generate(1).planets[0]
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
        let g = SolarSystem::generate(1);
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

    #[test]
    fn the_solar_day_follows_the_scaled_day_length() {
        let mut p = SolarSystem::generate(1).planets[2];
        p.orbit_tilt = 0.0;
        p.traits.day_scale = 1.5;
        let day = p.day_length();
        assert!((day - 1.5 * p.planet_type.def().day_length_secs as f64).abs() < 1e-9);
        let star = DVec3::ZERO;
        let a = p.sun_dir_in_planet_frame(star, 5.0);
        let b = p.sun_dir_in_planet_frame(star, 5.0 + day);
        assert!((a - b).length() < 1e-3, "{a:?} vs {b:?}");
    }

    #[test]
    fn bake_carries_the_traits() {
        let mut p = SolarSystem::generate(1).planets[6];
        p.radius = 40.0; // keeps the bake quick in a debug test build
        p.traits.gravity = 0.7;
        assert_eq!(p.bake().traits.gravity, 0.7);
    }

    // the spin includes the orbit rate, so a day lasts exactly the planet type's day length (the
    // same as on the home planet) despite the planet also moving around the star
    #[test]
    fn solar_day_equals_the_planet_types_day_length() {
        let g = SolarSystem::generate(1);
        // exact for an orbit in the common plane; on a tilted orbit (spin axis still +Y) the sun's height
        // swings by up to +-tilt over the year (small seasons), so within one day it can shift by up to twice
        // the tilt's sine (inner planets cover much of their orbit in one of their days)
        let flat = Planet {
            orbit_tilt: 0.0,
            ..g.planets[0]
        };
        for p in g.planets.iter().chain([&flat]) {
            let tolerance = 2.0 * p.orbit_tilt.sin() as f32 + 1e-3;
            let day = p.day_length();
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

    // the bake of the largest planet (radius 1000, resolution 2000); run with
    // `cargo test --release largest_planet_bake -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn largest_planet_bake_time() {
        let mut p = inner_planet();
        p.radius = MAX_PLANET_RADIUS;
        let t = std::time::Instant::now();
        let data = p.bake();
        println!("bake radius {}: {:?}", p.radius, t.elapsed());
        drop(data);
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
        let mut p = SolarSystem::generate(1).planets[1]; // Volcanic
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
    fn generate_produces_eight_planets() {
        let g = SolarSystem::generate(1);
        assert_eq!(g.planet_count(), 8);
    }

    #[test]
    fn generate_orbit_radii_strictly_increase() {
        let g = SolarSystem::generate(1);
        for pair in g.planets[..g.planet_count()].windows(2) {
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
        let a = SolarSystem::generate(42);
        let b = SolarSystem::generate(42);
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
        let a = SolarSystem::generate(1);
        let b = SolarSystem::generate(2);
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
        let g = SolarSystem::generate(7);
        for p in g.planets.iter().filter(|p| !p.is_moon()) {
            assert!(
                (MIN_PLANET_RADIUS..=MAX_PLANET_RADIUS).contains(&p.radius),
                "radius {} out of range",
                p.radius
            );
        }
    }

    // regression test for a real bug: the hash's finalizer was missing a round and didn't
    // avalanche small sequential `index` values enough, so an entire system's planets came out
    // clustered within a few units of each other (observed: 160-164 across all 7 planets) instead
    // of spanning anywhere near the intended 40-250 range. "in range" alone doesn't catch this —
    // a tightly clustered set of values is still technically in range.
    #[test]
    fn generate_planet_radii_are_well_spread_not_clustered() {
        for seed in [1, 2, 7, 42] {
            let g = SolarSystem::generate(seed);
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
        let g = SolarSystem::generate(1);
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
        let g = SolarSystem::generate(1);
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
        let g = SolarSystem::generate(1);
        for p in g.planets.iter().filter(|p| !p.is_moon()) {
            let expected = 180.0 * (p.orbit_radius / INNER_ORBIT_RADIUS).powf(1.5);
            assert!((std::f64::consts::TAU / p.orbit_speed - expected).abs() < 1e-6 * expected);
        }
    }

    // starting angles are seeded, not the old even spread, and differ between systems
    #[test]
    fn starting_angles_are_seeded() {
        let (a, b) = (SolarSystem::generate(1), SolarSystem::generate(2));
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
            let g = SolarSystem::generate(seed);
            for (i, p) in g.planets[..g.planet_count()].iter().enumerate() {
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
        let varied = (0..20u64).map(|s| SolarSystem::generate(s).planets[3].orbit_radius);
        let (lo, hi) = varied.fold((f64::MAX, 0.0f64), |(l, h), r| (l.min(r), h.max(r)));
        assert!(hi - lo > 0.05 * lo, "radii don't vary between seeds");
    }

    // orbits are tilted a little (up to MAX_ORBIT_TILT) out of the common plane, staying circles
    #[test]
    fn orbits_are_slightly_tilted() {
        let g = SolarSystem::generate(1);
        let mut out_of_plane = false;
        for p in g.planets.iter().filter(|p| !p.is_moon()) {
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
        let g = SolarSystem::generate(1);
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
        let g = SolarSystem::generate(9);
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
        let system = SolarSystem::generate(1);
        let r = system.star.radius;
        assert_eq!(star_heat(HEAT_ZONE_RADII * r, r), 0.0);
        assert_eq!(star_heat(5.0 * r, r), 0.0);
        let inner_orbit = system
            .planets
            .iter()
            .filter(|p| !p.is_moon())
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
        let system = SolarSystem::generate(1);
        let r = system.star.radius;
        let mut flight = SpaceFlight::new(DVec3::new(0.0, 0.0, 1.9 * r));
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
        let mut at_limit = SpaceFlight::new(DVec3::new(0.0, 0.0, HEAT_LIMIT_RADII * r));
        at_limit.velocity = DVec3::new(0.0, 0.0, -(SPACE_BOOST_SPEED as f64));
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
    fn space_flight_brakes_from_boost_within_a_second() {
        let mut flight = SpaceFlight::new(DVec3::ZERO);
        flight.velocity = DVec3::new(0.0, 0.0, -SPACE_BOOST_SPEED as f64);
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
            flight.velocity.length() < 50.0 * SYSTEM_SCALE, // speeds scale with the system
            "{}",
            flight.velocity.length()
        );
    }

    #[test]
    fn space_flight_update_moves_position_in_input_direction() {
        let mut flight = SpaceFlight::new(DVec3::ZERO);
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
        let g = SolarSystem::generate(1);
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
        let a = SolarSystem::generate(42);
        let b = SolarSystem::generate(42);
        for (pa, pb) in a.planets.iter().zip(b.planets.iter()) {
            assert_eq!(pa.noise_seed, pb.noise_seed);
        }
    }

    #[test]
    fn noise_seed_differs_from_planet_radius_hash() {
        // sanity check that the two per-planet hashes aren't accidentally the same function
        // reused (which would correlate radius and terrain shape in a confusing way)
        let g = SolarSystem::generate(1);
        let p = &g.planets[0];
        assert_ne!(
            p.noise_seed as f32, p.radius,
            "noise_seed and radius should come from independent hashes"
        );
    }
}
