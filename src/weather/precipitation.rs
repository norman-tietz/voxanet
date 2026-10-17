// precipitation.rs
// Rain, snow, ash and dust around the camera (spec docs/superpowers/specs/2026-10-11-precipitation-design.md),
// the CPU side: what falls (the planet type's kind, the weather at the camera), how much (instances
// drawn), the wind, the particles' accumulated offset in the camera's local frame (so the rain stays
// fixed in the world while the camera moves through it), and the roof grid that keeps precipitation
// out from under roofs. precip.wgsl draws it: stateless particles, position = hash(index) + offset,
// wrapped in a box around the camera.

use glam::Vec3;

use crate::planet::data::PlanetData;
use crate::weather::{WeatherForce, WeatherKind, WeatherSettings};

pub const BOX_SIZE: Vec3 = Vec3::new(40.0, 30.0, 40.0); // east, up, north (precip.wgsl BOX_SIZE)
pub const WIND_MAX: f32 = 8.0; // units/s at storm 1
const JUMP: f32 = 30.0; // camera movement in one frame beyond which it's a teleport, not a walk
pub const GRID: usize = 64; // roof grid cells per side (precip.wgsl ROOF_GRID)
pub const CELL: f32 = 1.0; // roof grid cell size, units
const REBUILD_DIST: f32 = 8.0; // tangent-plane distance from the grid origin that rebuilds it

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecipKind {
    Rain,
    Snow,
    Ash,
    Dust,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindParams {
    pub n_max: u32,
    pub fall: f32, // units/s along local down
    pub sway: f32, // sideways amplitude, units
    pub size: f32, // flake size, or streak width for rain, units
    pub alpha: f32,
    pub color: [f32; 3], // linear; rain is lit sky colour (the shader ignores this for rain)
}

impl PrecipKind {
    pub fn params(self) -> KindParams {
        match self {
            PrecipKind::Rain => KindParams {
                n_max: 16000,
                fall: 9.0,
                sway: 0.0,
                size: 0.012,
                alpha: 0.25,
                color: [0.0; 3],
            },
            PrecipKind::Snow => KindParams {
                n_max: 8000,
                fall: 1.2,
                sway: 0.3,
                size: 0.12,
                alpha: 0.9,
                color: [0.9, 0.9, 0.95],
            },
            PrecipKind::Ash => KindParams {
                n_max: 8000,
                fall: 1.6,
                sway: 0.2,
                size: 0.12,
                alpha: 0.8,
                color: [0.12, 0.12, 0.12],
            },
            PrecipKind::Dust => KindParams {
                n_max: 6000,
                fall: 0.3,
                sway: 0.0,
                size: 0.03,
                alpha: 0.5,
                color: [0.55, 0.40, 0.22],
            },
        }
    }

    // the shader's kind (precip.wgsl)
    pub fn index(self) -> u32 {
        match self {
            PrecipKind::Rain => 0,
            PrecipKind::Snow => 1,
            PrecipKind::Ash => 2,
            PrecipKind::Dust => 3,
        }
    }
}

// what falls on a planet of this weather kind, given the snow share at the camera
pub fn kind_for(kind: WeatherKind, snow_share: f32) -> Option<PrecipKind> {
    match kind {
        WeatherKind::Rain if snow_share > 0.5 => Some(PrecipKind::Snow),
        WeatherKind::Rain | WeatherKind::AcidRain => Some(PrecipKind::Rain),
        WeatherKind::Snow => Some(PrecipKind::Snow),
        WeatherKind::Ash => Some(PrecipKind::Ash),
        WeatherKind::Dust => Some(PrecipKind::Dust),
        WeatherKind::None => None,
    }
}

pub fn instances(kind: PrecipKind, intensity: f32) -> u32 {
    let n = kind.params().n_max;
    ((n as f32 * intensity.clamp(0.0, 1.0)).round() as u32).min(n)
}

// none from the cloud base up (`radius_ratio` = camera radius / planet radius)
pub fn altitude_fade(radius_ratio: f32) -> f32 {
    let base = crate::planet::data::CLOUD_ALT;
    1.0 - crate::weather::smoothstep(0.9 * base, base, radius_ratio)
}

// the weather's wind at direction `dir`: the fronts travel with the weather field's rotation about
// the spin axis (weather.wgsl turns the noise domain by +angle about Y, so features move the other
// way: velocity ∝ dir × Y), east–west, fading toward the poles; × storm
pub fn wind_at(dir: Vec3, storm: f32) -> Vec3 {
    dir.normalize_or_zero().cross(Vec3::Y) * WIND_MAX * storm
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub east: Vec3,
    pub up: Vec3,
    pub north: Vec3,
}

// the local frame at `pos`: up from the planet centre, east around the spin axis (X near the poles)
pub fn frame_at(pos: Vec3) -> Frame {
    let up = pos.normalize_or_zero();
    let up = if up == Vec3::ZERO { Vec3::Y } else { up };
    let east = Vec3::Y.cross(up);
    let east = if east.length() < 1e-3 {
        Vec3::X.cross(up).normalize()
    } else {
        east.normalize()
    };
    let north = up.cross(east);
    Frame { east, up, north }
}

pub fn wrap_box(v: Vec3) -> Vec3 {
    v - BOX_SIZE * (v / BOX_SIZE).floor()
}

#[derive(Clone, Debug)]
pub struct RoofGrid {
    pub origin: Vec3, // the camera's point on its column's surface when built
    pub east: Vec3,
    pub north: Vec3,
    pub radii: Vec<f32>, // radii[j * GRID + i]: i along east, j along north
}

impl RoofGrid {
    // world position of cell (i, j)'s centre, on the plane through the origin
    pub fn cell_centre(&self, i: usize, j: usize) -> Vec3 {
        let half = GRID as f32 * 0.5;
        self.origin
            + self.east * ((i as f32 + 0.5 - half) * CELL)
            + self.north * ((j as f32 + 0.5 - half) * CELL)
    }

    // the cell whose column a world point is looked up in: the point projected radially onto the
    // grid plane, then its plane coordinates / CELL. Line-by-line the same maths as vs_precip in
    // precip.wgsl; None outside the grid or when the point is not above the plane's hemisphere.
    #[allow(dead_code)] // the shader does this at runtime; the Rust copy is for tests
    pub fn cell_of(&self, world: Vec3) -> Option<(usize, usize)> {
        let n = world.normalize();
        let r0 = self.origin.length();
        let gu = self.origin / r0;
        let dn = n.dot(gu);
        if dn <= 0.0 {
            return None;
        }
        let q = n * (r0 / dn);
        let g = q - self.origin;
        let x = (g.dot(self.east) / CELL).floor() as i32 + GRID as i32 / 2;
        let y = (g.dot(self.north) / CELL).floor() as i32 + GRID as i32 / 2;
        if x < 0 || y < 0 || x >= GRID as i32 || y >= GRID as i32 {
            return None;
        }
        Some((x as usize, y as usize))
    }

    // each cell: the radius of its column's highest surface, the top block (edits and plants included,
    // so canopies are roofs) or the water, whichever is higher; 0 where no column is found
    pub fn build(planet: &PlanetData, cam: Vec3) -> RoofGrid {
        let res = planet.resolution;
        let f = frame_at(cam);
        let top = |p: Vec3| -> f32 {
            planet.cell_at(p).map_or(0.0, |id| {
                let block = crate::meshing::gen::CoordSystem::get_layer_radius(
                    planet.top(id.face, id.u, id.v) + 1,
                    res,
                );
                block.max(planet.water_surface_radius(id.face, id.u, id.v))
            })
        };
        // the plane sits on the ground under the camera, so every cell centre is inside the planet's
        // layers (cell_at needs a position, its radius picks the layer only)
        let ground_r = planet.cell_at(f.up * res as f32 * 0.5).map_or(0.0, |id| {
            crate::meshing::gen::CoordSystem::get_layer_radius(
                planet.surface(id.face, id.u, id.v) + 1,
                res,
            )
            .max(planet.water_surface_radius(id.face, id.u, id.v))
        });
        let r0 = ground_r.max(res as f32 * 0.25);
        let mut g = RoofGrid {
            origin: f.up * r0,
            east: f.east,
            north: f.north,
            radii: vec![0.0; GRID * GRID],
        };
        for j in 0..GRID {
            for i in 0..GRID {
                let p = g.cell_centre(i, j);
                g.radii[j * GRID + i] = top(p.normalize() * r0);
            }
        }
        g
    }
}

pub struct PrecipFrame {
    pub kind: Option<PrecipKind>,
    pub instances: u32,
    pub frame: Frame,
    pub offset: Vec3, // east, up, north; inside the box
    pub wind: Vec3,   // world, units/s
    pub grid_rebuilt: bool,
}

pub struct Precipitation {
    pub offset: Vec3,
    prev_cam: Option<Vec3>,
    grid: Option<RoofGrid>,
    grid_edits: u64,
}

impl Precipitation {
    pub fn new() -> Self {
        Self {
            offset: Vec3::ZERO,
            prev_cam: None,
            grid: None,
            grid_edits: 0,
        }
    }

    // a planet switch or Space mode: start over (the grid is rebuilt on the next update)
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn grid(&self) -> Option<&RoofGrid> {
        self.grid.as_ref()
    }

    // once per planet frame: `cam` in the planet frame, `t` the clock wrapped at 3600
    pub fn update(
        &mut self,
        dt: f32,
        cam: Vec3,
        planet: &PlanetData,
        weather: &WeatherSettings,
        force: &WeatherForce,
        t: f32,
    ) -> PrecipFrame {
        let frame = frame_at(cam);
        let dir = frame.up;
        let s = crate::weather::sample(dir, t, force);
        let kind = crate::weather::WeatherKind::for_planet(planet.planet_type);
        let precip = kind_for(kind, s[3]);
        let ratio = cam.length() / (planet.resolution as f32 * 0.5).max(1.0);
        let intensity = crate::weather::precipitation(s[0]) * altitude_fade(ratio);
        let on = weather.on && weather.particles;
        let wind = wind_at(dir, s[0]);

        // the offset: this frame's fall and wind, minus the camera's own movement (a jump is a
        // teleport: no camera term, or the rain would streak past)
        let fall = precip.map_or(0.0, |k| k.params().fall);
        let v = Vec3::new(wind.dot(frame.east), -fall, wind.dot(frame.north));
        let mut offset = self.offset + v * dt;
        if let Some(prev) = self.prev_cam {
            let d = cam - prev;
            if d.length() < JUMP {
                offset -= Vec3::new(d.dot(frame.east), d.dot(frame.up), d.dot(frame.north));
            }
        }
        self.offset = wrap_box(offset);
        self.prev_cam = Some(cam);

        let instances = match precip {
            Some(k) if on => instances(k, intensity),
            _ => 0,
        };
        // the roof grid: rebuilt when the camera left its middle, or after an edit, but only on a
        // frame that draws (a stale grid waits for the first drawing frame, before the upload)
        let edits = planet.edit_generation();
        let stale = match &self.grid {
            None => true,
            Some(g) => {
                let d = cam.normalize_or_zero() * g.origin.length() - g.origin;
                let tangent = Vec3::new(d.dot(g.east), 0.0, d.dot(g.north)).length();
                tangent > REBUILD_DIST || edits != self.grid_edits
            }
        } && instances > 0;
        if stale {
            self.grid = Some(RoofGrid::build(planet, cam));
            self.grid_edits = edits;
        }

        PrecipFrame {
            kind: precip,
            instances,
            frame,
            offset: self.offset,
            wind,
            grid_rebuilt: stale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biome::PlanetType;
    use crate::planet::data::PlanetData;
    use crate::weather::{WeatherCommand, WeatherKind, WeatherSettings};
    use glam::Vec3;

    fn planet() -> PlanetData {
        PlanetData::new_for_type(64, crate::planet::noise::HOME_SEED, PlanetType::EarthLike)
    }

    // a point on the ground of `p` in direction `dir`, eye height above its column's surface
    fn ground(p: &PlanetData, dir: Vec3) -> Vec3 {
        let id = p
            .cell_at(dir.normalize() * p.resolution as f32 * 0.5)
            .unwrap();
        let r = crate::meshing::gen::CoordSystem::get_layer_radius(
            p.surface(id.face, id.u, id.v) + 1,
            p.resolution,
        );
        dir.normalize() * (r + 1.6)
    }

    fn storm_everywhere() -> (WeatherSettings, crate::weather::WeatherForce) {
        let mut s = WeatherSettings::default();
        s.apply(
            WeatherCommand::Force(crate::weather::Force::Storm),
            WeatherKind::Rain,
            Vec3::X,
            0.0,
        );
        // region test helper: force centred on +X, fully faded in at t = 10
        let f = s.force_uniform();
        (s, f)
    }

    #[test]
    fn frame_is_orthonormal_everywhere() {
        for dir in [
            Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::new(0.0, 0.999_999, 0.001),
            Vec3::new(0.3, -0.5, 0.8),
        ] {
            let f = frame_at(dir * 100.0);
            for v in [f.east, f.up, f.north] {
                assert!(
                    (v.length() - 1.0).abs() < 1e-4 && v.is_finite(),
                    "{dir}: {v}"
                );
            }
            assert!(
                f.east.dot(f.up).abs() < 1e-4
                    && f.north.dot(f.up).abs() < 1e-4
                    && f.east.dot(f.north).abs() < 1e-4
            );
            assert!((f.up - dir.normalize()).length() < 1e-4);
        }
    }

    #[test]
    fn wrap_keeps_the_offset_inside_the_box() {
        for v in [-1000.3, -40.0, -0.1, 0.0, 39.99, 41.0, 1e5] {
            let w = wrap_box(Vec3::splat(v));
            assert!(
                w.x >= 0.0
                    && w.x < BOX_SIZE.x
                    && w.y >= 0.0
                    && w.y < BOX_SIZE.y
                    && w.z >= 0.0
                    && w.z < BOX_SIZE.z,
                "{v}: {w}"
            );
        }
    }

    // the rain stays put in the world: moving the camera by d moves the offset by −d (in the frame)
    #[test]
    fn camera_movement_makes_the_rain_world_fixed() {
        let p = planet();
        let (s, f) = storm_everywhere();
        let mut pr = Precipitation::new();
        let cam = ground(&p, Vec3::X);
        pr.update(0.0, cam, &p, &s, &f, 10.0);
        let before = pr.offset;
        let fr = frame_at(cam);
        let step = fr.east * 0.7;
        pr.update(0.0, cam + step, &p, &s, &f, 10.0);
        let moved = wrap_box(pr.offset - before + BOX_SIZE * 0.5) - BOX_SIZE * 0.5;
        // the frame turns slightly with the step around the planet (~0.02 rad here): a small tolerance
        assert!(
            (moved.x + 0.7).abs() < 0.02 && moved.y.abs() < 0.02 && moved.z.abs() < 0.02,
            "{moved}"
        );
    }

    // Review Focus 1
    #[test]
    fn a_jump_resets_instead_of_streaking() {
        let p = planet();
        let (s, f) = storm_everywhere();
        let mut pr = Precipitation::new();
        pr.update(0.016, ground(&p, Vec3::X), &p, &s, &f, 10.0);
        let before = pr.offset;
        pr.update(0.016, ground(&p, Vec3::Z), &p, &s, &f, 10.016);
        // far more than a box away: the offset only got this frame's fall, no camera term
        let moved = wrap_box(pr.offset - before + BOX_SIZE * 0.5) - BOX_SIZE * 0.5;
        assert!(moved.length() < 1.0, "{moved}");
    }

    #[test]
    fn wind_blows_east_west_with_the_storm() {
        let w = wind_at(Vec3::X, 1.0);
        assert!(w.dot(Vec3::X).abs() < 1e-5, "tangent: {w}");
        assert!((w.length() - WIND_MAX).abs() < 1e-3);
        assert!(
            (w.normalize() - Vec3::X.cross(Vec3::Y)).length() < 1e-4,
            "the fronts' travel: {w}"
        );
        assert_eq!(wind_at(Vec3::X, 0.0), Vec3::ZERO);
        assert!(wind_at(Vec3::Y, 1.0).length() < 1e-5, "none at the pole");
        assert!(wind_at(Vec3::new(1.0, 1.0, 0.0).normalize(), 1.0).length() < WIND_MAX * 0.75);
    }

    #[test]
    fn kinds_follow_the_planet_type_and_snow_share() {
        assert_eq!(kind_for(WeatherKind::Rain, 0.2), Some(PrecipKind::Rain));
        assert_eq!(kind_for(WeatherKind::Rain, 0.8), Some(PrecipKind::Snow));
        assert_eq!(kind_for(WeatherKind::AcidRain, 0.8), Some(PrecipKind::Rain));
        assert_eq!(kind_for(WeatherKind::Snow, 0.0), Some(PrecipKind::Snow));
        assert_eq!(kind_for(WeatherKind::Ash, 0.0), Some(PrecipKind::Ash));
        assert_eq!(kind_for(WeatherKind::Dust, 0.0), Some(PrecipKind::Dust));
        assert_eq!(kind_for(WeatherKind::None, 0.0), None);
    }

    #[test]
    fn instances_grow_with_the_rain() {
        assert_eq!(instances(PrecipKind::Rain, 0.0), 0);
        assert_eq!(instances(PrecipKind::Rain, 1.0), 16000);
        assert!(instances(PrecipKind::Rain, 0.3) < instances(PrecipKind::Rain, 0.6));
        assert_eq!(instances(PrecipKind::Snow, 1.0), 8000);
        assert_eq!(instances(PrecipKind::Dust, 2.0), 6000, "clamped to N_max");
    }

    // Review Focus 4
    #[test]
    fn nothing_above_the_cloud_base() {
        assert_eq!(altitude_fade(1.0), 1.0);
        assert_eq!(altitude_fade(crate::planet::data::CLOUD_ALT), 0.0);
        assert_eq!(altitude_fade(crate::planet::data::CLOUD_ALT * 2.0), 0.0);
        let p = planet();
        let (s, f) = storm_everywhere();
        let mut pr = Precipitation::new();
        let high = Vec3::X * p.resolution as f32 * 0.5 * crate::planet::data::CLOUD_ALT * 1.01;
        assert_eq!(pr.update(0.016, high, &p, &s, &f, 10.0).instances, 0);
    }

    // Review Focus 5
    #[test]
    fn off_draws_nothing() {
        let p = planet();
        let (mut s, f) = storm_everywhere();
        let mut pr = Precipitation::new();
        let cam = ground(&p, Vec3::X);
        assert!(pr.update(0.016, cam, &p, &s, &f, 10.0).instances > 0);
        s.apply(
            WeatherCommand::Particles(Some(false)),
            WeatherKind::Rain,
            Vec3::X,
            10.0,
        );
        assert_eq!(pr.update(0.016, cam, &p, &s, &f, 10.0).instances, 0);
        s.apply(
            WeatherCommand::Particles(Some(true)),
            WeatherKind::Rain,
            Vec3::X,
            10.0,
        );
        s.apply(WeatherCommand::Off, WeatherKind::Rain, Vec3::X, 10.0);
        let f = s.force_uniform();
        assert_eq!(pr.update(0.016, cam, &p, &s, &f, 10.0).instances, 0);
    }

    // a canopy is a roof: with plants the grid is never lower than without, and higher somewhere
    #[test]
    fn the_roof_grid_includes_plants() {
        let bare = PlanetData::new_for_type(96, 5, PlanetType::EarthLike);
        let forest = bare.clone().with_flora(true, 3.0);
        let mut higher = 0;
        for k in 0..60 {
            let a = k as f32 * 2.399_963;
            let y = 1.0 - 2.0 * (k as f32 + 0.5) / 60.0;
            let r = (1.0 - y * y).sqrt();
            let cam = ground(&bare, Vec3::new(r * a.cos(), y, r * a.sin()));
            let (g, flat) = (RoofGrid::build(&forest, cam), RoofGrid::build(&bare, cam));
            for (a, b) in g.radii.iter().zip(&flat.radii) {
                assert!(a >= b);
                higher += (a > b) as usize;
            }
        }
        assert!(higher > 0, "some cells are under a canopy");
    }

    #[test]
    fn the_grid_holds_roofs_and_water() {
        let mut p = planet();
        let cam = ground(&p, Vec3::X);
        let g = RoofGrid::build(&p, cam);
        assert_eq!(g.radii.len(), GRID * GRID);
        let centre = g.radii[(GRID / 2) * GRID + GRID / 2];
        // place a block well above the centre cell's column: that cell's radius rises
        let id = p
            .cell_at(g.cell_centre(GRID / 2, GRID / 2).normalize() * g.origin.length())
            .unwrap(); // as build() finds the column
        let top = p.surface(id.face, id.u, id.v);
        // the start spot is under the sea: the roof goes above the water, which the grid counts too
        let roof = crate::planet::data::BlockId {
            layer: top.max(p.resolution / 2) + 4,
            ..id
        };
        p.add_block(roof, crate::planet::material::BlockType::Stone)
            .unwrap();
        let g2 = RoofGrid::build(&p, cam);
        assert!(g2.radii[(GRID / 2) * GRID + GRID / 2] > centre + 2.0);
        // every cell is at least its column's water surface
        for j in 0..GRID {
            for i in 0..GRID {
                if let Some(id) = p.cell_at(g2.cell_centre(i, j)) {
                    assert!(g2.radii[j * GRID + i] >= p.water_surface_radius(id.face, id.u, id.v));
                }
            }
        }
    }

    // Review Focus 3, and the movement rule
    #[test]
    fn an_edit_rebuilds_the_grid() {
        let mut p = planet();
        let (s, f) = storm_everywhere();
        let mut pr = Precipitation::new();
        let cam = ground(&p, Vec3::X);
        assert!(
            pr.update(0.016, cam, &p, &s, &f, 10.0).grid_rebuilt,
            "first frame builds"
        );
        assert!(!pr.update(0.016, cam, &p, &s, &f, 10.0).grid_rebuilt);
        let fr = frame_at(cam);
        assert!(
            !pr.update(0.016, cam + fr.east * 5.0, &p, &s, &f, 10.0)
                .grid_rebuilt
        );
        assert!(
            pr.update(0.016, cam + fr.east * 9.0, &p, &s, &f, 10.0)
                .grid_rebuilt
        );
        let id = p.cell_at(cam).unwrap();
        let top = p.surface(id.face, id.u, id.v);
        p.add_block(
            crate::planet::data::BlockId {
                layer: top + 5,
                ..id
            },
            crate::planet::material::BlockType::Stone,
        )
        .unwrap();
        assert!(
            pr.update(0.016, cam + fr.east * 9.0, &p, &s, &f, 10.0)
                .grid_rebuilt
        );
    }

    #[test]
    fn no_grid_is_built_while_nothing_is_drawn() {
        let p = planet();
        let (s, f) = storm_everywhere();
        let mut off = s.clone();
        off.particles = false;
        let mut pr = Precipitation::new();
        let cam = ground(&p, Vec3::X);
        assert!(!pr.update(0.016, cam, &p, &off, &f, 10.0).grid_rebuilt);
        assert!(pr.grid().is_none());
        assert!(
            pr.update(0.016, cam, &p, &s, &f, 10.0).grid_rebuilt,
            "first drawing frame builds"
        );
    }

    // the shader's lookup (cell_of mirrors it) finds the cell a point radially above a centre is in
    #[test]
    fn cell_of_round_trips_radially() {
        let p = planet();
        let g = RoofGrid::build(&p, ground(&p, Vec3::X));
        let r0 = g.origin.length();
        for &(i, j) in &[
            (0, 0),
            (GRID - 1, 0),
            (0, GRID - 1),
            (GRID - 1, GRID - 1),
            (32, 32),
            (5, 50),
            (60, 11),
        ] {
            for h in [0.0, 10.0, 25.0] {
                let w = g.cell_centre(i, j).normalize() * (r0 + h);
                assert_eq!(g.cell_of(w), Some((i, j)), "cell ({i}, {j}) at +{h}");
            }
        }
        assert_eq!(g.cell_of(-g.origin), None);
    }

    #[test]
    fn precip_shader_fades_with_the_storm_fog() {
        let src = include_str!("precip.wgsl");
        assert!(src.contains("fog_storm_weight(global.fog.x)"));
    }

    #[test]
    fn particles_command_parses() {
        assert_eq!(
            WeatherCommand::parse(&["particles"]),
            Ok(WeatherCommand::Particles(None))
        );
        assert_eq!(
            WeatherCommand::parse(&["particles", "off"]),
            Ok(WeatherCommand::Particles(Some(false)))
        );
        assert_eq!(
            WeatherCommand::parse(&["particles", "maybe"]),
            Err(crate::weather::WEATHER_USAGE)
        );
    }
}

// --- GPU: precip.wgsl's uniform, the roof grid texture and the draw ---

// precip.wgsl's Precip uniform
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PrecipUniform {
    east: [f32; 4],   // w: time
    up: [f32; 4],     // w: kind
    north: [f32; 4],  // w: fall speed
    offset: [f32; 4], // w: sway
    wind: [f32; 4],   // w: size
    color: [f32; 4],
    grid_origin: [f32; 4], // w: cell size
    grid_east: [f32; 4],
    grid_north: [f32; 4],
}

pub struct PrecipGpu {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    grid_tex: wgpu::Texture,
    bind: wgpu::BindGroup,
    instances: u32,
}

impl PrecipGpu {
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule, // the scene shader (precip.wgsl is appended to it)
        global_layout: &wgpu::BindGroupLayout,
        sample_layout: &wgpu::BindGroupLayout,
        textures_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("precip_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Precip Uniform"),
            size: std::mem::size_of::<PrecipUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Roof Grid"),
            size: wgpu::Extent3d {
                width: GRID as u32,
                height: GRID as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let grid_view = grid_tex.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("precip_bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&grid_view),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("precip_pipeline_layout"),
            bind_group_layouts: &[
                Some(global_layout),
                Some(&layout),
                Some(sample_layout),
                Some(textures_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Precipitation"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_precip"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_precip"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::render::deferred::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform,
            grid_tex,
            bind,
            instances: 0,
        }
    }

    // this frame's parameters and, when rebuilt, the roof grid; `t`: the clock wrapped at 3600
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        f: &PrecipFrame,
        grid: Option<&RoofGrid>,
        t: f32,
    ) {
        self.instances = f.instances;
        let (Some(kind), Some(grid)) = (f.kind, grid) else {
            self.instances = 0;
            return;
        };
        if f.grid_rebuilt {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.grid_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&grid.radii),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(GRID as u32 * 4),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: GRID as u32,
                    height: GRID as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        if self.instances == 0 {
            return;
        }
        let k = kind.params();
        let v4 = |v: Vec3, w: f32| [v.x, v.y, v.z, w];
        let u = PrecipUniform {
            east: v4(f.frame.east, t),
            up: v4(f.frame.up, kind.index() as f32),
            north: v4(f.frame.north, k.fall),
            offset: v4(f.offset, k.sway),
            wind: v4(f.wind, k.size),
            color: [k.color[0], k.color[1], k.color[2], k.alpha],
            grid_origin: v4(grid.origin, CELL),
            grid_east: v4(grid.east, 0.0),
            grid_north: v4(grid.north, 0.0),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[u]));
    }

    // in the lighting pass after the water: groups 0, 2 and 3 are bound there already
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        if self.instances > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(1, &self.bind, &[]);
            pass.draw(0..4, 0..self.instances);
        }
    }
}
