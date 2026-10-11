// map.rs
// The solar-system map's geometry (spec docs/superpowers/specs/2026-10-12-system-map-travel-design.md §1):
// orbits compressed with a square root so inner and outer planets both fit, one size factor k for every
// planet (relative sizes kept, no overlaps), picking, and the map camera orbiting the star. The renderer
// draws the map with real-unit offsets (render_offset): perspective then shows each body at its mapped
// apparent size while all shading runs at true scale.

use glam::{DVec3, Mat4, Vec2, Vec3};

use crate::space::solar_system::SolarSystem;

pub const D_MAX: f32 = 100.0; // map distance of the outermost orbit
pub const STAR_MAP_RADIUS: f32 = 5.0; // the star's radius on the map
pub const MAX_FILL: f32 = 0.35; // the largest planet's share of the smallest orbit gap

pub struct MapLayout {
    pub k: f32, // map units per real unit for planet radii
    r_outer: f64,
}

impl MapLayout {
    pub fn new(system: &SolarSystem) -> Self {
        let r_outer = system
            .planets
            .iter()
            .map(|p| p.orbit_radius)
            .fold(1.0, f64::max);
        let mut m = Self { k: 1.0, r_outer };
        let mut d: Vec<f32> = system
            .planets
            .iter()
            .map(|p| m.orbit_distance(p.orbit_radius))
            .collect();
        d.sort_by(f32::total_cmp);
        let mut gap = d.first().map_or(D_MAX, |first| first - STAR_MAP_RADIUS);
        for w in d.windows(2) {
            gap = gap.min(w[1] - w[0]);
        }
        let largest = system.planets.iter().map(|p| p.radius).fold(1.0, f32::max);
        m.k = MAX_FILL * gap.max(1e-3) / largest;
        m
    }

    pub fn orbit_distance(&self, r: f64) -> f32 {
        D_MAX * (r / self.r_outer).max(0.0).sqrt() as f32
    }

    pub fn position(&self, rel: DVec3) -> Vec3 {
        (rel.normalize_or_zero() * self.orbit_distance(rel.length()) as f64).as_vec3()
    }

    pub fn radius(&self, real_radius: f32) -> f32 {
        real_radius * self.k
    }

    pub fn render_offset(&self, map_pos: Vec3, eye: Vec3) -> DVec3 {
        (map_pos - eye).as_dvec3() / self.k as f64
    }
}

// the nearest sphere a ray hits (index, centre, radius); None if it misses all
pub fn pick(origin: Vec3, dir: Vec3, spheres: &[(usize, Vec3, f32)]) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for &(i, c, r) in spheres {
        let oc = origin - c;
        let b = oc.dot(dir);
        let disc = b * b - (oc.length_squared() - r * r);
        if disc < 0.0 {
            continue;
        }
        let t = -b - disc.sqrt();
        if t > 0.0 && best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, i));
        }
    }
    best.map(|(_, i)| i)
}

pub struct MapCamera {
    pub yaw: f32,
    pub pitch: f32,    // above the orbital plane, 5°..85°
    pub distance: f32, // from the star
}

const PITCH_MIN: f32 = 5.0 * std::f32::consts::PI / 180.0;
const PITCH_MAX: f32 = 85.0 * std::f32::consts::PI / 180.0;
const ROTATE_PER_PX: f32 = 0.005;

impl MapCamera {
    pub fn default_view() -> Self {
        Self {
            yaw: 0.0,
            pitch: 35f32.to_radians(),
            distance: D_MAX * 2.6,
        }
    }

    pub fn eye(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        ) * self.distance
    }

    pub fn rotate(&mut self, dx_px: f32, dy_px: f32) {
        self.yaw -= dx_px * ROTATE_PER_PX;
        self.pitch = (self.pitch + dy_px * ROTATE_PER_PX).clamp(PITCH_MIN, PITCH_MAX);
    }

    // positive steps zoom in
    pub fn zoom(&mut self, steps: f32) {
        self.distance = (self.distance * 0.9f32.powf(steps)).clamp(D_MAX * 0.05, D_MAX * 4.0);
    }

    pub fn view_proj(&self, aspect: f32, fov_y: f32) -> Mat4 {
        let proj = glam::camera::rh::proj::directx::perspective(fov_y, aspect, 0.1, D_MAX * 20.0);
        proj * glam::camera::rh::view::look_at_mat4(self.eye(), Vec3::ZERO, Vec3::Y)
    }

    // the ray through a point in normalised device coordinates (-1..1, y up)
    pub fn ray(&self, ndc: Vec2, aspect: f32, fov_y: f32) -> (Vec3, Vec3) {
        let eye = self.eye();
        let fwd = (-eye).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let h = (fov_y * 0.5).tan();
        let dir = (fwd + right * ndc.x * h * aspect + up * ndc.y * h).normalize();
        (eye, dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::solar_system::SolarSystem;
    use glam::{DVec3, Vec2, Vec3};

    #[test]
    fn outermost_orbit_maps_to_d_max_and_order_is_kept() {
        let s = SolarSystem::generate(1);
        let m = MapLayout::new(&s);
        let mut radii: Vec<f64> = s.planets.iter().map(|p| p.orbit_radius).collect();
        radii.sort_by(f64::total_cmp);
        assert!((m.orbit_distance(*radii.last().unwrap()) - D_MAX).abs() < 1e-3);
        for w in radii.windows(2) {
            assert!(m.orbit_distance(w[0]) < m.orbit_distance(w[1]));
        }
    }

    fn no_overlap(s: &SolarSystem) {
        let m = MapLayout::new(s);
        let mut maps: Vec<(f32, f32)> = s
            .planets
            .iter()
            .map(|p| (m.orbit_distance(p.orbit_radius), m.radius(p.radius)))
            .collect();
        maps.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(
            maps[0].0 - maps[0].1 > STAR_MAP_RADIUS,
            "first planet clears the star"
        );
        for w in maps.windows(2) {
            assert!(w[1].0 - w[1].1 > w[0].0 + w[0].1, "{w:?} overlap");
        }
        // relative sizes kept
        let (a, b) = (&s.planets[0], &s.planets[1]);
        assert!(((m.radius(a.radius) / m.radius(b.radius)) - a.radius / b.radius).abs() < 1e-4);
    }

    #[test]
    fn mapped_planets_never_overlap() {
        for seed in [1, 2, 8, 13] {
            no_overlap(&SolarSystem::generate(seed));
        }
    }

    // Review Focus 4
    #[test]
    fn map_fits_fifteen_planets_without_overlap() {
        let mut s = SolarSystem::generate(1);
        while s.planets.len() < 15 {
            s.add_planet(crate::planet::biome::PlanetType::Ice, 400.0)
                .unwrap();
        }
        no_overlap(&s);
    }

    #[test]
    fn render_offset_keeps_the_mapped_apparent_size() {
        let s = SolarSystem::generate(1);
        let m = MapLayout::new(&s);
        let (map_pos, eye) = (Vec3::new(30.0, 0.0, 10.0), Vec3::new(0.0, 40.0, 120.0));
        let real_r = s.planets[0].radius;
        let off = m.render_offset(map_pos, eye);
        let mapped_angle = m.radius(real_r) / (map_pos - eye).length();
        let rendered_angle = real_r / off.length() as f32;
        assert!((mapped_angle - rendered_angle).abs() < 1e-6);
    }

    #[test]
    fn picking_hits_the_nearest_sphere() {
        let spheres = [
            (0, Vec3::new(0.0, 0.0, -10.0), 1.0),
            (1, Vec3::new(0.0, 0.0, -20.0), 3.0),
            (2, Vec3::new(5.0, 0.0, -10.0), 1.0),
        ];
        assert_eq!(pick(Vec3::ZERO, Vec3::NEG_Z, &spheres), Some(0));
        assert_eq!(
            pick(Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Z, &spheres),
            Some(1)
        );
        assert_eq!(pick(Vec3::new(0.0, 9.0, 0.0), Vec3::NEG_Z, &spheres), None);
    }

    #[test]
    fn camera_pitch_and_zoom_stay_in_range_and_rays_go_through_the_centre() {
        let mut c = MapCamera::default_view();
        assert!((c.pitch.to_degrees() - 35.0).abs() < 0.5);
        c.rotate(0.0, 1e6);
        assert!(c.pitch.to_degrees() <= 85.0 + 1e-3);
        c.rotate(0.0, -1e6);
        assert!(c.pitch.to_degrees() >= 5.0 - 1e-3);
        let d = c.distance;
        c.zoom(3.0);
        assert!(c.distance < d);
        c.zoom(-1000.0);
        assert!(c.distance <= D_MAX * 4.0 + 1e-3);
        let (o, dir) = c.ray(Vec2::ZERO, 1.6, 0.8);
        assert!((o - c.eye()).length() < 1e-4);
        assert!(
            dir.dot((-c.eye()).normalize()) > 0.9999,
            "screen centre looks at the star"
        );
    }
}
