// map.rs
// The solar-system map's geometry (spec docs/superpowers/specs/2026-10-12-system-map-travel-design.md §1):
// orbits evenly spaced (order and direction kept; the spec's square root left the inner orbits so close
// that every planet had to be a few pixels), one size factor k for every
// planet (relative sizes kept, no overlaps), picking, and the map camera orbiting the star. The renderer
// draws the map with real-unit offsets (render_offset): perspective then shows each body at its mapped
// apparent size while all shading runs at true scale.

use glam::{DVec3, Mat4, Vec2, Vec3};

use crate::space::solar_system::SolarSystem;

pub const D_MAX: f32 = 100.0; // map distance of the outermost orbit
pub const STAR_MAP_RADIUS: f32 = 5.0; // the star's radius on the map
const ZOOM_MIN: f32 = STAR_MAP_RADIUS * 2.0; // the map camera's closest distance to the star
pub const MAX_FILL: f32 = 0.35; // the largest planet's share of the smallest orbit gap

pub struct MapLayout {
    pub k: f32,       // map units per real unit for planet radii
    orbits: Vec<f64>, // the real orbit radii, sorted
}

impl MapLayout {
    pub fn new(system: &SolarSystem) -> Self {
        let mut orbits: Vec<f64> = system.planets.iter().map(|p| p.orbit_radius).collect();
        orbits.sort_by(f64::total_cmp);
        let mut m = Self { k: 1.0, orbits };
        // evenly spaced orbits: every gap (and the first one, from the star) is the same
        let gap = (D_MAX - STAR_MAP_RADIUS) / m.orbits.len().max(1) as f32;
        let largest = system.planets.iter().map(|p| p.radius).fold(1.0, f32::max);
        m.k = MAX_FILL * gap / largest;
        m
    }

    // a real distance from the star on the map: the orbits evenly spaced from the star to D_MAX (order
    // and direction kept, distances compressed), linear in between, so ships and travel lines map too
    pub fn orbit_distance(&self, r: f64) -> f32 {
        let n = self.orbits.len();
        if n == 0 {
            return D_MAX * (r.max(0.0) as f32).sqrt().min(1.0);
        }
        let step = (D_MAX - STAR_MAP_RADIUS) as f64 / n as f64;
        let d = |i: usize| STAR_MAP_RADIUS as f64 + step * (i + 1) as f64;
        let r = r.max(0.0);
        let mapped = if r <= self.orbits[0] {
            d(0) * r / self.orbits[0].max(1e-9)
        } else if r >= self.orbits[n - 1] {
            d(n - 1) + step * (r - self.orbits[n - 1]) / self.orbits[n - 1].max(1e-9)
        } else {
            let i = self.orbits.partition_point(|&o| o <= r) - 1;
            let f = (r - self.orbits[i]) / (self.orbits[i + 1] - self.orbits[i]).max(1e-9);
            d(i) + step * f
        };
        mapped as f32
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

// the sphere (index, centre, radius) a ray hits most centrally: the smallest distance between the ray and
// the centre relative to the radius, so among the overlapping pick spheres of screen-sized markers (each
// the same pixel radius) the one closest to the cursor wins, not the one nearest the camera; ties (a ray
// through two centres) go to the front one. None if it misses all
pub fn pick(origin: Vec3, dir: Vec3, spheres: &[(usize, Vec3, f32)]) -> Option<usize> {
    let mut best: Option<(f32, f32, usize)> = None;
    for &(i, c, r) in spheres {
        let oc = origin - c;
        let b = oc.dot(dir);
        let disc = b * b - (oc.length_squared() - r * r);
        if disc < 0.0 {
            continue;
        }
        let t = -b - disc.sqrt();
        if t <= 0.0 {
            continue;
        }
        let off = (oc - dir * oc.dot(dir)).length() / r.max(1e-9);
        if best.is_none_or(|(bo, bt, _)| off < bo || (off == bo && t < bt)) {
            best = Some((off, t, i));
        }
    }
    best.map(|(_, _, i)| i)
}

// the planet types of a system for the galaxy view's info panel, each named once, in orbit order
pub fn distinct_kinds<'a>(kinds: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::new();
    for k in kinds {
        if !out.contains(&k) {
            out.push(k);
        }
    }
    out
}

// an info panel's width in pixels for its lines (text, font size) in the HUD's monospace font (an
// advance of 0.6 × the size), plus `padding` on each side
pub fn panel_width_px<'a>(lines: impl IntoIterator<Item = (&'a str, f32)>, padding: f32) -> f32 {
    lines
        .into_iter()
        .map(|(text, size)| text.chars().count() as f32 * size * MONO_ADVANCE)
        .fold(0.0, f32::max)
        + 2.0 * padding
}

const MONO_ADVANCE: f32 = 0.6; // the monospace font's advance per character, × its size

pub struct MapCamera {
    pub yaw: f32,
    pub pitch: f32,    // above the orbital plane, 5°..85°
    pub distance: f32, // from the star
    pub min_distance: f32,
    pub max_distance: f32,
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
            min_distance: ZOOM_MIN,
            max_distance: D_MAX * 4.0,
        }
    }

    pub fn galaxy_view() -> Self {
        Self {
            yaw: 0.0,
            pitch: 45f32.to_radians(),
            distance: D_MAX * 2.4,
            min_distance: D_MAX * 0.08,
            max_distance: D_MAX * 5.0,
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

    // positive steps zoom in; the camera orbits the star, so the closest it gets keeps the eye
    // outside the star (ZOOM_MIN)
    pub fn zoom(&mut self, steps: f32) {
        self.distance =
            (self.distance * 0.9f32.powf(steps)).clamp(self.min_distance, self.max_distance);
    }

    #[cfg_attr(not(test), allow(dead_code))] // the renderer builds its own; tests check the rays against it
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

// the field of view the map is drawn and picked with
pub const MAP_FOV_Y: f32 = 50.0 * std::f32::consts::PI / 180.0;
// the smallest on-screen radius a planet can be clicked at, pixels
const PICK_MIN_PX: f32 = 12.0;
// mouse movement (pixels) after which a press is a drag, not a click
const DRAG_THRESHOLD_PX: f32 = 4.0;

pub const LY_TO_VIEW: f32 = D_MAX / 50.0; // galaxy view units: the galaxy's radius maps to D_MAX
const PICK_SYSTEM_MIN_PX: f32 = 10.0;

pub fn galaxy_view_position(p: DVec3) -> Vec3 {
    p.as_vec3() * LY_TO_VIEW
}

pub fn pick_system(
    galaxy: &crate::space::galaxy::Galaxy,
    camera: &MapCamera,
    cursor: Vec2,
    screen: (f32, f32),
) -> Option<usize> {
    let ndc = Vec2::new(
        cursor.x / screen.0 * 2.0 - 1.0,
        1.0 - cursor.y / screen.1 * 2.0,
    );
    let (origin, dir) = camera.ray(ndc, screen.0 / screen.1, MAP_FOV_Y);
    let px_angle = MAP_FOV_Y / screen.1;
    let spheres: Vec<(usize, Vec3, f32)> = galaxy
        .systems
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let c = galaxy_view_position(s.position);
            (
                i,
                c,
                PICK_SYSTEM_MIN_PX * 0.5 * px_angle * (c - origin).length(),
            )
        })
        .collect();
    pick(origin, dir, &spheres)
}

pub fn shows_label(
    i: usize,
    current: usize,
    hover: Option<usize>,
    selected: Option<usize>,
    visited: bool,
) -> bool {
    i == current || hover == Some(i) || selected == Some(i) || visited
}

// the open map's state (main.rs): camera, mouse, hover and selection
pub struct MapView {
    pub camera: MapCamera,
    pub cursor: Option<Vec2>,        // pixels, y down
    pub press: Option<(Vec2, bool)>, // the left button held: where it last was, whether it dragged
    pub hover: Option<usize>,
    pub selected: Option<usize>,
    pub travel_request: Option<usize>, // a planet picked for travel, taken by the game loop
    pub restore_mouse_released: bool,  // the controller's mouse state before the map opened
}

impl MapView {
    pub fn new(restore_mouse_released: bool, camera: MapCamera) -> Self {
        Self {
            camera,
            cursor: None,
            press: None,
            hover: None,
            selected: None,
            travel_request: None,
            restore_mouse_released,
        }
    }

    pub fn cursor_moved(&mut self, pos: Vec2) {
        if let (Some((last, moved)), Some(_)) = (&mut self.press, self.cursor) {
            let d = pos - *last;
            if *moved || d.length() > DRAG_THRESHOLD_PX {
                *moved = true;
                self.camera.rotate(d.x, d.y);
                *last = pos;
            }
        }
        self.cursor = Some(pos);
    }

    pub fn press(&mut self) {
        if let Some(c) = self.cursor {
            self.press = Some((c, false));
        }
    }

    // a release without dragging is a click: select the hovered planet, or travel to the selected one
    pub fn release(&mut self) {
        let clicked = matches!(self.press.take(), Some((_, false)));
        if clicked {
            self.click(self.hover);
        }
    }

    pub fn click(&mut self, planet: Option<usize>) {
        match planet {
            Some(i) if self.selected == Some(i) => self.travel_request = Some(i),
            Some(i) => self.selected = Some(i),
            None => {}
        }
    }

    // `Enter`: travel to the selection
    pub fn confirm(&mut self) {
        if let Some(i) = self.selected {
            self.travel_request = Some(i);
        }
    }
}

// the planet under the mouse: each planet a sphere at its mapped place, at least PICK_MIN_PX across
pub fn pick_planet(
    system: &SolarSystem,
    map: &MapLayout,
    camera: &MapCamera,
    cursor: Vec2,
    screen: (f32, f32),
    t: f64,
) -> Option<usize> {
    let aspect = screen.0 / screen.1;
    let ndc = Vec2::new(
        cursor.x / screen.0 * 2.0 - 1.0,
        1.0 - cursor.y / screen.1 * 2.0,
    );
    let (origin, dir) = camera.ray(ndc, aspect, MAP_FOV_Y);
    let px_angle = MAP_FOV_Y / screen.1;
    let star = system.star.position();
    let spheres: Vec<(usize, Vec3, f32)> = system
        .planets
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let c = map.position(p.position_at(t) - star);
            let min = PICK_MIN_PX * px_angle * (c - origin).length();
            (i, c, map.radius(p.radius).max(min))
        })
        .collect();
    pick(origin, dir, &spheres)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::solar_system::SolarSystem;
    use glam::{Vec2, Vec3};

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
            no_overlap(&SolarSystem::generate_varied(seed));
        }
    }

    // Review Focus 4
    #[test]
    fn map_fits_fifteen_planets_without_overlap() {
        let mut s = SolarSystem::generate(1);
        while s.planet_count() < crate::space::solar_system::MAX_PLANET_COUNT {
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
    fn picking_prefers_the_sphere_closest_to_the_cursor() {
        // two equal-sized markers overlapping on screen: the ray passes 0.8 r from the front one's centre
        // and 0.1 r from the back one's, so the back one (under the cursor) wins
        let spheres = [
            (0, Vec3::new(0.8, 0.0, -10.0), 1.0),
            (1, Vec3::new(0.2, 0.0, -20.0), 2.0),
        ];
        assert_eq!(pick(Vec3::ZERO, Vec3::NEG_Z, &spheres), Some(1));
        // a big sphere hit through its centre stays ahead of a small one it hides
        let spheres = [
            (0, Vec3::new(0.0, 0.0, -10.0), 3.0),
            (1, Vec3::new(0.5, 0.0, -30.0), 1.0),
        ];
        assert_eq!(pick(Vec3::ZERO, Vec3::NEG_Z, &spheres), Some(0));
    }

    #[test]
    fn the_info_panel_fits_its_longest_line() {
        let kinds = distinct_kinds(["Ice", "Desert", "Ice", "Toxic", "Desert"]);
        assert_eq!(kinds, vec!["Ice", "Desert", "Toxic"]);
        let long = "7 planets: Earth-like, Volcanic, Ice, Desert, Toxic, Ocean, Shattered";
        let w = panel_width_px([("Name (#1)", 18.0), (long, 15.0)], 10.0);
        assert!((w - (long.len() as f32 * 15.0 * 0.6 + 20.0)).abs() < 1e-3);
        assert!(w > 420.0, "the old fixed width was too narrow: {w}");
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

    #[test]
    fn a_click_selects_then_travels_and_a_drag_rotates() {
        let mut v = MapView::new(false, MapCamera::default_view());
        v.cursor_moved(Vec2::new(100.0, 100.0));
        v.hover = Some(2);
        v.press();
        v.release();
        assert_eq!((v.selected, v.travel_request), (Some(2), None));
        v.press();
        v.release();
        assert_eq!(v.travel_request, Some(2));
        // a drag doesn't click, it turns the camera
        let mut v = MapView::new(false, MapCamera::default_view());
        v.cursor_moved(Vec2::new(100.0, 100.0));
        v.hover = Some(1);
        let yaw = v.camera.yaw;
        v.press();
        v.cursor_moved(Vec2::new(160.0, 100.0));
        v.release();
        assert_eq!(v.selected, None);
        assert!(v.camera.yaw != yaw);
    }

    #[test]
    fn the_planet_under_the_cursor_is_picked() {
        let s = SolarSystem::generate(1);
        let m = MapLayout::new(&s);
        let cam = MapCamera::default_view();
        let screen = (1600.0, 900.0);
        // project planet 2's mapped centre to the screen with the camera's own matrix, then pick there
        let c = m.position(s.planets[2].position_at(0.0) - s.star.position());
        let clip = cam.view_proj(screen.0 / screen.1, MAP_FOV_Y) * c.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let px = Vec2::new(
            (ndc.x * 0.5 + 0.5) * screen.0,
            (0.5 - ndc.y * 0.5) * screen.1,
        );
        assert_eq!(pick_planet(&s, &m, &cam, px, screen, 0.0), Some(2));
        assert_eq!(
            pick_planet(&s, &m, &cam, Vec2::new(2.0, 2.0), screen, 0.0),
            None
        );
    }

    // review minor 3: zooming all the way in keeps the eye well outside the star
    #[test]
    fn zoom_stops_outside_the_star() {
        let mut c = MapCamera::default_view();
        c.zoom(1000.0);
        assert!(c.eye().length() >= 2.0 * STAR_MAP_RADIUS - 1e-4);
        c.zoom(-1000.0);
        assert!((c.distance - D_MAX * 4.0).abs() < 1e-3);
    }

    #[test]
    fn galaxy_camera_has_its_own_range() {
        let mut c = MapCamera::galaxy_view();
        c.zoom(-1000.0);
        assert!((c.distance - c.max_distance).abs() < 1e-3);
        c.zoom(1000.0);
        assert!((c.distance - c.min_distance).abs() < 1e-3);
        let m = MapCamera::default_view();
        assert!(
            (m.max_distance - D_MAX * 4.0).abs() < 1e-3,
            "system map range unchanged"
        );
    }

    #[test]
    fn the_system_under_the_cursor_is_picked() {
        let g = crate::space::galaxy::Galaxy::generate(1);
        let cam = MapCamera::galaxy_view();
        let screen = (1600.0, 900.0);
        let c = galaxy_view_position(g.systems[7].position);
        let clip = cam.view_proj(screen.0 / screen.1, MAP_FOV_Y) * c.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let px = Vec2::new(
            (ndc.x * 0.5 + 0.5) * screen.0,
            (0.5 - ndc.y * 0.5) * screen.1,
        );
        let hit = pick_system(&g, &cam, px, screen).expect("something under the cursor");
        let ch = galaxy_view_position(g.systems[hit].position);
        let clip = cam.view_proj(screen.0 / screen.1, MAP_FOV_Y) * ch.extend(1.0);
        let n2 = clip.truncate() / clip.w;
        let p2 = Vec2::new((n2.x * 0.5 + 0.5) * screen.0, (0.5 - n2.y * 0.5) * screen.1);
        assert!((p2 - px).length() < 12.0);
    }

    #[test]
    fn label_rules() {
        assert!(shows_label(3, 3, None, None, false), "current");
        assert!(shows_label(5, 3, Some(5), None, false), "hovered");
        assert!(shows_label(5, 3, None, Some(5), false), "selected");
        assert!(shows_label(5, 3, None, None, true), "visited");
        assert!(!shows_label(5, 3, None, None, false));
    }
}
