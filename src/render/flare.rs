// flare.rs
// Screen-space lens flares for the visible sun (galaxy mode and planet mode): layout and intensity
// (pure, tested here) and the uniform flare.wgsl reads. Occlusion is measured on the GPU from depth
// (galaxy) or G-buffer distance (planet) taps around the sun in flare.wgsl's vertex shader.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

// the ghosts (mirror of flare.wgsl GHOSTS, which draws them; used here by the layout tests): position along the sun->centre line (0 = the sun, 1 = mirrored through
// the centre), size in screen heights, edge softness 0..1, ring 0/1
#[cfg_attr(not(test), allow(dead_code))]
pub const GHOSTS: [(f32, f32, f32, f32); 6] = [
    (0.35, 0.05, 0.6, 0.0),
    (0.6, 0.025, 0.4, 0.0),
    (0.8, 0.09, 0.85, 1.0),
    (1.15, 0.04, 0.5, 0.0),
    (1.4, 0.12, 0.9, 1.0),
    (1.75, 0.06, 0.7, 0.0),
];
// the strongest flare: a close sun must not white out the screen
pub const MAX_FLARE: f32 = 0.8;
// how far outside the screen (in NDC) a sun still flares, fading out
pub const OFF_SCREEN_MARGIN: f32 = 0.3;

// a ghost's screen position (NDC, centre = 0) at `k` along the line from the sun through the centre
// (flare.wgsl vs_flare places them so; tested here)
#[cfg_attr(not(test), allow(dead_code))]
pub fn ghost_position(sun: Vec2, k: f32) -> Vec2 {
    sun * (1.0 - 2.0 * k)
}

// 1 while the sun is on screen (|ndc| <= 1), falling to 0 over `margin` outside it
pub fn on_screen_factor(sun_ndc: Vec2, margin: f32) -> f32 {
    let over = (sun_ndc.abs() - Vec2::ONE).max_element().max(0.0);
    (1.0 - over / margin).clamp(0.0, 1.0)
}

// flare strength from the visible share of the sun, the on-screen factor and its apparent size
// (angular radius, radians): bigger, closer suns flare harder, up to MAX_FLARE
pub fn flare_intensity(visibility: f32, on_screen: f32, angular_radius: f32) -> f32 {
    let brightness = (0.25 + 6.0 * angular_radius).min(1.0);
    (visibility * on_screen * brightness).min(MAX_FLARE)
}

// the sun's apparent radius in pixels on a screen `height` px tall with vertical field of view `fov_y`
pub fn tap_radius_px(angular_radius: f32, fov_y: f32, height: f32) -> f32 {
    angular_radius.tan() / (fov_y * 0.5).tan() * height * 0.5
}

// how much of the sun stands above the planet's geometric horizon (sphere of `horizon_r` around the
// origin) seen from `cam_pos`: 1 above, 0 below, fading across the sun's own disc. High up the horizon
// dips below the local horizontal, so the sun stays up over the limb, as in galaxy mode at the handover
pub fn horizon_visibility(
    sun_dir: glam::Vec3,
    cam_pos: glam::Vec3,
    horizon_r: f32,
    angular_radius: f32,
) -> f32 {
    let d = cam_pos.length().max(1e-6);
    let up = cam_pos / d;
    let sun_elevation = sun_dir.normalize().dot(up).clamp(-1.0, 1.0).asin();
    let horizon_elevation = -(horizon_r / d).min(1.0).acos();
    let r = angular_radius.max(1e-4);
    let x = ((sun_elevation - horizon_elevation + r) / (2.0 * r)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

// must match `Flare` in flare.wgsl
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlareUniform {
    pub sun: [f32; 4], // xy: sun NDC, z: galaxy mode: the star's expected reversed-Z depth, w: intensity before occlusion
    pub tint: [f32; 4], // corona colour
    pub tint2: [f32; 4], // surface colour
    pub mode: [f32; 4], // x: 0 = galaxy depth taps, 1 = G-buffer taps; y: tap radius px; zw: screen size
    pub camera: [f32; 4], // planet mode: camera position in the planet frame (xyz), planet radius (w)
    pub light: [f32; 4],  // planet mode: direction to the sun (xyz), cloud time (w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn tap_radius_follows_the_field_of_view() {
        // a sun 2 degrees across, on a 1000 px tall screen
        let r = 2f32.to_radians();
        let wide = tap_radius_px(r, 80f32.to_radians(), 1000.0);
        let narrow = tap_radius_px(r, 45f32.to_radians(), 1000.0);
        assert!((wide - r.tan() / 40f32.to_radians().tan() * 500.0).abs() < 1e-3);
        // the third-person camera's 45 degrees magnifies the disc ~2x
        assert!(
            (narrow / wide - 40f32.to_radians().tan() / 22.5f32.to_radians().tan()).abs() < 1e-3
        );
    }

    #[test]
    fn on_the_ground_the_sun_sets_at_the_local_horizon() {
        let up = glam::Vec3::Y;
        let cam = up * 100.0;
        let sun = |deg: f32| glam::Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
        let r = 0.5f32.to_radians();
        assert_eq!(horizon_visibility(sun(5.0), cam, 100.0, r), 1.0);
        assert_eq!(horizon_visibility(sun(-5.0), cam, 100.0, r), 0.0);
        assert!((horizon_visibility(sun(0.0), cam, 100.0, r) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn high_up_the_sun_stays_visible_below_the_local_horizontal() {
        // at twice the horizon radius the horizon dips 60 degrees below the local horizontal
        let cam = glam::Vec3::Y * 200.0;
        let sun = |deg: f32| glam::Vec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
        let r = 0.5f32.to_radians();
        assert_eq!(horizon_visibility(sun(-30.0), cam, 100.0, r), 1.0);
        assert_eq!(horizon_visibility(sun(-80.0), cam, 100.0, r), 0.0);
        assert!((horizon_visibility(sun(-60.0), cam, 100.0, r) - 0.5).abs() < 1e-2);
    }

    // ghosts lie on the line from the sun through the screen centre (and beyond it)
    #[test]
    fn ghosts_lie_on_the_line_through_the_centre() {
        let sun = Vec2::new(0.4, -0.3);
        for k in GHOSTS.iter().map(|g| g.0) {
            let p = ghost_position(sun, k);
            assert!(p.perp_dot(sun).abs() < 1e-6, "ghost at {k} off the line");
        }
        assert!(
            ghost_position(sun, 1.0).dot(sun) < 0.0,
            "no ghost beyond the centre"
        );
    }

    #[test]
    fn on_screen_factor_fades_outside_the_view() {
        assert_eq!(on_screen_factor(Vec2::new(0.5, 0.5), 0.3), 1.0);
        assert_eq!(on_screen_factor(Vec2::new(1.5, 0.0), 0.3), 0.0);
        let edge = on_screen_factor(Vec2::new(1.15, 0.0), 0.3);
        assert!(0.0 < edge && edge < 1.0);
    }

    #[test]
    fn a_hidden_sun_gives_no_flare() {
        assert_eq!(flare_intensity(0.0, 1.0, 0.05), 0.0);
        assert!(flare_intensity(1.0, 1.0, 0.05) > flare_intensity(0.5, 1.0, 0.05));
        assert!(
            flare_intensity(1.0, 1.0, 0.2) <= MAX_FLARE,
            "a close sun whites out"
        );
        assert!(
            flare_intensity(1.0, 1.0, 0.0005) > 0.0,
            "a distant sun has no flare"
        );
    }

    #[test]
    fn flare_uniform_is_six_vec4s() {
        assert_eq!(std::mem::size_of::<FlareUniform>(), 6 * 16);
    }
}
