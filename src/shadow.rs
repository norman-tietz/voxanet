// Soft sun shadows (docs/superpowers/specs/2026-10-11-soft-shadows-design.md): the sun is an area light
// of a fixed, tunable angular size, so a shadow is sharp where its caster touches the receiver and
// softens with the distance between them. cs_shadow (rt_hw.wgsl) stores the shadow ray's hit distance,
// blur.wgsl's search passes spread the mean blocker distance around each shadow and its blur uses the
// resulting width per texel. Settings, the /shadow command and Rust mirrors of the WGSL maths (tests).

pub const MIN_WIDTH: f32 = 0.03; // world units: contact shadows still get about a pixel of smoothing
pub const MAX_BLOCKER: f32 = 32.0; // blocks: beyond, the edge stops widening (and the search radius ends)
pub const FIXED_WIDTH: f32 = 0.3; // world units: /shadow off and the ray-march path (the old look)
pub const MAX_RADIUS_PX: f32 = 32.0; // blur.wgsl MAX_RADIUS
#[cfg_attr(not(test), allow(dead_code))]
pub const TAPS: i32 = 8; // blur.wgsl TAPS, per side
#[cfg_attr(not(test), allow(dead_code))]
pub const SEARCH_TAPS: i32 = 4; // blur.wgsl SEARCH_TAPS: the search only averages distances
pub const DEFAULT_SUN_DEG: f32 = 2.5; // the sun's angular radius
const MAX_SUN_DEG: f32 = 10.0;
pub const SHADOW_USAGE: &str = "Usage: /shadow [on|off|sun <0-10>]";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSettings {
    pub sun_deg: f32,
    pub soft: bool, // off: FIXED_WIDTH everywhere, sun_deg kept for /shadow on
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            sun_deg: DEFAULT_SUN_DEG,
            soft: true,
        }
    }
}

impl ShadowSettings {
    pub fn tan_sun(&self) -> f32 {
        self.sun_deg.to_radians().tan()
    }

    // applies a /shadow command; returns the line to log
    pub fn apply(&mut self, cmd: ShadowCommand) -> String {
        match cmd {
            ShadowCommand::Get => {}
            ShadowCommand::On => self.soft = true,
            ShadowCommand::Off => self.soft = false,
            ShadowCommand::Sun(v) => (self.sun_deg, self.soft) = (v, true),
        }
        format!(
            "Shadow: {}, sun {:.1}°",
            if self.soft { "soft" } else { "fixed width" },
            self.sun_deg
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowCommand {
    Get,
    On,
    Off,
    Sun(f32),
}

impl ShadowCommand {
    // the words after /shadow
    pub fn parse(args: &[&str]) -> Result<ShadowCommand, &'static str> {
        match args {
            [] => Ok(ShadowCommand::Get),
            ["on"] => Ok(ShadowCommand::On),
            ["off"] => Ok(ShadowCommand::Off),
            ["sun", v] => v
                .parse::<f32>()
                .ok()
                .filter(|v| (0.0..=MAX_SUN_DEG).contains(v))
                .map(ShadowCommand::Sun)
                .ok_or(SHADOW_USAGE),
            _ => Err(SHADOW_USAGE),
        }
    }
}

// blur.wgsl penumbra_width: the edge width at a receiver whose blocker is `t` away
#[cfg_attr(not(test), allow(dead_code))]
pub fn penumbra_width(t: f32, tan_sun: f32) -> f32 {
    t.min(MAX_BLOCKER) * tan_sun + MIN_WIDTH
}

// blur.wgsl's radius in pixels for a world width at camera distance `cam_dist` (focal: pixels per world
// unit at distance 1), capped
#[cfg_attr(not(test), allow(dead_code))]
pub fn blur_radius_px(width: f32, focal: f32, cam_dist: f32) -> f32 {
    (width * focal / cam_dist).min(MAX_RADIUS_PX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penumbra_grows_with_the_blocker_distance_and_stops_at_max() {
        let tan = 2.5f32.to_radians().tan();
        assert_eq!(penumbra_width(0.0, tan), MIN_WIDTH);
        let w10 = penumbra_width(10.0, tan);
        assert!((w10 - (10.0 * tan + MIN_WIDTH)).abs() < 1e-6);
        assert!((penumbra_width(20.0, tan) - MIN_WIDTH - 2.0 * (w10 - MIN_WIDTH)).abs() < 1e-5);
        assert_eq!(penumbra_width(100.0, tan), penumbra_width(MAX_BLOCKER, tan));
    }

    #[test]
    fn blur_radius_halves_at_twice_the_distance_and_is_capped() {
        let r = blur_radius_px(0.4, 600.0, 20.0);
        assert!((r - 12.0).abs() < 1e-4);
        assert!((blur_radius_px(0.4, 600.0, 40.0) - r / 2.0).abs() < 1e-4);
        assert_eq!(blur_radius_px(5.0, 600.0, 1.0), MAX_RADIUS_PX);
    }

    // the search's two passes sum (Σ w·t, Σ w) per row, then those sums per column: the same as one
    // 2D sum, so a lit texel beside a shadow gets the shadow's mean blocker distance (Review Focus 1)
    #[test]
    fn separable_blocker_sum_equals_the_2d_sum() {
        // 3×3 neighbourhood, blocker distances (0 = lit)
        let t = [[0.0, 4.0, 6.0], [0.0, 0.0, 8.0], [0.0, 2.0, 0.0f32]];
        let rows: Vec<(f32, f32)> = t
            .iter()
            .map(|r| {
                r.iter()
                    .filter(|&&x| x > 0.0)
                    .fold((0.0, 0.0), |(s, n), &x| (s + x, n + 1.0))
            })
            .collect();
        let (s, n) = rows
            .iter()
            .fold((0.0, 0.0), |(s, n), &(a, b)| (s + a, n + b));
        assert_eq!((s, n), (20.0, 4.0));
        assert_eq!(s / n, 5.0);
    }

    #[test]
    fn fixed_mode_is_todays_width() {
        assert_eq!(FIXED_WIDTH, 0.3);
        let mut s = ShadowSettings::default();
        assert!(s.soft);
        s.apply(ShadowCommand::Off);
        assert!(!s.soft);
        assert_eq!(s.sun_deg, 2.5); // kept for /shadow on
    }

    #[test]
    fn shadow_commands_parse_and_apply() {
        assert_eq!(ShadowCommand::parse(&[]), Ok(ShadowCommand::Get));
        assert_eq!(ShadowCommand::parse(&["on"]), Ok(ShadowCommand::On));
        assert_eq!(ShadowCommand::parse(&["off"]), Ok(ShadowCommand::Off));
        assert_eq!(
            ShadowCommand::parse(&["sun", "4"]),
            Ok(ShadowCommand::Sun(4.0))
        );
        assert_eq!(ShadowCommand::parse(&["sun", "11"]), Err(SHADOW_USAGE));
        assert_eq!(ShadowCommand::parse(&["sun", "-1"]), Err(SHADOW_USAGE));
        assert_eq!(ShadowCommand::parse(&["soft"]), Err(SHADOW_USAGE));
        let mut s = ShadowSettings::default();
        s.apply(ShadowCommand::Off);
        assert_eq!(s.apply(ShadowCommand::Sun(4.0)), "Shadow: soft, sun 4.0°");
        assert!(s.soft);
        assert_eq!(s.apply(ShadowCommand::Off), "Shadow: fixed width, sun 4.0°");
    }
}
