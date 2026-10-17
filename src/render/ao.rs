// Hardware ray-traced ambient occlusion (docs/superpowers/specs/2026-10-09-rt-ao-design.md): settings,
// the /ao console command, the rule that picks between it and the mesher's vertex AO, and the frame
// bookkeeping of the progressive AO (one sample per 4x4 block and frame, accumulated while nothing
// changes). No GPU code.

pub const DEFAULT_RADIUS: f32 = 3.0; // world units (blocks): gaps, wall feet, steps, narrow gorges
pub const DEFAULT_RAYS: u32 = 2; // per AO sample (one sample per BLOCK x BLOCK texels and frame)
pub const DEFAULT_SAMPLES: u32 = 64; // accumulation cap per texel while still; beyond, a moving average
                                     // caps while the camera moves (motion_cap): fast and slow screen motion
pub const DEFAULT_MOTION_FAST: u32 = 4;
pub const DEFAULT_MOTION_SLOW: u32 = 32;
// AO budget: one sample per BLOCK x BLOCK shadow texels and frame (a ray query costs ~10 ns on the
// reference GPU, so full resolution was 33 ms per ray in fullscreen)
pub const BLOCK: u32 = 4;
pub const DEFAULT_STRENGTH: f32 = 1.0;
// camera distance (world units) where AO has faded out, from full at 0.375 × that; no AO rays beyond.
// Out there a 3-block AO is under a shadow texel and the blur no longer smooths its noise, and on
// large planets the LOD terrain begins, drawn morphed but traced unmorphed (rays would start under it)
pub const FADE_END: f32 = 400.0;
pub const AO_USAGE: &str =
    "Usage: /ao [on|off|radius <0.5-8>|rays <1-16>|strength <0-1>|samples <1-256>|motion <fast> <slow>]";

// the AO look, tuned with /ao (cmd.rs); HwParams.ao = uniform()
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoSettings {
    pub radius: f32,
    pub rays: u32,
    pub strength: f32,
    pub samples: u32,     // accumulation cap per texel while still
    pub motion_fast: u32, // ... while the camera moves fast (motion_cap)
    pub motion_slow: u32, // ... while it moves slowly
    pub enabled: bool,    // off: no AO rays, the mesher's vertex AO instead (vertex_ao_wanted)
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            radius: DEFAULT_RADIUS,
            rays: DEFAULT_RAYS,
            strength: DEFAULT_STRENGTH,
            samples: DEFAULT_SAMPLES,
            motion_fast: DEFAULT_MOTION_FAST,
            motion_slow: DEFAULT_MOTION_SLOW,
            enabled: true,
        }
    }
}

impl AoSettings {
    // rt_hw.wgsl HwParams.ao: radius, ray count (0 while off: no AO rays, ao = 1), strength, fade end
    pub fn uniform(&self) -> [f32; 4] {
        let rays = if self.enabled { self.rays as f32 } else { 0.0 };
        [self.radius, rays, self.strength, FADE_END]
    }

    // applies an /ao command; returns the line to log
    pub fn apply(&mut self, cmd: AoCommand) -> String {
        match cmd {
            AoCommand::Get => {}
            AoCommand::On => self.enabled = true,
            AoCommand::Off => self.enabled = false,
            AoCommand::Radius(r) => {
                self.radius = r;
                self.enabled = true;
            }
            AoCommand::Rays(n) => {
                self.rays = n;
                self.enabled = true;
            }
            AoCommand::Strength(s) => {
                self.strength = s;
                self.enabled = true;
            }
            AoCommand::Samples(n) => {
                self.samples = n;
                self.enabled = true;
            }
            AoCommand::Motion(fast, slow) => {
                (self.motion_fast, self.motion_slow) = (fast, slow);
                self.enabled = true;
            }
        }
        format!(
            "AO: {}, radius {:.2}, rays {}, strength {:.2}, samples {}, motion {}/{}",
            if self.enabled {
                "ray traced"
            } else {
                "off (vertex AO)"
            },
            self.radius,
            self.rays,
            self.strength,
            self.samples,
            self.motion_fast,
            self.motion_slow
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AoCommand {
    Get,
    On,
    Off,
    Radius(f32),
    Rays(u32),
    Strength(f32),
    Samples(u32),
    Motion(u32, u32), // fast, slow
}

impl AoCommand {
    // the words after /ao
    pub fn parse(args: &[&str]) -> Result<AoCommand, &'static str> {
        let float = |s: &str, lo: f32, hi: f32| {
            s.parse::<f32>()
                .ok()
                .filter(|v| (lo..=hi).contains(v))
                .ok_or(AO_USAGE)
        };
        match args {
            [] => Ok(AoCommand::Get),
            ["on"] => Ok(AoCommand::On),
            ["off"] => Ok(AoCommand::Off),
            ["radius", v] => float(v, 0.5, 8.0).map(AoCommand::Radius),
            ["rays", v] => v
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=16).contains(n))
                .map(AoCommand::Rays)
                .ok_or(AO_USAGE),
            ["strength", v] => float(v, 0.0, 1.0).map(AoCommand::Strength),
            ["motion", fast, slow] => match (fast.parse::<u32>(), slow.parse::<u32>()) {
                (Ok(f), Ok(s)) if 1 <= f && f <= s && s <= 256 => Ok(AoCommand::Motion(f, s)),
                _ => Err(AO_USAGE),
            },
            ["samples", v] => v
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=256).contains(n))
                .map(AoCommand::Samples)
                .ok_or(AO_USAGE),
            _ => Err(AO_USAGE),
        }
    }
}

// exactly one kind of AO: the mesher's vertex AO unless ray-traced AO runs (hardware ray queries,
// hardware shadows in use, /ao on)
pub fn vertex_ao_wanted(hw_available: bool, hw_shadows: bool, ao_enabled: bool) -> bool {
    !(hw_available && hw_shadows && ao_enabled)
}

// the texel of each BLOCK x BLOCK block that gets this frame's AO sample: 4x4 Bayer order, so the first
// samples after a reset spread over the block and all 16 positions come round in 16 frames
pub fn sample_offset(frame: u32) -> (u32, u32) {
    const BAYER: [(u32, u32); 16] = [
        (0, 0),
        (2, 2),
        (2, 0),
        (0, 2),
        (1, 1),
        (3, 3),
        (3, 1),
        (1, 3),
        (1, 0),
        (3, 2),
        (3, 0),
        (1, 2),
        (0, 1),
        (2, 3),
        (2, 1),
        (0, 3),
    ];
    BAYER[(frame % 16) as usize]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// mirror of rt_hw.wgsl's AO fill radius, in shadow texels, from the centre texel's sample count: wide
// enough to bridge the gaps between sparse samples where a surface just appeared, narrow (sharp
// contact AO) where it has many
#[cfg_attr(not(test), allow(dead_code))]
pub fn fill_radius_for_count(count: f32) -> f32 {
    (2.0 * BLOCK as f32 / (count + 1.0).sqrt()).clamp(1.5, 8.0)
}

// mirror of rt_hw.wgsl's accumulation cap for a texel whose history moved `m` texels on screen since
// the last frame: the full cap while still, toward `slow` up to half a texel, `fast` from 8 texels on
#[cfg_attr(not(test), allow(dead_code))]
pub fn motion_cap(m: f32, samples: f32, slow: f32, fast: f32) -> f32 {
    let still_to_slow = samples + (slow - samples) * smoothstep(0.05, 0.5, m);
    still_to_slow + (fast - still_to_slow) * smoothstep(0.5, 8.0, m)
}

// mirror of rt_hw.wgsl's reprojection: world point p (planet frame) through last frame's view-projection
// onto the shadow targets (`size`), in texel coordinates with texel centres at integers; None behind the
// camera
#[cfg_attr(not(test), allow(dead_code))]
pub fn reproject(p: glam::Vec3, view_proj: glam::Mat4, size: (u32, u32)) -> Option<glam::Vec2> {
    let clip = view_proj * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate().truncate() / clip.w;
    let uv = glam::Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    Some(uv * glam::Vec2::new(size.0 as f32, size.1 as f32) - 0.5)
}

// mirror of rt_hw.wgsl's history offset: where last frame's view-projection put world point p minus
// where this frame's puts it, in texels; read at the texel plus this. Measured as a difference of the
// same formula, a still camera gives exactly 0, also where the shadow targets are smaller than the
// screen (their world points lie at screen-pixel centres, which don't reproject onto texel centres)
#[cfg_attr(not(test), allow(dead_code))]
pub fn history_offset(
    p: glam::Vec3,
    prev_view_proj: glam::Mat4,
    view_proj: glam::Mat4,
    size: (u32, u32),
) -> Option<glam::Vec2> {
    Some(reproject(p, prev_view_proj, size)? - reproject(p, view_proj, size)?)
}

// what the accumulated AO depends on besides block edits (camera movement is reprojected)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoKey {
    pub settings: AoSettings,
    pub size: (u32, u32), // shadow targets
}

// frames since the progressive AO was last fully reset, and the running frame index (sample rotation)
#[derive(Default)]
pub struct AoProgress {
    frame: u32,
    frames: u32,
    epoch: Option<u64>,
    key: Option<AoKey>,
}

impl AoProgress {
    // the renderer's edit epoch (bumped by block edits) and the AO key: returns the frames since the
    // last full reset, 0 when either changed (the accumulation starts over); camera movement and
    // chunk streaming don't reset, the reprojection carries the accumulation across them
    pub fn advance(&mut self, edit_epoch: u64, key: AoKey) -> u32 {
        if self.epoch == Some(edit_epoch) && self.key == Some(key) {
            self.frames = self.frames.saturating_add(1);
        } else {
            self.frames = 0;
        }
        self.epoch = Some(edit_epoch);
        self.key = Some(key);
        self.frame = self.frame.wrapping_add(1);
        self.frames
    }

    pub fn frame(&self) -> u32 {
        self.frame
    }

    // frames since the last reset (the debug overlay shows it)
    pub fn frames(&self) -> u32 {
        self.frames
    }
}

// geometry that changes the AO without changing the depth (a chunk or LOD mesh streaming in or out
// next to converged ground) isn't caught by the reprojection's depth test: for REFRESH_FRAMES after
// such a change within reach of the AO, the cap while still drops to motion_slow, so the stale part of
// the running mean fades in ~motion_slow frames instead of ~samples (a full reset would flash)
pub const REFRESH_FRAMES: u32 = 120;

#[derive(Default)]
pub struct AoRefresh {
    epoch: Option<u64>,
    since: u32,
}

impl AoRefresh {
    // the renderer's geometry epoch (bumped by mesh changes within reach of the AO): this frame's cap
    // while still
    pub fn still_cap(&mut self, geometry_epoch: u64, s: &AoSettings) -> u32 {
        if self.epoch == Some(geometry_epoch) {
            self.since = self.since.saturating_add(1);
        } else {
            self.since = if self.epoch.is_some() {
                0
            } else {
                REFRESH_FRAMES
            };
        }
        self.epoch = Some(geometry_epoch);
        if self.since < REFRESH_FRAMES {
            s.samples.min(s.motion_slow)
        } else {
            s.samples
        }
    }
}

// whether a mesh's bounding sphere reaches the points that get AO (closer than FADE_END) or their AO
// rays (radius)
pub fn within_ao_reach(
    center: glam::Vec3,
    radius: f32,
    camera: glam::Vec3,
    ao_radius: f32,
) -> bool {
    center.distance(camera) - radius <= FADE_END + ao_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    // glam's deprecated Mat4::perspective_rh, kept on purpose: its replacement
    // (glam::camera::rh::proj::directx::perspective) computes the depth terms through 1 / (far - near),
    // not far / (near - far), so its matrices are not bit-identical to the ones these tests were written
    // against (look_at_rh's replacement, look_at_mat4, is, and is used directly)
    fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> glam::Mat4 {
        #[allow(deprecated)]
        glam::Mat4::perspective_rh(fov_y, aspect, near, far)
    }

    #[test]
    fn default_settings_feed_the_uniform() {
        let s = AoSettings::default();
        assert_eq!(
            (DEFAULT_RADIUS, DEFAULT_RAYS, DEFAULT_STRENGTH),
            (3.0, 2, 1.0)
        );
        assert_eq!(DEFAULT_SAMPLES, 64);
        assert_eq!(s.samples, DEFAULT_SAMPLES);
        assert_eq!(s.uniform(), [3.0, 2.0, 1.0, FADE_END]);
    }

    // AO fades out with distance and no AO rays are cast beyond FADE_END: there it's under a shadow
    // texel, unblurred noise, and on large planets the start of the morphed LOD terrain (whose ray
    // geometry is unmorphed)
    #[test]
    fn ao_fades_out_with_distance() {
        assert_eq!(FADE_END, 400.0);
        assert_eq!(AoSettings::default().uniform()[3], FADE_END);
    }

    #[test]
    fn ao_command_parses_and_checks_ranges() {
        assert_eq!(AoCommand::parse(&[]), Ok(AoCommand::Get));
        assert_eq!(AoCommand::parse(&["on"]), Ok(AoCommand::On));
        assert_eq!(AoCommand::parse(&["off"]), Ok(AoCommand::Off));
        assert_eq!(
            AoCommand::parse(&["radius", "2.5"]),
            Ok(AoCommand::Radius(2.5))
        );
        assert_eq!(AoCommand::parse(&["rays", "8"]), Ok(AoCommand::Rays(8)));
        assert_eq!(
            AoCommand::parse(&["strength", "0.5"]),
            Ok(AoCommand::Strength(0.5))
        );
        assert_eq!(
            AoCommand::parse(&["samples", "128"]),
            Ok(AoCommand::Samples(128))
        );
        assert_eq!(AoCommand::parse(&["samples", "0"]), Err(AO_USAGE));
        assert_eq!(AoCommand::parse(&["samples", "257"]), Err(AO_USAGE));
        for bad in [
            &["radius", "0.2"][..],
            &["radius", "9"],
            &["rays", "0"],
            &["rays", "17"],
            &["rays", "2.5"],
            &["strength", "1.5"],
            &["strength", "x"],
            &["radius"],
            &["nope"],
        ] {
            assert_eq!(AoCommand::parse(bad), Err(AO_USAGE), "{bad:?}");
        }
    }

    // off casts no AO rays (ray count 0) but keeps the values; on and setting a value bring them back
    #[test]
    fn off_keeps_the_values_for_on() {
        let mut s = AoSettings::default();
        s.apply(AoCommand::Rays(8));
        s.apply(AoCommand::Off);
        assert_eq!(s.uniform()[1], 0.0);
        assert!(!s.enabled);
        s.apply(AoCommand::On);
        assert_eq!(s.uniform(), [3.0, 8.0, 1.0, FADE_END]);
        s.apply(AoCommand::Off);
        s.apply(AoCommand::Radius(1.5));
        assert!(s.enabled);
        assert_eq!(s.uniform()[0], 1.5);
        assert!(s.apply(AoCommand::Get).contains("radius 1.50"));
    }

    // exactly one kind of AO: vertex AO unless hardware rays are available, used and AO is on
    #[test]
    fn vertex_ao_is_used_exactly_when_rt_ao_is_not() {
        for hw in [false, true] {
            for shadows in [false, true] {
                for on in [false, true] {
                    assert_eq!(
                        vertex_ao_wanted(hw, shadows, on),
                        !(hw && shadows && on),
                        "{hw} {shadows} {on}"
                    );
                }
            }
        }
    }
    // one sample per 4x4 block and frame: every position of the block once in 16 frames, then again
    #[test]
    fn sample_offsets_cover_the_block_in_bayer_order() {
        let first: Vec<_> = (0..16).map(sample_offset).collect();
        let mut sorted = first.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 16);
        assert!(first.iter().all(|&(x, y)| x < BLOCK && y < BLOCK));
        assert_eq!((16..32).map(sample_offset).collect::<Vec<_>>(), first);
        // Bayer: the first four samples fall into the four 2x2 quadrants
        let quads: std::collections::HashSet<_> =
            first[..4].iter().map(|&(x, y)| (x / 2, y / 2)).collect();
        assert_eq!(quads.len(), 4);
    }

    // the fill bridges the 4x4 gaps where a texel has no samples yet; with samples it's narrow
    #[test]
    fn fill_radius_follows_the_sample_count() {
        assert_eq!(fill_radius_for_count(0.0), 8.0);
        assert!(fill_radius_for_count(15.0) <= 2.0);
        assert_eq!(fill_radius_for_count(1000.0), 1.5);
        assert!((0..100)
            .all(|n| fill_radius_for_count(n as f32 + 1.0) <= fill_radius_for_count(n as f32)));
    }

    // still: the full cap; slow: toward `slow`; fast: `fast`
    #[test]
    fn cap_shrinks_with_screen_motion() {
        assert_eq!(motion_cap(0.0, 64.0, 32.0, 4.0), 64.0);
        assert!((motion_cap(0.5, 64.0, 32.0, 4.0) - 32.0).abs() < 1e-4);
        assert_eq!(motion_cap(8.0, 64.0, 32.0, 4.0), 4.0);
        assert_eq!(motion_cap(50.0, 64.0, 32.0, 4.0), 4.0);
        let caps: Vec<f32> = (0..200)
            .map(|i| motion_cap(i as f32 * 0.05, 64.0, 32.0, 4.0))
            .collect();
        assert!(caps.windows(2).all(|w| w[1] <= w[0] + 1e-5));
    }

    // a world point onto the previous frame's shadow-target texel (centres at integer coordinates)
    #[test]
    fn reprojection_lands_on_the_right_texel() {
        use glam::Vec3;
        let size = (200u32, 100u32);
        let proj = perspective_rh(1.2, 2.0, 0.1, 1000.0);
        let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let vp = proj * view;
        // straight ahead: the centre of the target (texel centres at k + 0.5 → coordinate k)
        let q = reproject(Vec3::new(0.0, 0.0, -10.0), vp, size).unwrap();
        assert!(
            (q.x - 99.5).abs() < 1e-3 && (q.y - 49.5).abs() < 1e-3,
            "{q}"
        );
        // up on screen is a smaller y (rows count down from the top)
        let up = reproject(Vec3::new(0.0, 1.0, -10.0), vp, size).unwrap();
        assert!(up.y < q.y);
        // a camera moved sideways by one texel's width at the point's depth: one texel off
        let texel_world = 2.0 * 10.0 * (1.2f32 / 2.0).tan() / size.1 as f32;
        let moved = glam::camera::rh::view::look_at_mat4(
            Vec3::X * texel_world,
            Vec3::X * texel_world + Vec3::NEG_Z,
            Vec3::Y,
        );
        let q2 = reproject(Vec3::new(0.0, 0.0, -10.0), proj * moved, size).unwrap();
        assert!((q.x - q2.x - 1.0).abs() < 1e-2, "{q} {q2}");
        // behind the camera: no history
        assert!(reproject(Vec3::new(0.0, 0.0, 10.0), vp, size).is_none());
    }

    // only settings, target size and block edits start the accumulation over; camera movement and
    // chunk streaming don't (reprojection carries it)
    #[test]
    fn progress_resets_on_settings_size_and_edits() {
        let key = AoKey {
            settings: AoSettings::default(),
            size: (100, 50),
        };
        let mut a = AoProgress::default();
        assert_eq!(a.advance(7, key), 0);
        assert_eq!(a.advance(7, key), 1);
        assert_eq!(a.advance(7, key), 2);
        assert_eq!(a.advance(8, key), 0); // a block edit
        let mut other = key;
        other.settings.rays = 4;
        assert_eq!(a.advance(8, other), 0); // settings
        other.size = (101, 50);
        assert_eq!(a.advance(8, other), 0); // targets resized
        assert_eq!(a.advance(8, other), 1);
        assert_eq!(a.frames(), 1);
        let before = a.frame();
        a.advance(8, other);
        assert_eq!(a.frame(), before + 1);
    }

    // a mesh change within reach of the AO lowers the cap while still to motion_slow for a while;
    // the first frame (no epoch yet) isn't a change
    #[test]
    fn refresh_lowers_the_still_cap_after_geometry_changes() {
        let s = AoSettings::default();
        let mut r = AoRefresh::default();
        assert_eq!(r.still_cap(3, &s), s.samples);
        assert_eq!(r.still_cap(4, &s), s.motion_slow);
        for _ in 1..REFRESH_FRAMES {
            assert_eq!(r.still_cap(4, &s), s.motion_slow);
        }
        assert_eq!(r.still_cap(4, &s), s.samples);
        // never above samples
        let few = AoSettings { samples: 8, ..s };
        assert_eq!(r.still_cap(5, &few), 8);
    }

    #[test]
    fn ao_reach_covers_the_fade_range_and_the_rays() {
        let cam = glam::Vec3::ZERO;
        assert!(within_ao_reach(
            glam::Vec3::new(410.0, 0.0, 0.0),
            8.0,
            cam,
            3.0
        ));
        assert!(!within_ao_reach(
            glam::Vec3::new(420.0, 0.0, 0.0),
            8.0,
            cam,
            3.0
        ));
        assert!(within_ao_reach(
            glam::Vec3::new(5.0, 0.0, 0.0),
            1.0,
            cam,
            3.0
        ));
    }

    #[test]
    fn motion_caps_are_set_and_checked() {
        let mut s = AoSettings::default();
        assert_eq!(
            (s.motion_fast, s.motion_slow),
            (DEFAULT_MOTION_FAST, DEFAULT_MOTION_SLOW)
        );
        assert_eq!((DEFAULT_MOTION_FAST, DEFAULT_MOTION_SLOW), (4, 32));
        assert_eq!(
            AoCommand::parse(&["motion", "4", "32"]),
            Ok(AoCommand::Motion(4, 32))
        );
        for bad in [
            &["motion", "8", "4"][..],
            &["motion", "0", "4"],
            &["motion", "4", "300"],
            &["motion", "4"],
        ] {
            assert_eq!(AoCommand::parse(bad), Err(AO_USAGE), "{bad:?}");
        }
        s.apply(AoCommand::Motion(2, 16));
        assert_eq!((s.motion_fast, s.motion_slow), (2, 16));
        assert!(s.apply(AoCommand::Get).contains("motion 2/16"));
    }
    // the history offset is measured as last frame's projection minus this frame's of the same world
    // point: with a still camera exactly 0, also where the shadow targets are smaller than the screen
    // (their world points sit at screen-pixel centres, which don't reproject onto texel centres)
    #[test]
    fn history_offset_is_zero_for_a_still_camera() {
        use glam::{Mat4, Vec3};
        let size = (1549u32, 968u32);
        let vp = perspective_rh(1.4, 1920.0 / 1200.0, 0.1, 1000.0)
            * glam::camera::rh::view::look_at_mat4(
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(0.3, -0.2, -5.0),
                Vec3::Y,
            );
        for p in [
            Vec3::new(0.7, 0.1, -4.3),
            Vec3::new(-2.0, 1.5, -9.9),
            Vec3::new(3.3, -0.4, -2.2),
        ] {
            assert_eq!(
                history_offset(p, vp, vp, size),
                Some(glam::Vec2::ZERO),
                "{p}"
            );
        }
        // a camera turned a little: the offset is the projection difference
        let turned = vp * Mat4::from_rotation_y(0.001);
        let p = Vec3::new(0.7, 0.1, -4.3);
        let off = history_offset(p, turned, vp, size).unwrap();
        let expected = reproject(p, turned, size).unwrap() - reproject(p, vp, size).unwrap();
        assert!(
            (off - expected).length() < 1e-6 && off.length() > 0.1,
            "{off}"
        );
        assert!(history_offset(Vec3::new(1.0, 2.0, 30.0), vp, vp, size).is_none());
        // behind
    }
}
