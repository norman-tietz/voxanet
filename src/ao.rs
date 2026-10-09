// Hardware ray-traced ambient occlusion (docs/superpowers/specs/2026-10-09-rt-ao-design.md): settings,
// the /ao console command, the rule that picks between it and the mesher's vertex AO, and the frame
// bookkeeping of the progressive AO (one sample per 4x4 block and frame, accumulated while nothing
// changes). No GPU code.

pub const DEFAULT_RADIUS: f32 = 3.0; // world units (blocks): gaps, wall feet, steps, narrow gorges
pub const DEFAULT_RAYS: u32 = 2; // per AO sample (one sample per BLOCK x BLOCK texels and frame)
pub const DEFAULT_SAMPLES: u32 = 64; // accumulation cap per texel; beyond, a moving average
                                     // AO budget: one sample per BLOCK x BLOCK shadow texels and frame (a ray query costs ~10 ns on the
                                     // reference GPU, so full resolution was 33 ms per ray in fullscreen)
pub const BLOCK: u32 = 4;
pub const DEFAULT_STRENGTH: f32 = 1.0;
// camera distance (world units) where AO has faded out, from full at 0.375 × that; no AO rays beyond.
// Out there a 3-block AO is under a shadow texel and the blur no longer smooths its noise, and on
// large planets the LOD terrain begins, drawn morphed but traced unmorphed (rays would start under it)
pub const FADE_END: f32 = 400.0;
pub const AO_USAGE: &str =
    "Usage: /ao get|on|off|radius <0.5-8>|rays <1-16>|strength <0-1>|samples <1-256>";

// the AO look, tuned with /ao (cmd.rs); HwParams.ao = uniform()
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoSettings {
    pub radius: f32,
    pub rays: u32,
    pub strength: f32,
    pub samples: u32,  // accumulation cap per texel
    pub enabled: bool, // off: no AO rays, the mesher's vertex AO instead (vertex_ao_wanted)
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            radius: DEFAULT_RADIUS,
            rays: DEFAULT_RAYS,
            strength: DEFAULT_STRENGTH,
            samples: DEFAULT_SAMPLES,
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
        }
        format!(
            "AO: {}, radius {:.2}, rays {}, strength {:.2}, samples {}",
            if self.enabled {
                "ray traced"
            } else {
                "off (vertex AO)"
            },
            self.radius,
            self.rays,
            self.strength,
            self.samples
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
            ["get"] => Ok(AoCommand::Get),
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

// the AO fill's radius in shadow texels, `frames` after a reset: wide enough to bridge the gaps between
// the sparse samples right after it, narrow (sharp contact AO) once every texel has its own
pub fn fill_radius_texels(frames: u32) -> f32 {
    (2.0 * BLOCK as f32 / (frames as f32 + 1.0).sqrt()).max(1.5)
}

// what the accumulated AO depends on besides the camera and the meshes
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoKey {
    pub settings: AoSettings,
    pub size: (u32, u32), // shadow targets
}

// frames since the progressive AO was last reset, and the running frame index (sample rotation)
#[derive(Default)]
pub struct AoProgress {
    frame: u32,
    frames: u32,
    pose: Option<(glam::Vec3, glam::Vec3, glam::Vec3)>,
    epoch: u64,
    key: Option<AoKey>,
}

impl AoProgress {
    // this frame's camera (position, forward, up), the renderer's mesh epoch and the AO key: returns the
    // frames since the last reset, 0 when anything changed (the accumulation starts over)
    pub fn advance(
        &mut self,
        pos: glam::Vec3,
        forward: glam::Vec3,
        up: glam::Vec3,
        epoch: u64,
        key: AoKey,
    ) -> u32 {
        let still = self.pose.is_some_and(|(p, f, u)| {
            p.distance(pos) <= 1e-4 && f.dot(forward) > 1.0 - 1e-7 && u.dot(up) > 1.0 - 1e-7
        });
        if still && self.epoch == epoch && self.key == Some(key) {
            self.frames = self.frames.saturating_add(1);
        } else {
            self.frames = 0;
        }
        self.pose = Some((pos, forward, up));
        self.epoch = epoch;
        self.key = Some(key);
        self.frame = self.frame.wrapping_add(1);
        self.frames
    }

    pub fn frame(&self) -> u32 {
        self.frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(AoCommand::parse(&["get"]), Ok(AoCommand::Get));
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
            &[],
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

    // right after a reset the fill bridges the 4x4 gaps; converged it's narrow
    #[test]
    fn fill_radius_shrinks_as_samples_arrive() {
        assert_eq!(fill_radius_texels(0), 8.0);
        assert!(fill_radius_texels(15) <= 2.0);
        assert_eq!(fill_radius_texels(1000), 1.5);
        assert!((0..100).all(|n| fill_radius_texels(n + 1) <= fill_radius_texels(n)));
    }

    // frames since the last reset: counts up while nothing changes, back to 0 on any change
    #[test]
    fn progress_resets_on_any_change() {
        use glam::Vec3;
        let key = AoKey {
            settings: AoSettings::default(),
            size: (100, 50),
        };
        let (p, f, u) = (Vec3::new(1.0, 2.0, 3.0), Vec3::Z, Vec3::Y);
        let mut a = AoProgress::default();
        assert_eq!(a.advance(p, f, u, 7, key), 0);
        assert_eq!(a.advance(p, f, u, 7, key), 1);
        assert_eq!(a.advance(p + Vec3::splat(1e-6), f, u, 7, key), 2); // below tolerance
        let q = p + Vec3::X * 0.01;
        assert_eq!(a.advance(q, f, u, 7, key), 0); // moved
        assert_eq!(a.advance(q, f, u, 7, key), 1);
        let turned = (f + Vec3::X * 0.01).normalize();
        assert_eq!(a.advance(q, turned, u, 7, key), 0); // turned
        assert_eq!(a.advance(q, turned, u, 8, key), 0); // meshes changed
        let mut other = key;
        other.settings.rays = 4;
        assert_eq!(a.advance(q, turned, u, 8, other), 0); // settings
        other.size = (101, 50);
        assert_eq!(a.advance(q, turned, u, 8, other), 0); // targets resized
        assert_eq!(a.advance(q, turned, u, 8, other), 1);
        // the frame index keeps running across resets (the Bayer rotation)
        let before = a.frame();
        a.advance(p, f, u, 9, key);
        assert_eq!(a.frame(), before + 1);
    }
}
