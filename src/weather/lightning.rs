// lightning.rs
// Lightning (spec docs/superpowers/specs/2026-10-12-lightning-design.md), the CPU side: a deterministic
// strike schedule – a strike is a hash of (spatial cell, time slot) gated by the storm value and the
// planet type, a pure function of the planet-frame position and the shared clock – the strikes active
// near a camera, their flash envelope, and the values for the GPU. atmosphere.wgsl adds the cloud glow
// and the flash; bolt.rs builds the bolt geometry.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::{smoothstep, storm, WeatherForce, WeatherKind};

pub const SLOT_SECS: f32 = 1.5; // 3600 / 1.5 = 2400 slots: the clock wrap falls on a slot edge
pub const CELL_SIZE: f32 = 80.0; // strike cells on the surface, about this wide
pub const RANGE: f32 = 3000.0; // the voxel frame's strike distance
pub const MAX_STRIKES: usize = 6;
pub const LIFE: f32 = 0.65; // seconds a strike lasts (main flash plus re-strikes)
pub const FLASH_RANGE: f32 = 600.0; // distance at which a strike's ambient flash is a quarter (small planets: 1.2 R)
const RATE_FULL: f32 = 0.03; // strikes per cell-slot at storm 1 (an 80-unit cell: one per 50 s)
const STORM_MIN: f32 = 0.5;
const RISE: f32 = 0.03;
const DECAY: f32 = 0.07;
const WRAP: f32 = 3600.0;

// the glow and flash constants shared with atmosphere.wgsl (name there, value)
#[cfg(test)]
pub const GLOW_RADIUS: f32 = 0.13; // cloud glow radius, in planet radii at the cloud shell
#[cfg(test)]
pub const CONSTANTS: [(&str, f32); 1] = [("LIGHTNING_GLOW_RADIUS", GLOW_RADIUS)];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strike {
    pub pos: Vec3,      // on the planet's surface sphere, planet frame
    pub age: f32,       // seconds since it started, 0..=LIFE
    pub intensity: f32, // the envelope now, 0..1
    pub seed: u32,      // the cell/slot hash: seeds the bolt's shape
}

pub const USAGE: &str = "Usage: /weather lightning [on|off|rate <0-3>]";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightningSettings {
    pub on: bool,
    pub rate: f32,
}

impl Default for LightningSettings {
    fn default() -> Self {
        Self {
            on: true,
            rate: 1.0,
        }
    }
}

// how often a planet type's storms strike, relative to rain storms
pub fn kind_rate(kind: WeatherKind) -> f32 {
    match kind {
        WeatherKind::Rain | WeatherKind::AcidRain => 1.0,
        WeatherKind::Ash => 0.4,
        WeatherKind::Dust => 0.1,
        WeatherKind::Snow | WeatherKind::None => 0.0,
    }
}

// the PCG integer hash (also bolt.rs's random sequence)
pub(super) fn pcg(mut h: u32) -> u32 {
    h = h.wrapping_mul(747796405).wrapping_add(2891336453);
    h = ((h >> ((h >> 28) + 4)) ^ h).wrapping_mul(277803737);
    (h >> 22) ^ h
}

fn hash(ix: i32, iy: i32, iz: i32, slot: u32, salt: u32) -> u32 {
    let mut h = pcg(ix as u32 ^ 0x68bc21eb);
    h = pcg(h ^ iy as u32);
    h = pcg(h ^ iz as u32);
    h = pcg(h ^ slot);
    pcg(h ^ salt)
}

// cells per cube-face edge: about CELL_SIZE world units wide on the surface
fn grid_size(planet_radius: f32) -> i32 {
    ((2.0 * planet_radius / CELL_SIZE).round() as i32).clamp(4, 32)
}

// the unit direction of point (u, v) in -1..1 on cube face `face` (any consistent cube-to-sphere mapping)
fn face_dir(face: i32, u: f32, v: f32) -> Vec3 {
    let sign = if face & 1 == 0 { 1.0 } else { -1.0 };
    let p = match face >> 1 {
        0 => Vec3::new(sign, u, v),
        1 => Vec3::new(u, sign, v),
        _ => Vec3::new(u, v, sign),
    };
    p.normalize()
}

// a hash's top 24 bits as a float in [0, 1)
pub(super) fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

fn pulse(age: f32) -> f32 {
    if age < 0.0 {
        0.0
    } else if age < RISE {
        age / RISE
    } else {
        (-(age - RISE) / DECAY).exp()
    }
}

// the flash over a strike's life: the main stroke, then a dip and one or two weaker re-strikes, the timing
// from the seed
pub fn envelope(age: f32, seed: u32) -> f32 {
    let r1 = 0.14 + 0.08 * unit(pcg(seed ^ 0x9e37_79b9));
    let r2 = r1 + 0.08 + 0.06 * unit(pcg(seed ^ 0x85eb_ca6b));
    (pulse(age) + 0.5 * pulse(age - r1) + 0.3 * pulse(age - r2)).min(1.0)
}

// seconds from `start` to `t` across the clock wrap
fn age_since(t: f32, start: f32) -> f32 {
    let a = t - start;
    if a < -(SLOT_SECS * 2.0) {
        a + WRAP
    } else {
        a
    }
}

// the planet a strike schedule runs on
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrikePlanet {
    pub kind: WeatherKind, // how often its storms strike (kind_rate)
    pub radius: f32,       // the surface sphere the strikes lie on
    pub seed: u32,         // varies the sites per planet
}

// The strikes alive at clock `t` (wrapped at 3600) seen from `cam` (planet frame): every cell of a cube-face
// grid over the planet's surface (n x n per face, cells about CELL_SIZE wide), a strike per cell and slot
// jittered inside the cell, within `range` of the camera (`f32::MAX`: the whole planet), strongest first, at
// most MAX_STRIKES. `rate` is the /weather lightning setting.
pub fn active_strikes(
    planet: StrikePlanet,
    rate: f32,
    cam: Vec3,
    range: f32,
    t: f32,
    force: &WeatherForce,
) -> Vec<Strike> {
    let StrikePlanet {
        kind,
        radius: planet_radius,
        seed: planet_seed,
    } = planet;
    let per_slot = RATE_FULL * kind_rate(kind) * rate;
    if per_slot <= 0.0 || planet_radius <= 0.0 {
        return vec![];
    }
    let t = t.rem_euclid(WRAP);
    let slot = (t / SLOT_SECS).floor() as u32;
    let slot_start = slot as f32 * SLOT_SECS;
    let n = grid_size(planet_radius);
    let salt = pcg(planet_seed ^ 0x51ed_270b);
    let mut out = Vec::new();
    for face in 0..6 {
        for i in 0..n {
            for j in 0..n {
                // the cheap test first: the chance can't exceed per_slot
                let h = hash(face, i, j, slot, salt);
                let u = unit(h);
                if u >= per_slot {
                    continue;
                }
                let jx = unit(hash(face, i, j, slot, salt ^ 1));
                let jy = unit(hash(face, i, j, slot, salt ^ 2));
                let dir = face_dir(
                    face,
                    -1.0 + 2.0 * (i as f32 + jx) / n as f32,
                    -1.0 + 2.0 * (j as f32 + jy) / n as f32,
                );
                let s = storm(dir, slot_start, force);
                if u >= per_slot * smoothstep(STORM_MIN, 1.0, s) {
                    continue;
                }
                let start = slot_start + unit(pcg(h ^ 5)) * (SLOT_SECS - LIFE);
                let age = age_since(t, start);
                if !(0.0..=LIFE).contains(&age) {
                    continue;
                }
                let seed = pcg(h ^ 6);
                let intensity = envelope(age, seed);
                if intensity < 0.01 {
                    continue;
                }
                let pos = dir * planet_radius;
                if range < f32::MAX && pos.distance(cam) > range {
                    continue;
                }
                out.push(Strike {
                    pos,
                    age,
                    intensity,
                    seed,
                });
            }
        }
    }
    out.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    out.truncate(MAX_STRIKES);
    out
}

// the scene flash at `cam` on a planet of `planet_radius`: each strike's intensity, falling off with distance
// (scale FLASH_RANGE, at most 1.2 R) and counted only when its cloud point is above the camera's horizon
pub fn flash_at(strikes: &[Strike], cam: Vec3, planet_radius: f32) -> f32 {
    let range = FLASH_RANGE.min(1.2 * planet_radius).max(1.0);
    let cloud = crate::planet::data::CLOUD_ALT * planet_radius;
    strikes
        .iter()
        .map(|s| {
            let d = s.pos.distance(cam) / range;
            let c = s.pos.normalize_or_zero() * cloud;
            let seg = c - cam;
            let l2 = seg.length_squared();
            let t = if l2 > 1e-6 {
                (-cam.dot(seg) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let clear = (cam + seg * t).length();
            let visible = smoothstep(planet_radius - 3.0, planet_radius + 1.0, clear);
            visible * s.intensity / (1.0 + d * d)
        })
        .sum::<f32>()
        .min(1.0)
}

// Global.strikes / PlanetUniform.strikes: xyz position, w intensity
pub fn gpu_strikes(strikes: &[Strike]) -> [[f32; 4]; MAX_STRIKES] {
    let mut out = [[0.0; 4]; MAX_STRIKES];
    for (o, s) in out.iter_mut().zip(strikes) {
        *o = [s.pos.x, s.pos.y, s.pos.z, s.intensity];
    }
    out
}

// the strikes to draw this frame in the voxel frame: none with lightning or the weather off
pub fn frame_strikes(
    planet: StrikePlanet,
    settings: &LightningSettings,
    weather_on: bool,
    cam: Vec3,
    t: f32,
    force: &WeatherForce,
) -> Vec<Strike> {
    if !settings.on || !weather_on {
        return vec![];
    }
    active_strikes(planet, settings.rate, cam, RANGE, t, force)
}

// one bolt segment for lightning.wgsl: a (xyz, width), b (xyz, brightness)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BoltInstance {
    pub a: [f32; 4],
    pub b: [f32; 4],
}

// the bolts of `strikes` as segments: from the cloud shell down to the ground under each strike
pub fn bolt_instances(
    strikes: &[Strike],
    planet_radius: f32,
    cloud_radius: f32,
    ground_radius: impl Fn(Vec3) -> f32,
) -> Vec<BoltInstance> {
    let mut out = Vec::with_capacity(strikes.len() * super::bolt::MAX_SEGMENTS_PER_BOLT);
    for s in strikes {
        let dir = s.pos.normalize_or_zero();
        if dir == Vec3::ZERO {
            continue;
        }
        let ground = ground_radius(dir).max(planet_radius * 0.5);
        let segs = super::bolt::build(s.seed, dir * cloud_radius, dir * ground);
        for g in segs {
            // keep every vertex above the ground under it (the jag may dip towards the surface)
            let clamp = |p: Vec3| {
                let r = p.length();
                if r < ground {
                    p / r * ground
                } else {
                    p
                }
            };
            let (a, b) = (clamp(g.a), clamp(g.b));
            out.push(BoltInstance {
                a: [a.x, a.y, a.z, g.width],
                b: [b.x, b.y, b.z, s.intensity * g.glow],
            });
        }
    }
    out
}

pub struct LightningGpu {
    pipeline: wgpu::RenderPipeline,
    instances: wgpu::Buffer,
    count: u32,
}

impl LightningGpu {
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule, // the scene shader (lightning.wgsl is appended to it)
        global_layout: &wgpu::BindGroupLayout,
        sample_layout: &wgpu::BindGroupLayout,
        textures_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lightning_pipeline_layout"),
            bind_group_layouts: &[
                Some(global_layout),
                None,
                Some(sample_layout),
                Some(textures_layout),
            ],
            immediate_size: 0,
        });
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lightning"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_bolt"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_bolt"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: additive,
                        alpha: wgpu::BlendComponent::OVER,
                    }),
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
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bolt Instances"),
            size: (MAX_STRIKES * super::bolt::MAX_SEGMENTS_PER_BOLT * 32) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            instances,
            count: 0,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, instances: &[BoltInstance]) {
        let n = instances
            .len()
            .min(MAX_STRIKES * super::bolt::MAX_SEGMENTS_PER_BOLT);
        self.count = n as u32;
        if n > 0 {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances[..n]));
        }
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..4, 0..self.count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::biome::PlanetType;
    use crate::weather::{WeatherForce, WeatherSettings};

    const R: f32 = 250.0;
    const SEED: u32 = 42;

    #[test]
    fn bolt_instances_run_from_the_cloud_shell_to_the_ground() {
        let s = Strike {
            pos: Vec3::new(0.0, R, 0.0),
            age: 0.03,
            intensity: 1.0,
            seed: 9,
        };
        let cloud = R * 1.4;
        let inst = bolt_instances(&[s], R, cloud, |_| R - 3.0);
        assert!(
            !inst.is_empty()
                && inst.len() <= MAX_STRIKES * crate::weather::bolt::MAX_SEGMENTS_PER_BOLT
        );
        // the first segment starts on the cloud shell, the main channel ends on the ground, never below it
        let top = Vec3::new(inst[0].a[0], inst[0].a[1], inst[0].a[2]);
        assert!((top.length() - cloud).abs() < 1e-2);
        let lowest = inst
            .iter()
            .map(|i| Vec3::new(i.b[0], i.b[1], i.b[2]).length())
            .fold(f32::MAX, f32::min);
        assert!(lowest >= R - 3.0 - 1e-2, "{lowest}");
        assert!(
            inst.iter().all(|i| i.b[3] > 0.0),
            "brightness follows the envelope"
        );
    }

    #[test]
    fn bolt_instances_are_empty_without_strikes() {
        assert!(bolt_instances(&[], R, R * 1.4, |_| R).is_empty());
    }

    fn kind(t: PlanetType) -> WeatherKind {
        WeatherKind::for_planet(t)
    }

    // the test planet of radius R with that weather
    fn on_r(kind: WeatherKind) -> StrikePlanet {
        StrikePlanet {
            kind,
            radius: R,
            seed: SEED,
        }
    }

    // a force that makes the region within about 60 degrees of +Y one storm of value 1 (natural storms go on elsewhere)
    fn full_storm() -> WeatherForce {
        let mut s = WeatherSettings::default();
        s.apply(
            crate::weather::WeatherCommand::Force(crate::weather::Force::Storm),
            WeatherKind::Rain,
            Vec3::Y,
            0.0,
        );
        s.tick(1000.0);
        s.force_uniform()
    }

    fn clear_sky() -> WeatherForce {
        let mut s = WeatherSettings::default();
        s.apply(
            crate::weather::WeatherCommand::Force(crate::weather::Force::Clear),
            WeatherKind::Rain,
            Vec3::Y,
            0.0,
        );
        s.force_uniform()
    }

    // every strike over an hour of slots, as (slot time, strike) – used to count and compare
    fn hour(kind: WeatherKind, force: &WeatherForce) -> Vec<(f32, Strike)> {
        let mut all = vec![];
        let mut t = 0.0;
        while t < 3600.0 {
            // sample mid-life of the slot's strikes: ask for all cells at several ages
            for s in active_strikes(on_r(kind), 1.0, Vec3::new(0.0, R, 0.0), f32::MAX, t, force) {
                all.push((t, s));
            }
            t += 0.25;
        }
        all
    }

    #[test]
    fn wgsl_constants_match_the_rust_mirror() {
        let w = include_str!("../render/atmosphere.wgsl");
        crate::weather::tests::assert_wgsl_constants(w, &CONSTANTS);
        assert!(
            w.contains("array<vec4<f32>, 6>"),
            "MAX_STRIKES is 6 in the WGSL arrays"
        );
        assert_eq!(MAX_STRIKES, 6);
    }

    #[test]
    fn the_schedule_is_deterministic() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let a = active_strikes(on_r(k), 1.0, Vec3::new(0.0, R, 0.0), f32::MAX, 123.4, &f);
        let b = active_strikes(on_r(k), 1.0, Vec3::new(0.0, R, 0.0), f32::MAX, 123.4, &f);
        assert_eq!(a, b);
    }

    #[test]
    fn a_full_storm_strikes_at_roughly_the_planned_rate() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        // distinct strikes over the hour, counted by (position, seed)
        let mut seen = std::collections::HashSet::new();
        for (t, s) in hour(k, &f) {
            seen.insert((s.seed, (t - s.age).round() as i64));
        }
        // 216 cells (R 250) × 2400 slots × 0.03 ≈ 15,500 at a full storm everywhere; the forced region is about a
        // quarter of the planet and natural storms add a little elsewhere, so a few thousand. The band only
        // guards the order of magnitude.
        assert!(seen.len() > 1000 && seen.len() < 20_000, "{}", seen.len());
    }

    #[test]
    fn no_storm_no_lightning() {
        let k = kind(PlanetType::EarthLike);
        // the force covers a region of about 60 degrees around +Y (elsewhere the natural storms go on), so look
        // only at the neighbourhood of the camera there, and after the force has faded in
        let mut t = 10.0;
        while t < 600.0 {
            assert!(active_strikes(on_r(k), 1.0, Vec3::Y * R, 100.0, t, &clear_sky()).is_empty());
            t += 0.5;
        }
    }

    #[test]
    fn weather_off_means_no_lightning() {
        let mut s = WeatherSettings::default();
        s.on = false;
        let f = s.force_uniform();
        let k = kind(PlanetType::EarthLike);
        assert!(hour(k, &f).is_empty());
    }

    #[test]
    fn ice_and_shattered_never_strike_desert_rarely() {
        let f = full_storm();
        assert!(hour(kind(PlanetType::Ice), &f).is_empty());
        assert!(hour(kind(PlanetType::Shattered), &f).is_empty());
        let wet = hour(kind(PlanetType::EarthLike), &f).len();
        let dry = hour(kind(PlanetType::Desert), &f).len();
        assert!(dry > 0, "a forced storm still thunders on a desert");
        assert!(dry * 4 < wet, "{dry} vs {wet}");
        assert_eq!(kind_rate(kind(PlanetType::Ice)), 0.0);
    }

    #[test]
    fn the_rate_setting_scales_the_strikes() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let count = |rate: f32| {
            let mut n = 0;
            let mut t = 0.0;
            while t < 1200.0 {
                n += active_strikes(on_r(k), rate, Vec3::Y * R, f32::MAX, t, &f).len();
                t += 0.25;
            }
            n
        };
        assert_eq!(count(0.0), 0);
        assert!(count(2.0) > count(1.0));
    }

    #[test]
    fn strikes_lie_on_the_planet_surface_and_age_within_a_life() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        for (_, s) in hour(k, &f).into_iter().take(500) {
            assert!((s.pos.length() - R).abs() < 1e-2, "{}", s.pos.length());
            assert!((0.0..=LIFE).contains(&s.age), "{}", s.age);
            assert!(s.intensity > 0.0 && s.intensity <= 1.0);
        }
    }

    #[test]
    fn the_active_set_is_capped_and_strongest_first() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let mut t = 0.0;
        while t < 300.0 {
            let v = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, t, &f);
            assert!(v.len() <= MAX_STRIKES);
            let score = |s: &Strike| s.intensity;
            for w in v.windows(2) {
                assert!(score(&w[0]) >= score(&w[1]) - 1e-6);
            }
            t += 0.25;
        }
    }

    #[test]
    fn the_range_limits_the_strikes_to_the_neighbourhood() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let cam = Vec3::new(0.0, R, 0.0);
        let mut t = 0.0;
        while t < 300.0 {
            for s in active_strikes(on_r(k), 3.0, cam, 800.0, t, &f) {
                assert!(s.pos.distance(cam) <= 800.0, "{}", s.pos.distance(cam));
            }
            t += 0.25;
        }
    }

    #[test]
    fn the_schedule_is_continuous_across_the_clock_wrap() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        // t = 3599.9 and 3599.9 - 3600 are the same moment: the same strikes (up to f32 rounding)
        let a = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, 3599.9, &f);
        let b = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, 3599.9 - WRAP, &f);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!(x.seed == y.seed && x.pos == y.pos && (x.age - y.age).abs() < 1e-2);
        }
        // strikes never outlive their slot, so across the wrap itself there are no pairs by construction;
        // the same age arithmetic is checked within slots (10.0 and 10.1 share one), and the wrap maps t
        // = 3600 + x onto x
        let mut pairs = 0;
        for i in 0..200 {
            let t0 = 10.0 + i as f32 * 1.5;
            let a = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, t0, &f);
            let b = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, t0 + 0.1, &f);
            for x in &a {
                if let Some(y) = b.iter().find(|y| y.seed == x.seed) {
                    pairs += 1;
                    assert!(
                        (y.age - (x.age + 0.1)).abs() < 1e-3,
                        "{} vs {}",
                        y.age,
                        x.age
                    );
                }
            }
        }
        assert!(pairs > 0);
        let a = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, 3600.1, &f);
        let b = active_strikes(on_r(k), 3.0, Vec3::Y * R, f32::MAX, 0.1, &f);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!(x.seed == y.seed && x.pos == y.pos && (x.age - y.age).abs() < 1e-2);
        }
    }

    #[test]
    fn the_envelope_flashes_dips_and_dies() {
        let seed = 12345;
        assert_eq!(envelope(-0.1, seed), 0.0);
        let peak = (0..60)
            .map(|i| envelope(i as f32 * 0.005, seed))
            .fold(0.0, f32::max);
        assert!(peak > 0.9 && peak <= 1.0, "{peak}");
        assert!(envelope(LIFE, seed) < 0.02);
        // dips between the main flash and a re-strike
        let dip = (0..40)
            .map(|i| envelope(0.1 + i as f32 * 0.003, seed))
            .fold(1.0, f32::min);
        assert!(dip < 0.5 * peak);
    }

    fn strike_at(pos: Vec3) -> Strike {
        Strike {
            pos,
            age: 0.05,
            intensity: 1.0,
            seed: 1,
        }
    }

    #[test]
    fn flash_falls_off_with_distance() {
        // same side: a camera on the surface, strikes at growing angles up to the horizon
        let cam = Vec3::Y * (R + 1.0);
        let at = |deg: f32| {
            let a = deg.to_radians();
            flash_at(&[strike_at(Vec3::new(a.sin(), a.cos(), 0.0) * R)], cam, R)
        };
        assert!(at(0.0) > 0.9 && at(0.0) <= 1.0);
        assert!(
            at(0.0) > at(10.0) && at(10.0) > at(30.0),
            "{} {} {}",
            at(0.0),
            at(10.0),
            at(30.0)
        );
        // the old falloff: intensity / (1 + (d / range)^2) for a visible strike
        let d = Vec3::new(0.0, R + 1.0, 0.0)
            .distance(Vec3::new(10f32.to_radians().sin(), 10f32.to_radians().cos(), 0.0) * R);
        let range = FLASH_RANGE.min(1.2 * R);
        assert!((at(10.0) - 1.0 / (1.0 + (d / range).powi(2))).abs() < 1e-4);
        assert_eq!(flash_at(&[], cam, R), 0.0);
    }

    #[test]
    fn flash_does_not_go_through_the_planet() {
        let cam = Vec3::Y * (R + 1.0);
        assert_eq!(flash_at(&[strike_at(-Vec3::Y * R)], cam, R), 0.0);
        assert_eq!(flash_at(&[strike_at(-Vec3::X * R)], cam, R), 0.0);
    }

    #[test]
    fn flash_range_scales_with_small_planets() {
        let r = 40.0;
        let cam = Vec3::Y * (r + 1.0);
        let near = flash_at(&[strike_at(Vec3::Y * r)], cam, r);
        let far = flash_at(
            &[strike_at(Vec3::new(0.5, 1.0, 0.0).normalize() * r)],
            cam,
            r,
        );
        assert!(near > far && far > 0.0);
    }

    fn key(p: Vec3) -> (u32, u32, u32) {
        (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
    }

    #[test]
    fn strikes_spread_over_the_surface_on_every_planet_size() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        for r in [40.0, 100.0, 168.0, 250.0, 500.0, 1000.0] {
            let mut seen = std::collections::HashSet::new();
            let mut t = 0.0;
            while t < 120.0 {
                for s in active_strikes(
                    StrikePlanet {
                        kind: k,
                        radius: r,
                        seed: SEED,
                    },
                    1.0,
                    Vec3::Y * r,
                    f32::MAX,
                    t,
                    &f,
                ) {
                    seen.insert(key(s.pos));
                }
                t += 0.25;
            }
            assert!(seen.len() >= 20, "R {r}: {} distinct sites", seen.len());
        }
    }

    #[test]
    fn planets_with_other_seeds_strike_elsewhere() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let sites = |seed: u32| {
            let mut v = std::collections::HashSet::new();
            let mut t = 0.0;
            while t < 120.0 {
                for s in active_strikes(
                    StrikePlanet {
                        kind: k,
                        radius: R,
                        seed,
                    },
                    1.0,
                    Vec3::Y * R,
                    f32::MAX,
                    t,
                    &f,
                ) {
                    v.insert(key(s.pos));
                }
                t += 0.25;
            }
            v
        };
        let (a, b) = (sites(1), sites(2));
        assert!(!a.is_empty() && !b.is_empty());
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn a_small_planet_strikes_near_a_camera_on_its_surface() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let cam = Vec3::Y * 168.0;
        let mut near = false;
        let mut t = 0.0;
        while t < 60.0 {
            near |= !active_strikes(
                StrikePlanet {
                    kind: k,
                    radius: 168.0,
                    seed: SEED,
                },
                1.0,
                cam,
                300.0,
                t,
                &f,
            )
            .is_empty();
            t += 0.25;
        }
        assert!(near);
    }

    #[test]
    fn the_site_changes_from_slot_to_slot() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        // rate 3, a small grid: collect the sites per slot and check no site repeats in the next slot
        let sites_in_slot = |slot: u32| {
            let mut v = std::collections::HashSet::new();
            let mut t = slot as f32 * SLOT_SECS;
            while t < (slot + 1) as f32 * SLOT_SECS {
                for s in active_strikes(
                    StrikePlanet {
                        kind: k,
                        radius: 40.0,
                        seed: SEED,
                    },
                    3.0,
                    Vec3::Y * 40.0,
                    f32::MAX,
                    t,
                    &f,
                ) {
                    v.insert(key(s.pos));
                }
                t += 0.05;
            }
            v
        };
        let mut compared = 0;
        for slot in 10..60 {
            let (a, b) = (sites_in_slot(slot), sites_in_slot(slot + 1));
            compared += a.len().min(b.len());
            assert!(a.is_disjoint(&b), "slot {slot} repeats a site");
        }
        assert!(compared > 0);
    }

    #[test]
    fn gpu_packing_pads_with_zeros() {
        let s = Strike {
            pos: Vec3::new(1.0, 2.0, 3.0),
            age: 0.0,
            intensity: 0.5,
            seed: 0,
        };
        let g = gpu_strikes(&[s]);
        assert_eq!(g[0], [1.0, 2.0, 3.0, 0.5]);
        assert_eq!(g[1], [0.0; 4]);
    }

    #[test]
    fn a_camera_at_the_strike_or_the_centre_gives_finite_values() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        for cam in [Vec3::ZERO, Vec3::Y * R] {
            for s in active_strikes(on_r(k), 3.0, cam, f32::MAX, 10.0, &f) {
                assert!(s.pos.is_finite() && s.intensity.is_finite());
            }
            assert!(flash_at(
                &active_strikes(on_r(k), 3.0, cam, f32::MAX, 10.0, &f),
                cam,
                R
            )
            .is_finite());
        }
    }

    #[test]
    fn frame_strikes_are_empty_when_lightning_or_weather_is_off() {
        let f = full_storm();
        let k = kind(PlanetType::EarthLike);
        let on = LightningSettings::default();
        let off = LightningSettings { on: false, ..on };
        let mut found = false;
        let mut t = 0.0;
        while t < 120.0 {
            found |= !frame_strikes(on_r(k), &on, true, Vec3::Y * R, t, &f).is_empty();
            assert!(frame_strikes(on_r(k), &off, true, Vec3::Y * R, t, &f).is_empty());
            assert!(
                frame_strikes(on_r(k), &on, false, Vec3::Y * R, t, &f).is_empty(),
                "weather off"
            );
            t += 0.25;
        }
        assert!(found);
    }
}
