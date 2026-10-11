// weather.rs
// The deterministic weather (spec docs/superpowers/specs/2026-10-11-weather-design.md): a Rust mirror of
// weather.wgsl (keep them identical, line by line) for the /weather readout, the overcast sky and the
// lens flare at the camera, and the tests; WeatherSettings and the /weather console command; the weather
// map's snapshot schedule. The GPU map itself (WeatherMap) is further down.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::biome::PlanetType;

#[cfg(test)]
pub(crate) const WEATHER_FN_SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    include_str!("star.wgsl"),
    "\n",
    include_str!("weather.wgsl")
);

pub(crate) const WEATHER_BAKE_SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    include_str!("star.wgsl"),
    "\n",
    include_str!("weather.wgsl"),
    "\n",
    include_str!("weather_bake.wgsl")
);

pub const SNAPSHOT_SECS: f64 = 5.0;
const SLOTS: u32 = 3;
pub const WEATHER_USAGE: &str = "Usage: /weather [on|off|auto|clear|rain|storm|snow]";

const WEATHER_PERIOD: f32 = 3600.0;
const WEATHER_SCALE: f32 = 1.6;
const WEATHER_SPIN_CYCLES: f32 = 2.0;
const WEATHER_TILT_CYCLES: f32 = 1.0;
const WEATHER_EVOLVE_CYCLES: f32 = 3.0;
const WEATHER_EVOLVE_RADIUS: f32 = 0.6;
const STORM_LO: f32 = 0.46;
const STORM_HI: f32 = 0.62;
const RAIN_LO: f32 = 0.3;
const RAIN_HI: f32 = 0.8;
const SNOW_LAT_LO: f32 = 0.6;
const SNOW_LAT_HI: f32 = 0.8;
const WEATHER_AGE_LOOKBACK: f32 = 200.0;
const FORCE_FADE_SECS: f32 = 4.0;
const FORCE_EDGE_LO: f32 = 0.35;
const FORCE_EDGE_HI: f32 = 0.65;
#[cfg_attr(not(test), allow(dead_code))] // the cloud bake's copy (weather.wgsl) is the one in use
pub const STORM_COVER: f32 = 0.35;

#[cfg(test)]
const CONSTANTS: [(&str, f32); 17] = [
    ("WEATHER_PERIOD", WEATHER_PERIOD),
    ("WEATHER_SCALE", WEATHER_SCALE),
    ("WEATHER_SPIN_CYCLES", WEATHER_SPIN_CYCLES),
    ("WEATHER_TILT_CYCLES", WEATHER_TILT_CYCLES),
    ("WEATHER_EVOLVE_CYCLES", WEATHER_EVOLVE_CYCLES),
    ("WEATHER_EVOLVE_RADIUS", WEATHER_EVOLVE_RADIUS),
    ("STORM_LO", STORM_LO),
    ("STORM_HI", STORM_HI),
    ("RAIN_LO", RAIN_LO),
    ("RAIN_HI", RAIN_HI),
    ("SNOW_LAT_LO", SNOW_LAT_LO),
    ("SNOW_LAT_HI", SNOW_LAT_HI),
    ("FORCE_FADE_SECS", FORCE_FADE_SECS),
    ("FORCE_EDGE_LO", FORCE_EDGE_LO),
    ("FORCE_EDGE_HI", FORCE_EDGE_HI),
    ("STORM_COVER", STORM_COVER),
    ("WEATHER_AGE_LOOKBACK", WEATHER_AGE_LOOKBACK),
];

const ACTIVE_END: f32 = 1e9; // WeatherForce.timing.y while a force is active

// must match WeatherForce in weather.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct WeatherForce {
    pub dir_value: [f32; 4], // xyz: forced region centre, w: forced storm value
    pub timing: [f32; 4], // x: start, y: end (ACTIVE_END while active), z: snow share (< 0: none), w: on
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// atmosphere.wgsl hash31 / value_noise3
fn hash31(p: Vec3) -> f32 {
    let mut p3 = Vec3::new(
        fract(p.x * 0.1031),
        fract(p.y * 0.1031),
        fract(p.z * 0.1031),
    );
    let d = p3.dot(Vec3::new(p3.y, p3.z, p3.x) + Vec3::splat(33.33));
    p3 += Vec3::splat(d);
    fract((p3.x + p3.y) * p3.z)
}

fn value_noise3(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let h = |x: f32, y: f32, z: f32| hash31(i + Vec3::new(x, y, z));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(h(0.0, 0.0, 0.0), h(1.0, 0.0, 0.0), u.x);
    let x10 = lerp(h(0.0, 1.0, 0.0), h(1.0, 1.0, 0.0), u.x);
    let x01 = lerp(h(0.0, 0.0, 1.0), h(1.0, 0.0, 1.0), u.x);
    let x11 = lerp(h(0.0, 1.0, 1.0), h(1.0, 1.0, 1.0), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}

fn rotate(v: Vec3, k: Vec3, a: f32) -> Vec3 {
    v * a.cos() + k.cross(v) * a.sin() + k * k.dot(v) * (1.0 - a.cos())
}

fn noise(dir: Vec3, t: f32) -> f32 {
    let w = std::f32::consts::TAU / WEATHER_PERIOD;
    let mut d = rotate(dir, Vec3::Y, t * w * WEATHER_SPIN_CYCLES);
    d = rotate(d, Vec3::new(0.6, 0.0, 0.8), t * w * WEATHER_TILT_CYCLES);
    let e = t * w * WEATHER_EVOLVE_CYCLES;
    let p = d * WEATHER_SCALE
        + Vec3::new(e.cos(), e.sin(), 0.0) * WEATHER_EVOLVE_RADIUS
        + Vec3::new(11.3, 4.7, 8.1);
    0.65 * value_noise3(p) + 0.35 * value_noise3(p * 2.03 + Vec3::new(17.1, 3.3, 9.7))
}

fn age(t: f32, since: f32) -> f32 {
    let a = t - since;
    if a < -WEATHER_AGE_LOOKBACK {
        a + WEATHER_PERIOD
    } else {
        a
    }
}

pub(crate) fn weather_force_weight(dir: Vec3, t: f32, f: &WeatherForce) -> f32 {
    let centre = Vec3::new(f.dir_value[0], f.dir_value[1], f.dir_value[2]);
    let region = smoothstep(FORCE_EDGE_LO, FORCE_EDGE_HI, dir.dot(centre));
    let fade_in = (age(t, f.timing[0]) / FORCE_FADE_SECS).clamp(0.0, 1.0);
    let fade_out = if f.timing[1] < 1e8 {
        1.0 - (age(t, f.timing[1]) / FORCE_FADE_SECS).clamp(0.0, 1.0)
    } else {
        1.0
    };
    region * fade_in * fade_out
}

pub fn storm(dir: Vec3, t: f32, f: &WeatherForce) -> f32 {
    if f.timing[3] < 0.5 {
        return 0.0;
    }
    let s = smoothstep(STORM_LO, STORM_HI, noise(dir, t));
    let w = weather_force_weight(dir, t, f);
    s + (f.dir_value[3] - s) * w
}

fn rain(dir: Vec3, t: f32, f: &WeatherForce) -> f32 {
    smoothstep(RAIN_LO, RAIN_HI, storm(dir, t, f))
}

pub fn sample(dir: Vec3, t: f32, f: &WeatherForce) -> [f32; 4] {
    let s0 = storm(dir, t, f);
    let r0 = smoothstep(RAIN_LO, RAIN_HI, s0);
    let wet = r0
        .max(0.7 * rain(dir, t - 30.0, f))
        .max(0.45 * rain(dir, t - 60.0, f))
        .max(0.2 * rain(dir, t - 90.0, f));
    let cover = r0
        .max(0.7 * rain(dir, t - 60.0, f))
        .max(0.45 * rain(dir, t - 120.0, f))
        .max(0.2 * rain(dir, t - 180.0, f));
    let mut share = smoothstep(SNOW_LAT_LO, SNOW_LAT_HI, dir.y.abs());
    if f.timing[2] >= 0.0 {
        share += (f.timing[2] - share) * weather_force_weight(dir, t, f);
    }
    [s0, wet, cover, share]
}

// weather.wgsl weather_surface: (wet, cover)
pub fn surface(s: [f32; 4], rule: [f32; 4]) -> (f32, f32) {
    if rule[1] > 0.5 {
        (s[1] * (1.0 - s[3]) * rule[0], s[2] * s[3] * rule[2])
    } else {
        (s[1] * rule[0], s[2] * rule[2])
    }
}

// the overcast sky at the camera: the storm overhead, gone from the cloud shell up
pub fn overcast(storm_at_camera: f32, radius_ratio: f32) -> f32 {
    let alt = crate::common::CLOUD_ALT;
    storm_at_camera * (1.0 - smoothstep(alt * 0.9, alt, radius_ratio))
}

// what weather does to a planet type's surfaces
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// AcidRain, Dust and None await the planet types to come (Toxic; Desert; Shattered)
#[cfg_attr(not(test), allow(dead_code))]
pub enum WeatherKind {
    Rain,     // rain, snow toward the poles
    AcidRain, // wet everywhere, no snow
    Snow,
    Ash,
    Dust,
    None, // storm clouds only
}

impl WeatherKind {
    pub fn for_planet(t: PlanetType) -> Self {
        match t {
            PlanetType::EarthLike => WeatherKind::Rain,
            PlanetType::Ice => WeatherKind::Snow,
            PlanetType::Volcanic => WeatherKind::Ash,
        }
    }

    // weather.wgsl weather_surface's rule: wet on, split by snow share, cover on, puddles on
    pub fn rule(self) -> [f32; 4] {
        match self {
            WeatherKind::Rain => [1.0, 1.0, 1.0, 1.0],
            WeatherKind::AcidRain => [1.0, 0.0, 0.0, 1.0],
            WeatherKind::Snow | WeatherKind::Ash | WeatherKind::Dust => [0.0, 0.0, 1.0, 0.0],
            WeatherKind::None => [0.0; 4],
        }
    }

    // the cover's linear colour (rgb) and roughness (w)
    pub fn cover(self) -> [f32; 4] {
        match self {
            WeatherKind::Ash => [0.12, 0.12, 0.12, 0.9],
            WeatherKind::Dust => [0.55, 0.40, 0.22, 0.95],
            _ => [0.9, 0.9, 0.95, 0.6],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            WeatherKind::Rain => "rain",
            WeatherKind::AcidRain => "acid rain",
            WeatherKind::Snow => "snow",
            WeatherKind::Ash => "ash",
            WeatherKind::Dust => "dust",
            WeatherKind::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Force {
    Clear,
    Rain,
    Storm,
    Snow,
}

impl Force {
    fn storm(self) -> f32 {
        match self {
            Force::Clear => 0.0,
            Force::Rain | Force::Snow => 0.8,
            Force::Storm => 1.0,
        }
    }
    fn snow_share(self) -> f32 {
        match self {
            Force::Snow => 1.0,
            Force::Rain => 0.0,
            Force::Clear | Force::Storm => -1.0,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Force::Clear => "clear",
            Force::Rain => "rain",
            Force::Storm => "storm",
            Force::Snow => "snow",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ForceState {
    force: Force,
    dir: Vec3,
    start: f32,
    end: f32, // ACTIVE_END while active; the release time after /weather auto (fades out)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherSettings {
    pub on: bool,
    force: Option<ForceState>,
}

impl Default for WeatherSettings {
    fn default() -> Self {
        Self {
            on: true,
            force: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeatherCommand {
    Show,
    On,
    Off,
    Auto,
    Force(Force),
}

impl WeatherCommand {
    pub fn parse(args: &[&str]) -> Result<WeatherCommand, &'static str> {
        match args {
            [] => Ok(WeatherCommand::Show),
            ["on"] => Ok(WeatherCommand::On),
            ["off"] => Ok(WeatherCommand::Off),
            ["auto"] => Ok(WeatherCommand::Auto),
            ["clear"] => Ok(WeatherCommand::Force(Force::Clear)),
            ["rain"] => Ok(WeatherCommand::Force(Force::Rain)),
            ["storm"] => Ok(WeatherCommand::Force(Force::Storm)),
            ["snow"] => Ok(WeatherCommand::Force(Force::Snow)),
            _ => Err(WEATHER_USAGE),
        }
    }
}

impl WeatherSettings {
    // the bake's WeatherForce (weather.wgsl)
    pub fn force_uniform(&self) -> WeatherForce {
        let on = if self.on { 1.0 } else { 0.0 };
        match self.force {
            Some(f) => WeatherForce {
                dir_value: [f.dir.x, f.dir.y, f.dir.z, f.force.storm()],
                timing: [f.start, f.end, f.force.snow_share(), on],
            },
            // no force: a region that never weighs in (start in the far future is unsafe across the
            // wrap, so the weight is cut by an empty region instead)
            None => WeatherForce {
                dir_value: [0.0, 0.0, 0.0, 0.0],
                timing: [0.0, ACTIVE_END, -1.0, on],
            },
        }
    }

    // a released force is gone once its fade-out is over (`t`: the clock wrapped at 3600); true if
    // it was dropped, so the bakes must start over
    pub fn expire(&mut self, t: f32) -> bool {
        match self.force {
            Some(f) if f.end < ACTIVE_END && age(t, f.end) > FORCE_FADE_SECS => {
                self.force = None;
                true
            }
            _ => false,
        }
    }

    // applies /weather at the player (`dir`: the player's direction from the planet centre, `t`: the
    // clock wrapped at 3600); returns the line to log and whether the bakes must start over
    pub fn apply(
        &mut self,
        cmd: WeatherCommand,
        kind: WeatherKind,
        dir: Vec3,
        t: f32,
    ) -> (String, bool) {
        let dir = dir.try_normalize().unwrap_or(Vec3::Y);
        let changed = match cmd {
            WeatherCommand::Show => false,
            WeatherCommand::On => !std::mem::replace(&mut self.on, true),
            WeatherCommand::Off => std::mem::replace(&mut self.on, false),
            WeatherCommand::Auto => match &mut self.force {
                Some(f) if f.end >= ACTIVE_END => {
                    f.end = t;
                    true
                }
                _ => false,
            },
            WeatherCommand::Force(force) => {
                let rule = kind.rule();
                if force == Force::Rain && rule[0] == 0.0 {
                    return ("No rain on this planet type.".into(), false);
                }
                if force == Force::Snow && rule[2] == 0.0 {
                    return ("No snow on this planet type.".into(), false);
                }
                self.force = Some(ForceState {
                    force,
                    dir,
                    start: t,
                    end: ACTIVE_END,
                });
                self.on = true;
                true
            }
        };
        (self.readout(kind, dir, t), changed)
    }

    fn readout(&self, kind: WeatherKind, dir: Vec3, t: f32) -> String {
        let mode = match self.force {
            _ if !self.on => "off".to_string(),
            Some(f) if f.end >= ACTIVE_END => format!("forced {}", f.force.name()),
            _ => "auto".to_string(),
        };
        let s = sample(dir, t, &self.force_uniform());
        let (wet, cover) = surface(s, kind.rule());
        format!(
            "Weather ({mode}): storm {:.2}, wet {wet:.2}, cover {cover:.2}; this planet: {}",
            s[0],
            kind.name()
        )
    }
}

// which snapshots the weather map holds (time = n * SNAPSHOT_SECS); pure bookkeeping, tested.
// Every snapshot is baked whole in one pass (64² per face is cheap), the next one ahead of the blend.
#[derive(Debug, Default)]
struct Schedule {
    held: [Option<i64>; SLOTS as usize],
}

fn slot(n: i64) -> u32 {
    n.rem_euclid(SLOTS as i64) as u32
}

impl Schedule {
    // the snapshots to bake now and the blend (older slot, newer slot, factor, 0)
    fn update(&mut self, t: f64) -> (Vec<i64>, [f32; 4]) {
        let n0 = (t / SNAPSHOT_SECS).floor() as i64;
        let mut work = Vec::new();
        for n in [n0, n0 + 1, n0 + 2] {
            if self.held[slot(n) as usize] != Some(n) {
                self.held[slot(n) as usize] = Some(n);
                work.push(n);
            }
        }
        let blend = (t / SNAPSHOT_SECS - n0 as f64) as f32;
        (work, [slot(n0) as f32, slot(n0 + 1) as f32, blend, 0.0])
    }

    // the force or on/off changed: every snapshot is outdated
    fn reset(&mut self) {
        self.held = [None; SLOTS as usize];
    }
}

const MAP_SIZE: u32 = 64; // weather_bake.wgsl WEATHER_MAP_SIZE
const FACES: u32 = 6;
const MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// must match WeatherBakeParams in weather_bake.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WeatherBakeParams {
    times: [f32; 4],
    force: WeatherForce,
}

pub struct WeatherMap {
    schedule: Schedule,
    times: [f32; 4],
    sample_view: wgpu::TextureView,
    params_buf: wgpu::Buffer, // WeatherMapParams (weather_map.wgsl): the blend
    bake_buf: wgpu::Buffer,
    bake_bind: wgpu::BindGroup,
    bake_pipeline: wgpu::RenderPipeline,
    targets: Vec<wgpu::TextureView>, // one per layer (slot * 6 + face)
}

impl WeatherMap {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Weather Map"),
            size: wgpu::Extent3d {
                width: MAP_SIZE,
                height: MAP_SIZE,
                depth_or_array_layers: SLOTS * FACES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MAP_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Weather Map (cube array)"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weather Map Params"),
            contents: bytemuck::cast_slice(&[[0.0f32; 4]]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bake_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weather Bake Params"),
            contents: bytemuck::cast_slice(&[WeatherBakeParams::zeroed()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Weather Bake Shader"),
            source: wgpu::ShaderSource::Wgsl(WEATHER_BAKE_SHADER.into()),
        });
        let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("weather_bake_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bake_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Weather Bake"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bake_layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_weather_face"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_weather_bake"),
                compilation_options: Default::default(),
                targets: &[Some(MAP_FORMAT.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let bake_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("weather_bake_bind"),
            layout: &bake_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bake_buf.as_entire_binding(),
            }],
        });
        let targets = (0..SLOTS * FACES)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Self {
            schedule: Schedule::default(),
            times: [0.0; 4],
            sample_view,
            params_buf,
            bake_buf,
            bake_bind,
            bake_pipeline,
            targets,
        }
    }

    // scene group 0 and galaxy camera group 0 (weather_map.wgsl); sampled with the cloud map's sampler
    pub fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
        let fragment = wgpu::ShaderStages::FRAGMENT;
        [
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::CubeArray,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 14,
                visibility: fragment,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }

    pub fn bind_entries(&self) -> [wgpu::BindGroupEntry<'_>; 2] {
        [
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(&self.sample_view),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: self.params_buf.as_entire_binding(),
            },
        ]
    }

    pub fn reset(&mut self) {
        self.schedule.reset();
    }

    // once per frame before anything reads the map: bakes the missing snapshots (own submission) and
    // sets the blend; `t`: the clock wrapped at 3600
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        t: f64,
        force: WeatherForce,
    ) {
        let (work, params) = self.schedule.update(t);
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[params]));
        if work.is_empty() {
            return;
        }
        for &n in &work {
            self.times[slot(n) as usize] = (n as f64 * SNAPSHOT_SECS) as f32;
        }
        queue.write_buffer(
            &self.bake_buf,
            0,
            bytemuck::cast_slice(&[WeatherBakeParams {
                times: self.times,
                force,
            }]),
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Weather Map Bake"),
        });
        for n in work {
            for face in 0..FACES {
                let layer = slot(n) * FACES + face;
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Weather Map"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        depth_slice: None,
                        view: &self.targets[layer as usize],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.bake_pipeline);
                pass.set_bind_group(0, &self.bake_bind, &[]);
                pass.draw(0..3, layer..layer + 1);
            }
        }
        queue.submit(std::iter::once(enc.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biome::PlanetType;
    use glam::Vec3;

    fn auto() -> WeatherForce {
        WeatherSettings::default().force_uniform()
    }

    fn dirs() -> Vec<Vec3> {
        // a Fibonacci sphere: even coverage
        let n = 400;
        (0..n)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                let r = (1.0 - y * y).sqrt();
                let a = i as f32 * 2.399_963;
                Vec3::new(r * a.cos(), y, r * a.sin())
            })
            .collect()
    }

    #[test]
    fn weather_is_periodic_in_the_clock() {
        let f = auto();
        for d in dirs().into_iter().step_by(37) {
            for t in [0.0, 100.0, 1234.5] {
                let a = sample(d, t, &f);
                let b = sample(d, t + 3600.0, &f);
                for k in 0..4 {
                    assert!((a[k] - b[k]).abs() < 2e-3, "{d} {t}: {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn about_a_third_of_the_planet_is_stormy() {
        let f = auto();
        let mut stormy = 0;
        let mut total = 0;
        for t in (0..3600).step_by(300) {
            for d in dirs() {
                total += 1;
                if storm(d, t as f32, &f) > 0.5 {
                    stormy += 1;
                }
            }
        }
        let share = stormy as f32 / total as f32;
        assert!((0.2..0.4).contains(&share), "{share}");
    }

    // a front passes a point in ~2-5 min: storms at a point last neither seconds nor most of the hour
    #[test]
    fn storms_last_minutes() {
        let f = auto();
        let mut runs = Vec::new();
        for d in dirs().into_iter().step_by(7) {
            let mut run = 0;
            for t in 0..3600 {
                if storm(d, t as f32, &f) > 0.5 {
                    run += 1;
                } else if run > 0 {
                    runs.push(run);
                    run = 0;
                }
            }
        }
        runs.sort();
        let median = runs[runs.len() / 2];
        assert!((60..=400).contains(&median), "median storm {median} s");
    }

    // a storm forced until t = 300 (fading out by 304) where the natural weather is calm afterwards:
    // still wet 20 s later, dry ~1.5 min after, the cover still there
    #[test]
    fn wetness_lags_behind_the_rain_and_cover_lasts_longer() {
        let natural = auto();
        let calm = dirs()
            .into_iter()
            .find(|&d| {
                (300..=460)
                    .step_by(5)
                    .all(|t| storm(d, t as f32, &natural) < 0.05)
            })
            .expect("a calm direction");
        let f = WeatherForce {
            dir_value: [calm.x, calm.y, calm.z, 1.0],
            timing: [0.0, 300.0, -1.0, 1.0],
        };
        let after = sample(calm, 320.0, &f);
        assert!(after[1] > 0.3, "still wet 20 s after: {after:?}");
        let later = sample(calm, 400.0, &f);
        assert!(later[1] < 0.05, "dry after ~1.5 min: {later:?}");
        assert!(later[2] > 0.1, "cover outlasts wetness: {later:?}");
        // the histories never fall below the current precipitation
        for d in dirs().into_iter().step_by(11) {
            let s = sample(d, 777.0, &natural);
            let r0 = smoothstep(RAIN_LO, RAIN_HI, s[0]);
            assert!(s[1] >= r0 - 1e-6 && s[2] >= r0 - 1e-6);
        }
    }

    #[test]
    fn weather_is_deterministic() {
        let f = auto();
        let d = Vec3::new(0.3, 0.5, -0.81).normalize();
        assert_eq!(sample(d, 512.25, &f), sample(d, 512.25, &f));
    }

    // Review Focus 2
    #[test]
    fn off_means_no_storm_anywhere() {
        let mut s = WeatherSettings::default();
        s.apply(WeatherCommand::Off, WeatherKind::Rain, Vec3::Y, 10.0);
        let f = s.force_uniform();
        for d in dirs() {
            for t in [0.0, 900.0, 2700.0] {
                assert_eq!(&sample(d, t, &f)[..3], &[0.0, 0.0, 0.0]);
            }
        }
    }

    // Review Focus 3
    #[test]
    fn surface_rules_follow_the_planet_type() {
        let soaked = [1.0, 1.0, 1.0, 0.0]; // rain everywhere, no snow share
        let polar = [1.0, 1.0, 1.0, 1.0]; // all snow share
        for t in PlanetType::ALL {
            let k = WeatherKind::for_planet(t);
            let (wet, cover) = surface(soaked, k.rule());
            let (pwet, pcover) = surface(polar, k.rule());
            match k {
                WeatherKind::Rain => {
                    assert_eq!((wet, cover), (1.0, 0.0));
                    assert_eq!((pwet, pcover), (0.0, 1.0));
                }
                WeatherKind::AcidRain => {
                    assert_eq!((wet, cover, pwet, pcover), (1.0, 0.0, 1.0, 0.0))
                }
                WeatherKind::Snow | WeatherKind::Ash | WeatherKind::Dust => {
                    assert_eq!((wet, pwet), (0.0, 0.0), "{t:?} never wet");
                    assert_eq!((cover, pcover), (1.0, 1.0));
                }
                WeatherKind::None => assert_eq!((wet, cover, pwet, pcover), (0.0, 0.0, 0.0, 0.0)),
            }
        }
        assert_eq!(WeatherKind::for_planet(PlanetType::Ice), WeatherKind::Snow);
        assert_eq!(
            WeatherKind::for_planet(PlanetType::Volcanic),
            WeatherKind::Ash
        );
        assert_eq!(
            WeatherKind::for_planet(PlanetType::EarthLike),
            WeatherKind::Rain
        );
    }

    #[test]
    fn forced_weather_fades_in_around_the_player() {
        let mut s = WeatherSettings::default();
        let here = Vec3::new(0.0, 0.0, 1.0);
        let (_, changed) = s.apply(
            WeatherCommand::Force(Force::Storm),
            WeatherKind::Rain,
            here,
            100.0,
        );
        assert!(changed);
        let f = s.force_uniform();
        assert!(storm(here, 104.5, &f) > 0.99);
        // half-way through the fade it is partly there (unless the natural storm is already strong)
        let w = weather_force_weight(here, 102.0, &f);
        assert!((w - 0.5).abs() < 1e-4);
        // the far side of the planet is untouched
        assert_eq!(weather_force_weight(-here, 104.5, &f), 0.0);
        // auto fades it out again
        s.apply(WeatherCommand::Auto, WeatherKind::Rain, here, 200.0);
        let f = s.force_uniform();
        assert_eq!(weather_force_weight(here, 204.5, &f), 0.0);
        assert!(weather_force_weight(here, 201.0, &f) > 0.7);
    }

    #[test]
    fn released_forces_expire_active_ones_do_not() {
        let mut s = WeatherSettings::default();
        s.apply(
            WeatherCommand::Force(Force::Storm),
            WeatherKind::Rain,
            Vec3::Z,
            90.0,
        );
        assert!(!s.expire(3000.0));
        assert!(s.force.is_some());
        s.apply(WeatherCommand::Auto, WeatherKind::Rain, Vec3::Z, 100.0);
        assert!(!s.expire(103.0) && s.force.is_some());
        assert!(s.expire(105.0));
        assert!(s.force.is_none());
        assert!(!s.expire(106.0));
        // released just before the wrap: expires after it
        s.apply(
            WeatherCommand::Force(Force::Storm),
            WeatherKind::Rain,
            Vec3::Z,
            3590.0,
        );
        s.apply(WeatherCommand::Auto, WeatherKind::Rain, Vec3::Z, 3598.0);
        assert!(!s.expire(1.0) && s.force.is_some());
        assert!(s.expire(3.0));
    }

    #[test]
    fn forced_snow_and_rain_set_the_snow_share() {
        let here = Vec3::new(1.0, 0.0, 0.0); // the equator: natural share 0
        let mut s = WeatherSettings::default();
        s.apply(
            WeatherCommand::Force(Force::Snow),
            WeatherKind::Rain,
            here,
            10.0,
        );
        assert!(sample(here, 20.0, &s.force_uniform())[3] > 0.99);
        let pole = Vec3::Y;
        s.apply(
            WeatherCommand::Force(Force::Rain),
            WeatherKind::Rain,
            pole,
            30.0,
        );
        assert!(sample(pole, 40.0, &s.force_uniform())[3] < 0.01);
    }

    // Review Focus 1
    #[test]
    fn force_survives_the_clock_wrap() {
        let mut s = WeatherSettings::default();
        let here = Vec3::Z;
        s.apply(
            WeatherCommand::Force(Force::Storm),
            WeatherKind::Rain,
            here,
            3598.0,
        );
        let f = s.force_uniform();
        assert!((weather_force_weight(here, 3.0, &f) - 1.0).abs() < 1e-5); // 5 s after, past the wrap
    }

    #[test]
    fn refusals_and_readout() {
        let mut s = WeatherSettings::default();
        let (msg, changed) = s.apply(
            WeatherCommand::Force(Force::Rain),
            WeatherKind::Snow,
            Vec3::Z,
            0.0,
        );
        assert!(!changed && msg.contains("No rain"), "{msg}");
        let (msg, changed) = s.apply(
            WeatherCommand::Force(Force::Snow),
            WeatherKind::AcidRain,
            Vec3::Z,
            0.0,
        );
        assert!(!changed && msg.contains("No snow"), "{msg}");
        let (msg, changed) = s.apply(
            WeatherCommand::Force(Force::Snow),
            WeatherKind::Ash,
            Vec3::Z,
            0.0,
        );
        assert!(changed && msg.contains("ash"), "{msg}");
        let (msg, changed) = s.apply(WeatherCommand::Show, WeatherKind::Ash, Vec3::Z, 1.0);
        assert!(
            !changed && msg.starts_with("Weather (forced snow)"),
            "{msg}"
        );
    }

    // Review Focus 5
    #[test]
    fn readout_handles_a_zero_direction() {
        let mut s = WeatherSettings::default();
        let (msg, _) = s.apply(WeatherCommand::Show, WeatherKind::Rain, Vec3::ZERO, 1.0);
        assert!(!msg.contains("NaN"), "{msg}");
        let (_, changed) = s.apply(
            WeatherCommand::Force(Force::Rain),
            WeatherKind::Rain,
            Vec3::ZERO,
            1.0,
        );
        assert!(changed);
        assert!(s.force_uniform().dir_value.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn commands_parse() {
        assert_eq!(WeatherCommand::parse(&[]), Ok(WeatherCommand::Show));
        assert_eq!(WeatherCommand::parse(&["off"]), Ok(WeatherCommand::Off));
        assert_eq!(WeatherCommand::parse(&["on"]), Ok(WeatherCommand::On));
        assert_eq!(WeatherCommand::parse(&["auto"]), Ok(WeatherCommand::Auto));
        assert_eq!(
            WeatherCommand::parse(&["storm"]),
            Ok(WeatherCommand::Force(Force::Storm))
        );
        assert_eq!(WeatherCommand::parse(&["hail"]), Err(WEATHER_USAGE));
        assert_eq!(WeatherCommand::parse(&["rain", "now"]), Err(WEATHER_USAGE));
    }

    #[test]
    fn overcast_only_below_the_clouds() {
        assert_eq!(overcast(1.0, 1.0), 1.0);
        assert_eq!(overcast(0.5, 1.1), 0.5);
        assert_eq!(overcast(1.0, crate::common::CLOUD_ALT), 0.0);
        assert!(overcast(1.0, crate::common::CLOUD_ALT * 0.95) < 1.0);
    }

    #[test]
    fn schedule_keeps_the_blended_pair_and_bakes_ahead() {
        let mut s = Schedule::default();
        let (work, p) = s.update(12.0); // snapshots 2 and 3 blended, 4 ahead
        assert_eq!(work, vec![2, 3, 4]);
        assert_eq!(p[0], slot(2) as f32);
        assert_eq!(p[1], slot(3) as f32);
        assert!((p[2] - 0.4).abs() < 1e-5);
        assert!(s.update(13.0).0.is_empty());
        assert_eq!(s.update(15.1).0, vec![5]); // 3, 4 held; 5 baked ahead
        s.reset();
        assert_eq!(s.update(15.2).0, vec![3, 4, 5]);
        // the wrap jumps back: everything again
        assert_eq!(s.update(0.5).0, vec![0, 1, 2]);
    }

    #[test]
    fn wgsl_constants_match_the_rust_mirror() {
        let w = include_str!("weather.wgsl");
        for (name, v) in CONSTANTS {
            let line = w
                .lines()
                .find(|l| l.starts_with(&format!("const {name}: f32 = ")))
                .unwrap_or_else(|| panic!("{name} missing"));
            let value: f32 = line
                .split('=')
                .nth(1)
                .unwrap()
                .split(';')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            assert_eq!(value, v, "{name}");
        }
    }

    #[test]
    fn weather_shader_is_valid_wgsl() {
        crate::renderer::tests::assert_valid_wgsl(WEATHER_FN_SHADER);
    }
}
