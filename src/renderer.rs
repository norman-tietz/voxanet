// engine renderer

use crate::cmd::Console;
use crate::common::*;
use crate::controller::Controller;
use crate::deferred::Deferred;
use crate::entity::Player;
use crate::gen::{CoordSystem, MeshGen};
use crate::gpu_timer::{self, GpuTimer};
use crate::hw_rt::HwRt;
use crate::lod_animation::{AnyKey, LodAnimator};
use crate::rt_blur::RtBlur;
use crate::rt_shadow::{RtParams, ShadowWindow};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use glyphon::{
    Attrs, Buffer, Cache, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea,
    TextAtlas, TextBounds, TextRenderer as GlyphRenderer, Viewport,
};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::PresentMode;
use winit::window::Window;

// the scene shader: the shared atmosphere maths (atmosphere.wgsl, also used by the galaxy impostors)
// followed by the voxel engine's own shader
// the lens flare (flare.wgsl), with the atmosphere (atmo_cloud_shadow) and star code it uses
pub(crate) const FLARE_SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    include_str!("star.wgsl"),
    "\n",
    include_str!("flare.wgsl")
);

pub(crate) const SCENE_SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    include_str!("star.wgsl"),
    "\n",
    include_str!("shader.wgsl")
);

// --- UNIFORMS ---

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GlobalUniform {
    pub view_proj: [f32; 16],
    // camera ray through screen pixel (x, y) = col0 + x/width * col1 + y/height * col2 (unnormalised, from
    // the camera); computed in f64, since an f32 inverse of the projection (near 0.1, far 20000) is too
    // imprecise to reconstruct distant surfaces from their camera distance (deferred lighting)
    pub ray_dirs: [f32; 16],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub screen: [f32; 4], // width, height in pixels, sea surface radius, time in seconds (water animation)
    pub motion: [f32; 4], // x: radial motion blur strength 0..1 (fs_light), from the player's current speed
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LocalUniform {
    pub model: [f32; 16],
    pub params: [f32; 4], // x = opacity
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BiomeUniform {
    pub liquid_shallow: [f32; 4], // w: 0.0 = reflective, 1.0 = glowing
    pub liquid_deep: [f32; 4],    // unused (all zero) when the planet type has no liquid
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub cloud_light: [f32; 4],
    pub cloud_dark: [f32; 4],
    pub space_color: [f32; 4],
    pub sun: [f32; 4], // rgb: the star type's sunlight (Atmosphere.sun_color); w: star angular radius
    // the star type's look (star.wgsl StarLook) for the sky's sun disc (atmo_sun_disc)
    pub star_surface: [f32; 4],
    pub star_limb: [f32; 4],
    pub star_corona: [f32; 4], // rgb, w: corona_size
    pub star_params: [f32; 4], // granulation, granule_scale, sunspots, unused
    pub star_frame: [f32; 4], // quaternion planet frame -> galaxy space, so the sky disc shows the star's own surface
}

impl BiomeUniform {
    // `angular_radius`: the star's apparent radius from the installed planet (GalaxyPlanet::star_angular_radius)
    pub fn from_def(
        def: &crate::biome::PlanetTypeDef,
        star: &crate::galaxy::Star,
        angular_radius: f32,
        star_frame: glam::Quat,
    ) -> Self {
        let look = star.star_type.def();
        let (shallow, deep, behavior) = match def.liquid {
            Some(l) => (
                l.shallow_color,
                l.deep_color,
                if matches!(l.behavior, crate::biome::LiquidBehavior::Glowing) {
                    1.0
                } else {
                    0.0
                },
            ),
            None => ([0.0; 3], [0.0; 3], 0.0),
        };
        let v4 = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
        Self {
            liquid_shallow: v4(shallow, behavior),
            liquid_deep: v4(deep, 0.0),
            sky_zenith: v4(def.atmosphere.sky_zenith, 0.0),
            sky_horizon: v4(def.atmosphere.sky_horizon_warm, 0.0),
            cloud_light: v4(def.atmosphere.cloud_light, 0.0),
            cloud_dark: v4(def.atmosphere.cloud_dark, 0.0),
            space_color: v4(def.atmosphere.space_color, 0.0),
            sun: v4(look.sunlight, angular_radius),
            star_surface: v4(look.surface_color, 0.0),
            star_limb: v4(look.limb_color, 0.0),
            star_corona: v4(look.corona_color, look.corona_size),
            star_params: [look.granulation, look.granule_scale, look.sunspots, 0.0],
            star_frame: star_frame.to_array(),
        }
    }
}

// a voxel chunk's meshes from a worker thread: key, terrain vertices/indices, water vertices/indices
// mesh worker results, tagged with the reload generation they were started in (Renderer::generation)
type ChunkGeometry = (u64, ChunkKey, Vec<Vertex>, Vec<u32>, Vec<Vertex>, Vec<u32>);
type LodGeometry = (u64, LodKey, Vec<Vertex>, Vec<u32>);

// --- RENDERER STRUCT ---

pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    // --- TEXT ENGINE ---
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_viewport: Viewport,
    text_renderer: GlyphRenderer,

    // --- UI ---
    pipeline_ui: wgpu::RenderPipeline,
    console_v_buf: wgpu::Buffer,
    console_i_buf: wgpu::Buffer,
    console_inds: u32,

    // --- CORE ---
    animator: LodAnimator,
    local_layout: wgpu::BindGroupLayout,

    pipeline_fill: wgpu::RenderPipeline,
    pipeline_line: wgpu::RenderPipeline,

    chunks: HashMap<ChunkKey, ChunkMesh>,
    lod_chunks: HashMap<LodKey, ChunkMesh>,

    // --- UNIFORMS ---
    global_buf: wgpu::Buffer,
    global_bind: wgpu::BindGroup,
    biome_buf: wgpu::Buffer,

    local_bind_identity: wgpu::BindGroup,

    local_buf_player: wgpu::Buffer,
    local_bind_player: wgpu::BindGroup,

    deferred: Deferred, // full-resolution G-buffer, geometry and lighting pipelines
    galaxy: crate::galaxy_render::GalaxyRenderer,
    global_bind_identity: wgpu::BindGroup, // for UI: identity camera, no ray-marched shadows

    // --- MESHES ---
    player_v_buf: wgpu::Buffer,
    player_i_buf: wgpu::Buffer,
    player_inds: u32,

    cross_v_buf: wgpu::Buffer,
    cross_i_buf: wgpu::Buffer,
    cross_inds: u32,

    cursor_v_buf: wgpu::Buffer,
    cursor_i_buf: wgpu::Buffer,
    cursor_inds: u32,

    collision_v_buf: wgpu::Buffer,
    collision_i_buf: wgpu::Buffer,
    collision_inds: u32,
    frozen_frustum: Option<crate::common::Frustum>,

    // --- THREADING ---
    load_queue: Vec<ChunkKey>,
    player_chunk_pos: Option<ChunkKey>,

    mesh_tx: Sender<ChunkGeometry>,
    mesh_rx: Receiver<ChunkGeometry>,
    pending_chunks: HashSet<ChunkKey>,

    lod_tx: Sender<LodGeometry>,
    lod_rx: Receiver<LodGeometry>,
    // bumped by force_reload_all: workers still meshing the old planet (e.g. before a resize) send results
    // that must be dropped, or a mesh of the old resolution would be kept for good (a hole into the planet)
    generation: u64,
    pending_lods: HashSet<LodKey>,
    view_missing: usize, // required meshes not loaded yet, as of the last update_view

    // --- FPS ---
    last_fps_time: std::time::Instant,
    frame_count: u32,
    current_fps: u32,
    output_p3: bool, // surface tagged Display P3, fs_main converts to P3 primaries

    // --- RAY-MARCHED SHADOWS ---
    rt_params_buf: wgpu::Buffer,
    rt_bits_buf: wgpu::Buffer,
    rt_center: Option<BlockId>,
    rt_dirty: bool,
    rt_blur: RtBlur,
    gpu_timer: Option<GpuTimer>,        // None without timestamp queries
    hw_rt: Option<HwRt>,                // hardware ray-traced shadows, None without ray queries
    pub hw_shadows: bool,               // use hw_rt instead of the ray march (console: /hw_shadows)
    screenshot_request: Option<String>, // set by /screenshot, consumed at the end of the next render()
    flash_pipeline: wgpu::RenderPipeline,
    flare_pipeline: wgpu::RenderPipeline, // lens flares (flare.wgsl), drawn before the HUD text
    flare_layout: wgpu::BindGroupLayout,
    flare_buf: wgpu::Buffer,
    flare_bind: wgpu::BindGroup, // rebuilt on resize (it binds the G-buffer distance and depth)
    flash_buf: wgpu::Buffer,
    flash_bind: wgpu::BindGroup,
    heat_buf: wgpu::Buffer, // the star heat glow (update_heat_glow), drawn with flash_pipeline
    heat_bind: wgpu::BindGroup,
    status: Option<(String, std::time::Instant)>, // timed HUD status line, see show_status
    screenshot_flash: Option<std::time::Instant>, // set right after a capture, so the flash itself is never in the PNG
}

// the star heat glow (galaxy mode): opacity at full heat (the view still shows through) and colour
const HEAT_GLOW_OPACITY: f32 = 0.85;
const HEAT_GLOW_COLOR: [f32; 3] = [1.0, 0.62, 0.28];
// the HUD warns from this heat on
const HEAT_WARNING: f32 = 0.35;

// timed HUD status line (Renderer::show_status): fully visible, then fading out over the last
// STATUS_FADE_SECONDS
const STATUS_SECONDS: f32 = 2.5;
const STATUS_FADE_SECONDS: f32 = 0.5;

fn status_alpha(age_secs: f32) -> f32 {
    ((STATUS_SECONDS - age_secs) / STATUS_FADE_SECONDS).clamp(0.0, 1.0)
}

// the debug console drops down over this fraction of the screen height when fully open
const CONSOLE_SCREEN_FRACTION: f32 = 0.25;

// how far the console reaches down at `height_fraction` (0..1, its slide animation): in pixels from
// the top for the text layout, and as the NDC y of the background panel's bottom edge (+1 = top) —
// both from here, so panel and text can't disagree
fn console_height_px(screen_height: f32, height_fraction: f32) -> f32 {
    screen_height * CONSOLE_SCREEN_FRACTION * height_fraction
}

fn console_bottom_ndc(height_fraction: f32) -> f32 {
    1.0 - 2.0 * CONSOLE_SCREEN_FRACTION * height_fraction
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .unwrap();

        // log GPU info
        crate::system_diagnostics::SystemDiagnostics::log_gpu(&adapter.get_info());

        let target_buffer_size: u64 = 8 * 1024 * 1024 * 1024;
        let mut limits = adapter.limits();
        // we are requiring a maximum of 8gb but we take as much as the platform is capable of
        limits.max_buffer_size = target_buffer_size.min(limits.max_buffer_size);

        let mut features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            features |= wgpu::Features::POLYGON_MODE_LINE;
        }
        features |= GpuTimer::required_features(&adapter);
        features |= HwRt::required_features(&adapter);
        // ray queries are an experimental wgpu feature and need the explicit opt-in
        let experimental_features = if features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                experimental_features,
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .unwrap();

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        config.usage |= wgpu::TextureUsages::COPY_SRC; // lets /screenshot read back the swapchain texture
                                                       // macOS only colour-manages the window when its layer has a colour space; for 8-bit surfaces wgpu
                                                       // offers sRGB (which leaves the layer untagged) or Display P3, so tag it P3 and convert the
                                                       // output to P3 primaries in fs_main. Otherwise sRGB colours show oversaturated on P3 displays.
        let output_p3 = surface
            .get_capabilities(&adapter)
            .format_capabilities
            .iter()
            .any(|c| {
                c.format == config.format
                    && c.color_spaces
                        .contains(wgpu::SurfaceColorSpaces::DISPLAY_P3)
            });
        if output_p3 {
            config.color_space = wgpu::SurfaceColorSpace::DisplayP3;
        }
        let p3_flag = if output_p3 { 1.0 } else { 0.0 }; // sun_dir.w, read by fs_main

        let available_present_modes = surface.get_capabilities(&adapter).present_modes;

        config.present_mode = [
            // presentation preference order.
            PresentMode::Immediate,
            PresentMode::Mailbox,
        ]
        .into_iter()
        .find(|&mode| available_present_modes.contains(&mode))
        .unwrap_or(PresentMode::Fifo);

        surface.configure(&device, &config);

        let font_system = FontSystem::new();

        let swash_cache = SwashCache::new();
        let glyph_cache = Cache::new(&device);
        let text_viewport = Viewport::new(&device, &glyph_cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyph_cache, config.format);
        let text_renderer = GlyphRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        let global_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: ray-marched shadow window params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: ray-marched shadow window solid/air bits
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: biome (liquid colors/behavior, atmosphere colors) — written once per frame
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("global_layout"),
        });

        let local_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("local_layout"),
        });

        // --- BUFFERS ---
        let global_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Uniform"),
            size: std::mem::size_of::<GlobalUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ray-marched shadows; the UI bind group gets rt_off_buf so it doesn't read the shadow term
        let rt_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RT Shadow Params"),
            size: std::mem::size_of::<RtParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rt_off_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Shadow Params (off)"),
            contents: bytemuck::cast_slice(&[RtParams::zeroed()]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let rt_bits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RT Shadow Bits"),
            size: ShadowWindow::max_words() * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let biome_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Biome Uniform"),
            size: std::mem::size_of::<BiomeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &global_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rt_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rt_bits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: biome_buf.as_entire_binding(),
                },
            ],
            label: None,
        });

        let identity_mat = glam::Mat4::IDENTITY;
        let default_local = LocalUniform {
            model: identity_mat.to_cols_array(),
            params: [1.0, 0.0, 1.0, 0.0],
        };

        // console buffers
        let console_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Console V"),
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let console_i_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Console I"),
            size: 1024,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let local_buf_identity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Identity Uniform"),
            contents: bytemuck::cast_slice(&[default_local]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let local_bind_identity = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: local_buf_identity.as_entire_binding(),
            }],
            label: None,
        });

        // player uniform
        let local_buf_player = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Player Uniform"),
            contents: bytemuck::cast_slice(&[default_local]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let local_bind_player = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: local_buf_player.as_entire_binding(),
            }],
            label: None,
        });

        // screenshot flash (F2 / /screenshot feedback): opacity-only, reuses params.x like fading chunks
        let flash_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Flash Uniform"),
            contents: bytemuck::cast_slice(&[default_local]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let flash_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: flash_buf.as_entire_binding(),
            }],
            label: None,
        });

        let heat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Heat Glow Uniform"),
            contents: bytemuck::cast_slice(&[default_local]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let heat_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: heat_buf.as_entire_binding(),
            }],
            label: None,
        });

        // --- PIPELINES ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SCENE_SHADER.into()),
        });
        // group 2: the blurred ray-marched shadow term, read by fs_main
        let rt_sample_layout = RtBlur::sample_layout(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&global_layout),
                Some(&local_layout),
                Some(&rt_sample_layout),
            ],
            immediate_size: 0,
        });
        let rt_blur = RtBlur::new(
            &device,
            rt_sample_layout,
            &global_layout,
            &shader,
            config.width,
            config.height,
        );

        let pipeline_fill = Self::create_pipeline(
            &device,
            &config,
            &layout,
            &shader,
            wgpu::PrimitiveTopology::TriangleList,
            false,
        );
        let pipeline_line = Self::create_pipeline(
            &device,
            &config,
            &layout,
            &shader,
            wgpu::PrimitiveTopology::LineList,
            false,
        );
        let deferred = Deferred::new(
            &device,
            &layout,
            &global_layout,
            &local_layout,
            &rt_blur.sample_layout,
            &shader,
            config.format,
            features.contains(wgpu::Features::POLYGON_MODE_LINE),
            config.width,
            config.height,
        );
        let galaxy = crate::galaxy_render::GalaxyRenderer::new(&device, &config);

        // --- UI PIPELINE ---
        let pipeline_ui = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &Vertex::ATTRIBUTES,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- FLASH PIPELINE --- (full-screen triangle, only needs local.params.x for its opacity)
        // lens flare (flare.wgsl): its uniform plus the depth / G-buffer distance it reads for occlusion
        let flare_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flare_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let flare_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flare Uniform"),
            size: std::mem::size_of::<crate::flare::FlareUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let flare_bind = Self::make_flare_bind(&device, &flare_layout, &flare_buf, &deferred);
        let flare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flare.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FLARE_SHADER.into()),
        });
        let flare_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("flare_pipeline_layout"),
                bind_group_layouts: &[Some(&flare_layout)],
                immediate_size: 0,
            });
        let flare_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Flare Pipeline"),
            layout: Some(&flare_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &flare_shader,
                entry_point: Some("vs_flare"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &flare_shader,
                entry_point: Some("fs_flare"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let flash_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[None, Some(&local_layout)],
            immediate_size: 0,
        });
        let flash_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Flash Pipeline"),
            layout: Some(&flash_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_full"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_flash"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- MESHES ---
        let (pv, pi) = MeshGen::generate_cylinder(0.4, 1.8, 16);
        let player_v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let player_i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pi),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (cv, ci) = MeshGen::generate_crosshair();
        let cross_v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&cv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cross_i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&ci),
            usage: wgpu::BufferUsages::INDEX,
        });

        let cursor_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cursor V"),
            size: 4096,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cursor_i_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cursor I"),
            size: 4096,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let collision_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collision V"),
            size: 65536,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let collision_i_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collision I"),
            size: 65536,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // global identity
        let identity_global_data = GlobalUniform {
            view_proj: identity_mat.to_cols_array(),
            ray_dirs: [0.0; 16],
            cam_pos: [0.0, 0.0, 0.0, 0.0],
            screen: [config.width as f32, config.height as f32, 0.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, p3_flag],
            motion: [0.0, 0.0, 0.0, 0.0],
        };

        let global_buf_identity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Identity Buffer"),
            contents: bytemuck::cast_slice(&[identity_global_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let global_bind_identity = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &global_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buf_identity.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rt_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rt_bits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: biome_buf.as_entire_binding(),
                },
            ],
            label: Some("Identity Bind Group"),
        });

        let gpu_timer = GpuTimer::new(&device, &queue);
        let hw_rt = HwRt::supported(&device).then(|| HwRt::new(&device));
        let hw_shadows = hw_rt.is_some();
        println!(
            "Shadows: {}",
            if hw_shadows {
                "hardware ray tracing"
            } else {
                "ray marching (no hardware ray queries)"
            }
        );
        let (mesh_tx, mesh_rx) = channel();
        let (lod_tx, lod_rx) = channel();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline_fill,
            pipeline_line,
            chunks: HashMap::new(),
            lod_chunks: HashMap::new(),
            global_buf,
            global_bind,
            biome_buf,
            local_bind_identity,
            local_buf_player,
            local_bind_player,
            flash_pipeline,
            flare_pipeline,
            flare_layout,
            flare_buf,
            flare_bind,
            flash_buf,
            flash_bind,
            heat_buf,
            heat_bind,
            screenshot_flash: None,
            status: None,
            deferred,
            galaxy,

            font_system,
            swash_cache,
            text_atlas,
            text_viewport,
            text_renderer,
            collision_v_buf,
            collision_i_buf,
            collision_inds: 0,
            frozen_frustum: None,
            player_v_buf,
            player_i_buf,
            player_inds: pi.len() as u32,
            pipeline_ui,
            console_v_buf,
            console_i_buf,
            console_inds: 0,
            cross_v_buf,
            cross_i_buf,
            cross_inds: ci.len() as u32,
            global_bind_identity,
            cursor_v_buf,
            cursor_i_buf,
            cursor_inds: 0,
            animator: LodAnimator::new(),
            local_layout,
            load_queue: Vec::new(),
            player_chunk_pos: None,
            mesh_tx,
            mesh_rx,
            pending_chunks: HashSet::new(),
            lod_tx,
            lod_rx,
            generation: 0,
            pending_lods: HashSet::new(),
            view_missing: usize::MAX,

            last_fps_time: std::time::Instant::now(),
            frame_count: 0,
            current_fps: 0,
            output_p3,

            rt_params_buf,
            rt_bits_buf,
            rt_center: None,
            rt_dirty: true,
            rt_blur,
            gpu_timer,
            hw_rt,
            hw_shadows,
            screenshot_request: None,
        }
    }

    pub fn request_screenshot(&mut self, path: String) {
        self.screenshot_request = Some(path);
    }

    fn create_pipeline(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        topology: wgpu::PrimitiveTopology,
        wireframe: bool,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &Vertex::ATTRIBUTES,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology,
                cull_mode: None,
                polygon_mode: if wireframe {
                    wgpu::PolygonMode::Line
                } else {
                    wgpu::PolygonMode::Fill
                },
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn ray_dirs(view_proj: glam::Mat4, cam_pos: Vec3) -> [f32; 16] {
        let inv = view_proj.as_dmat4().inverse();
        // inv * (ndc, 1, 1) is linear in ndc in homogeneous coordinates; keeping the direction homogeneous
        // (xyz - cam * w) keeps it linear, so interpolating it across the screen is exact (w > 0)
        let toward = |x: f64, y: f64| {
            let h = inv * glam::DVec4::new(x, y, 1.0, 1.0);
            (h.truncate() - cam_pos.as_dvec3() * h.w) * h.w.signum()
        };
        let top_left = toward(-1.0, 1.0);
        let scale = 1.0 / top_left.length(); // keep the magnitudes near 1
        let top_left = top_left * scale;
        let dx = toward(1.0, 1.0) * scale - top_left;
        let dy = toward(-1.0, -1.0) * scale - top_left;
        let col = |v: glam::DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        [col(top_left), col(dx), col(dy), [0.0; 4]]
            .concat()
            .try_into()
            .unwrap()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.deferred.resize(&self.device, width, height);
        self.rt_blur.resize(&self.device, width, height);
        self.flare_bind = Self::make_flare_bind(
            &self.device,
            &self.flare_layout,
            &self.flare_buf,
            &self.deferred,
        );
    }

    fn make_flare_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buf: &wgpu::Buffer,
        deferred: &Deferred,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flare_bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(deferred.dist_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&deferred.depth),
                },
            ],
        })
    }

    // the lens flare's uniform for a sun at `ndc` (None: behind the camera), with `angular_radius`;
    // `boost` scales it (heat, night, underwater); returns whether there's anything to draw
    #[allow(clippy::too_many_arguments)]
    // the water surface radius over the camera's column (its lake or the sea), 0 when that column holds
    // no water: GlobalUniform.screen.z, for the underwater tint and fs_water's seen-from-below checks
    pub(crate) fn camera_water_radius(planet: &PlanetData, cam_pos: Vec3) -> f32 {
        CoordSystem::pos_to_id(cam_pos, planet.resolution)
            .map_or(0.0, |id| planet.water_surface_radius(id.face, id.u, id.v))
    }

    fn update_flare(
        &self,
        ndc: Option<glam::Vec2>,
        expected_depth: f32,
        angular_radius: f32,
        fov_y: f32,
        boost: f32,
        star: &crate::galaxy::StarTypeDef,
        planet_mode: bool,
        camera: [f32; 4],
        light: [f32; 4],
    ) -> bool {
        let Some(ndc) = ndc else {
            return false;
        };
        let intensity = boost
            * crate::flare::flare_intensity(
                1.0,
                crate::flare::on_screen_factor(ndc, crate::flare::OFF_SCREEN_MARGIN),
                angular_radius,
            );
        if intensity <= 0.001 {
            return false;
        }
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        // taps across half the sun's apparent radius
        let radius_px = crate::flare::tap_radius_px(angular_radius, fov_y, h);
        let v4 = |c: [f32; 3]| [c[0], c[1], c[2], 1.0];
        self.queue.write_buffer(
            &self.flare_buf,
            0,
            bytemuck::cast_slice(&[crate::flare::FlareUniform {
                sun: [ndc.x, ndc.y, expected_depth, intensity],
                tint: v4(star.corona_color),
                tint2: v4(star.surface_color),
                mode: [
                    if planet_mode { 1.0 } else { 0.0 },
                    (radius_px * 0.5).clamp(2.0, 40.0),
                    w,
                    h,
                ],
                camera,
                light,
            }]),
        );
        true
    }

    fn draw_flare(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.flare_pipeline);
        pass.set_bind_group(0, &self.flare_bind, &[]);
        pass.draw(0..6, 0..8);
    }

    pub fn update_console_mesh(&mut self, t: f32) {
        if t <= 0.001 {
            self.console_inds = 0;
            return;
        }

        let bottom_y = console_bottom_ndc(t);

        let color = [0.1, 0.1, 0.15];
        let normal = [0.0, 0.0, 1.0];

        let verts = vec![
            Vertex {
                pos: [-1.0, 1.0, 0.0],
                color,
                normal,
                water: 0.0,
            },
            Vertex {
                pos: [1.0, 1.0, 0.0],
                color,
                normal,
                water: 0.0,
            },
            Vertex {
                pos: [-1.0, bottom_y, 0.0],
                color,
                normal,
                water: 0.0,
            },
            Vertex {
                pos: [1.0, bottom_y, 0.0],
                color,
                normal,
                water: 0.0,
            },
        ];

        let inds = vec![0, 2, 1, 1, 2, 3];

        self.queue
            .write_buffer(&self.console_v_buf, 0, bytemuck::cast_slice(&verts));
        self.queue
            .write_buffer(&self.console_i_buf, 0, bytemuck::cast_slice(&inds));
        self.console_inds = inds.len() as u32;
    }

    pub fn update_view(&mut self, player_pos: Vec3, planet: &PlanetData) {
        let res = planet.resolution;
        let player_id = CoordSystem::pos_to_id(player_pos, res);
        let mut upload_count = 0;
        while let Ok((generation, key, v, i)) = self.lod_rx.try_recv() {
            if generation != self.generation {
                continue;
            } // meshed for a previous planet
            self.pending_lods.remove(&key);
            self.upload_lod_buffer(key, v, i);
            upload_count += 1;
            if upload_count > 20 {
                break;
            }
        }
        let mut required_voxels: HashSet<ChunkKey> = HashSet::new();
        let mut required_lods: HashSet<LodKey> = HashSet::new();
        let logical_size = res.next_power_of_two();

        for face in 0..6 {
            self.process_quadtree(
                face,
                0,
                0,
                logical_size,
                player_pos,
                planet,
                player_id,
                &mut required_voxels,
                &mut required_lods,
            );
        }

        // a mesh that is no longer required (voxel chunk or LOD node) stays until every required mesh
        // overlapping its area has loaded, so splits, merges and voxel <-> LOD hand-overs never leave a hole
        let missing: Vec<(u8, u32, u32, u32)> = required_voxels
            .iter()
            .filter(|k| !self.chunks.contains_key(k))
            .map(|k| {
                (
                    k.face,
                    k.u_idx * CHUNK_SIZE,
                    k.v_idx * CHUNK_SIZE,
                    CHUNK_SIZE,
                )
            })
            .chain(
                required_lods
                    .iter()
                    .filter(|k| !self.lod_chunks.contains_key(k))
                    .map(|k| (k.face, k.x, k.y, k.size)),
            )
            .collect();
        self.view_missing = missing.len();
        let uncovered = |face: u8, x: u32, y: u32, size: u32| {
            missing.iter().any(|&(f, mx, my, ms)| {
                f == face && x < mx + ms && mx < x + size && y < my + ms && my < y + size
            })
        };

        let current_lods: Vec<LodKey> = self.lod_chunks.keys().cloned().collect();
        for k in current_lods {
            if required_lods.contains(&k) || uncovered(k.face, k.x, k.y, k.size) {
                continue;
            }
            if let Some(mesh) = self.lod_chunks.remove(&k) {
                self.animator.retire(AnyKey::Lod(k), mesh);
            }
        }

        let mut spawn_count = 0;
        for key in required_lods {
            if !self.lod_chunks.contains_key(&key) && !self.pending_lods.contains(&key) {
                if spawn_count >= 8 {
                    break;
                }
                self.pending_lods.insert(key);
                let tx = self.lod_tx.clone();
                let generation = self.generation;
                let p = planet.clone();
                std::thread::spawn(move || {
                    let (v, i) = MeshGen::generate_lod_mesh(key, &p);
                    let _ = tx.send((generation, key, v, i));
                });
                spawn_count += 1;
            }
        }

        let current_voxels: Vec<ChunkKey> = self.chunks.keys().cloned().collect();
        for k in current_voxels {
            if !required_voxels.contains(&k)
                && !uncovered(
                    k.face,
                    k.u_idx * CHUNK_SIZE,
                    k.v_idx * CHUNK_SIZE,
                    CHUNK_SIZE,
                )
            {
                if let Some(mesh) = self.chunks.remove(&k) {
                    self.animator.retire(AnyKey::Voxel(k), mesh);
                }
            }
        }

        self.load_queue.retain(|k| required_voxels.contains(k));
        for k in required_voxels {
            if !self.chunks.contains_key(&k) && !self.load_queue.contains(&k) {
                self.load_queue.push(k);
            }
        }

        self.load_queue.sort_by(|a, b| {
            let get_center = |k: &ChunkKey| -> glam::Vec3 {
                // centre of the chunk's actual extent: the last chunk on a face can be partial, and a
                // u/v past the face edge makes cube_to_sphere return NaN
                let res = planet.resolution;
                let (u0, v0) = (k.u_idx * CHUNK_SIZE, k.v_idx * CHUNK_SIZE);
                let u = (u0 + (u0 + CHUNK_SIZE).min(res)) / 2;
                let v = (v0 + (v0 + CHUNK_SIZE).min(res)) / 2;
                CoordSystem::get_vertex_pos(k.face, u, v, res / 2, res)
            };
            let da = get_center(a).distance_squared(player_pos);
            let db = get_center(b).distance_squared(player_pos);
            db.total_cmp(&da)
        });

        self.process_load_queue(player_pos, planet);
    }

    // QUADTREE LOGIC
    fn process_quadtree(
        &self,
        face: u8,
        x: u32,
        y: u32,
        size: u32,
        cam_pos: Vec3,
        planet: &PlanetData,
        player_id: Option<BlockId>,
        voxels: &mut HashSet<ChunkKey>,
        lods: &mut HashSet<LodKey>,
    ) {
        if x >= planet.resolution || y >= planet.resolution {
            return;
        }

        let center_u = (x + size / 2).min(planet.resolution - 1);
        let center_v = (y + size / 2).min(planet.resolution - 1);
        let h = planet.resolution / 2;

        let world_pos = CoordSystem::get_vertex_pos(face, center_u, center_v, h, planet.resolution);

        let mut dist = world_pos.distance(cam_pos);

        if let Some(pid) = player_id {
            if pid.face == face {
                if pid.u >= x && pid.u < x + size && pid.v >= y && pid.v < y + size {
                    dist = 0.0;
                }
            }
        }

        let node_radius_world = (size as f32 * CoordSystem::get_layer_radius(h, planet.resolution))
            / planet.resolution as f32;

        let mut lod_factor = 4.0;
        if size <= CHUNK_SIZE * 8 {
            lod_factor = 5.0;
        }
        if size <= CHUNK_SIZE * 4 {
            lod_factor = 7.0;
        }
        if size <= CHUNK_SIZE * 2 {
            lod_factor = 12.0;
        }
        if size <= CHUNK_SIZE {
            lod_factor = 18.0;
        }

        let split_distance = node_radius_world * lod_factor;
        let is_smallest = size <= CHUNK_SIZE;

        if dist < split_distance && !is_smallest {
            let half = size / 2;
            self.process_quadtree(face, x, y, half, cam_pos, planet, player_id, voxels, lods);
            self.process_quadtree(
                face,
                x + half,
                y,
                half,
                cam_pos,
                planet,
                player_id,
                voxels,
                lods,
            );
            self.process_quadtree(
                face,
                x,
                y + half,
                half,
                cam_pos,
                planet,
                player_id,
                voxels,
                lods,
            );
            self.process_quadtree(
                face,
                x + half,
                y + half,
                half,
                cam_pos,
                planet,
                player_id,
                voxels,
                lods,
            );
        } else {
            if size <= CHUNK_SIZE {
                let key = ChunkKey {
                    face,
                    u_idx: x / CHUNK_SIZE,
                    v_idx: y / CHUNK_SIZE,
                };
                if (key.u_idx * CHUNK_SIZE) < planet.resolution
                    && (key.v_idx * CHUNK_SIZE) < planet.resolution
                {
                    voxels.insert(key);
                }
            } else {
                let key = LodKey { face, x, y, size };
                lods.insert(key);
            }
        }
    }

    fn upload_lod_buffer(&mut self, key: LodKey, v: Vec<Vertex>, i: Vec<u32>) {
        // with hardware ray tracing every chunk mesh also gets a BLAS, built from the same buffers
        let blas_input = if self.hw_rt.is_some() {
            wgpu::BufferUsages::BLAS_INPUT
        } else {
            wgpu::BufferUsages::empty()
        };
        let v_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | blas_input,
            });
        let i_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&i),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | blas_input,
            });
        let blas = self.hw_rt.is_some().then(|| {
            HwRt::build_blas(
                &self.device,
                &self.queue,
                &v_buf,
                v.len() as u32,
                &i_buf,
                i.len() as u32,
            )
        });

        let uniform_data = LocalUniform {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            params: [0.0, 0.0, 0.0, 0.0],
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LOD Uniform"),
                contents: bytemuck::cast_slice(&[uniform_data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.local_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
            label: None,
        });

        // bounds from the actual vertices
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for vert in &v {
            let p = Vec3::from_array(vert.pos);
            min = min.min(p);
            max = max.max(p);
        }
        let real_center = (min + max) * 0.5;
        let real_radius = min.distance(max) * 0.5;

        self.lod_chunks.insert(
            key,
            ChunkMesh {
                v_buf,
                i_buf,
                num_inds: i.len() as u32,
                num_verts: v.len(),
                uniform_buf,
                bind_group,
                center: real_center, // <--- ADDED
                radius: real_radius, // <--- ADDED
                blas,
                water: None,
            },
        );
        self.animator.start_spawn(AnyKey::Lod(key));
    }
    fn process_load_queue(&mut self, _player_pos: Vec3, planet: &PlanetData) {
        let mut upload_budget = 4;
        while let Ok((generation, key, v, i, wv, wi)) = self.mesh_rx.try_recv() {
            if generation != self.generation {
                continue;
            } // meshed for a previous planet
            self.pending_chunks.remove(&key);
            if !v.is_empty() {
                self.upload_chunk_buffers(key, v, i, wv, wi);
                upload_budget -= 1;
            }
            if upload_budget <= 0 {
                break;
            }
        }

        if upload_budget <= 0 {
            return;
        }
        if self.load_queue.is_empty() {
            return;
        }
        if self.pending_chunks.len() >= 12 {
            return;
        }

        let chunks_to_spawn = 4;
        for _ in 0..chunks_to_spawn {
            if let Some(key) = self.load_queue.pop() {
                if self.chunks.contains_key(&key) || self.pending_chunks.contains(&key) {
                    continue;
                }
                self.pending_chunks.insert(key);
                let planet_clone = planet.clone();
                let tx = self.mesh_tx.clone();
                let generation = self.generation;
                std::thread::spawn(move || {
                    let (v, i) = MeshGen::build_chunk(key, &planet_clone);
                    let (wv, wi) = MeshGen::build_water(key, &planet_clone);
                    let _ = tx.send((generation, key, v, i, wv, wi));
                });
            } else {
                break;
            }
        }
    }

    pub fn force_reload_all(&mut self, planet: &PlanetData, player_pos: Vec3) {
        self.generation += 1; // results still in flight belong to the old planet
        self.chunks.clear();
        self.lod_chunks.clear();
        self.load_queue.clear();
        self.pending_chunks.clear();
        self.pending_lods.clear();
        self.player_chunk_pos = None;
        self.rt_dirty = true;
        self.update_view(player_pos, planet);
    }

    // true once every mesh the current view requires is loaded (as of the last update_view): the
    // voxel world can be shown without holes — the landing handover waits for this
    pub fn view_covered(&self) -> bool {
        self.view_missing == 0
    }

    pub fn set_near_impostor(&mut self, planet_index: usize, verts: &[Vertex], indices: &[u32]) {
        self.galaxy
            .set_near_impostor(&self.device, planet_index, verts, indices);
    }

    pub fn refresh_neighbors(&mut self, id: BlockId, planet: &PlanetData) {
        self.rt_dirty = true;
        // the block's own chunk and the chunks of its four neighbours, which may lie on another cube face
        let mut keys = vec![PlanetData::chunk_key(id)];
        for (du, dv) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            if let Some((face, u, v)) = planet.neighbor_column(id.face, id.u, id.v, du, dv) {
                let key = PlanetData::chunk_key(BlockId {
                    face,
                    u,
                    v,
                    layer: 0,
                });
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
        for key in keys {
            if self.chunks.contains_key(&key) {
                let (v, i) = MeshGen::build_chunk(key, planet);
                if v.is_empty() {
                    self.chunks.remove(&key);
                } else {
                    let (wv, wi) = MeshGen::build_water(key, planet);
                    self.upload_chunk_buffers(key, v, i, wv, wi);
                }
            }
        }
    }

    // switch between hardware ray-traced and ray-marched shadows; returns the method now in use
    pub fn set_hw_shadows(&mut self, on: bool) -> bool {
        self.hw_shadows = on && self.hw_rt.is_some();
        if !self.hw_shadows {
            self.rt_dirty = true; // the ray-march window wasn't updated while hardware rays were used
        }
        self.hw_shadows
    }

    fn update_rt_window(&mut self, player_pos: Vec3, planet: &PlanetData) {
        if self.hw_shadows {
            // the ray-march windows aren't needed while hardware rays are used, but fs_main reads the shadow
            // texture only when rt.enabled is set (set_hw_shadows marks the windows dirty for switching back)
            let params = RtParams {
                enabled: 1,
                ..RtParams::zeroed()
            };
            self.queue
                .write_buffer(&self.rt_params_buf, 0, bytemuck::cast_slice(&[params]));
            return;
        }
        let Some(id) = CoordSystem::pos_to_id(player_pos, planet.resolution) else {
            return;
        };
        // small planets are stored completely, so only big ones follow the player
        let whole_planet = planet.resolution <= crate::rt_shadow::WINDOW_SIZE;
        let recenter = match self.rt_center {
            Some(c) => {
                let limit = (crate::rt_shadow::WINDOW_SIZE / 8) as i32;
                !whole_planet
                    && (c.face != id.face
                        || (c.u as i32 - id.u as i32).abs() > limit
                        || (c.v as i32 - id.v as i32).abs() > limit)
            }
            None => true,
        };
        if !recenter && !self.rt_dirty {
            return;
        }

        let start = std::time::Instant::now();
        let window = ShadowWindow::build(planet, id, player_pos.normalize_or_zero());
        self.queue.write_buffer(
            &self.rt_params_buf,
            0,
            bytemuck::cast_slice(&[window.params]),
        );
        self.queue
            .write_buffer(&self.rt_bits_buf, 0, bytemuck::cast_slice(&window.bits));
        let faces: Vec<u32> = (0..6)
            .filter(|&f| window.params.faces[f].size > 0)
            .map(|f| f as u32)
            .collect();
        println!(
            "RT shadow windows rebuilt: faces {:?}, {} KB in {:.2} ms",
            faces,
            window.bits.len() * 4 / 1024,
            start.elapsed().as_secs_f32() * 1000.0
        );
        self.rt_center = Some(id);
        self.rt_dirty = false;
    }

    fn upload_chunk_buffers(
        &mut self,
        key: ChunkKey,
        v: Vec<Vertex>,
        i: Vec<u32>,
        wv: Vec<Vertex>,
        wi: Vec<u32>,
    ) {
        let water = (!wi.is_empty()).then(|| WaterMesh {
            v_buf: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water V"),
                    contents: bytemuck::cast_slice(&wv),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            i_buf: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water I"),
                    contents: bytemuck::cast_slice(&wi),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            num_inds: wi.len() as u32,
        });
        // with hardware ray tracing every chunk mesh also gets a BLAS, built from the same buffers
        let blas_input = if self.hw_rt.is_some() {
            wgpu::BufferUsages::BLAS_INPUT
        } else {
            wgpu::BufferUsages::empty()
        };
        let v_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | blas_input,
            });
        let i_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&i),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | blas_input,
            });
        let blas = self.hw_rt.is_some().then(|| {
            HwRt::build_blas(
                &self.device,
                &self.queue,
                &v_buf,
                v.len() as u32,
                &i_buf,
                i.len() as u32,
            )
        });

        let is_update = self.chunks.contains_key(&key);
        let start_opacity = if is_update { 1.0 } else { 0.0 };

        let uniform_data = LocalUniform {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            params: [start_opacity, 0.0, 0.0, 0.0],
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chunk Uniform"),
                contents: bytemuck::cast_slice(&[uniform_data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.local_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
            label: None,
        });

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        if v.is_empty() {
            min = Vec3::ZERO;
            max = Vec3::ZERO;
        } else {
            for vert in v.iter().chain(&wv) {
                // the water surface can lie far above the sea floor
                let p = Vec3::from_array(vert.pos);
                min = min.min(p);
                max = max.max(p);
            }
        }
        let real_center = (min + max) * 0.5;
        let real_radius = min.distance(max) * 0.5;

        self.chunks.insert(
            key,
            ChunkMesh {
                v_buf,
                i_buf,
                num_inds: i.len() as u32,
                num_verts: v.len(),
                uniform_buf,
                bind_group,
                center: real_center,
                radius: real_radius,
                blas,
                water,
            },
        );

        if !is_update {
            self.animator.start_spawn(AnyKey::Voxel(key));
        }
    }
    pub fn log_memory(&self, planet: &PlanetData) {
        let mut total_v = 0;
        let mut total_i = 0;
        for c in self.chunks.values() {
            total_v += c.num_verts;
            total_i += c.num_inds as usize;
        }
        let bytes = (total_v * 36) + (total_i * 4);
        let mb = bytes as f32 / (1024.0 * 1024.0);
        println!("------------------------------------------");
        println!("RESOLUTION: {}", planet.resolution);
        println!("Active Chunks: {}", self.chunks.len());
        if mb > 1024.0 {
            println!("GPU Memory: {:.2} GB", mb / 1024.0);
        } else {
            println!("GPU Memory: {:.2} MB", mb);
        }
        println!("------------------------------------------");
    }

    pub fn update_cursor(&mut self, planet: &PlanetData, id: Option<BlockId>) {
        if let Some(id) = id {
            let res = planet.resolution;
            let p = |u, v, l| {
                CoordSystem::get_vertex_pos(id.face, id.u + u, id.v + v, id.layer + l, res)
            };

            let corners = [
                p(0, 0, 0),
                p(1, 0, 0),
                p(0, 1, 0),
                p(1, 1, 0),
                p(0, 0, 1),
                p(1, 0, 1),
                p(0, 1, 1),
                p(1, 1, 1),
            ];

            let edges = [
                (0, 1),
                (1, 3),
                (3, 2),
                (2, 0),
                (4, 5),
                (5, 7),
                (7, 6),
                (6, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ];

            let mut verts = Vec::new();
            let mut inds = Vec::new();
            let thickness = 0.025;
            let color = [1.0, 1.0, 0.0];
            let mut idx_base = 0;

            for (start, end) in edges {
                let a = corners[start];
                let b = corners[end];
                let dir = (b - a).normalize();
                let ref_up = if dir.dot(Vec3::Y).abs() > 0.9 {
                    Vec3::X
                } else {
                    Vec3::Y
                };
                let right = dir.cross(ref_up).normalize() * thickness;
                let up = dir.cross(right).normalize() * thickness;
                let offsets = [(-right - up), (right - up), (right + up), (-right + up)];

                for off in offsets {
                    verts.push(Vertex {
                        pos: (a + off).to_array(),
                        color,
                        normal: [0.0; 3],
                        water: 0.0,
                    });
                    verts.push(Vertex {
                        pos: (b + off).to_array(),
                        color,
                        normal: [0.0; 3],
                        water: 0.0,
                    });
                }

                let faces = [(0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 7, 6), (6, 7, 1, 0)];
                for (i0, i1, i2, i3) in faces {
                    inds.push(idx_base + i0);
                    inds.push(idx_base + i1);
                    inds.push(idx_base + i2);
                    inds.push(idx_base + i2);
                    inds.push(idx_base + i3);
                    inds.push(idx_base + i0);
                }
                idx_base += 8;
            }

            self.queue
                .write_buffer(&self.cursor_v_buf, 0, bytemuck::cast_slice(&verts));
            self.queue
                .write_buffer(&self.cursor_i_buf, 0, bytemuck::cast_slice(&inds));
            self.cursor_inds = inds.len() as u32;
        } else {
            self.cursor_inds = 0;
        }
    }

    // shows a short message in the planet-mode HUD for STATUS_SECONDS (fading out at the end) —
    // for game feedback the player must see without opening the console, e.g. F-landing messages
    pub fn show_status(&mut self, text: &str) {
        self.status = Some((text.to_string(), std::time::Instant::now()));
    }

    // shared by render() and render_galaxy() so the counter keeps ticking across a mode switch
    // instead of freezing at whatever it last read in planet mode
    fn update_fps(&mut self) {
        self.frame_count += 1;
        let now = std::time::Instant::now();
        if now.duration_since(self.last_fps_time).as_secs_f32() >= 1.0 {
            self.current_fps = self.frame_count;
            self.frame_count = 0;
            self.last_fps_time = now;
        }
    }

    // Screenshot flash (F2 / /screenshot feedback), shared by render() and render_galaxy(): a quick
    // white fade-out starting the frame *after* a capture (screenshot_flash is only set once the
    // capture is done), so the flash is visible on screen but never in the PNG. Uploads this
    // frame's opacity and returns it; draw it with draw_flash as the last thing in the frame.
    fn update_flash(&mut self, now: std::time::Instant) -> f32 {
        const FLASH_DURATION: f32 = 0.18;
        let mut alpha = 0.0f32;
        if let Some(start) = self.screenshot_flash {
            let t = now.duration_since(start).as_secs_f32();
            if t < FLASH_DURATION {
                alpha = 0.65 * (1.0 - t / FLASH_DURATION);
            } else {
                self.screenshot_flash = None;
            }
        }
        self.queue.write_buffer(
            &self.flash_buf,
            0,
            bytemuck::cast_slice(&[LocalUniform {
                model: glam::Mat4::IDENTITY.to_cols_array(),
                params: [alpha, 1.0, 1.0, 1.0],
            }]),
        );
        alpha
    }

    // the star's heat glow in galaxy mode (galaxy::star_heat): a warm full-screen tint over the view,
    // drawn under the HUD text; uploads this frame's opacity and returns it
    fn update_heat_glow(&mut self, heat: f32) -> f32 {
        let alpha = HEAT_GLOW_OPACITY * heat;
        self.queue.write_buffer(
            &self.heat_buf,
            0,
            bytemuck::cast_slice(&[LocalUniform {
                model: glam::Mat4::IDENTITY.to_cols_array(),
                params: [
                    alpha,
                    HEAT_GLOW_COLOR[0],
                    HEAT_GLOW_COLOR[1],
                    HEAT_GLOW_COLOR[2],
                ],
            }]),
        );
        alpha
    }

    // full-screen triangle over whatever the pass already holds; the pass must target the swapchain
    // format (flash_pipeline's only colour target) and needs no other bind groups
    fn draw_flash(&self, pass: &mut wgpu::RenderPass, alpha: f32) {
        self.draw_full_screen(pass, &self.flash_bind, alpha);
    }

    fn draw_full_screen(&self, pass: &mut wgpu::RenderPass, bind: &wgpu::BindGroup, alpha: f32) {
        if alpha > 0.001 {
            pass.set_pipeline(&self.flash_pipeline);
            pass.set_bind_group(1, bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        controller: &Controller,
        player: &Player,
        planet: &PlanetData,
        console: &Console,
        time: f64,     // shared game clock in seconds (Game::clock)
        sun_dir: Vec3, // toward the sun, in the planet frame; computed by the game (home vs galaxy planet)
        backdrop: &crate::galaxy_render::Backdrop, // the galaxy behind the voxel world, drawn first
    ) {
        self.update_console_mesh(console.height_fraction);

        if controller.show_collisions {
            let (v, i) = MeshGen::generate_collision_debug(player.position, planet);
            self.queue
                .write_buffer(&self.collision_v_buf, 0, bytemuck::cast_slice(&v));
            self.queue
                .write_buffer(&self.collision_i_buf, 0, bytemuck::cast_slice(&i));
            self.collision_inds = i.len() as u32;
        } else {
            self.collision_inds = 0;
        }

        if let Some(timer) = &mut self.gpu_timer {
            timer.poll(&self.device);
        }

        let out = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = out
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // the galaxy behind everything: submitted on its own before this frame's encoder, so it lands
        // in the swapchain first; it also uses Deferred::depth, which the geometry pass below clears
        // again, so the backdrop can never hide terrain. The lighting pass blends the sky over it.
        let screen = (self.config.width as f32, self.config.height as f32);
        self.galaxy.render(
            &self.device,
            &self.queue,
            &view,
            &self.deferred.depth,
            &backdrop.camera,
            backdrop.content,
            backdrop.galaxy,
            backdrop.t,
            screen,
        );

        self.update_rt_window(player.position, planet);

        // the sun direction comes from the game (the real star,
        // GalaxyPlanet::sun_dir_in_planet_frame); everything downstream (shade(), caustics, cloud_shadow, both
        // shadow paths) re-reads it fresh every frame. The water animation clock wraps hourly as before.
        let time = (time % 3600.0) as f32;

        // -- Camera Matrix --
        let mvp =
            controller.get_matrix(player, self.config.width as f32, self.config.height as f32);

        // before cull_frustum below, which may keep self borrowed for the rest of the frame
        let flash_alpha = self.update_flash(std::time::Instant::now());

        // --- FRUSTUM CULLING LOGIC ---
        let current_frustum = crate::common::Frustum::from_matrix(mvp);

        // determine which frustum to use for culling
        // if freeze is on, we use the stored one. if freeze is off, update the stored one (or just use current).
        let cull_frustum = if controller.freeze_culling {
            if self.frozen_frustum.is_none() {
                self.frozen_frustum = Some(crate::common::Frustum::from_matrix(mvp));
            }
            self.frozen_frustum.as_ref().unwrap()
        } else {
            self.frozen_frustum = None;
            &current_frustum
        };

        // debug Stats
        let mut rendered_lods = 0;
        let mut rendered_chunks = 0;

        let cam_pos = controller.get_camera_pos(player);
        let frustum = crate::common::Frustum::from_matrix(mvp);

        // radial motion blur (fs_light): eases in between these speeds so normal walking never
        // blurs, reaching full strength around flying-sprint speed (move_speed * 10)
        const MOTION_BLUR_MIN_SPEED: f32 = 10.0;
        const MOTION_BLUR_MAX_SPEED: f32 = 45.0;
        let speed = player.velocity.length();
        let motion_blur = ((speed - MOTION_BLUR_MIN_SPEED)
            / (MOTION_BLUR_MAX_SPEED - MOTION_BLUR_MIN_SPEED))
            .clamp(0.0, 1.0);

        // update main global uni
        let global_data = GlobalUniform {
            view_proj: mvp.to_cols_array(),
            ray_dirs: Self::ray_dirs(mvp, cam_pos),
            cam_pos: [
                cam_pos.x,
                cam_pos.y,
                cam_pos.z,
                planet.resolution as f32 * 0.5,
            ], // w: planet radius (shader.wgsl clouds/sky; rt.resolution is 0 in hardware shadow mode)
            screen: [
                self.config.width as f32,
                self.config.height as f32,
                // the water surface over the camera's own column (a lake's or the sea's), 0 when
                // that column holds no water: the underwater tint and fs_water's below-the-surface checks
                Self::camera_water_radius(planet, cam_pos),
                time,
            ], // wrapped: keeps f32 wave phases precise
            sun_dir: [
                sun_dir.x,
                sun_dir.y,
                sun_dir.z,
                if self.output_p3 { 1.0 } else { 0.0 },
            ],
            motion: [motion_blur, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.global_buf, 0, bytemuck::cast_slice(&[global_data]));

        // the installed planet is the one the backdrop leaves out (the voxel engine draws it)
        let (angular_radius, star_frame) = match backdrop.content {
            crate::galaxy_render::GalaxyContent::AllButPlanet(i) => {
                let p = &backdrop.galaxy.planets[i];
                (
                    p.star_angular_radius(&backdrop.galaxy.star),
                    p.orientation(backdrop.t),
                )
            }
            crate::galaxy_render::GalaxyContent::Everything => (0.0, glam::Quat::IDENTITY),
        };
        let biome_data = BiomeUniform::from_def(
            &planet.planet_type.def(),
            &backdrop.galaxy.star,
            angular_radius,
            star_frame,
        );
        // lens flare: the sun projected far along its direction; none once the sun is below the planet's
        // geometric horizon (high up that's below the local horizontal, as in galaxy mode) or under
        // water, dimmed by clouds and terrain on the GPU
        let flare = {
            let sun_far = mvp * (cam_pos + sun_dir.normalize() * 1000.0).extend(1.0);
            let ndc = (sun_far.w > 0.0)
                .then(|| glam::Vec2::new(sun_far.x / sun_far.w, sun_far.y / sun_far.w));
            let water_radius = Self::camera_water_radius(planet, cam_pos);
            let horizon_radius =
                CoordSystem::get_layer_radius(planet.terrain.sea_level() + 1, planet.resolution);
            let daylight =
                crate::flare::horizon_visibility(sun_dir, cam_pos, horizon_radius, angular_radius);
            let above_water = if cam_pos.length() < water_radius {
                0.0
            } else {
                1.0
            };
            self.update_flare(
                ndc,
                0.0,
                angular_radius,
                controller.fov_y(),
                daylight * above_water,
                backdrop.galaxy.star.star_type.def(),
                true,
                [
                    cam_pos.x,
                    cam_pos.y,
                    cam_pos.z,
                    planet.resolution as f32 * 0.5,
                ],
                [sun_dir.x, sun_dir.y, sun_dir.z, time],
            )
        };
        self.queue
            .write_buffer(&self.biome_buf, 0, bytemuck::cast_slice(&[biome_data]));

        let model_mat = player.get_model_matrix();
        self.queue.write_buffer(
            &self.local_buf_player,
            0,
            bytemuck::cast_slice(model_mat.as_ref()),
        );

        let now = std::time::Instant::now();

        let dying_status = self.animator.update_dying(now);
        for (key, alpha) in dying_status {
            if let Some(state) = self.animator.dying_chunks.get(&key) {
                let data = LocalUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array(),
                    params: [alpha, 1.0, 0.0, 0.0],
                };
                self.queue
                    .write_buffer(&state.mesh.uniform_buf, 0, bytemuck::cast_slice(&[data]));
            }
        }

        let queue = &self.queue;
        let animator = &mut self.animator;

        let mut update_opacity = |key: AnyKey, mesh: &ChunkMesh| {
            let alpha = animator.get_opacity(key, now);
            if alpha < 1.0 {
                let data = LocalUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array(),
                    params: [alpha, 0.0, 0.0, 0.0],
                };
                queue.write_buffer(&mesh.uniform_buf, 0, bytemuck::cast_slice(&[data]));
            } else if animator.spawning_chunks.contains_key(&key) {
                let data = LocalUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array(),
                    params: [1.0, 0.0, 0.0, 0.0],
                };
                queue.write_buffer(&mesh.uniform_buf, 0, bytemuck::cast_slice(&[data]));
                animator.spawning_chunks.remove(&key);
            }
        };

        for (key, mesh) in &self.lod_chunks {
            update_opacity(AnyKey::Lod(*key), mesh);
        }
        for (key, mesh) in &self.chunks {
            update_opacity(AnyKey::Voxel(*key), mesh);
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // hardware ray tracing: rebuild the TLAS over the current chunk BLASes before pass 2 uses it
        if let Some(hw) = &mut self.hw_rt {
            if self.hw_shadows {
                // all loaded chunks, also those outside the view: they cast shadows into it
                let blases = self
                    .chunks
                    .values()
                    .chain(self.lod_chunks.values())
                    .filter_map(|m| m.blas.as_ref());
                hw.update(&self.device, &mut enc, blases);
            }
        }
        // --- PASS 1: GEOMETRY (DEFERRED G-BUFFER, FULL RESOLUTION) ---
        // all scene geometry is drawn once, here; lighting runs per pixel in pass 4
        {
            let geometry_targets = self.deferred.geometry_targets();
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Geometry Pass"),
                color_attachments: &geometry_targets,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.deferred.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self
                    .gpu_timer
                    .as_ref()
                    .and_then(|t| t.writes(gpu_timer::GEOMETRY, true, true)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(if controller.is_wireframe {
                &self.deferred.geom_wire
            } else {
                &self.deferred.geom_fill
            });
            pass.set_bind_group(0, &self.global_bind, &[]);
            pass.set_bind_group(2, &self.rt_blur.sample_bind, &[]);

            for mesh in self.lod_chunks.values() {
                if cull_frustum.intersects_sphere(mesh.center, mesh.radius) {
                    rendered_lods += 1; // Count
                    pass.set_bind_group(1, &mesh.bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
                    pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_inds, 0, 0..1);
                }
            }
            for mesh in self.chunks.values() {
                if cull_frustum.intersects_sphere(mesh.center, mesh.radius) {
                    rendered_chunks += 1; // Count
                    pass.set_bind_group(1, &mesh.bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
                    pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_inds, 0, 0..1);
                }
            }
            // fading chunks are dithered (fs_geom discards), so they work in the G-buffer too
            for state in self.animator.dying_chunks.values() {
                if frustum.intersects_sphere(state.mesh.center, state.mesh.radius) {
                    pass.set_bind_group(1, &state.mesh.bind_group, &[]);
                    pass.set_vertex_buffer(0, state.mesh.v_buf.slice(..));
                    pass.set_index_buffer(state.mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..state.mesh.num_inds, 0, 0..1);
                }
            }
            if !controller.first_person {
                pass.set_bind_group(1, &self.local_bind_player, &[]);
                pass.set_vertex_buffer(0, self.player_v_buf.slice(..));
                pass.set_index_buffer(self.player_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.player_inds, 0, 0..1);
            }
        }

        // --- PASS 2: SHADOWS (HARDWARE RAY TRACING OR RAY MARCHING) + BLUR ---
        // the shadow-resolution G-buffer comes from pass 1, then one compute invocation per shadow texel
        {
            let fov: f32 = if controller.first_person { 80.0 } else { 45.0 }; // Controller::get_matrix
            self.rt_blur.set_fov(&self.queue, fov.to_radians());
            self.deferred.downsample(
                &self.device,
                &mut enc,
                &self.global_bind,
                &self.rt_blur,
                self.gpu_timer
                    .as_ref()
                    .and_then(|t| t.compute_writes(gpu_timer::RAYS, true, false)),
            );
            let compute_writes = self
                .gpu_timer
                .as_ref()
                .and_then(|t| t.compute_writes(gpu_timer::RAYS, false, true));
            match self.hw_rt.as_ref().filter(|_| self.hw_shadows) {
                Some(hw) => hw.trace(
                    &self.device,
                    &self.queue,
                    &mut enc,
                    &self.rt_blur,
                    sun_dir,
                    compute_writes,
                ),
                None => self
                    .rt_blur
                    .march(&mut enc, &self.global_bind, compute_writes),
            }
            self.rt_blur.blur(&mut enc, self.gpu_timer.as_ref());
        }

        // --- PASS 3: LIGHTING (ONCE PER PIXEL) + FORWARD OVERLAYS ---
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lighting Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // the galaxy backdrop is already in the swapchain; fs_light blends over it
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // the G-buffer depth, so the overlays below are hidden behind terrain
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.deferred.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self
                    .gpu_timer
                    .as_ref()
                    .and_then(|t| t.writes(gpu_timer::LIGHTING, true, true)),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.deferred.light_pipeline);
            pass.set_bind_group(0, &self.global_bind, &[]);
            pass.set_bind_group(2, &self.rt_blur.sample_bind, &[]);
            pass.set_bind_group(3, &self.deferred.textures_bind, &[]);
            pass.draw(0..3, 0..1);

            // translucent water over the lit sea floor
            pass.set_pipeline(&self.deferred.water_pipeline);
            for mesh in self.chunks.values() {
                if let Some(water) = &mesh.water {
                    if cull_frustum.intersects_sphere(mesh.center, mesh.radius) {
                        pass.set_bind_group(1, &mesh.bind_group, &[]);
                        pass.set_vertex_buffer(0, water.v_buf.slice(..));
                        pass.set_index_buffer(water.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..water.num_inds, 0, 0..1);
                    }
                }
            }

            if self.collision_inds > 0 {
                pass.set_pipeline(&self.pipeline_line); // Use line pipeline
                pass.set_bind_group(0, &self.global_bind, &[]);
                pass.set_bind_group(1, &self.local_bind_identity, &[]);
                pass.set_vertex_buffer(0, self.collision_v_buf.slice(..));
                pass.set_index_buffer(self.collision_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.collision_inds, 0, 0..1);
            }

            if self.cursor_inds > 0 {
                pass.set_pipeline(&self.pipeline_fill);
                pass.set_bind_group(0, &self.global_bind, &[]);
                pass.set_bind_group(1, &self.local_bind_identity, &[]);
                pass.set_vertex_buffer(0, self.cursor_v_buf.slice(..));
                pass.set_index_buffer(self.cursor_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.cursor_inds, 0, 0..1);
            }

            if controller.first_person {
                pass.set_pipeline(&self.pipeline_line);
                pass.set_bind_group(0, &self.global_bind_identity, &[]);
                pass.set_bind_group(1, &self.local_bind_identity, &[]);
                pass.set_vertex_buffer(0, self.cross_v_buf.slice(..));
                pass.set_index_buffer(self.cross_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.cross_inds, 0, 0..1);
            }

            if self.console_inds > 0 {
                pass.set_pipeline(&self.pipeline_ui);
                pass.set_bind_group(0, &self.global_bind_identity, &[]);
                pass.set_bind_group(1, &self.local_bind_identity, &[]);
                pass.set_vertex_buffer(0, self.console_v_buf.slice(..));
                pass.set_index_buffer(self.console_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.console_inds, 0, 0..1);
            }
        }

        // --- FPS CALCULATION ---
        self.update_fps();

        // --- PASS 4: TEXT RENDER ---
        // run this pass every frame to show FPS
        {
            let mut text_buffers = Vec::new();
            if console.height_fraction > 0.0 {
                let console_pixel_height =
                    console_height_px(self.config.height as f32, console.height_fraction);
                let start_y = console_pixel_height - 40.0;
                let line_height = 20.0;

                for (i, (line_text, color)) in console.history.iter().rev().enumerate() {
                    let y = start_y - (i as f32 * line_height);
                    if y < 0.0 {
                        break;
                    }

                    let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
                    buffer.set_size(
                        Some(self.config.width as f32),
                        Some(self.config.height as f32),
                    );
                    buffer.set_text(
                        line_text,
                        &Attrs::new()
                            .family(Family::Monospace)
                            .color(glyphon::Color::rgb(
                                (color[0] * 255.0) as u8,
                                (color[1] * 255.0) as u8,
                                (color[2] * 255.0) as u8,
                            )),
                        Shaping::Advanced,
                        None,
                    );
                    buffer.shape_until_scroll(&mut self.font_system, false);
                    text_buffers.push((buffer, y));
                }

                let input_y = console_pixel_height - 20.0;
                let mut input_buf = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
                input_buf.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                let time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                let cursor = if (time / 500) % 2 == 0 { "_" } else { " " };
                input_buf.set_text(
                    &format!("> {}{}", console.input_buffer, cursor),
                    &Attrs::new()
                        .family(Family::Monospace)
                        .color(glyphon::Color::rgb(255, 255, 0)),
                    Shaping::Advanced,
                    None,
                );
                input_buf.shape_until_scroll(&mut self.font_system, false);
                text_buffers.push((input_buf, input_y));
            }

            // 2. FPS Text
            let mut fps_buffer = Buffer::new(&mut self.font_system, Metrics::new(20.0, 24.0));
            fps_buffer.set_size(
                Some(self.config.width as f32),
                Some(self.config.height as f32),
            );
            fps_buffer.set_text(
                &format!("FPS: {}", self.current_fps),
                &Attrs::new()
                    .family(Family::Monospace)
                    .color(glyphon::Color::rgb(0, 255, 0)),
                Shaping::Advanced,
                None,
            );
            fps_buffer.shape_until_scroll(&mut self.font_system, false);

            let mut block_buf = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
            block_buf.set_size(
                Some(self.config.width as f32),
                Some(self.config.height as f32),
            );
            block_buf.set_text(
                &format!("Block: {}", controller.selected_block.name()),
                &Attrs::new()
                    .family(Family::Monospace)
                    .color(glyphon::Color::rgb(220, 220, 220)),
                Shaping::Advanced,
                None,
            );
            block_buf.shape_until_scroll(&mut self.font_system, false);

            let mut hp_buf = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
            hp_buf.set_size(
                Some(self.config.width as f32),
                Some(self.config.height as f32),
            );
            let hp_frac = (player.health / player.max_health).clamp(0.0, 1.0);
            let hp_color =
                glyphon::Color::rgb((255.0 * (1.0 - hp_frac)) as u8, (255.0 * hp_frac) as u8, 40);
            hp_buf.set_text(
                &format!("HP: {:.0}/{:.0}", player.health, player.max_health),
                &Attrs::new().family(Family::Monospace).color(hp_color),
                Shaping::Advanced,
                None,
            );
            hp_buf.shape_until_scroll(&mut self.font_system, false);

            // timed status line (show_status), centred in the upper part of the screen
            let status_text = self
                .status
                .as_ref()
                .map(|(text, at)| (text.clone(), status_alpha(at.elapsed().as_secs_f32())))
                .filter(|(_, alpha)| *alpha > 0.0);
            let mut status_buf = Buffer::new(&mut self.font_system, Metrics::new(20.0, 24.0));
            if let Some((text, alpha)) = &status_text {
                status_buf.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                status_buf.set_text(
                    text,
                    &Attrs::new()
                        .family(Family::Monospace)
                        .color(glyphon::Color::rgba(255, 235, 180, (alpha * 255.0) as u8)),
                    Shaping::Advanced,
                    None,
                );
                status_buf.shape_until_scroll(&mut self.font_system, false);
            } else {
                self.status = None;
            }

            let mut debug_buf = Buffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));

            if player.debug_mode {
                let status = if controller.freeze_culling {
                    "FROZEN"
                } else {
                    "ACTIVE"
                };
                let info = format!(
                    "Culling: {}\nChunks: {} / {}\nLODs:   {} / {}\nQueue:  {}\n\nScreen  {}x{}\nShadows {}x{} {}\n{}", 
                    status,
                    rendered_chunks, self.chunks.len(),
                    rendered_lods, self.lod_chunks.len(),
                    self.load_queue.len(),
                    self.config.width, self.config.height,
                    self.rt_blur.size.0, self.rt_blur.size.1,
                    if self.hw_shadows { "HW" } else { "march" },
                    self.gpu_timer.as_ref().map_or("GPU timing unavailable".to_string(), |t| t.summary())
                );

                debug_buf.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                debug_buf.set_text(
                    &info,
                    &Attrs::new()
                        .family(Family::Monospace)
                        .color(glyphon::Color::rgb(200, 200, 200)),
                    Shaping::Advanced,
                    None,
                );
                debug_buf.shape_until_scroll(&mut self.font_system, false);
            }

            // create text areas
            let mut text_areas: Vec<TextArea> = text_buffers
                .iter()
                .map(|(buf, y)| TextArea {
                    buffer: buf,
                    left: 10.0,
                    top: *y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                })
                .collect();

            text_areas.push(TextArea {
                buffer: &fps_buffer,
                left: self.config.width as f32 - 120.0,
                top: 10.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            });

            text_areas.push(TextArea {
                buffer: &block_buf,
                left: self.config.width as f32 - 120.0,
                top: 36.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            });

            text_areas.push(TextArea {
                buffer: &hp_buf,
                left: self.config.width as f32 - 120.0,
                top: 62.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            });

            if let Some((text, _)) = &status_text {
                text_areas.push(TextArea {
                    buffer: &status_buf,
                    left: self.config.width as f32 / 2.0 - text.chars().count() as f32 * 6.0,
                    top: self.config.height as f32 * 0.25,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                });
            }

            if player.debug_mode {
                text_areas.push(TextArea {
                    buffer: &debug_buf,
                    left: self.config.width as f32 - 180.0,
                    top: 88.0, // below the FPS, block, and HP lines
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                });
            }

            self.text_viewport.update(
                &self.queue,
                Resolution {
                    width: self.config.width,
                    height: self.config.height,
                },
            );
            self.text_renderer
                .prepare(
                    &self.device,
                    &self.queue,
                    &mut self.font_system,
                    &mut self.text_atlas,
                    &self.text_viewport,
                    text_areas,
                    &mut self.swash_cache,
                )
                .unwrap();

            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self
                    .gpu_timer
                    .as_ref()
                    .and_then(|t| t.writes(gpu_timer::TEXT, true, true)),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if flare {
                self.draw_flare(&mut pass);
            }
            self.text_renderer
                .render(&self.text_atlas, &self.text_viewport, &mut pass)
                .unwrap();

            // screenshot flash: drawn last (over the text too), so it's visible feedback on screen but,
            // since it's timed to start only after this frame's capture (below), never in the PNG itself
            self.draw_flash(&mut pass, flash_alpha);
        }

        if let Some(timer) = &mut self.gpu_timer {
            timer.resolve(&mut enc);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        if let Some(timer) = &mut self.gpu_timer {
            timer.after_submit();
        }
        if let Some(path) = self.screenshot_request.take() {
            crate::screenshot::capture(
                &self.device,
                &self.queue,
                &out.texture,
                self.config.format,
                self.config.width,
                self.config.height,
                &path,
            );
            self.screenshot_flash = Some(std::time::Instant::now());
        }
        self.queue.present(out);
        self.text_atlas.trim();
    }

    pub fn render_galaxy(
        &mut self,
        camera: &crate::galaxy_render::GalaxyCamera,
        galaxy: &crate::galaxy::Galaxy,
        t: f64,
    ) {
        let out = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = out
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let screen = (self.config.width as f32, self.config.height as f32);
        self.galaxy.render(
            &self.device,
            &self.queue,
            &view,
            &self.deferred.depth,
            camera,
            crate::galaxy_render::GalaxyContent::Everything,
            galaxy,
            t,
            screen,
        );
        self.update_fps();
        let heat = crate::galaxy::star_heat(
            camera.position.distance(galaxy.star.position()),
            galaxy.star.radius,
        );
        let on_screen = crate::galaxy_render::star_on_screen(camera, galaxy, screen.0 / screen.1);
        let flare = self.update_flare(
            on_screen.map(|s| s.0),
            on_screen.map_or(0.0, |s| s.1),
            on_screen.map_or(0.0, |s| s.2),
            camera.fov_y,
            1.0 - heat, // the heat glow already dominates near the star
            galaxy.star.star_type.def(),
            false,
            [0.0; 4],
            [0.0; 4],
        );
        self.render_galaxy_overlay(&view, camera, galaxy, t, heat, flare);
        if let Some(path) = self.screenshot_request.take() {
            crate::screenshot::capture(
                &self.device,
                &self.queue,
                &out.texture,
                self.config.format,
                self.config.width,
                self.config.height,
                &path,
            );
            self.screenshot_flash = Some(std::time::Instant::now());
        }
        self.queue.present(out);
    }

    // Galaxy-mode HUD, drawn over the finished galaxy frame: the same FPS readout as planet mode,
    // plus a compass strip across the top showing where the star and each planet are relative to
    // the view direction, with a distance to its surface. Text-only (glyphon), no extra pipeline.
    fn render_galaxy_overlay(
        &mut self,
        view: &wgpu::TextureView,
        camera: &crate::galaxy_render::GalaxyCamera,
        galaxy: &crate::galaxy::Galaxy,
        t: f64,
        heat: f32,   // the star's heat at the camera (galaxy::star_heat): glow and warning
        flare: bool, // draw the lens flare (update_flare wrote its uniform)
    ) {
        const COMPASS_HALF_WIDTH: f32 = 240.0; // px, covers ±COMPASS_SPAN_DEG
        const COMPASS_SPAN_DEG: f32 = 90.0;
        const COMPASS_TOP: f32 = 10.0;
        const MARKER_ROW_TOP: f32 = 30.0;
        const MARKER_ROW_HEIGHT: f32 = 18.0;
        const MARKER_CHAR_WIDTH: f32 = 8.4; // approx. advance of the 14px monospace font
        const PITCH_HINT_DEG: f32 = 20.0; // show ^/v once a target is this far above/below

        let width = self.config.width as f32;
        let height = self.config.height as f32;
        let center_x = width / 2.0;
        let mut texts: Vec<(String, f32, f32, f32, glyphon::Color)> = Vec::new(); // text, x, y, size, color

        // tick bar: a mark every 15 degrees, the centre one highlighted
        for i in -6..=6 {
            let x = center_x + (i as f32 / 6.0) * COMPASS_HALF_WIDTH;
            let (mark, color) = if i == 0 {
                ("+", glyphon::Color::rgb(255, 255, 255))
            } else {
                ("|", glyphon::Color::rgb(120, 120, 120))
            };
            texts.push((mark.to_string(), x - 4.0, COMPASS_TOP, 14.0, color));
        }

        // markers: (label, x, color), surface distance so "0" means you're touching it
        let mut markers: Vec<(String, f32, glyphon::Color)> = Vec::new();
        let mut add_marker = |name: String, pos: glam::DVec3, radius: f64, rgb: [f32; 3]| {
            let b = crate::galaxy::compass_bearing(camera.rotation, camera.position, pos);
            let dist = crate::galaxy::format_distance((pos - camera.position).length() - radius);
            let pitch = if b.pitch_deg > PITCH_HINT_DEG {
                " ^"
            } else if b.pitch_deg < -PITCH_HINT_DEG {
                " v"
            } else {
                ""
            };
            let (label, frac) = if b.yaw_deg > COMPASS_SPAN_DEG {
                (format!("{name} {dist}{pitch} >"), 1.0)
            } else if b.yaw_deg < -COMPASS_SPAN_DEG {
                (format!("< {name} {dist}{pitch}"), -1.0)
            } else {
                (
                    format!("{name} {dist}{pitch}"),
                    b.yaw_deg / COMPASS_SPAN_DEG,
                )
            };
            let label_w = label.chars().count() as f32 * MARKER_CHAR_WIDTH;
            // centre on the bearing, but keep the whole label inside the strip (an edge-pinned one
            // would otherwise spill half past it, into the FPS readout on the right)
            let x = (center_x + frac * COMPASS_HALF_WIDTH - label_w / 2.0).clamp(
                center_x - COMPASS_HALF_WIDTH,
                center_x + COMPASS_HALF_WIDTH - label_w,
            );
            let c = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8; // brightened for legibility
            markers.push((
                label,
                x,
                glyphon::Color::rgb(c(rgb[0]), c(rgb[1]), c(rgb[2])),
            ));
        };
        add_marker(
            "Sun".to_string(),
            glam::DVec3::ZERO,
            galaxy.star.radius,
            [1.0, 0.9, 0.4],
        );
        for (i, p) in galaxy.planets.iter().enumerate() {
            let def = p.planet_type.def();
            add_marker(
                format!("#{} {}", i + 1, def.name),
                p.position_at(t),
                p.radius as f64,
                def.palette.ground.color(),
            );
        }

        // stack overlapping labels into rows instead of drawing them on top of each other
        markers.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut row_ends: Vec<f32> = Vec::new();
        for (label, x, color) in markers {
            let w = label.chars().count() as f32 * MARKER_CHAR_WIDTH;
            let row = match row_ends.iter().position(|&end| end + 8.0 <= x) {
                Some(r) => r,
                None => {
                    row_ends.push(f32::NEG_INFINITY);
                    row_ends.len() - 1
                }
            };
            row_ends[row] = x + w;
            let y = MARKER_ROW_TOP + row as f32 * MARKER_ROW_HEIGHT;
            texts.push((label, x, y, 14.0, color));
        }

        texts.push((
            format!("FPS: {}", self.current_fps),
            width - 120.0,
            10.0,
            20.0,
            glyphon::Color::rgb(0, 255, 0),
        ));
        if heat >= HEAT_WARNING {
            // centred below the compass, dark red so it reads against the glow
            let text = "Too close to the star";
            let height = self.config.height as f32;
            texts.push((
                text.to_string(),
                (width - text.len() as f32 * 12.6) / 2.0,
                height * 0.3,
                22.0,
                glyphon::Color::rgb(140, 20, 0),
            ));
        }

        let buffers: Vec<(Buffer, f32, f32)> = texts
            .into_iter()
            .map(|(text, x, y, size, color)| {
                let mut buf = Buffer::new(&mut self.font_system, Metrics::new(size, size * 1.2));
                buf.set_size(Some(width), Some(height));
                buf.set_text(
                    &text,
                    &Attrs::new().family(Family::Monospace).color(color),
                    Shaping::Advanced,
                    None,
                );
                buf.shape_until_scroll(&mut self.font_system, false);
                (buf, x, y)
            })
            .collect();
        let text_areas: Vec<TextArea> = buffers
            .iter()
            .map(|(buf, x, y)| TextArea {
                buffer: buf,
                left: *x,
                top: *y,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            })
            .collect();

        self.text_viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
        );
        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.text_viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();

        let flash_alpha = self.update_flash(std::time::Instant::now());
        let heat_alpha = self.update_heat_glow(heat);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Galaxy HUD"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Galaxy HUD Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // the heat glow and the lens flare under the HUD text, so the warning and compass stay readable
            self.draw_full_screen(&mut pass, &self.heat_bind, heat_alpha);
            if flare {
                self.draw_flare(&mut pass);
            }
            self.text_renderer
                .render(&self.text_atlas, &self.text_viewport, &mut pass)
                .unwrap();
            // drawn last, over the HUD text too, same as planet mode
            self.draw_flash(&mut pass, flash_alpha);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn flare_shader_is_valid_wgsl() {
        assert_valid_wgsl(FLARE_SHADER);
    }

    // the Rust BiomeUniform must match the WGSL struct: 13 vec4s (the sun, the star's look and frame)
    #[test]
    fn biome_uniform_matches_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<BiomeUniform>(), 13 * 16);
    }

    // a Yellow star lights its planets exactly as the old SUN_COLOR did
    #[test]
    fn yellow_sunlight_is_todays_sun_color() {
        let def = crate::biome::PlanetType::EarthLike.def();
        let galaxy = crate::galaxy::Galaxy::generate(1);
        assert_eq!(galaxy.star.star_type, crate::galaxy::StarType::Yellow);
        let u = BiomeUniform::from_def(&def, &galaxy.star, 0.3, glam::Quat::IDENTITY);
        assert_eq!(&u.sun[..3], &[1.6, 1.5, 1.3]);
        assert_eq!(u.sun[3], 0.3, "the star's angular radius");
    }

    #[test]
    fn console_covers_a_quarter_of_the_screen() {
        assert_eq!(console_height_px(800.0, 1.0), 200.0);
        assert_eq!(console_height_px(800.0, 0.0), 0.0);
        // the background panel's bottom edge (NDC, +1 = top) sits on the same screen row as the
        // text layout's bottom (pixels down from the top), at any point of the slide animation
        for t in [0.3, 0.6, 1.0] {
            let px = console_height_px(800.0, t);
            let ndc = console_bottom_ndc(t);
            assert!(((1.0 - ndc) / 2.0 * 800.0 - px).abs() < 1e-3, "t={t}");
        }
    }

    // validates WGSL the way wgpu does at pipeline creation, so a shader mistake fails `cargo test`
    // instead of panicking at startup
    pub(crate) fn assert_valid_wgsl(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("WGSL parse error:\n{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error: {e:?}"));
    }

    #[test]
    fn scene_shader_is_valid_wgsl() {
        assert_valid_wgsl(SCENE_SHADER);
    }

    // landing messages (F) must be seen without opening the console: a short HUD status line
    #[test]
    fn status_message_shows_then_fades_out() {
        assert_eq!(status_alpha(0.0), 1.0);
        assert_eq!(status_alpha(STATUS_SECONDS - STATUS_FADE_SECONDS), 1.0);
        let mid = status_alpha(STATUS_SECONDS - STATUS_FADE_SECONDS / 2.0);
        assert!(mid > 0.0 && mid < 1.0, "{mid}");
        assert_eq!(status_alpha(STATUS_SECONDS), 0.0);
        assert_eq!(status_alpha(STATUS_SECONDS + 5.0), 0.0);
    }
    #[test]
    fn camera_water_radius_is_zero_in_dry_columns() {
        let (planet, (face, u, v)) =
            crate::common::tests::lake_planet(crate::biome::PlanetType::EarthLike);
        let level = planet.terrain.water_level(face, u, v);
        let in_lake =
            crate::gen::CoordSystem::get_block_center(face, u, v, level, planet.resolution);
        assert!(
            (Renderer::camera_water_radius(&planet, in_lake)
                - planet.water_surface_radius(face, u, v))
            .abs()
                < 1e-3
        );
        // a dry column: the first one standing more than 3 layers above the sea
        let peak_dir = (0..planet.resolution)
            .find_map(|uu| {
                let h = planet.terrain.get_height(0, uu, 3);
                (h > planet.terrain.sea_level() + 3).then(|| {
                    crate::gen::CoordSystem::get_block_center(0, uu, 3, h + 2, planet.resolution)
                })
            })
            .unwrap();
        assert_eq!(Renderer::camera_water_radius(&planet, peak_dir), 0.0);
    }
}
