// engine renderer

use std::collections::{HashMap, HashSet};
use wgpu::PresentMode;
use winit::window::Window;
use wgpu::util::DeviceExt;
use glyphon::{Cache, FontSystem, SwashCache, TextAtlas, TextArea, TextRenderer as GlyphRenderer, TextBounds, Resolution, Buffer, Metrics, Shaping, Attrs, Family, Viewport};
use std::sync::Arc;
use crate::cmd::Console;
use crate::common::*;
use crate::gen::{MeshGen, CoordSystem};
use crate::controller::Controller;
use crate::entity::Player;
use glam::Vec3;
use crate::lod_animation::{LodAnimator, AnyKey};
use crate::rt_shadow::{RtParams, ShadowWindow};
use crate::rt_blur::RtBlur;
use bytemuck::{Pod, Zeroable};
use std::sync::mpsc::{channel, Receiver, Sender};

// --- UNIFORMS ---

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GlobalUniform {
    pub view_proj: [f32; 16],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],   
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LocalUniform {
    pub model: [f32; 16],
    pub params: [f32; 4], // x = opacity
}

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
    pipeline_wire: wgpu::RenderPipeline,
    pipeline_line: wgpu::RenderPipeline,
    
    chunks: HashMap<ChunkKey, ChunkMesh>,     
    lod_chunks: HashMap<LodKey, ChunkMesh>, 

    // --- UNIFORMS ---
    global_buf: wgpu::Buffer,
    global_bind: wgpu::BindGroup,
    
    local_bind_identity: wgpu::BindGroup,
    
    local_buf_player: wgpu::Buffer,
    local_bind_player: wgpu::BindGroup,


    depth: wgpu::TextureView,
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
    
    mesh_tx: Sender<(ChunkKey, Vec<Vertex>, Vec<u32>)>,
    mesh_rx: Receiver<(ChunkKey, Vec<Vertex>, Vec<u32>)>,
    pending_chunks: HashSet<ChunkKey>, 

    lod_tx: Sender<(LodKey, Vec<Vertex>, Vec<u32>)>,
    lod_rx: Receiver<(LodKey, Vec<Vertex>, Vec<u32>)>,
    pending_lods: HashSet<LodKey>,

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
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }).await.unwrap();
        
        // log GPU info
        crate::system_diagnostics::SystemDiagnostics::log_gpu(&adapter.get_info());

        let target_buffer_size: u64 = 8 * 1024 * 1024 * 1024;
        let mut limits = adapter.limits();
        // we are requiring a maximum of 8gb but we take as much as the platform is capable of
        limits.max_buffer_size = target_buffer_size.min(limits.max_buffer_size);

        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::POLYGON_MODE_LINE) {
            features |= wgpu::Features::POLYGON_MODE_LINE;
        }

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None, required_features: features, required_limits: limits,
            experimental_features: Default::default(), memory_hints: Default::default(), trace: Default::default(),
        }).await.unwrap();

let size = window.inner_size();
        let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        // macOS only colour-manages the window when its layer has a colour space; for 8-bit surfaces wgpu
        // offers sRGB (which leaves the layer untagged) or Display P3, so tag it P3 and convert the
        // output to P3 primaries in fs_main. Otherwise sRGB colours show oversaturated on P3 displays.
        let output_p3 = surface.get_capabilities(&adapter).format_capabilities.iter()
            .any(|c| c.format == config.format && c.color_spaces.contains(wgpu::SurfaceColorSpaces::DISPLAY_P3));
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
        let text_renderer = GlyphRenderer::new(&mut text_atlas, &device, wgpu::MultisampleState::default(), None);

        let global_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[

                wgpu::BindGroupLayoutEntry { 
                    binding: 0, 
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, 
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, 
                    count: None 
                },
                // 1: ray-marched shadow window params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 2: ray-marched shadow window solid/air bits
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
            label: Some("global_layout"),
        });

        let local_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry { 
                binding: 0, 
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, 
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, 
                count: None 
            }],
            label: Some("local_layout"),
        });

        // --- BUFFERS ---
        let global_buf = device.create_buffer(&wgpu::BufferDescriptor { 
            label: Some("Global Uniform"), 
            size: std::mem::size_of::<GlobalUniform>() as u64, 
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
            mapped_at_creation: false 
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

        let global_bind = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            layout: &global_layout, 
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: global_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rt_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rt_bits_buf.as_entire_binding() },
            ], 
            label: None 
        });

        let identity_mat = glam::Mat4::IDENTITY;
        let default_local = LocalUniform {
            model: identity_mat.to_cols_array(),
            params: [1.0, 0.0, 1.0, 0.0], 
        };

        // console buffers
        let console_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Console V"), size: 1024, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });
        let console_i_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Console I"), size: 1024, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });

        let local_buf_identity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: Some("Identity Uniform"), 
            contents: bytemuck::cast_slice(&[default_local]), 
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST 
        });
        
        let local_bind_identity = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            layout: &local_layout, 
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: local_buf_identity.as_entire_binding() }], 
            label: None 
        });

        // player uniform
        let local_buf_player = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: Some("Player Uniform"), 
            contents: bytemuck::cast_slice(&[default_local]), 
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
        });
        let local_bind_player = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            layout: &local_layout, 
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: local_buf_player.as_entire_binding() }], 
            label: None 
        });

        // --- PIPELINES ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()) });
        // group 2: the blurred ray-marched shadow term, read by fs_main
        let rt_sample_layout = RtBlur::sample_layout(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&global_layout), Some(&local_layout), Some(&rt_sample_layout)], immediate_size: 0 });
        let rt_blur = RtBlur::new(&device, rt_sample_layout, &layout, &shader, config.width, config.height);

        let pipeline_fill = Self::create_pipeline(&device, &config, &layout, &shader, wgpu::PrimitiveTopology::TriangleList, false);
        let pipeline_wire = Self::create_pipeline(&device, &config, &layout, &shader, wgpu::PrimitiveTopology::TriangleList, true);
        let pipeline_line = Self::create_pipeline(&device, &config, &layout, &shader, wgpu::PrimitiveTopology::LineList, false);
        let depth = Self::mk_depth(&device, &config);

        // --- UI PIPELINE ---
        let pipeline_ui = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as _, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 }] })]},
            fragment: Some(wgpu::FragmentState { 
                module: &shader, 
                entry_point: Some("fs_main"), compilation_options: Default::default(), 
                targets: &[Some(wgpu::ColorTargetState { 
                    format: config.format, 
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL 
                })] 
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });

        // --- MESHES ---
        let (pv, pi) = MeshGen::generate_cylinder(0.4, 1.8, 16);
        let player_v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&pv), usage: wgpu::BufferUsages::VERTEX });
        let player_i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&pi), usage: wgpu::BufferUsages::INDEX });

        let (cv, ci) = MeshGen::generate_crosshair();
        let cross_v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&cv), usage: wgpu::BufferUsages::VERTEX });
        let cross_i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&ci), usage: wgpu::BufferUsages::INDEX });

        let cursor_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cursor V"), size: 4096, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });
        let cursor_i_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cursor I"), size: 4096, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });



        let collision_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collision V"), size: 65536, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });
        let collision_i_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collision I"), size: 65536, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
        });





        // global identity
        let identity_global_data = GlobalUniform {
            view_proj: identity_mat.to_cols_array(),
            cam_pos: [0.0, 0.0, 0.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, p3_flag],
        };
        
        let global_buf_identity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Identity Buffer"),
            contents: bytemuck::cast_slice(&[identity_global_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        });

        let global_bind_identity = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &global_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: global_buf_identity.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rt_off_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rt_bits_buf.as_entire_binding() },
            ],
            label: Some("Identity Bind Group"), 
        });

        let (mesh_tx, mesh_rx) = channel(); 
        let (lod_tx, lod_rx) = channel();

        Self { 
            window, surface, device, queue, config, 
            pipeline_fill, pipeline_wire, pipeline_line,
            chunks: HashMap::new(), 
            lod_chunks: HashMap::new(),
            global_buf, global_bind, 
            local_bind_identity,
            local_buf_player, local_bind_player,
            depth,

            font_system,
            swash_cache,
            text_atlas,
            text_viewport,
            text_renderer,
            collision_v_buf, collision_i_buf, collision_inds: 0,
            frozen_frustum: None,
            player_v_buf, player_i_buf, player_inds: pi.len() as u32,
            pipeline_ui,
            console_v_buf,
            console_i_buf,
            console_inds: 0,
            cross_v_buf, cross_i_buf, cross_inds: ci.len() as u32,
            global_bind_identity,
            cursor_v_buf, cursor_i_buf, cursor_inds: 0,
            animator: LodAnimator::new(),
            local_layout,
            load_queue: Vec::new(),
            player_chunk_pos: None,
            mesh_tx,
            mesh_rx,
            pending_chunks: HashSet::new(),
            lod_tx,
            lod_rx,
            pending_lods: HashSet::new(),
            
            last_fps_time: std::time::Instant::now(),
            frame_count: 0,
            current_fps: 0,
            output_p3,

            rt_params_buf,
            rt_bits_buf,
            rt_center: None,
            rt_dirty: true,
            rt_blur,
        }
    }

    fn create_pipeline(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, layout: &wgpu::PipelineLayout, shader: &wgpu::ShaderModule, topology: wgpu::PrimitiveTopology, wireframe: bool) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(layout),
            vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as _, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 }] })]},
            fragment: Some(wgpu::FragmentState { module: shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(config.format.into())] }),
            primitive: wgpu::PrimitiveState { 
                topology, 
                cull_mode: None, 
                polygon_mode: if wireframe { wgpu::PolygonMode::Line } else { wgpu::PolygonMode::Fill }, 
                ..Default::default() 
            },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        })
    }

    fn mk_depth(dev: &wgpu::Device, cfg: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
        dev.create_texture(&wgpu::TextureDescriptor { size: wgpu::Extent3d { width: cfg.width, height: cfg.height, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, label: None, view_formats: &[] }).create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = Self::mk_depth(&self.device, &self.config);
        self.rt_blur.resize(&self.device, width, height);
    }

    pub fn update_console_mesh(&mut self, t: f32) {
        if t <= 0.001 {
            self.console_inds = 0;
            return;
        }

        let height = t * 1.0; 
        let bottom_y = 1.0 - height;

        let color = [0.1, 0.1, 0.15]; 
        let normal = [0.0, 0.0, 1.0];

        let verts = vec![
            Vertex { pos: [-1.0, 1.0, 0.0], color, normal },      
            Vertex { pos: [ 1.0, 1.0, 0.0], color, normal },      
            Vertex { pos: [-1.0, bottom_y, 0.0], color, normal }, 
            Vertex { pos: [ 1.0, bottom_y, 0.0], color, normal }, 
        ];

        let inds = vec![0, 2, 1, 1, 2, 3];

        self.queue.write_buffer(&self.console_v_buf, 0, bytemuck::cast_slice(&verts));
        self.queue.write_buffer(&self.console_i_buf, 0, bytemuck::cast_slice(&inds));
        self.console_inds = inds.len() as u32;
    }

    pub fn update_view(&mut self, player_pos: Vec3, planet: &PlanetData) {
        let res = planet.resolution;        
        let player_id = CoordSystem::pos_to_id(player_pos, res);
        let mut upload_count = 0;
        while let Ok((key, v, i)) = self.lod_rx.try_recv() {
            self.pending_lods.remove(&key);
            self.upload_lod_buffer(key, v, i);
            upload_count += 1;
            if upload_count > 20 { break; }
        }
        let mut required_voxels: HashSet<ChunkKey> = HashSet::new();
        let mut required_lods: HashSet<LodKey> = HashSet::new();
        let logical_size = res.next_power_of_two();

        for face in 0..6 {
            self.process_quadtree(
                face, 0, 0, logical_size, 
                player_pos, planet, 
                player_id, 
                &mut required_voxels, 
                &mut required_lods
            );
        }

        let missing_voxels: Vec<ChunkKey> = required_voxels.iter()
            .filter(|k| !self.chunks.contains_key(k))
            .cloned()
            .collect();

        let current_lods: Vec<LodKey> = self.lod_chunks.keys().cloned().collect();
        
        for k in current_lods {
            if required_lods.contains(&k) { continue; }
            
            let mut children_missing = false;
            for v_key in &missing_voxels {
                if v_key.face != k.face { continue; }
                let v_x = v_key.u_idx * CHUNK_SIZE as u32;
                let v_y = v_key.v_idx * CHUNK_SIZE as u32;
                let v_s = CHUNK_SIZE as u32;
                let overlap = k.x < v_x + v_s && k.x + k.size > v_x &&
                              k.y < v_y + v_s && k.y + k.size > v_y;
                if overlap { children_missing = true; break; }
            }

            if children_missing {
                required_lods.insert(k);
            } else {
                if let Some(mesh) = self.lod_chunks.remove(&k) {
                    self.animator.retire(AnyKey::Lod(k), mesh);
                }
            }
        }

        let mut spawn_count = 0;
        for key in required_lods {
            if !self.lod_chunks.contains_key(&key) && !self.pending_lods.contains(&key) {
                if spawn_count >= 8 { break; }
                self.pending_lods.insert(key);
                let tx = self.lod_tx.clone();
                let p = planet.clone();
                std::thread::spawn(move || {
                    let (v, i) = MeshGen::generate_lod_mesh(key, &p);
                    let _ = tx.send((key, v, i));
                });
                spawn_count += 1;
            }
        }

        let current_voxels: Vec<ChunkKey> = self.chunks.keys().cloned().collect();
        for k in current_voxels {
            if !required_voxels.contains(&k) {
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
        face: u8, x: u32, y: u32, size: u32, 
        cam_pos: Vec3, 
        planet: &PlanetData,
        player_id: Option<BlockId>, 
        voxels: &mut HashSet<ChunkKey>,
        lods: &mut HashSet<LodKey>
    ) {
        if x >= planet.resolution || y >= planet.resolution { return; }

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

        let node_radius_world = (size as f32 * CoordSystem::get_layer_radius(h, planet.resolution)) / planet.resolution as f32;
        
        let mut lod_factor = 4.0; 
        if size <= CHUNK_SIZE * 8 { lod_factor = 5.0; }
        if size <= CHUNK_SIZE * 4 { lod_factor = 7.0; }
        if size <= CHUNK_SIZE * 2 { lod_factor = 12.0; } 
        if size <= CHUNK_SIZE     { lod_factor = 18.0; } 

        let split_distance = node_radius_world * lod_factor;
        let is_smallest = size <= CHUNK_SIZE;
        
        if dist < split_distance && !is_smallest {
            let half = size / 2;
            self.process_quadtree(face, x, y, half, cam_pos, planet, player_id, voxels, lods);
            self.process_quadtree(face, x + half, y, half, cam_pos, planet, player_id, voxels, lods);
            self.process_quadtree(face, x, y + half, half, cam_pos, planet, player_id, voxels, lods);
            self.process_quadtree(face, x + half, y + half, half, cam_pos, planet, player_id, voxels, lods);
        } else {
            if size <= CHUNK_SIZE {
                let key = ChunkKey { face, u_idx: x / CHUNK_SIZE, v_idx: y / CHUNK_SIZE };
                if (key.u_idx * CHUNK_SIZE) < planet.resolution && (key.v_idx * CHUNK_SIZE) < planet.resolution {
                    voxels.insert(key);
                }
            } else {
                let key = LodKey { face, x, y, size };
                lods.insert(key);
            }
        }
    }

    fn upload_lod_buffer(&mut self, key: LodKey, v: Vec<Vertex>, i: Vec<u32>) {
        let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST });
        let i_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&i), usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST });

        let uniform_data = LocalUniform {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            params: [0.0, 0.0, 0.0, 0.0], 
        };
        
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LOD Uniform"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.local_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
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

        self.lod_chunks.insert(key, ChunkMesh { 
            v_buf, i_buf, num_inds: i.len() as u32, num_verts: v.len(), uniform_buf, bind_group,
            center: real_center, // <--- ADDED
            radius: real_radius  // <--- ADDED
        });
        self.animator.start_spawn(AnyKey::Lod(key));
    }
    fn process_load_queue(&mut self, _player_pos: Vec3, planet: &PlanetData) {
        let mut upload_budget = 4; 
        while let Ok((key, v, i)) = self.mesh_rx.try_recv() {
            self.pending_chunks.remove(&key);
            if !v.is_empty() {
                self.upload_chunk_buffers(key, v, i);
                upload_budget -= 1;
            }
            if upload_budget <= 0 { break; }
        }

        if upload_budget <= 0 { return; }
        if self.load_queue.is_empty() { return; }
        if self.pending_chunks.len() >= 12 { return; } 

        let chunks_to_spawn = 4;
        for _ in 0..chunks_to_spawn {
            if let Some(key) = self.load_queue.pop() {
                if self.chunks.contains_key(&key) || self.pending_chunks.contains(&key) {
                    continue;
                }
                self.pending_chunks.insert(key);
                let planet_clone = planet.clone();
                let tx = self.mesh_tx.clone();
                std::thread::spawn(move || {
                    let (v, i) = MeshGen::build_chunk(key, &planet_clone);
                    let _ = tx.send((key, v, i));
                });
            } else {
                break;
            }
        }
    }

    pub fn force_reload_all(&mut self, planet: &PlanetData, player_pos: Vec3) {
        self.chunks.clear();
        self.lod_chunks.clear();
        self.load_queue.clear();
        self.pending_chunks.clear();
        self.pending_lods.clear(); 
        self.player_chunk_pos = None; 
        self.rt_dirty = true;
        self.update_view(player_pos, planet);
    }

    pub fn refresh_neighbors(&mut self, id: BlockId, planet: &PlanetData) {
        self.rt_dirty = true;
        // the block's own chunk and the chunks of its four neighbours, which may lie on another cube face
        let mut keys = vec![PlanetData::chunk_key(id)];
        for (du, dv) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            if let Some((face, u, v)) = planet.neighbor_column(id.face, id.u, id.v, du, dv) {
                let key = PlanetData::chunk_key(BlockId { face, u, v, layer: 0 });
                if !keys.contains(&key) { keys.push(key); }
            }
        }
        for key in keys {
            if self.chunks.contains_key(&key) {
                let (v, i) = MeshGen::build_chunk(key, planet);
                if v.is_empty() { 
                    self.chunks.remove(&key);
                } else {
                    self.upload_chunk_buffers(key, v, i);
                }
            }
        }
    }


    fn update_rt_window(&mut self, player_pos: Vec3, planet: &PlanetData) {
        let Some(id) = CoordSystem::pos_to_id(player_pos, planet.resolution) else { return };
        // small planets are stored completely, so only big ones follow the player
        let whole_planet = planet.resolution <= crate::rt_shadow::WINDOW_SIZE;
        let recenter = match self.rt_center {
            Some(c) => {
                let limit = (crate::rt_shadow::WINDOW_SIZE / 8) as i32;
                !whole_planet && (c.face != id.face || (c.u as i32 - id.u as i32).abs() > limit || (c.v as i32 - id.v as i32).abs() > limit)
            }
            None => true,
        };
        if !recenter && !self.rt_dirty { return; }

        let start = std::time::Instant::now();
        let window = ShadowWindow::build(planet, id, player_pos.normalize_or_zero());
        self.queue.write_buffer(&self.rt_params_buf, 0, bytemuck::cast_slice(&[window.params]));
        self.queue.write_buffer(&self.rt_bits_buf, 0, bytemuck::cast_slice(&window.bits));
        let faces: Vec<u32> = (0..6).filter(|&f| window.params.faces[f].size > 0).map(|f| f as u32).collect();
        println!("RT shadow windows rebuilt: faces {:?}, {} KB in {:.2} ms",
            faces, window.bits.len() * 4 / 1024, start.elapsed().as_secs_f32() * 1000.0);
        self.rt_center = Some(id);
        self.rt_dirty = false;
    }






    fn upload_chunk_buffers(&mut self, key: ChunkKey, v: Vec<Vertex>, i: Vec<u32>) {
        let v_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST });
        let i_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&i), usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST });
        
        let is_update = self.chunks.contains_key(&key);
        let start_opacity = if is_update { 1.0 } else { 0.0 };

        let uniform_data = LocalUniform {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            params: [start_opacity, 0.0, 0.0, 0.0], 
        };
        
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Uniform"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.local_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
            label: None,
        });

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        if v.is_empty() {
             min = Vec3::ZERO; max = Vec3::ZERO;
        } else {
            for vert in &v {
                let p = Vec3::from_array(vert.pos);
                min = min.min(p);
                max = max.max(p);
            }
        }
        let real_center = (min + max) * 0.5;
        let real_radius = min.distance(max) * 0.5;

        self.chunks.insert(key, ChunkMesh { 
            v_buf, i_buf, num_inds: i.len() as u32, num_verts: v.len(), uniform_buf, bind_group,
            center: real_center, 
            radius: real_radius  
        });
        
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
        if mb > 1024.0 { println!("GPU Memory: {:.2} GB", mb / 1024.0); } 
        else { println!("GPU Memory: {:.2} MB", mb); }
        println!("------------------------------------------");
    }

    pub fn update_cursor(&mut self, planet: &PlanetData, id: Option<BlockId>) {
        if let Some(id) = id {
            let res = planet.resolution;
            let p = |u, v, l| CoordSystem::get_vertex_pos(id.face, id.u + u, id.v + v, id.layer + l, res);
            
            let corners = [
                p(0,0,0), p(1,0,0), p(0,1,0), p(1,1,0), 
                p(0,0,1), p(1,0,1), p(0,1,1), p(1,1,1)  
            ];

            let edges = [
                (0,1), (1,3), (3,2), (2,0), 
                (4,5), (5,7), (7,6), (6,4), 
                (0,4), (1,5), (2,6), (3,7)  
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
                let ref_up = if dir.dot(Vec3::Y).abs() > 0.9 { Vec3::X } else { Vec3::Y };
                let right = dir.cross(ref_up).normalize() * thickness;
                let up = dir.cross(right).normalize() * thickness;
                let offsets = [(-right - up), (right - up), (right + up), (-right + up)];
                
                for off in offsets {
                    verts.push(Vertex { pos: (a + off).to_array(), color, normal: [0.0;3] });
                    verts.push(Vertex { pos: (b + off).to_array(), color, normal: [0.0;3] });
                }

                let faces = [(0,1,3,2), (2,3,5,4), (4,5,7,6), (6,7,1,0)];
                for (i0, i1, i2, i3) in faces {
                    inds.push(idx_base + i0); inds.push(idx_base + i1); inds.push(idx_base + i2);
                    inds.push(idx_base + i2); inds.push(idx_base + i3); inds.push(idx_base + i0);
                }
                idx_base += 8;
            }

            self.queue.write_buffer(&self.cursor_v_buf, 0, bytemuck::cast_slice(&verts));
            self.queue.write_buffer(&self.cursor_i_buf, 0, bytemuck::cast_slice(&inds));
            self.cursor_inds = inds.len() as u32;
        } else {
            self.cursor_inds = 0;
        }
    }


pub fn render(&mut self, controller: &Controller, player: &Player, planet: &PlanetData, console: &Console) {
        self.update_console_mesh(console.height_fraction);

if controller.show_collisions {
             let (v, i) = MeshGen::generate_collision_debug(player.position, planet);
             self.queue.write_buffer(&self.collision_v_buf, 0, bytemuck::cast_slice(&v));
             self.queue.write_buffer(&self.collision_i_buf, 0, bytemuck::cast_slice(&i));
             self.collision_inds = i.len() as u32;
        } else {
             self.collision_inds = 0;
        }



        let out = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o) | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = out.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.update_rt_window(player.position, planet);

        // directional sun, fixed in world space
        let sun_dir = glam::Vec3::new(0.5, 0.8, 0.4).normalize();

        // -- Camera Matrix --
        let mvp = controller.get_matrix(player, self.config.width as f32, self.config.height as f32);
        
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

        // update main global uni
        let global_data = GlobalUniform {
            view_proj: mvp.to_cols_array(),
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, if self.output_p3 { 1.0 } else { 0.0 }],
        };
        self.queue.write_buffer(&self.global_buf, 0, bytemuck::cast_slice(&[global_data]));

        let model_mat = player.get_model_matrix();
        self.queue.write_buffer(&self.local_buf_player, 0, bytemuck::cast_slice(model_mat.as_ref()));

        let now = std::time::Instant::now();
        let dying_status = self.animator.update_dying(now);
        for (key, alpha) in dying_status {
            if let Some(state) = self.animator.dying_chunks.get(&key) {
                let data = LocalUniform { 
                    model: glam::Mat4::IDENTITY.to_cols_array(), 
                    params: [alpha, 1.0, 0.0, 0.0] 
                };
                self.queue.write_buffer(&state.mesh.uniform_buf, 0, bytemuck::cast_slice(&[data]));
            }
        }

        let queue = &self.queue;
        let animator = &mut self.animator;
        
        let mut update_opacity = |key: AnyKey, mesh: &ChunkMesh| {
            let alpha = animator.get_opacity(key, now);
            if alpha < 1.0 {
                let data = LocalUniform { 
                    model: glam::Mat4::IDENTITY.to_cols_array(), 
                    params: [alpha, 0.0, 0.0, 0.0] 
                };
                queue.write_buffer(&mesh.uniform_buf, 0, bytemuck::cast_slice(&[data]));
            } else if animator.spawning_chunks.contains_key(&key) {
                let data = LocalUniform { 
                    model: glam::Mat4::IDENTITY.to_cols_array(), 
                    params: [1.0, 0.0, 0.0, 0.0] 
                };
                queue.write_buffer(&mesh.uniform_buf, 0, bytemuck::cast_slice(&[data]));
                animator.spawning_chunks.remove(&key);
            }
        };

        for (key, mesh) in &self.lod_chunks { update_opacity(AnyKey::Lod(*key), mesh); }
        for (key, mesh) in &self.chunks { update_opacity(AnyKey::Voxel(*key), mesh); }

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        
        // --- PASS 1: RAY-MARCHED SHADOWS + BLUR ---
        {
            let fov: f32 = if controller.first_person { 80.0 } else { 45.0 }; // Controller::get_matrix
            self.rt_blur.set_focal(&self.queue, self.config.height as f32 * 0.5 / (fov.to_radians() * 0.5).tan());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("RT Shadow Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { depth_slice: None, view: &self.rt_blur.target, resolve_target: None, ops: RtBlur::clear_ops() })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.rt_blur.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                multiview_mask: None,
                });
                pass.set_pipeline(&self.rt_blur.rt_pipeline);
                pass.set_bind_group(0, &self.global_bind, &[]);
                pass.set_bind_group(2, &self.rt_blur.sample_bind, &[]);
                // the same meshes the main pass draws below
                let meshes = self.lod_chunks.values().chain(self.chunks.values()).filter(|m| cull_frustum.intersects_sphere(m.center, m.radius))
                    .chain(self.animator.dying_chunks.values().map(|d| &d.mesh).filter(|m| frustum.intersects_sphere(m.center, m.radius)));
                for mesh in meshes {
                    pass.set_bind_group(1, &mesh.bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
                    pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_inds, 0, 0..1);
                }
                if !controller.first_person {
                    pass.set_bind_group(1, &self.local_bind_player, &[]);
                    pass.set_vertex_buffer(0, self.player_v_buf.slice(..));
                    pass.set_index_buffer(self.player_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.player_inds, 0, 0..1);
                }
            }
            self.rt_blur.blur(&mut enc);
        }

        // --- PASS 2: MAIN RENDER ---
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {

            label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                depth_slice: None, 
                view: &view, 
                resolve_target: None, 
                ops: wgpu::Operations { 
                    // Matches the atmospheric fog color in shader

                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.03, b: 0.05, a: 1.0 }),
                    store: wgpu::StoreOp::Store 
                } 
            })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            
            if controller.is_wireframe { pass.set_pipeline(&self.pipeline_wire); } 
            else { pass.set_pipeline(&self.pipeline_fill); }
            
            pass.set_bind_group(0, &self.global_bind, &[]);
            pass.set_bind_group(2, &self.rt_blur.sample_bind, &[]);
            
            // DRAW LOD CHUNKS
            for mesh in self.lod_chunks.values() {
                if cull_frustum.intersects_sphere(mesh.center, mesh.radius) {
                    rendered_lods += 1; // Count
                    pass.set_bind_group(1, &mesh.bind_group, &[]); 
                    pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
                    pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_inds, 0, 0..1);
                }
            }

            // DRAW VOXEL CHUNKS
            for mesh in self.chunks.values() {
                if cull_frustum.intersects_sphere(mesh.center, mesh.radius) {
                    rendered_chunks += 1; // Count
                    pass.set_bind_group(1, &mesh.bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
                    pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_inds, 0, 0..1);
                }
            }

            // DRAW DYING ANIMATIONS
            for state in self.animator.dying_chunks.values() {
                if frustum.intersects_sphere(state.mesh.center, state.mesh.radius) {
                    pass.set_bind_group(1, &state.mesh.bind_group, &[]);
                    pass.set_vertex_buffer(0, state.mesh.v_buf.slice(..));
                    pass.set_index_buffer(state.mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..state.mesh.num_inds, 0, 0..1);
                }
            }

            if !controller.first_person {
                if controller.is_wireframe { pass.set_pipeline(&self.pipeline_wire); } 
                else { pass.set_pipeline(&self.pipeline_fill); }
                pass.set_bind_group(1, &self.local_bind_player, &[]);
                pass.set_vertex_buffer(0, self.player_v_buf.slice(..));
                pass.set_index_buffer(self.player_i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.player_inds, 0, 0..1);
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
        self.frame_count += 1;
        let now = std::time::Instant::now();
        if now.duration_since(self.last_fps_time).as_secs_f32() >= 1.0 {
            self.current_fps = self.frame_count;
            self.frame_count = 0;
            self.last_fps_time = now;
        }

        // --- PASS 3: TEXT RENDER ---
        // run this pass every frame to show FPS
        {
            let mut text_buffers = Vec::new();
            if console.height_fraction > 0.0 {
                let console_pixel_height = (self.config.height as f32 / 2.0) * console.height_fraction;
                let start_y = console_pixel_height - 40.0;
                let line_height = 20.0;
                
                for (i, (line_text, color)) in console.history.iter().rev().enumerate() {
                    let y = start_y - (i as f32 * line_height);
                    if y < 0.0 { break; } 
                    
                    let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
                    buffer.set_size(Some(self.config.width as f32), Some(self.config.height as f32));
                    buffer.set_text(line_text, &Attrs::new().family(Family::Monospace).color(glyphon::Color::rgb(
                        (color[0] * 255.0) as u8, 
                        (color[1] * 255.0) as u8, 
                        (color[2] * 255.0) as u8
                    )), Shaping::Advanced, None);
                    buffer.shape_until_scroll(&mut self.font_system, false);
                    text_buffers.push((buffer, y));
                }

                let input_y = console_pixel_height - 20.0;
                let mut input_buf = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
                input_buf.set_size(Some(self.config.width as f32), Some(self.config.height as f32));
                let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
                let cursor = if (time / 500) % 2 == 0 { "_" } else { " " };
                input_buf.set_text(&format!("> {}{}", console.input_buffer, cursor), &Attrs::new().family(Family::Monospace).color(glyphon::Color::rgb(255, 255, 0)), Shaping::Advanced, None);
                input_buf.shape_until_scroll(&mut self.font_system, false);
                text_buffers.push((input_buf, input_y));
            }

            // 2. FPS Text
            let mut fps_buffer = Buffer::new(&mut self.font_system, Metrics::new(20.0, 24.0));
            fps_buffer.set_size(Some(self.config.width as f32), Some(self.config.height as f32));
            fps_buffer.set_text(
                &format!("FPS: {}", self.current_fps), 
                &Attrs::new().family(Family::Monospace).color(glyphon::Color::rgb(0, 255, 0)), 
                Shaping::Advanced,
                None
            );
            fps_buffer.shape_until_scroll(&mut self.font_system, false);


            let mut block_buf = Buffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
            block_buf.set_size(Some(self.config.width as f32), Some(self.config.height as f32));
            block_buf.set_text(
                &format!("Block: {}", controller.selected_block.name()),
                &Attrs::new().family(Family::Monospace).color(glyphon::Color::rgb(220, 220, 220)),
                Shaping::Advanced,
                None
            );
            block_buf.shape_until_scroll(&mut self.font_system, false);

          
            let mut debug_buf = Buffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
            
            if player.debug_mode {
                let status = if controller.freeze_culling { "FROZEN" } else { "ACTIVE" };
                let info = format!(
                    "Culling: {}\nChunks: {} / {}\nLODs:   {} / {}\nQueue:  {}", 
                    status,
                    rendered_chunks, self.chunks.len(),
                    rendered_lods, self.lod_chunks.len(),
                    self.load_queue.len()
                );

                debug_buf.set_size(Some(self.config.width as f32), Some(self.config.height as f32));
                debug_buf.set_text(
                    &info, 
                    &Attrs::new().family(Family::Monospace).color(glyphon::Color::rgb(200, 200, 200)), 
                    Shaping::Advanced,
                    None
                );
                debug_buf.shape_until_scroll(&mut self.font_system, false);
            }
           
            // create text areas
            let mut text_areas: Vec<TextArea> = text_buffers.iter().map(|(buf, y)| {
                TextArea {
                    buffer: buf,
                    left: 10.0,
                    top: *y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0, top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                }
            }).collect();

            text_areas.push(TextArea {
                buffer: &fps_buffer,
                left: self.config.width as f32 - 120.0, 
                top: 10.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0, top: 0,
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
                    left: 0, top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            });

            if player.debug_mode {
                text_areas.push(TextArea {
                    buffer: &debug_buf,
                    left: self.config.width as f32 - 180.0,
                    top: 40.0,
                    scale: 1.0,
                    bounds: TextBounds { left: 0, top: 0, right: self.config.width as i32, bottom: self.config.height as i32 },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                });
            }

            self.text_viewport.update(&self.queue, Resolution { width: self.config.width, height: self.config.height });
            self.text_renderer.prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.text_viewport,
                text_areas,
                &mut self.swash_cache
            ).unwrap();

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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            
            self.text_renderer.render(&self.text_atlas, &self.text_viewport, &mut pass).unwrap();
        }

        self.queue.submit(std::iter::once(enc.finish()));
        self.queue.present(out);
        self.text_atlas.trim();
    }
}
