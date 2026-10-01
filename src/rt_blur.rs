// rt_blur.rs
// Shadow targets shared by both shadow paths, and the soft-shadow blur:
// 1. a G-buffer pass draws the scene with fs_gbuf (shader.wgsl): world position + camera distance, normal
// 2. a compute pass computes the sharp shadow term once per texel into `target`: cs_march (shader.wgsl,
//    ray marching, `march`) or hw_rt.rs (hardware rays); the G-buffer means no work for hidden surfaces
// 3. blur.wgsl blurs `target` horizontally into `tmp` and vertically into `out`; the main pass upsamples
//    `out` at its pixel (group 2).
//
// Shadows cost per texel, so these targets are capped at MAX_RT_PIXELS: on large screens the shadow
// term is computed at a lower resolution and upsampled depth-aware in fs_main (shadow_factor).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::common::Vertex;
use crate::gpu_timer::{self, GpuTimer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const POS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float; // positions need f32 precision
const NRM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const PENUMBRA_WIDTH: f32 = 0.3; // world units
pub const MAX_RT_PIXELS: f32 = 1.5e6;

// size of the shadow targets for a screen size, keeping the aspect ratio
fn rt_size(width: u32, height: u32) -> (u32, u32) {
    let scale = (MAX_RT_PIXELS / (width.max(1) * height.max(1)) as f32).sqrt().min(1.0);
    (((width as f32 * scale).round() as u32).max(1), ((height as f32 * scale).round() as u32).max(1))
}

// must match BlurParams in blur.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BlurParams {
    dir: [f32; 2],
    focal: f32,
    width: f32,
}

struct Targets {
    target: wgpu::TextureView,
    depth: wgpu::TextureView,
    g_pos: wgpu::TextureView,
    g_nrm: wgpu::TextureView,
    march_bind: wgpu::BindGroup,
    tmp: wgpu::TextureView,
    out: wgpu::TextureView,
    sample_bind: wgpu::BindGroup,
    bind_h: wgpu::BindGroup,
    bind_v: wgpu::BindGroup,
}

pub struct RtBlur {
    pub size: (u32, u32), // of the shadow targets, at most MAX_RT_PIXELS
    pub target: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub g_pos: wgpu::TextureView, // xyz world position, w camera distance (0 = sky)
    pub g_nrm: wgpu::TextureView, // xyz world normal
    pub gbuf_pipeline: wgpu::RenderPipeline,
    march_pipeline: wgpu::ComputePipeline,
    march_layout: wgpu::BindGroupLayout,
    march_bind: wgpu::BindGroup, // group 3 of cs_march: G-buffer in, target out
    tmp: wgpu::TextureView,
    out: wgpu::TextureView,
    pub sample_layout: wgpu::BindGroupLayout,
    pub sample_bind: wgpu::BindGroup, // group 2 of the scene pipelines: `out`
    blur_layout: wgpu::BindGroupLayout,
    blur_pipeline: wgpu::RenderPipeline,
    params_h: wgpu::Buffer,
    params_v: wgpu::Buffer,
    bind_h: wgpu::BindGroup, // target -> tmp
    bind_v: wgpu::BindGroup, // tmp -> out
}

impl RtBlur {
    pub fn sample_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_sample_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        })
    }

    // `sample_layout` must come from RtBlur::sample_layout and be group 2 of `scene_layout`; `global_layout`
    // is group 0 (cs_march reads the camera, sun and ray-march data from it)
    pub fn new(device: &wgpu::Device, sample_layout: wgpu::BindGroupLayout, global_layout: &wgpu::BindGroupLayout, scene_layout: &wgpu::PipelineLayout, scene_shader: &wgpu::ShaderModule, width: u32, height: u32) -> Self {
        let gbuf_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow G-Buffer Pipeline"),
            layout: Some(scene_layout),
            vertex: wgpu::VertexState { module: scene_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as _, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 }] })] },
            fragment: Some(wgpu::FragmentState { module: scene_shader, entry_point: Some("fs_gbuf"), compilation_options: Default::default(), targets: &[Some(POS_FORMAT.into()), Some(NRM_FORMAT.into())] }),
            // same rasterisation as the main fill pipeline, so both passes see the same front surface
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview_mask: None, cache: None,
        });

        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE, count: None,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        };
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("march_layout"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, count: None, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: FORMAT, view_dimension: wgpu::TextureViewDimension::D2 } },
            ],
        });
        let march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow March Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(global_layout), None, None, Some(&march_layout)], immediate_size: 0 })),
            module: scene_shader,
            entry_point: Some("cs_march"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blur"), source: wgpu::ShaderSource::Wgsl(include_str!("blur.wgsl").into()) });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&blur_layout)], immediate_size: 0 });
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RT Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState { module: &blur_shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &blur_shader, entry_point: Some("fs_blur"), compilation_options: Default::default(), targets: &[Some(FORMAT.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None, cache: None,
        });

        let params = |dir: [f32; 2]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur Params"),
            contents: bytemuck::cast_slice(&[BlurParams { dir, focal: 1.0, width: PENUMBRA_WIDTH }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_h = params([1.0, 0.0]);
        let params_v = params([0.0, 1.0]);

        let size = rt_size(width, height);
        let t = Self::make_targets(device, &sample_layout, &blur_layout, &march_layout, &params_h, &params_v, size.0, size.1);
        Self {
            size, target: t.target, depth: t.depth, g_pos: t.g_pos, g_nrm: t.g_nrm, gbuf_pipeline, march_pipeline, march_layout, march_bind: t.march_bind,
            tmp: t.tmp, out: t.out, sample_layout, sample_bind: t.sample_bind, blur_layout, blur_pipeline, params_h, params_v, bind_h: t.bind_h, bind_v: t.bind_v,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_targets(device: &wgpu::Device, sample_layout: &wgpu::BindGroupLayout, blur_layout: &wgpu::BindGroupLayout, march_layout: &wgpu::BindGroupLayout, params_h: &wgpu::Buffer, params_v: &wgpu::Buffer, width: u32, height: u32) -> Targets {
        let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
        let tex = |format: wgpu::TextureFormat, usage: wgpu::TextureUsages| device.create_texture(&wgpu::TextureDescriptor {
            label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());
        let color = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let target = tex(FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING); // written by compute
        let g_pos = tex(POS_FORMAT, color);
        let g_nrm = tex(NRM_FORMAT, color);
        let tmp = tex(FORMAT, color);
        let out = tex(FORMAT, color);
        let depth = tex(wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);

        let sample_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_sample_bind"), layout: sample_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&out) }],
        });
        let blur_bind = |src: &wgpu::TextureView, params: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_bind"), layout: blur_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            ],
        });
        let bind_h = blur_bind(&target, params_h);
        let bind_v = blur_bind(&tmp, params_v);
        let march_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("march_bind"), layout: march_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&g_pos) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&g_nrm) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&target) },
            ],
        });
        Targets { target, depth, g_pos, g_nrm, march_bind, tmp, out, sample_bind, bind_h, bind_v }
    }

    // width/height: the screen size
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.size = rt_size(width, height);
        let t = Self::make_targets(device, &self.sample_layout, &self.blur_layout, &self.march_layout, &self.params_h, &self.params_v, self.size.0, self.size.1);
        (self.target, self.depth, self.g_pos, self.g_nrm, self.march_bind) = (t.target, t.depth, t.g_pos, t.g_nrm, t.march_bind);
        (self.tmp, self.out, self.sample_bind, self.bind_h, self.bind_v) = (t.tmp, t.out, t.sample_bind, t.bind_h, t.bind_v);
    }

    pub fn gbuf_clear_ops() -> wgpu::Operations<wgpu::Color> {
        wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store }
    }

    // ray-marched shadow term, one invocation per texel (cs_march); global_bind: group 0 of the scene
    pub fn march(&self, enc: &mut wgpu::CommandEncoder, global_bind: &wgpu::BindGroup, timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Shadow March Pass"), timestamp_writes });
        pass.set_pipeline(&self.march_pipeline);
        pass.set_bind_group(0, global_bind, &[]);
        pass.set_bind_group(3, &self.march_bind, &[]);
        pass.dispatch_workgroups(self.size.0.div_ceil(8), self.size.1.div_ceil(8), 1);
    }

    // the camera's vertical field of view, so the blur radius covers PENUMBRA_WIDTH at the targets' size
    pub fn set_fov(&self, queue: &wgpu::Queue, fov_y: f32) {
        let focal = self.size.1 as f32 * 0.5 / (fov_y * 0.5).tan(); // pixels per world unit at distance 1
        for (buf, dir) in [(&self.params_h, [1.0, 0.0]), (&self.params_v, [0.0, 1.0])] {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&[BlurParams { dir, focal, width: PENUMBRA_WIDTH }]));
        }
    }

    pub fn clear_ops() -> wgpu::Operations<wgpu::Color> {
        wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store }
    }

    // timer: the horizontal pass writes the blur's begin timestamp, the vertical pass its end
    pub fn blur(&self, enc: &mut wgpu::CommandEncoder, timer: Option<&GpuTimer>) {
        for (i, (dst, bind)) in [(&self.tmp, &self.bind_h), (&self.out, &self.bind_v)].into_iter().enumerate() {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RT Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { depth_slice: None, view: dst, resolve_target: None, ops: Self::clear_ops() })],
                depth_stencil_attachment: None,
                timestamp_writes: timer.and_then(|t| t.writes(gpu_timer::BLUR, i == 0, i == 1)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
