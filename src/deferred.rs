// deferred.rs
// Deferred shading: the scene geometry is drawn once per frame, into a full-resolution G-buffer, and lit
// once per pixel afterwards.
// 1. geometry pass (fs_geom): vertex colour, normal and camera distance per screen pixel, plus depth
// 2. shadow prep (cs_gbuf_down): the shadow-resolution G-buffer of rt_blur.rs, from the full-resolution one
//    (then the shadow compute pass and the blur run as before)
// 3. lighting pass (fs_light): a full-screen triangle that shades each pixel once (shade() in shader.wgsl)
// The translucent water surface (fs_water) and the overlays (cursor box, collision lines, crosshair,
// console) are drawn forward after the lighting pass, depth-tested against the G-buffer depth.

use crate::common::Vertex;
use crate::rt_blur::RtBlur;

const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm; // normal * 0.5 + 0.5
const DIST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float; // camera distance, full precision
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Deferred {
    pub geom_fill: wgpu::RenderPipeline,
    pub geom_wire: wgpu::RenderPipeline, // the same as geom_fill on devices without POLYGON_MODE_LINE
    pub light_pipeline: wgpu::RenderPipeline,
    pub water_pipeline: wgpu::RenderPipeline,
    down_pipeline: wgpu::ComputePipeline,
    textures_layout: wgpu::BindGroupLayout, // group 3: the full-resolution G-buffer
    down_layout: wgpu::BindGroupLayout,     // group 1 of cs_gbuf_down: shadow G-buffer outputs
    albedo: wgpu::TextureView,
    normal: wgpu::TextureView,
    dist: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub textures_bind: wgpu::BindGroup,
}

impl Deferred {
    // scene_layout: groups 0-2 of the scene pipelines; global/local/sample_layout: its groups 0, 1 and 2
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        scene_layout: &wgpu::PipelineLayout,
        global_layout: &wgpu::BindGroupLayout,
        local_layout: &wgpu::BindGroupLayout,
        sample_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        surface_format: wgpu::TextureFormat,
        wireframe: bool,
        width: u32,
        height: u32,
    ) -> Self {
        let geom = |polygon_mode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Geometry Pipeline"),
                layout: Some(scene_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as _,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 24,
                                shader_location: 2,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_geom"),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(ALBEDO_FORMAT.into()),
                        Some(NORMAL_FORMAT.into()),
                        Some(DIST_FORMAT.into()),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    polygon_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let geom_fill = geom(wgpu::PolygonMode::Fill);
        let geom_wire = geom(if wireframe {
            wgpu::PolygonMode::Line
        } else {
            wgpu::PolygonMode::Fill
        });

        let tex_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        };
        let both = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let textures_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred_textures_layout"),
            entries: &[tex_entry(3, both), tex_entry(4, both), tex_entry(5, both)],
        });

        // the lighting pass writes the swapchain and keeps the depth for the overlays drawn after it
        let light_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light_layout"),
            bind_group_layouts: &[
                Some(global_layout),
                None,
                Some(sample_layout),
                Some(&textures_layout),
            ],
            immediate_size: 0,
        });
        let light_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lighting Pipeline"),
            layout: Some(&light_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_full"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_light"),
                compilation_options: Default::default(),
                // alpha-blended over the galaxy backdrop: fs_light's alpha is the sky's opacity
                // (1 for terrain and for a thick, lit atmosphere)
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // water: scene vertices, alpha-blended over the lit image, depth-tested but not written
        let water_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_layout"),
            bind_group_layouts: &[
                Some(global_layout),
                Some(local_layout),
                Some(sample_layout),
                Some(&textures_layout),
            ],
            immediate_size: 0,
        });
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&water_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as _,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_water"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let storage = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
        };
        let down_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gbuf_down_layout"),
            entries: &[
                storage(1, wgpu::TextureFormat::Rgba32Float),
                storage(2, wgpu::TextureFormat::Rgba16Float),
            ],
        });
        let down_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow G-Buffer Downsample"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        Some(global_layout),
                        Some(&down_layout),
                        None,
                        Some(&textures_layout),
                    ],
                    immediate_size: 0,
                }),
            ),
            module: shader,
            entry_point: Some("cs_gbuf_down"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (albedo, normal, dist, depth, textures_bind) =
            Self::make_targets(device, &textures_layout, width, height);
        Self {
            geom_fill,
            geom_wire,
            light_pipeline,
            water_pipeline,
            down_pipeline,
            textures_layout,
            down_layout,
            albedo,
            normal,
            dist,
            depth,
            textures_bind,
        }
    }

    fn make_targets(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> (
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::BindGroup,
    ) {
        let tex = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("G-Buffer"),
                    size: wgpu::Extent3d {
                        width: width.max(1),
                        height: height.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let target = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let albedo = tex(ALBEDO_FORMAT, target);
        let normal = tex(NORMAL_FORMAT, target);
        let dist = tex(DIST_FORMAT, target);
        let depth = tex(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_textures_bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&dist),
                },
            ],
        });
        (albedo, normal, dist, depth, bind)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        (
            self.albedo,
            self.normal,
            self.dist,
            self.depth,
            self.textures_bind,
        ) = Self::make_targets(device, &self.textures_layout, width, height);
    }

    // colour attachments of the geometry pass, cleared to "sky" (distance 0)
    pub fn geometry_targets(&self) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 3] {
        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };
        [&self.albedo, &self.normal, &self.dist].map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                depth_slice: None,
                view,
                resolve_target: None,
                ops: clear,
            })
        })
    }

    // fill the shadow-resolution G-buffer of `shadows` from the full-resolution one
    pub fn downsample(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        global_bind: &wgpu::BindGroup,
        shadows: &RtBlur,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        // the shadow targets are replaced on resize, so this bind group is made per frame
        let out_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuf_down_bind"),
            layout: &self.down_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadows.g_pos),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadows.g_nrm),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shadow G-Buffer Downsample"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.down_pipeline);
        pass.set_bind_group(0, global_bind, &[]);
        pass.set_bind_group(1, &out_bind, &[]);
        pass.set_bind_group(3, &self.textures_bind, &[]);
        pass.dispatch_workgroups(shadows.size.0.div_ceil(8), shadows.size.1.div_ceil(8), 1);
    }
}
