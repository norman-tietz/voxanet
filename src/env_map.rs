// env_map.rs
// Pre-filtered environment map for the plastic's sky reflection (docs/superpowers/specs/
// 2026-10-11-env-map-design.md): the sky around the camera (dome × opacity with the clouds, no sun
// disc), premultiplied with its coverage above the local horizon, in a 2-slot cube-map array whose
// levels are GGX-filtered for roughness level / 5. Each frame one step: a face of the source cube
// (fs_env_bake, shader.wgsl), or one filtered level of the back slot (env_filter.wgsl); after the last
// level the slots swap, so nothing reads a half-updated map. shade() samples the read slot.

pub const ENV_SIZE: u32 = 128; // texels per face edge (shader.wgsl ENV_SIZE, env_filter.wgsl)
pub const ENV_LEVELS: u32 = 6; // 128 … 4 (shader.wgsl ENV_LEVELS)
pub const SRC_MIPS: u32 = 8; // the source cube's box mips, 128 … 1 (the filter's undersampling guard)
pub const SLOTS: u32 = 2;
pub const FACES: u32 = 6;
#[cfg_attr(not(test), allow(dead_code))]
pub const FILTER_SAMPLES: u32 = 32; // env_filter.wgsl SAMPLES (checked by a test)
pub const STEPS: u32 = FACES + ENV_LEVELS; // one renewal

// one frame's GPU work
#[derive(Debug, PartialEq)]
pub enum EnvBake {
    Whole(u32), // every face, the source mips and every level into this slot (start-up, reset)
    Face(u32),  // one face of the source cube
    Level(u32, u32), // (slot, level): one filtered level; level 0 first builds the source's mips
}

// which slot is read and where the renewal of the other one stands; pure bookkeeping (tested)
#[derive(Debug, Default)]
pub struct Schedule {
    read: u32,
    step: u32,
    valid: bool, // the read slot holds a complete map
}

impl Schedule {
    // this frame's work; `reset`: the map no longer fits (planet switch, back from galaxy mode)
    pub fn update(&mut self, reset: bool) -> Vec<EnvBake> {
        if reset || !self.valid {
            self.valid = true;
            self.step = 0;
            return vec![EnvBake::Whole(self.read)];
        }
        let back = 1 - self.read;
        let work = if self.step < FACES {
            EnvBake::Face(self.step)
        } else {
            EnvBake::Level(back, self.step - FACES)
        };
        self.step += 1;
        if self.step == STEPS {
            self.read = back;
            self.step = 0;
        }
        vec![work]
    }

    pub fn read_slot(&self) -> u32 {
        self.read
    }
}

// the level shade() samples for a roughness (shader.wgsl env_sample)
#[cfg_attr(not(test), allow(dead_code))]
pub fn roughness_to_level(r: f32) -> f32 {
    r.clamp(0.0, 1.0) * (ENV_LEVELS - 1) as f32
}

// shade() reflects only the analytic sky from ENV_FADE_TOP × the cloud shell's radius up: the map is
// baked from the camera, and from the cloud shell up its sky is space (black reflections at the
// landing handover, no clouds); shader.wgsl ENV_FADE_TOP
pub const ENV_FADE_TOP: f32 = 1.5;

// how much of the map shade() reflects with the camera `cam_r` from the centre of a planet of radius
// `planet_r` (shader.wgsl): all of it below the clouds, none from ENV_FADE_TOP × their radius
#[cfg_attr(not(test), allow(dead_code))]
pub fn map_weight(cam_r: f32, planet_r: f32) -> f32 {
    let lo = crate::common::CLOUD_ALT * planet_r;
    let hi = lo * ENV_FADE_TOP;
    let t = ((cam_r - lo) / (hi - lo)).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

// the GGX roughness level `level` is filtered with (env_filter.wgsl)
pub fn level_roughness(level: u32) -> f32 {
    level as f32 / (ENV_LEVELS - 1) as f32
}

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// shader.wgsl EnvParams
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EnvParams {
    slot: [f32; 4], // x = the read slot
}

// env_filter.wgsl FilterParams
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FilterParams {
    roughness: f32,
    size: f32,
    src_size: f32,
    _pad: f32,
}

pub struct EnvMap {
    schedule: Schedule,
    sample_view: wgpu::TextureView, // the 2-slot cube array, all levels (group 0 binding 11)
    params_buf: wgpu::Buffer,       // EnvParams (binding 12)
    src_faces: Vec<wgpu::TextureView>, // source (face, mip) as D2 targets: face * SRC_MIPS + mip
    src_mip_reads: Vec<wgpu::BindGroup>, // the same subresources, bound as fs_env_mip's mip_src
    targets: Vec<wgpu::TextureView>, // env (slot, face, level) as D2 targets: (slot * FACES + face) * ENV_LEVELS + level
    filter_binds: Vec<wgpu::BindGroup>, // per level: the source cube, sampler, that level's FilterParams
    filter_pipeline: wgpu::RenderPipeline,
    mip_pipeline: wgpu::RenderPipeline,
    bake_pipeline: Option<wgpu::RenderPipeline>, // fs_env_bake lives in the scene shader (init_bake)
}

impl EnvMap {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = |label, layers, mips| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: ENV_SIZE,
                    height: ENV_SIZE,
                    depth_or_array_layers: layers,
                },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let layer_view = |t: &wgpu::Texture, layer, mip| {
            t.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let env = texture("Env Map", SLOTS * FACES, ENV_LEVELS);
        let sample_view = env.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Env Map (cube array)"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let targets = (0..SLOTS * FACES)
            .flat_map(|layer| (0..ENV_LEVELS).map(move |level| (layer, level)))
            .map(|(layer, level)| layer_view(&env, layer, level))
            .collect();
        let src = texture("Env Source", FACES, SRC_MIPS);
        let src_cube = src.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Env Source (cube)"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Env Map Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Env Map Params"),
            contents: bytemuck::cast_slice(&[EnvParams::zeroed()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Env Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("env_filter.wgsl").into()),
        });
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let filter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_filter_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: fragment,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: fragment,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let mip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_mip_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline = |label, layout: &wgpu::BindGroupLayout, entry_point| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(layout)],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_env_face"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    targets: &[Some(FORMAT.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let filter_pipeline = pipeline("Env Filter", &filter_layout, "fs_env_filter");
        let mip_pipeline = pipeline("Env Mips", &mip_layout, "fs_env_mip");

        let filter_binds = (0..ENV_LEVELS)
            .map(|level| {
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Env Filter Params"),
                    contents: bytemuck::cast_slice(&[FilterParams {
                        roughness: level_roughness(level),
                        size: (ENV_SIZE >> level) as f32,
                        src_size: ENV_SIZE as f32,
                        _pad: 0.0,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &filter_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&src_cube),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        let mut src_faces = Vec::new();
        let mut src_mip_reads = Vec::new();
        for face in 0..FACES {
            for mip in 0..SRC_MIPS {
                let view = layer_view(&src, face, mip);
                src_mip_reads.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &mip_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                }));
                src_faces.push(view);
            }
        }

        Self {
            schedule: Schedule::default(),
            sample_view,
            params_buf,
            src_faces,
            src_mip_reads,
            targets,
            filter_binds,
            filter_pipeline,
            mip_pipeline,
            bake_pipeline: None,
        }
    }

    // the bindings the scene's group 0 adds (shader.wgsl env_map, env_params)
    pub fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
        [
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::CubeArray,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::FRAGMENT,
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
                binding: 11,
                resource: wgpu::BindingResource::TextureView(&self.sample_view),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: self.params_buf.as_entire_binding(),
            },
        ]
    }

    // the face bake is the scene shader's fs_env_bake, drawn with the scene's group 0
    pub fn init_bake(
        &mut self,
        device: &wgpu::Device,
        scene_shader: &wgpu::ShaderModule,
        global_layout: &wgpu::BindGroupLayout,
    ) {
        self.bake_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Env Bake"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(global_layout)],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: scene_shader,
                    entry_point: Some("vs_env_face"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: scene_shader,
                    entry_point: Some("fs_env_bake"),
                    compilation_options: Default::default(),
                    targets: &[Some(FORMAT.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            }),
        );
    }

    // once per planet-mode frame, after the frame's global and biome uniforms are written (the bake
    // reads the camera, sun and atmosphere from them): this frame's step, submitted on its own
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        global_bind: &wgpu::BindGroup,
        reset: bool,
    ) {
        let work = self.schedule.update(reset);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Env Map"),
        });
        for step in work {
            match step {
                EnvBake::Whole(slot) => {
                    for face in 0..FACES {
                        self.bake_face(&mut enc, global_bind, face);
                    }
                    for level in 0..ENV_LEVELS {
                        self.filter_level(&mut enc, slot, level);
                    }
                }
                EnvBake::Face(face) => self.bake_face(&mut enc, global_bind, face),
                EnvBake::Level(slot, level) => self.filter_level(&mut enc, slot, level),
            }
        }
        queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::cast_slice(&[EnvParams {
                slot: [self.schedule.read_slot() as f32, 0.0, 0.0, 0.0],
            }]),
        );
        queue.submit(std::iter::once(enc.finish()));
    }

    fn pass<'a>(
        enc: &'a mut wgpu::CommandEncoder,
        target: &'a wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Env Map"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                depth_slice: None,
                view: target,
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
        })
    }

    // the face goes to the shader as the instance index
    fn bake_face(&self, enc: &mut wgpu::CommandEncoder, global_bind: &wgpu::BindGroup, face: u32) {
        let mut pass = Self::pass(enc, &self.src_faces[(face * SRC_MIPS) as usize]);
        pass.set_pipeline(self.bake_pipeline.as_ref().expect("EnvMap::init_bake"));
        pass.set_bind_group(0, global_bind, &[]);
        pass.draw(0..3, face..face + 1);
    }

    // level 0 first builds the source's mip chain (every face is baked by then)
    fn filter_level(&self, enc: &mut wgpu::CommandEncoder, slot: u32, level: u32) {
        if level == 0 {
            for face in 0..FACES {
                for mip in 1..SRC_MIPS {
                    let i = (face * SRC_MIPS + mip) as usize;
                    let mut pass = Self::pass(enc, &self.src_faces[i]);
                    pass.set_pipeline(&self.mip_pipeline);
                    pass.set_bind_group(0, &self.src_mip_reads[i - 1], &[]);
                    pass.draw(0..3, face..face + 1);
                }
            }
        }
        for face in 0..FACES {
            let i = ((slot * FACES + face) * ENV_LEVELS + level) as usize;
            let mut pass = Self::pass(enc, &self.targets[i]);
            pass.set_pipeline(&self.filter_pipeline);
            pass.set_bind_group(0, &self.filter_binds[level as usize], &[]);
            pass.draw(0..3, face..face + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // start-up / reset: the whole map into the read slot in that very frame (Review Focus 1)
    #[test]
    fn reset_bakes_whole_into_the_read_slot() {
        let mut s = Schedule::default();
        assert_eq!(s.update(false), vec![EnvBake::Whole(0)]);
        assert_eq!(s.read_slot(), 0);
        // a later reset bakes whole again, into whatever slot is read then
        for _ in 0..5 {
            s.update(false);
        }
        assert_eq!(s.update(true), vec![EnvBake::Whole(s.read_slot())]);
    }

    // after the whole bake: faces 0–5 into the source, then levels 0–5 into the back slot, then swap
    #[test]
    fn renewal_is_six_faces_then_six_levels() {
        let mut s = Schedule::default();
        s.update(false);
        let mut steps = Vec::new();
        for _ in 0..STEPS {
            steps.extend(s.update(false));
        }
        let mut want: Vec<EnvBake> = (0..FACES).map(EnvBake::Face).collect();
        want.extend((0..ENV_LEVELS).map(|l| EnvBake::Level(1, l)));
        assert_eq!(steps, want);
    }

    // Review Focus 5
    #[test]
    fn read_slot_changes_only_after_a_full_renewal() {
        let mut s = Schedule::default();
        s.update(false);
        for i in 0..STEPS {
            assert_eq!(s.read_slot(), 0, "step {i}");
            s.update(false);
        }
        assert_eq!(s.read_slot(), 1);
        for _ in 0..STEPS {
            s.update(false);
        }
        assert_eq!(s.read_slot(), 0);
    }

    // Review Focus 4
    #[test]
    fn roughness_maps_to_levels() {
        assert_eq!(roughness_to_level(0.0), 0.0);
        assert_eq!(roughness_to_level(1.0), (ENV_LEVELS - 1) as f32);
        assert!((roughness_to_level(0.35) - 1.75).abs() < 1e-6);
        assert_eq!(roughness_to_level(-1.0), 0.0);
        assert_eq!(roughness_to_level(3.0), (ENV_LEVELS - 1) as f32);
        for l in 0..ENV_LEVELS {
            assert!((roughness_to_level(level_roughness(l)) - l as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn sizes_are_consistent() {
        assert_eq!(ENV_SIZE >> (ENV_LEVELS - 1), 4);
        assert_eq!(ENV_SIZE >> (SRC_MIPS - 1), 1);
    }

    // the map is baked from the camera: from the cloud shell up its sky is space, so shade() fades to
    // the analytic sky (which the impostors use too) — full map below the clouds, none from 1.5×
    #[test]
    fn map_weight_fades_out_above_the_clouds() {
        let r = 168.0;
        let clouds = crate::common::CLOUD_ALT * r;
        assert_eq!(map_weight(r + 5.0, r), 1.0);
        assert_eq!(map_weight(clouds, r), 1.0);
        let mid = map_weight(clouds * 1.25, r);
        assert!(mid > 0.0 && mid < 1.0, "{mid}");
        assert_eq!(map_weight(clouds * 1.5, r), 0.0);
        assert_eq!(map_weight(r * 3.0, r), 0.0);
        let s = include_str!("shader.wgsl");
        assert!(
            s.contains("smoothstep(CLOUD_ALT, CLOUD_ALT * ENV_FADE_TOP"),
            "shade() mirrors map_weight"
        );
        assert!(s.contains(&format!("const ENV_FADE_TOP: f32 = {ENV_FADE_TOP:?};")));
    }

    // three hand-copied cube-face direction tables must stay identical, or a face is rotated or
    // mirrored silently (the cloud map's is the tested reference)
    #[test]
    fn cube_face_directions_match_the_cloud_map() {
        let cases = |src: &str, func: &str| -> Vec<String> {
            let body = &src[src.find(func).unwrap()..];
            let body = &body[..body.find("\n}\n").unwrap()];
            body.lines()
                .map(str::trim)
                .filter(|l| l.starts_with("case") || l.starts_with("default"))
                .map(String::from)
                .collect()
        };
        let reference = cases(include_str!("cloud_bake.wgsl"), "fn cube_dir");
        assert_eq!(reference.len(), 6);
        assert_eq!(
            cases(include_str!("shader.wgsl"), "fn env_cube_dir"),
            reference
        );
        assert_eq!(
            cases(include_str!("env_filter.wgsl"), "fn cube_dir"),
            reference
        );
        assert_eq!(
            cases(include_str!("weather_bake.wgsl"), "fn cube_dir"),
            reference
        );
    }

    #[test]
    fn filter_shader_constants_match() {
        let f = include_str!("env_filter.wgsl");
        assert!(f.contains(&format!("const SAMPLES: u32 = {FILTER_SAMPLES}u;")));
        let s = include_str!("shader.wgsl");
        assert!(s.contains(&format!("const ENV_SIZE: f32 = {ENV_SIZE}.0;")));
        assert!(s.contains(&format!("const ENV_LEVELS: f32 = {ENV_LEVELS}.0;")));
    }
}
