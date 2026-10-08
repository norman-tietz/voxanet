// cloud_map.rs
// The cloud field (atmosphere.wgsl fbm_clouds: noise over the direction from the planet centre and the
// time) baked into a cube map, so the clouds cost a few filtered texture reads per pixel instead of ~160
// noise hashes (atmo_cloud_shade in clouds.wgsl samples it). Every planet shares the field, so one map
// serves the voxel engine and all galaxy impostors, and their clouds still match at the landing handover.
//
// The wind changes the field over time, so it is baked as snapshots SNAPSHOT_SECS apart and the shaders
// blend the two around the current time. A cube-map array holds SLOTS snapshots: the two being blended
// and the next one, baked ahead one cube face per frame (then its mip chain), so it is ready before the
// blend reaches it. The map stores the raw noise value; the coverage threshold is applied after
// filtering, which keeps cloud edges crisp under magnification.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub(crate) const CLOUD_BAKE_SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    include_str!("star.wgsl"), // atmosphere.wgsl's sun disc uses it
    "\n",
    include_str!("cloud_bake.wgsl")
);

pub const SIZE: u32 = 512; // texels per cube face edge (clouds.wgsl CLOUD_MAP_SIZE)
const MIPS: u32 = 10; // 512 down to 1 (clouds.wgsl CLOUD_MAP_MIPS)
const SLOTS: u32 = 3;
const FACES: u32 = 6;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub const SNAPSHOT_SECS: f64 = 0.5;

// what the shaders blend (must match CloudMapParams in clouds.wgsl)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CloudMapParams {
    slots: [f32; 4], // x: the older snapshot's slot, y: the newer one's, z: blend 0..1 toward y
}

// the snapshot time of each slot, for the bake (must match BakeParams in cloud_bake.wgsl)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BakeParams {
    times: [f32; 4],
}

// which snapshots exist and which is being baked; pure bookkeeping, no GPU (tested below)
#[derive(Debug, Default)]
struct Schedule {
    held: [Option<i64>; SLOTS as usize], // the snapshot number each slot holds (time = n * SNAPSHOT_SECS)
    baking: Option<(i64, u32)>,          // (snapshot, faces done) being baked into slot(n)
}

// GPU work for one frame
#[derive(Debug, PartialEq)]
enum Bake {
    Whole(i64),       // all faces and mips now (start-up, or time jumped)
    Face(i64, u32),   // one face of the snapshot baked ahead
    Finish(i64, u32), // its last face, then its mips
}

fn slot(n: i64) -> u32 {
    n.rem_euclid(SLOTS as i64) as u32
}

impl Schedule {
    // the work for time `t` (cloud time, seconds) and the blend (older slot, newer slot, factor)
    fn update(&mut self, t: f64) -> (Vec<Bake>, CloudMapParams) {
        let n0 = (t / SNAPSHOT_SECS).floor() as i64;
        let mut work = Vec::new();
        // the two blended snapshots must be there; missing ones are baked whole right away
        for n in [n0, n0 + 1] {
            if self.held[slot(n) as usize] != Some(n) {
                work.push(Bake::Whole(n));
                self.held[slot(n) as usize] = Some(n);
                if self.baking.is_some_and(|(b, _)| slot(b) == slot(n)) {
                    self.baking = None;
                }
            }
        }
        // the next one, a face per frame (its slot held n0 - 1, no longer blended)
        let next = n0 + 2;
        if self.held[slot(next) as usize] != Some(next) {
            let done = match self.baking {
                Some((b, done)) if b == next => done,
                _ => {
                    self.held[slot(next) as usize] = None;
                    0
                }
            };
            if done + 1 == FACES {
                work.push(Bake::Finish(next, done));
                self.held[slot(next) as usize] = Some(next);
                self.baking = None;
            } else {
                work.push(Bake::Face(next, done));
                self.baking = Some((next, done + 1));
            }
        }
        let blend = (t / SNAPSHOT_SECS - n0 as f64) as f32;
        let params = CloudMapParams {
            slots: [slot(n0) as f32, slot(n0 + 1) as f32, blend, 0.0],
        };
        (work, params)
    }
}

pub struct CloudMap {
    schedule: Schedule,
    times: [f32; 4], // BakeParams.times as last written
    sample_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    params_buf: wgpu::Buffer,
    bake_buf: wgpu::Buffer,
    bake_bind: wgpu::BindGroup,
    bake_pipeline: wgpu::RenderPipeline,
    mip_pipeline: wgpu::RenderPipeline,
    targets: Vec<wgpu::TextureView>, // one per (layer, mip): layer * MIPS + mip
    mip_sources: Vec<wgpu::BindGroup>, // the same subresources, bound for reading
}

impl CloudMap {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cloud Map"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: SLOTS * FACES,
            },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Cloud Map (cube array)"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cloud Map Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloud Map Params"),
            contents: bytemuck::cast_slice(&[CloudMapParams::zeroed()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bake_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloud Bake Params"),
            contents: bytemuck::cast_slice(&[BakeParams::zeroed()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Bake Shader"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_BAKE_SHADER.into()),
        });
        let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_bake_layout"),
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
        let mip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_mip_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
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
                    entry_point: Some("vs_cloud_face"),
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
        let bake_pipeline = pipeline("Cloud Bake", &bake_layout, "fs_cloud_bake");
        let mip_pipeline = pipeline("Cloud Mips", &mip_layout, "fs_cloud_mip");
        let bake_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud_bake_bind"),
            layout: &bake_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bake_buf.as_entire_binding(),
            }],
        });

        let mut targets = Vec::new();
        let mut mip_sources = Vec::new();
        for layer in 0..SLOTS * FACES {
            for mip in 0..MIPS {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                mip_sources.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &mip_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                }));
                targets.push(view);
            }
        }

        Self {
            schedule: Schedule::default(),
            times: [0.0; 4],
            sample_view,
            sampler,
            params_buf,
            bake_buf,
            bake_bind,
            bake_pipeline,
            mip_pipeline,
            targets,
            mip_sources,
        }
    }

    // the bindings the scene's group 0 and the galaxy camera's group 0 add (clouds.wgsl)
    pub fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
        let fragment = wgpu::ShaderStages::FRAGMENT;
        [
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::CubeArray,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: fragment,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
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

    pub fn bind_entries(&self) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&self.sample_view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: self.params_buf.as_entire_binding(),
            },
        ]
    }

    // once per frame, before anything that draws clouds: bakes what the schedule asks for (submitted
    // on its own) and sets the blend for cloud time `t` (seconds, wrapped like the shaders' clock)
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: f64) {
        let (work, params) = self.schedule.update(t);
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[params]));
        if work.is_empty() {
            return;
        }
        // a slot's time is set before any of its faces are baked and stays until it is reused, so
        // one write covers every pass of this submission
        for bake in &work {
            let (Bake::Whole(n) | Bake::Face(n, _) | Bake::Finish(n, _)) = *bake;
            self.times[slot(n) as usize] = (n as f64 * SNAPSHOT_SECS) as f32;
        }
        queue.write_buffer(
            &self.bake_buf,
            0,
            bytemuck::cast_slice(&[BakeParams { times: self.times }]),
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cloud Map Bake"),
        });
        for bake in work {
            match bake {
                Bake::Whole(n) => {
                    for face in 0..FACES {
                        self.bake_face(&mut enc, slot(n), face);
                    }
                    self.build_mips(&mut enc, slot(n));
                }
                Bake::Face(n, face) => self.bake_face(&mut enc, slot(n), face),
                Bake::Finish(n, face) => {
                    self.bake_face(&mut enc, slot(n), face);
                    self.build_mips(&mut enc, slot(n));
                }
            }
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    fn pass<'a>(
        enc: &'a mut wgpu::CommandEncoder,
        target: &'a wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Cloud Map"),
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

    // the layer goes to the shader as the instance index (cloud_bake.wgsl: slot = layer / 6)
    fn bake_face(&self, enc: &mut wgpu::CommandEncoder, slot: u32, face: u32) {
        let layer = slot * FACES + face;
        let mut pass = Self::pass(enc, &self.targets[(layer * MIPS) as usize]);
        pass.set_pipeline(&self.bake_pipeline);
        pass.set_bind_group(0, &self.bake_bind, &[]);
        pass.draw(0..3, layer..layer + 1);
    }

    fn build_mips(&self, enc: &mut wgpu::CommandEncoder, slot: u32) {
        for face in 0..FACES {
            let layer = slot * FACES + face;
            for mip in 1..MIPS {
                let i = (layer * MIPS + mip) as usize;
                let mut pass = Self::pass(enc, &self.targets[i]);
                pass.set_pipeline(&self.mip_pipeline);
                pass.set_bind_group(0, &self.mip_sources[i - 1], &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(s: &mut Schedule, t: f64) -> (Vec<Bake>, [f32; 4]) {
        let (work, p) = s.update(t);
        (work, p.slots)
    }

    #[test]
    fn start_bakes_the_blended_pair_whole_and_starts_the_next() {
        let mut s = Schedule::default();
        let (work, slots) = run(&mut s, 10.2);
        assert_eq!(
            work,
            vec![Bake::Whole(20), Bake::Whole(21), Bake::Face(22, 0)]
        );
        assert_eq!(slots[0], slot(20) as f32);
        assert_eq!(slots[1], slot(21) as f32);
        assert!((slots[2] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn the_next_snapshot_is_ready_before_the_blend_reaches_it() {
        let mut s = Schedule::default();
        let mut t = 10.0;
        run(&mut s, t);
        // at 60 fps, nothing is baked whole again: one face per frame, finished with its mips
        let mut finished = Vec::new();
        for _ in 0..600 {
            t += 1.0 / 60.0;
            let (work, _) = run(&mut s, t);
            for w in work {
                match w {
                    Bake::Whole(n) => panic!("baked {n} whole at {t}"),
                    Bake::Finish(n, face) => {
                        assert_eq!(face, FACES - 1);
                        assert!((n as f64) * SNAPSHOT_SECS > t + SNAPSHOT_SECS - 1e-9);
                        finished.push(n);
                    }
                    Bake::Face(..) => {}
                }
            }
        }
        // every snapshot from 22 on, in order
        assert_eq!(
            finished,
            (22..22 + finished.len() as i64).collect::<Vec<_>>()
        );
        assert!(finished.len() >= 19, "{}", finished.len());
    }

    #[test]
    fn slow_frames_and_time_jumps_bake_whole() {
        let mut s = Schedule::default();
        run(&mut s, 10.0);
        // a second later (2 fps): 21 is still held, 22 wasn't finished, so it is baked whole
        let (work, _) = run(&mut s, 11.0);
        assert_eq!(
            work,
            vec![Bake::Whole(22), Bake::Whole(23), Bake::Face(24, 0)]
        );
        // the clock wrapping at 3600 s jumps back
        let (work, slots) = run(&mut s, 0.1);
        assert_eq!(work, vec![Bake::Whole(0), Bake::Whole(1), Bake::Face(2, 0)]);
        assert_eq!(&slots[..2], &[0.0, 1.0]);
    }

    #[test]
    fn bake_shader_is_valid_wgsl() {
        crate::renderer::tests::assert_valid_wgsl(CLOUD_BAKE_SHADER);
    }
}
