// hw_rt.rs
// Hardware ray-traced sun shadows (wgpu EXPERIMENTAL_RAY_QUERY, e.g. Apple M3+, RTX, RDNA2+): one BLAS per
// chunk mesh (voxel and LOD) and a TLAS over all loaded chunks, rebuilt every frame.
//
// Per frame, after rt_blur.rs's G-buffer pass, a compute pass (rt_hw.wgsl) casts one ray per texel toward
// the sun and writes the same (shadow, distance) texel into the rt_blur target as the ray march, so the
// blur and the upsampling are shared. In the same pass the progressive AO (ao.rs) samples one texel per
// 4x4 block, accumulates into a ping-pong pair at shadow resolution and fills RtBlur::ao_out.
// Rays are not cast from the scene fragment shader: ray-query code there made every fragment several
// times slower, even where no ray was cast. Without the feature the renderer keeps using the ray march.

use crate::common::Vertex;
use crate::rt_blur::RtBlur;
use bytemuck::{Pod, Zeroable};
use std::iter;

// rt_hw.wgsl AoParams: one frame of the progressive AO
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AoFrameParams {
    pub ao: [f32; 4], // ao::AoSettings::uniform: radius, rays per sample, strength, fade end
    pub frame: u32,   // running frame index (ray rotation)
    pub frames: u32,  // since the last reset (0: start over)
    pub cap: u32,     // accumulation cap per texel
    pub block: u32,   // ao::BLOCK
    pub offset: [u32; 2], // ao::sample_offset(frame)
    pub fill_radius: f32, // ao::fill_radius_texels(frames)
    pub focal: f32,   // shadow-target pixels per world unit at distance 1
}

// the AO stages (rt_hw.wgsl entry point, group 1 bindings: storage writes, texture reads)
const AO_STAGES: [(&str, &[u32], &[u32]); 4] = [
    ("cs_ao_sample", &[0], &[]),
    ("cs_ao_accumulate", &[3], &[1, 2]),
    ("cs_ao_fill_h", &[5], &[4]),
    ("cs_ao_fill_v", &[7], &[6]),
];
const AO_PARAMS_BINDING: u32 = 8;

// the progressive AO's own textures, at the shadow targets' size
struct AoTargets {
    size: (u32, u32),
    samples: wgpu::TextureView,  // (ao, dist) per block
    acc: [wgpu::TextureView; 2], // (mean, count, dist) ping-pong
    fill_tmp: wgpu::TextureView, // (sum, weight, dist)
}

const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 3x4 row-major

pub struct HwRt {
    compute_pipeline: wgpu::ComputePipeline,
    compute_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    ao_pipelines: Vec<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>, // AO_STAGES order
    ao_params: wgpu::Buffer,
    ao: Option<AoTargets>,
    ao_flip: usize, // acc[ao_flip] is this frame's output, the other one the previous frame's
    tlas: wgpu::Tlas,
    capacity: u32,
    used: usize, // TLAS instance slots filled last frame
}

impl HwRt {
    pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::EXPERIMENTAL_RAY_QUERY
    }

    pub fn supported(device: &wgpu::Device) -> bool {
        device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_hw"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rt_hw.wgsl").into()),
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        };
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_hw_layout"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HW RT Shadow Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&compute_layout)],
                    immediate_size: 0,
                }),
            ),
            module: &module,
            entry_point: Some("cs_shadow"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HW RT Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // the AO stages: group 0 shared with the shadow, group 1 each stage's own textures + params
        let ao_pipelines = AO_STAGES
            .iter()
            .map(|&(entry, writes, reads)| {
                let mut entries: Vec<_> = writes
                    .iter()
                    .map(|&binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        count: None,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                    })
                    .collect();
                entries.extend(reads.iter().map(|&binding| tex_entry(binding)));
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: AO_PARAMS_BINDING,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                });
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(entry),
                    entries: &entries,
                });
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[Some(&compute_layout), Some(&layout)],
                            immediate_size: 0,
                        }),
                    ),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (pipeline, layout)
            })
            .collect();
        let ao_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AO Params"),
            size: std::mem::size_of::<AoFrameParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let capacity = 1024;
        Self {
            compute_pipeline,
            compute_layout,
            params_buf,
            ao_pipelines,
            ao_params,
            ao: None,
            ao_flip: 0,
            tlas: Self::make_tlas(device, capacity),
            capacity,
            used: 0,
        }
    }

    fn make_tlas(device: &wgpu::Device, capacity: u32) -> wgpu::Tlas {
        device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Scene TLAS"),
            max_instances: capacity,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        })
    }

    // the BLAS of a chunk mesh; v_buf/i_buf need BufferUsages::BLAS_INPUT
    pub fn build_blas(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        v_buf: &wgpu::Buffer,
        vertex_count: u32,
        i_buf: &wgpu::Buffer,
        index_count: u32,
    ) -> wgpu::Blas {
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(index_count),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("Chunk BLAS"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BLAS build"),
        });
        enc.build_acceleration_structures(
            iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: v_buf,
                        first_vertex: 0,
                        vertex_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress, // position is the first field
                        index_buffer: Some(i_buf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            iter::empty(),
        );
        queue.submit(iter::once(enc.finish()));
        blas
    }

    // rebuild the TLAS over these chunk BLASes; call before the compute pass in the same encoder
    pub fn update<'a>(
        &mut self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        blases: impl Iterator<Item = &'a wgpu::Blas>,
    ) {
        let blases: Vec<&wgpu::Blas> = blases.collect();
        if blases.len() as u32 > self.capacity {
            self.capacity = (blases.len() as u32).next_power_of_two();
            self.tlas = Self::make_tlas(device, self.capacity);
            self.used = 0;
        }
        for (i, blas) in blases.iter().enumerate() {
            *self.tlas.get_mut_single(i).unwrap() =
                Some(wgpu::TlasInstance::new(blas, IDENTITY, 0, 0xff));
        }
        for i in blases.len()..self.used {
            *self.tlas.get_mut_single(i).unwrap() = None;
        }
        self.used = blases.len();
        enc.build_acceleration_structures(iter::empty(), iter::once(&self.tlas));
    }

    // one ray per G-buffer texel of `shadows`; writes (shadow, distance) into its target
    // the shadow term, and the progressive AO when `ao` is given (it writes RtBlur::ao_out; without it the
    // caller clears that to 1)
    #[allow(clippy::too_many_arguments)]
    pub fn trace(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        shadows: &RtBlur,
        sun_dir: glam::Vec3,
        ao: Option<AoFrameParams>,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::cast_slice(&[sun_dir.x, sun_dir.y, sun_dir.z, 0.0]),
        );
        // targets and TLAS can be replaced (resize, capacity), so the bind groups are made per frame
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_hw_bind"),
            layout: &self.compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadows.g_pos),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadows.g_nrm),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadows.target),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });
        let ao_binds = ao.map(|params| {
            queue.write_buffer(&self.ao_params, 0, bytemuck::bytes_of(&params));
            self.ao_flip ^= 1;
            self.ensure_ao_targets(device, shadows.size);
            let t = self.ao.as_ref().unwrap();
            let (cur, prev) = (&t.acc[self.ao_flip], &t.acc[self.ao_flip ^ 1]);
            // per stage: the views for its group 1 bindings, in AO_STAGES order (writes, then reads)
            let views: [&[&wgpu::TextureView]; 4] = [
                &[&t.samples],
                &[cur, &t.samples, prev],
                &[&t.fill_tmp, cur],
                &[&shadows.ao_out, &t.fill_tmp],
            ];
            let binds: Vec<_> = AO_STAGES
                .iter()
                .zip(views)
                .zip(&self.ao_pipelines)
                .map(|((&(entry, writes, reads), views), (_, layout))| {
                    let mut entries: Vec<_> = writes
                        .iter()
                        .chain(reads)
                        .zip(views)
                        .map(|(&binding, view)| wgpu::BindGroupEntry {
                            binding,
                            resource: wgpu::BindingResource::TextureView(view),
                        })
                        .collect();
                    entries.push(wgpu::BindGroupEntry {
                        binding: AO_PARAMS_BINDING,
                        resource: self.ao_params.as_entire_binding(),
                    });
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(entry),
                        layout,
                        entries: &entries,
                    })
                })
                .collect();
            (binds, params.block)
        });

        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HW RT Shadow + AO Pass"),
            timestamp_writes,
        });
        let groups = |w: u32, h: u32| (w.div_ceil(8), h.div_ceil(8));
        let (gx, gy) = groups(shadows.size.0, shadows.size.1);
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
        if let Some((binds, block)) = &ao_binds {
            // each dispatch is its own usage scope, so a stage reads what the one before wrote
            for (i, ((pipeline, _), bind1)) in self.ao_pipelines.iter().zip(binds).enumerate() {
                let (w, h) = if i == 0 {
                    (
                        shadows.size.0.div_ceil(*block),
                        shadows.size.1.div_ceil(*block),
                    )
                } else {
                    shadows.size
                };
                let (gx, gy) = groups(w, h);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, bind1, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
    }

    // the AO textures for shadow targets of `size`, recreated when that changes
    fn ensure_ao_targets(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.ao.as_ref().is_none_or(|t| t.size != size) {
            let tex = |w: u32, h: u32, label: &str| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: w.max(1),
                            height: h.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            let b = crate::ao::BLOCK;
            self.ao = Some(AoTargets {
                size,
                samples: tex(size.0.div_ceil(b), size.1.div_ceil(b), "AO Samples"),
                acc: [
                    tex(size.0, size.1, "AO Acc A"),
                    tex(size.0, size.1, "AO Acc B"),
                ],
                fill_tmp: tex(size.0, size.1, "AO Fill"),
            });
        }
    }
}
