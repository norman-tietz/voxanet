// hw_rt.rs
// Hardware ray-traced sun shadows (wgpu EXPERIMENTAL_RAY_QUERY, e.g. Apple M3+, RTX, RDNA2+): one BLAS per
// chunk mesh (voxel and LOD) and a TLAS over all loaded chunks, rebuilt every frame.
//
// Per frame, after rt_blur.rs's G-buffer pass, a compute pass (rt_hw.wgsl) casts one ray per texel toward
// the sun and writes the same (shadow, distance) texel into the rt_blur target as the ray march, so the
// blur and the upsampling are shared.
// Rays are not cast from the scene fragment shader: ray-query code there made every fragment several
// times slower, even where no ray was cast. Without the feature the renderer keeps using the ray march.

use crate::common::Vertex;
use crate::rt_blur::RtBlur;
use std::iter;

const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 3x4 row-major

pub struct HwRt {
    compute_pipeline: wgpu::ComputePipeline,
    compute_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
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
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let capacity = 1024;
        Self {
            compute_pipeline,
            compute_layout,
            params_buf,
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
    pub fn trace(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        shadows: &RtBlur,
        sun_dir: glam::Vec3,
        ao: [f32; 4], // ao.rs AoSettings::uniform: radius, ray count, strength
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::cast_slice(&[
                sun_dir.x, sun_dir.y, sun_dir.z, 0.0, ao[0], ao[1], ao[2], ao[3],
            ]),
        );
        // targets and TLAS can be replaced (resize, capacity), so the bind group is made per frame
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
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HW RT Shadow Pass"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(shadows.size.0.div_ceil(8), shadows.size.1.div_ceil(8), 1);
    }
}
