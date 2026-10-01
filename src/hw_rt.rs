// hw_rt.rs
// Hardware ray-traced sun shadows (wgpu EXPERIMENTAL_RAY_QUERY, e.g. Apple M3+, RTX, RDNA2+): one BLAS per
// chunk mesh (voxel and LOD) and a TLAS over all loaded chunks, rebuilt every frame.
//
// Two steps per frame, at the shadow resolution of rt_blur.rs:
// 1. a G-buffer pass draws the scene with fs_gbuf (shader.wgsl): world position, camera distance, normal
// 2. a compute pass (rt_hw.wgsl) casts one ray per texel toward the sun and writes the same
//    (shadow, distance) texel into the rt_blur target as the ray march, so blur and upsampling are shared.
// Rays are not cast from the scene fragment shader: ray-query code there made every fragment several
// times slower, even where no ray was cast. Without the feature the renderer keeps using the ray march.

use std::iter;
use crate::common::Vertex;

const POS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float; // positions need f32 precision
const NRM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 3x4 row-major

pub struct HwRt {
    pub gbuf_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    compute_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    tlas: wgpu::Tlas,
    capacity: u32,
    used: usize, // TLAS instance slots filled last frame
    size: (u32, u32),
    pub g_pos: wgpu::TextureView,
    pub g_nrm: wgpu::TextureView,
}

impl HwRt {
    pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::EXPERIMENTAL_RAY_QUERY
    }

    pub fn supported(device: &wgpu::Device) -> bool {
        device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    }

    // scene_layout/scene_shader: those of the main pipelines (fs_gbuf lives in shader.wgsl)
    pub fn new(device: &wgpu::Device, scene_layout: &wgpu::PipelineLayout, scene_shader: &wgpu::ShaderModule, size: (u32, u32)) -> Self {
        let gbuf_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HW RT G-Buffer Pipeline"),
            layout: Some(scene_layout),
            vertex: wgpu::VertexState { module: scene_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as _, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 }, wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 }] })] },
            fragment: Some(wgpu::FragmentState { module: scene_shader, entry_point: Some("fs_gbuf"), compilation_options: Default::default(), targets: &[Some(POS_FORMAT.into()), Some(NRM_FORMAT.into())] }),
            // same rasterisation as the ray-march pipeline (rt_blur.rs)
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview_mask: None, cache: None,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("rt_hw"), source: wgpu::ShaderSource::Wgsl(include_str!("rt_hw.wgsl").into()) });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE, count: None,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        };
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_hw_layout"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, count: None, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 } },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, count: None, ty: wgpu::BindingType::AccelerationStructure { vertex_return: false } },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, count: None, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None } },
            ],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HW RT Shadow Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&compute_layout)], immediate_size: 0 })),
            module: &module,
            entry_point: Some("cs_shadow"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("HW RT Params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        let capacity = 1024;
        let (g_pos, g_nrm) = Self::make_gbuf(device, size);
        Self { gbuf_pipeline, compute_pipeline, compute_layout, params_buf, tlas: Self::make_tlas(device, capacity), capacity, used: 0, size, g_pos, g_nrm }
    }

    fn make_tlas(device: &wgpu::Device, capacity: u32) -> wgpu::Tlas {
        device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Scene TLAS"),
            max_instances: capacity,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        })
    }

    fn make_gbuf(device: &wgpu::Device, (width, height): (u32, u32)) -> (wgpu::TextureView, wgpu::TextureView) {
        let tex = |format| device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HW RT G-Buffer"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());
        (tex(POS_FORMAT), tex(NRM_FORMAT))
    }

    // keep the G-buffer at the shadow resolution (rt_blur.rs)
    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if size != self.size {
            self.size = size;
            (self.g_pos, self.g_nrm) = Self::make_gbuf(device, size);
        }
    }

    pub fn gbuf_clear_ops() -> wgpu::Operations<wgpu::Color> {
        wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store }
    }

    // the BLAS of a chunk mesh; v_buf/i_buf need BufferUsages::BLAS_INPUT
    pub fn build_blas(device: &wgpu::Device, queue: &wgpu::Queue, v_buf: &wgpu::Buffer, vertex_count: u32, i_buf: &wgpu::Buffer, index_count: u32) -> wgpu::Blas {
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(index_count),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor { label: Some("Chunk BLAS"), flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build },
            wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("BLAS build") });
        enc.build_acceleration_structures(
            iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: v_buf,
                    first_vertex: 0,
                    vertex_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress, // position is the first field
                    index_buffer: Some(i_buf),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            iter::empty(),
        );
        queue.submit(iter::once(enc.finish()));
        blas
    }

    // rebuild the TLAS over these chunk BLASes; call before the compute pass in the same encoder
    pub fn update<'a>(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, blases: impl Iterator<Item = &'a wgpu::Blas>) {
        let blases: Vec<&wgpu::Blas> = blases.collect();
        if blases.len() as u32 > self.capacity {
            self.capacity = (blases.len() as u32).next_power_of_two();
            self.tlas = Self::make_tlas(device, self.capacity);
            self.used = 0;
        }
        for (i, blas) in blases.iter().enumerate() {
            *self.tlas.get_mut_single(i).unwrap() = Some(wgpu::TlasInstance::new(blas, IDENTITY, 0, 0xff));
        }
        for i in blases.len()..self.used {
            *self.tlas.get_mut_single(i).unwrap() = None;
        }
        self.used = blases.len();
        enc.build_acceleration_structures(iter::empty(), iter::once(&self.tlas));
    }

    // one ray per G-buffer texel; writes (shadow, distance) into `target` (the rt_blur target)
    pub fn trace(&self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, sun_dir: glam::Vec3, timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[sun_dir.x, sun_dir.y, sun_dir.z, 0.0]));
        // targets and TLAS can be replaced (resize, capacity), so the bind group is made per frame
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_hw_bind"),
            layout: &self.compute_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.g_pos) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.g_nrm) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(target) },
                wgpu::BindGroupEntry { binding: 3, resource: self.tlas.as_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.params_buf.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("HW RT Shadow Pass"), timestamp_writes });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(self.size.0.div_ceil(8), self.size.1.div_ceil(8), 1);
    }
}
