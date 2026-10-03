// galaxy_render.rs
// Sub-project #1 (galaxy foundation): a deliberately separate, minimal forward-rendering pass for
// galaxy mode (a starfield background plus flat-shaded icosphere placeholders for the star and
// each planet). Does not touch the existing deferred G-buffer/shadow pipeline in deferred.rs —
// sub-project #2 replaces these placeholder spheres with real terrain-shaped impostors.

use crate::deferred::DEPTH_FORMAT;
use crate::galaxy::{Galaxy, GalaxyFlight};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

const ICOSPHERE_SUBDIVISIONS: u32 = 2;
const FOV_Y_RADIANS: f32 = 1.0;
const NEAR_PLANE: f32 = 1.0;
const FAR_PLANE: f32 = 200_000.0; // comfortably past the outermost orbit (galaxy.rs)
const MAX_BODIES: usize = 16; // star + up to 15 planets; galaxy.rs generates 7 today

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GalaxyCameraUniform {
    view_proj: [f32; 16],
    screen: [f32; 4],
    ray_dirs: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GalaxyBodyUniform {
    offset: [f32; 4],
    color: [f32; 4],
    light_dir: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GalaxyPlanetUniform {
    offset: [f32; 4],
    light_dir: [f32; 4],
    atmosphere_color: [f32; 4],
}

struct PlanetMesh {
    v_buf: wgpu::Buffer,
    i_buf: wgpu::Buffer,
    num_indices: u32,
}

const MAX_PLANETS: usize = MAX_BODIES - 1; // the star takes one conceptual slot; planets no longer
                                           // share the body path, but this keeps one shared cap
const ATMOSPHERE_GLOW_STRENGTH: f32 = 0.6;

pub struct GalaxyRenderer {
    camera_buf: wgpu::Buffer,
    camera_bind: wgpu::BindGroup,
    body_buf: wgpu::Buffer,
    body_bind: wgpu::BindGroup,
    body_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    v_buf: wgpu::Buffer,
    i_buf: wgpu::Buffer,
    num_indices: u32,
    planet_pipeline: wgpu::RenderPipeline,
    planet_uniform_buf: wgpu::Buffer,
    planet_uniform_bind: wgpu::BindGroup,
    planet_uniform_stride: u64,
    planet_meshes: std::collections::HashMap<(usize, u32), PlanetMesh>, // (planet index, subdivision)
}

impl GalaxyRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let (verts, indices) = crate::icosphere::generate(ICOSPHERE_SUBDIVISIONS);
        let vert_data: Vec<[f32; 3]> = verts.iter().map(|v| v.to_array()).collect();
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Galaxy Icosphere Vertices"),
            contents: bytemuck::cast_slice(&vert_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Galaxy Icosphere Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("galaxy_camera_layout"),
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
        });
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Galaxy Camera Uniform"),
            size: std::mem::size_of::<GalaxyCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("galaxy_camera_bind"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let body_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("galaxy_body_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let body_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Galaxy Body Storage"),
            size: (std::mem::size_of::<GalaxyBodyUniform>() * MAX_BODIES) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("galaxy_body_bind"),
            layout: &body_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: body_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("galaxy.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("galaxy.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("galaxy_pipeline_layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&body_layout)],
            immediate_size: 0,
        });

        let color_target = wgpu::ColorTargetState {
            format: config.format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };
        let depth_stencil = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        };

        let body_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Galaxy Body Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_body"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_body"),
                compilation_options: Default::default(),
                targets: &[Some(color_target.clone())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let planet_uniform_stride = {
            let min_align = device.limits().min_uniform_buffer_offset_alignment as u64;
            let unaligned = std::mem::size_of::<GalaxyPlanetUniform>() as u64;
            unaligned.div_ceil(min_align) * min_align
        };
        let planet_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("galaxy_planet_uniform_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let planet_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Galaxy Planet Uniform"),
            size: planet_uniform_stride * MAX_PLANETS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let planet_uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("galaxy_planet_uniform_bind"),
            layout: &planet_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &planet_uniform_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<GalaxyPlanetUniform>() as u64
                    ),
                }),
            }],
        });

        let planet_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("galaxy_planet_pipeline_layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&planet_uniform_layout)],
            immediate_size: 0,
        });

        let planet_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Galaxy Planet Pipeline"),
            layout: Some(&planet_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_planet"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::common::Vertex>() as u64,
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
                module: &shader,
                entry_point: Some("fs_planet"),
                compilation_options: Default::default(),
                targets: &[Some(color_target.clone())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // background: depth test on but no depth write, drawn first so it never occludes bodies
        // and is itself never occluded by the depth clear (compare Always, matches a skybox)
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Galaxy Background Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_background"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_background"),
                compilation_options: Default::default(),
                targets: &[Some(color_target)],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                ..depth_stencil
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            camera_buf,
            camera_bind,
            body_buf,
            body_bind,
            body_pipeline,
            background_pipeline,
            v_buf,
            i_buf,
            num_indices: indices.len() as u32,
            planet_pipeline,
            planet_uniform_buf,
            planet_uniform_bind,
            planet_uniform_stride,
            planet_meshes: std::collections::HashMap::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        flight: &GalaxyFlight,
        galaxy: &Galaxy,
        t: f64,
        screen: (f32, f32),
    ) {
        let proj = glam::camera::rh::proj::directx::perspective(
            FOV_Y_RADIANS,
            screen.0 / screen.1,
            NEAR_PLANE,
            FAR_PLANE,
        );
        let view = Mat4::from_quat(flight.rotation).inverse();
        let view_proj = proj * view;

        let ray_dirs = Self::ray_dirs(flight.rotation, FOV_Y_RADIANS, screen.0 / screen.1);
        queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[GalaxyCameraUniform {
                view_proj: view_proj.to_cols_array(),
                screen: [screen.0, screen.1, 0.0, 0.0],
                ray_dirs,
            }]),
        );

        let mut bodies = Vec::with_capacity(1 + galaxy.planets.len());
        let star_camera_relative = (-flight.position).as_vec3();
        bodies.push(GalaxyBodyUniform {
            offset: [
                star_camera_relative.x,
                star_camera_relative.y,
                star_camera_relative.z,
                galaxy.star.radius as f32,
            ],
            color: [1.6, 1.4, 0.9, 1.0],
            light_dir: [0.0, 1.0, 0.0, 0.0],
        });
        debug_assert!(
            bodies.len() <= MAX_BODIES,
            "galaxy has more bodies than the storage buffer was sized for"
        );
        queue.write_buffer(&self.body_buf, 0, bytemuck::cast_slice(&bodies));

        let mut planet_uniforms = Vec::with_capacity(galaxy.planets.len());
        for (i, p) in galaxy.planets.iter().enumerate() {
            self.ensure_planet_mesh(device, i, p, ICOSPHERE_SUBDIVISIONS);
            let body_pos = p.position_at(t);
            let camera_relative = (body_pos - flight.position).as_vec3();
            let light_dir = (-body_pos).normalize_or_zero().as_vec3();
            let atmosphere = p.planet_type.def().atmosphere;
            planet_uniforms.push(GalaxyPlanetUniform {
                offset: [camera_relative.x, camera_relative.y, camera_relative.z, 0.0],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                atmosphere_color: [
                    atmosphere.sky_zenith[0],
                    atmosphere.sky_zenith[1],
                    atmosphere.sky_zenith[2],
                    ATMOSPHERE_GLOW_STRENGTH,
                ],
            });
        }
        // every slot written before the pass begins — see this task's write-ordering design note
        for (i, u) in planet_uniforms.iter().enumerate() {
            queue.write_buffer(
                &self.planet_uniform_buf,
                i as u64 * self.planet_uniform_stride,
                bytemuck::cast_slice(&[*u]),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Galaxy Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_bind_group(0, &self.camera_bind, &[]);
            pass.set_bind_group(1, &self.body_bind, &[]);

            pass.set_pipeline(&self.background_pipeline);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&self.body_pipeline);
            pass.set_vertex_buffer(0, self.v_buf.slice(..));
            pass.set_index_buffer(self.i_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.num_indices, 0, 0..(bodies.len() as u32));

            pass.set_pipeline(&self.planet_pipeline);
            pass.set_bind_group(0, &self.camera_bind, &[]);
            for i in 0..galaxy.planets.len() {
                if let Some(mesh) = self.planet_meshes.get(&(i, ICOSPHERE_SUBDIVISIONS)) {
                    pass.set_bind_group(
                        1,
                        &self.planet_uniform_bind,
                        &[(i as u64 * self.planet_uniform_stride) as u32],
                    );
                    pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
                    pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    fn ensure_planet_mesh(
        &mut self,
        device: &wgpu::Device,
        index: usize,
        planet: &crate::galaxy::GalaxyPlanet,
        subdivision: u32,
    ) {
        if self.planet_meshes.contains_key(&(index, subdivision)) {
            return;
        }
        let (verts, indices) = crate::galaxy_terrain::generate_planet_mesh(planet, subdivision);
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Galaxy Planet Vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Galaxy Planet Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.planet_meshes.insert(
            (index, subdivision),
            PlanetMesh {
                v_buf,
                i_buf,
                num_indices: indices.len() as u32,
            },
        );
    }

    fn ray_dirs(rotation: Quat, fov_y: f32, aspect: f32) -> [[f32; 4]; 3] {
        let tan_half_fov = (fov_y * 0.5).tan();
        let forward = rotation * Vec3::NEG_Z;
        let right = rotation * Vec3::X * tan_half_fov * aspect;
        let up = rotation * Vec3::Y * tan_half_fov;
        let top_left = forward - right + up;
        let col1 = right * 2.0;
        let col2 = -up * 2.0;
        [
            [top_left.x, top_left.y, top_left.z, 0.0],
            [col1.x, col1.y, col1.z, 0.0],
            [col2.x, col2.y, col2.z, 0.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galaxy::Galaxy;

    // integration: Galaxy + GalaxyFlight together, at the same default entry point `Game::new` /
    // the `/galaxy enter` handler use (src/main.rs), should keep every body within the render
    // pipeline's near/far planes and within the body-storage buffer's capacity.
    #[test]
    // clippy's int_plus_one suggests `planets.len() < MAX_BODIES`, which is equivalent but hides
    // the "+1 for the star" reasoning the assertion is meant to document; keep the explicit form.
    #[allow(clippy::int_plus_one)]
    fn galaxy_and_flight_keep_every_body_within_render_planes() {
        let galaxy = Galaxy::generate(1);
        let flight = GalaxyFlight::new(glam::DVec3::new(0.0, 0.0, 120_000.0));

        let star_distance = flight.position.length();
        assert!(
            star_distance > NEAR_PLANE as f64 && star_distance < FAR_PLANE as f64,
            "star distance {star_distance} not within ({NEAR_PLANE}, {FAR_PLANE})"
        );

        for (i, p) in galaxy.planets.iter().enumerate() {
            let distance = (p.position_at(0.0) - flight.position).length();
            assert!(
                distance > NEAR_PLANE as f64 && distance < FAR_PLANE as f64,
                "planet {i} distance {distance} not within ({NEAR_PLANE}, {FAR_PLANE})"
            );
        }

        assert!(
            galaxy.planets.len() + 1 <= MAX_BODIES,
            "star + planets ({}) exceed MAX_BODIES ({MAX_BODIES})",
            galaxy.planets.len() + 1
        );
    }

    // ray_dirs packs the camera frustum as a corner (ray_dirs[0]) plus two edge vectors
    // (ray_dirs[1]/[2]) so the shader can interpolate `ray_dirs[0] + uv.x*ray_dirs[1] +
    // uv.y*ray_dirs[2]` per pixel (same scheme as GlobalUniform.ray_dirs / fs_light). At the
    // screen center (uv = 0.5, 0.5) that interpolation must reduce to the forward direction.
    #[test]
    fn ray_dirs_center_matches_forward_at_identity_rotation() {
        let ray_dirs = GalaxyRenderer::ray_dirs(Quat::IDENTITY, 1.0, 16.0 / 9.0);
        let top_left = Vec3::from_slice(&ray_dirs[0][..3]);
        let col1 = Vec3::from_slice(&ray_dirs[1][..3]);
        let col2 = Vec3::from_slice(&ray_dirs[2][..3]);
        let center = top_left + 0.5 * col1 + 0.5 * col2;
        let forward = Quat::IDENTITY * Vec3::NEG_Z;
        assert!(
            (center - forward).length() < 1e-5,
            "center ray {center:?} should match forward {forward:?}"
        );
    }

    #[test]
    fn ray_dirs_center_matches_forward_under_rotation() {
        let rotation = Quat::from_axis_angle(Vec3::Y, 1.2) * Quat::from_axis_angle(Vec3::X, -0.4);
        let ray_dirs = GalaxyRenderer::ray_dirs(rotation, 1.0, 16.0 / 9.0);
        let top_left = Vec3::from_slice(&ray_dirs[0][..3]);
        let col1 = Vec3::from_slice(&ray_dirs[1][..3]);
        let col2 = Vec3::from_slice(&ray_dirs[2][..3]);
        let center = top_left + 0.5 * col1 + 0.5 * col2;
        let forward = rotation * Vec3::NEG_Z;
        assert!(
            (center - forward).length() < 1e-5,
            "center ray {center:?} should match forward {forward:?}"
        );
    }
}
