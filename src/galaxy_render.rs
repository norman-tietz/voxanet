// galaxy_render.rs
// Sub-project #1 (galaxy foundation): a deliberately separate, minimal forward-rendering pass for
// galaxy mode (a starfield background plus flat-shaded icosphere placeholders for the star and
// each planet). Does not touch the existing deferred G-buffer/shadow pipeline in deferred.rs —
// sub-project #2 replaces these placeholder spheres with real terrain-shaped impostors.

use crate::deferred::DEPTH_FORMAT;
use crate::galaxy::{Galaxy, GalaxyFlight, GalaxyPlanet};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

// galaxy mode's shader: the shared atmosphere maths (atmosphere.wgsl) followed by galaxy.wgsl
pub(crate) const GALAXY_SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    "\n",
    include_str!("galaxy.wgsl")
);

const ICOSPHERE_SUBDIVISIONS: u32 = 2;
// the first-person planet camera's 80 degrees (Controller::fov_y): the landing/liftoff handover
// switches between the two cameras, and a different field of view would zoom the view right then
const FOV_Y_RADIANS: f32 = 80.0 * std::f32::consts::PI / 180.0;
const NEAR_PLANE: f32 = 1.0;
const FAR_PLANE: f32 = 200_000.0; // comfortably past the outermost orbit (galaxy.rs)
const MAX_BODIES: usize = MAX_PLANETS + 1; // the star + every planet the galaxy can hold

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
    offset: [f32; 4], // xyz: camera-relative planet centre, w: planet radius (voxel_resolution / 2)
    light_dir: [f32; 4], // direction from this planet toward the star (galaxy space)
    sky_zenith: [f32; 4], // the planet type's AtmosphereDef colours (rgb), as the engine's BiomeUniform
    sky_horizon: [f32; 4],
    space_color: [f32; 4],
    cloud_light: [f32; 4],
    cloud_dark: [f32; 4],
    model: [[f32; 4]; 3], // planet frame -> galaxy space rotation (GalaxyPlanet::orientation), mat3 columns
}

struct PlanetMesh {
    v_buf: wgpu::Buffer,
    i_buf: wgpu::Buffer,
    num_indices: u32,
}

use crate::galaxy::MAX_PLANETS; // galaxy.rs caps generated + added planets at this many

// starting points from the galaxy-terrain-impostors design discussion's faceting estimate
// (~10-13x radius before individual facets become visually obvious at this project's FOV);
// tune after a visual check, same as every other distance-based constant in this codebase
const LOD_MEDIUM_DISTANCE_MULTIPLIER: f32 = 13.0;
const LOD_NEAR_DISTANCE_MULTIPLIER: f32 = 6.0;

fn subdivision_for_distance(distance: f32, radius: f32) -> u32 {
    if distance > LOD_MEDIUM_DISTANCE_MULTIPLIER * radius {
        2
    } else if distance > LOD_NEAR_DISTANCE_MULTIPLIER * radius {
        4
    } else {
        5
    }
}

// where the galaxy is seen from: a position in galaxy space, an orientation (looking down local -Z)
// and a vertical field of view. Galaxy mode uses the flight; planet mode uses the planet camera
// converted into galaxy space, so the backdrop behind the voxel world lines up with it exactly.
#[derive(Clone, Copy, Debug)]
pub struct GalaxyCamera {
    pub position: DVec3,
    pub rotation: Quat,
    pub fov_y: f32,
}

impl GalaxyCamera {
    pub fn from_flight(flight: &GalaxyFlight) -> Self {
        Self {
            position: flight.position,
            rotation: flight.rotation,
            fov_y: FOV_Y_RADIANS,
        }
    }

    // galaxy mode's camera, whichever frame the flight is in (galaxy space, or captured by a planet)
    pub fn from_flight_in_frame(
        flight: &GalaxyFlight,
        frame: crate::galaxy::FlightFrame,
        galaxy: &Galaxy,
        t: f64,
    ) -> Self {
        match frame {
            crate::galaxy::FlightFrame::Free => Self::from_flight(flight),
            crate::galaxy::FlightFrame::Captured(i) => Self::on_galaxy_planet(
                &galaxy.planets[i],
                flight.position.as_vec3(),
                flight.rotation,
                FOV_Y_RADIANS,
                t,
            ),
        }
    }

    // the planet camera (eye and orientation in the planet frame) on a galaxy planet
    pub fn on_galaxy_planet(
        planet: &GalaxyPlanet,
        eye: Vec3,
        rotation: Quat,
        fov_y: f32,
        t: f64,
    ) -> Self {
        Self {
            position: planet.from_planet_frame(eye.as_dvec3(), t),
            rotation: planet.rotation_from_planet_frame(rotation, t),
            fov_y,
        }
    }
}

// which parts of the galaxy to draw: everything in galaxy mode; behind a planet's voxel world
// everything except that planet (the voxel engine draws it)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GalaxyContent {
    Everything,
    AllButPlanet(usize),
}

impl GalaxyContent {
    pub fn draws_planet(self, i: usize) -> bool {
        match self {
            Self::Everything => true,
            Self::AllButPlanet(skip) => i != skip,
        }
    }
}

// what Renderer::render draws behind the voxel world before anything else
pub struct Backdrop<'a> {
    pub galaxy: &'a Galaxy,
    pub camera: GalaxyCamera,
    pub content: GalaxyContent,
    pub t: f64,
}

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
    atmosphere_pipeline: wgpu::RenderPipeline,
    planet_uniform_buf: wgpu::Buffer,
    planet_uniform_bind: wgpu::BindGroup,
    planet_uniform_stride: u64,
    planet_meshes: std::collections::HashMap<(usize, u32), PlanetMesh>, // (planet index, subdivision)
    near_impostor: Option<(usize, PlanetMesh)>, // (planet index, mesh) — see set_near_impostor
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
            source: wgpu::ShaderSource::Wgsl(GALAXY_SHADER.into()),
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
                // no culling, like the voxel engine's own geometry pass: the near impostor is the engine's LOD
                // meshes, whose triangle winding isn't consistent (back-face culling punched holes at
                // cube-face corners and seams); the depth test hides the far side of the closed sphere
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // atmosphere shell: the star's unit icosphere scaled per planet in vs_atmosphere; back faces
        // only (so it also renders with the camera inside it), depth-tested against the planets but not
        // writing depth, premultiplied over whatever is behind
        let atmosphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Galaxy Atmosphere Pipeline"),
            layout: Some(&planet_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_atmosphere"),
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
                entry_point: Some("fs_atmosphere"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
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
            atmosphere_pipeline,
            planet_uniform_buf,
            planet_uniform_bind,
            planet_uniform_stride,
            planet_meshes: std::collections::HashMap::new(),
            near_impostor: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &GalaxyCamera,
        content: GalaxyContent,
        galaxy: &Galaxy,
        t: f64,
        screen: (f32, f32),
    ) {
        let proj = glam::camera::rh::proj::directx::perspective(
            camera.fov_y,
            screen.0 / screen.1,
            NEAR_PLANE,
            FAR_PLANE,
        );
        let view = Mat4::from_quat(camera.rotation).inverse();
        let view_proj = proj * view;

        let ray_dirs = Self::ray_dirs(camera.rotation, camera.fov_y, screen.0 / screen.1);
        queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[GalaxyCameraUniform {
                view_proj: view_proj.to_cols_array(),
                // z: cloud animation time, wrapped like the engine's GlobalUniform.screen.w
                screen: [screen.0, screen.1, (t % 3600.0) as f32, 0.0],
                ray_dirs,
            }]),
        );

        let mut bodies = Vec::with_capacity(1 + galaxy.planets.len());
        {
            // the star is always part of the view (only planets can be left out)
            let star_camera_relative = (galaxy.star.position() - camera.position).as_vec3();
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
        }
        debug_assert!(
            bodies.len() <= MAX_BODIES,
            "galaxy has more bodies than the storage buffer was sized for"
        );
        if !bodies.is_empty() {
            queue.write_buffer(&self.body_buf, 0, bytemuck::cast_slice(&bodies));
        }

        let mut planet_uniforms = Vec::with_capacity(galaxy.planets.len());
        for (i, p) in galaxy.planets.iter().enumerate() {
            let body_pos = p.position_at(t);
            let camera_relative = (body_pos - camera.position).as_vec3(); // already computed here
                                                                          // every planet keeps its uniform slot (slot i = planet i); only drawn ones need a mesh
            if content.draws_planet(i) && !self.has_near_impostor(i) {
                let subdivision = subdivision_for_distance(camera_relative.length(), p.radius);
                self.ensure_planet_mesh(device, i, p, subdivision);
            }
            let light_dir = (-body_pos).normalize_or_zero().as_vec3();
            let atmosphere = p.planet_type.def().atmosphere;
            let model = glam::Mat3::from_quat(p.orientation(t));
            let v4 = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
            planet_uniforms.push(GalaxyPlanetUniform {
                offset: [
                    camera_relative.x,
                    camera_relative.y,
                    camera_relative.z,
                    p.voxel_resolution() as f32 / 2.0,
                ],
                light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
                sky_zenith: v4(atmosphere.sky_zenith),
                sky_horizon: v4(atmosphere.sky_horizon_warm),
                space_color: v4(atmosphere.space_color),
                cloud_light: v4(atmosphere.cloud_light),
                cloud_dark: v4(atmosphere.cloud_dark),
                model: [
                    model.x_axis.extend(0.0).to_array(),
                    model.y_axis.extend(0.0).to_array(),
                    model.z_axis.extend(0.0).to_array(),
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

            if !bodies.is_empty() {
                pass.set_pipeline(&self.body_pipeline);
                pass.set_vertex_buffer(0, self.v_buf.slice(..));
                pass.set_index_buffer(self.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.num_indices, 0, 0..(bodies.len() as u32));
            }

            pass.set_pipeline(&self.planet_pipeline);
            pass.set_bind_group(0, &self.camera_bind, &[]);
            for (i, p) in galaxy.planets.iter().enumerate() {
                if !content.draws_planet(i) {
                    continue;
                }
                let body_pos = p.position_at(t);
                let camera_relative = (body_pos - camera.position).as_vec3();
                let subdivision = subdivision_for_distance(camera_relative.length(), p.radius);
                let mesh = match &self.near_impostor {
                    Some((n, near)) if *n == i => Some(near),
                    _ => self.planet_meshes.get(&(i, subdivision)),
                };
                if let Some(mesh) = mesh {
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

            // atmosphere shells last, over the planets and everything behind them
            pass.set_pipeline(&self.atmosphere_pipeline);
            pass.set_vertex_buffer(0, self.v_buf.slice(..));
            pass.set_index_buffer(self.i_buf.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..galaxy.planets.len() {
                if !content.draws_planet(i) {
                    continue;
                }
                pass.set_bind_group(
                    1,
                    &self.planet_uniform_bind,
                    &[(i as u64 * self.planet_uniform_stride) as u32],
                );
                pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // replaces planet `planet_index`'s noise impostor with the voxel engine's own distant-terrain
    // mesh (galaxy_terrain::near_impostor_mesh) once that planet is baked; one planet at a time
    pub fn set_near_impostor(
        &mut self,
        device: &wgpu::Device,
        planet_index: usize,
        verts: &[crate::common::Vertex],
        indices: &[u32],
    ) {
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Galaxy Near Impostor Vertices"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Galaxy Near Impostor Indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.near_impostor = Some((
            planet_index,
            PlanetMesh {
                v_buf,
                i_buf,
                num_indices: indices.len() as u32,
            },
        ));
    }

    fn has_near_impostor(&self, i: usize) -> bool {
        matches!(self.near_impostor, Some((n, _)) if n == i)
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

    // GalaxyPlanetUniform must mirror galaxy.wgsl's PlanetUniform: 7 vec4s + a mat3x3 (3 × 16 bytes)
    #[test]
    fn planet_uniform_matches_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<GalaxyPlanetUniform>(), 7 * 16 + 3 * 16);
    }

    #[test]
    fn galaxy_shader_is_valid_wgsl() {
        crate::renderer::tests::assert_valid_wgsl(GALAXY_SHADER);
    }

    // the landing handover switches from galaxy mode's camera to the first-person planet camera:
    // a different field of view would zoom the planet visibly at that moment
    #[test]
    fn galaxy_mode_uses_the_first_person_planet_cameras_field_of_view() {
        let mut controller = crate::controller::Controller::new();
        controller.first_person = true;
        assert!((FOV_Y_RADIANS - controller.fov_y()).abs() < 1e-6);
    }

    #[test]
    fn camera_from_a_captured_flight_is_placed_in_galaxy_space() {
        let g = Galaxy::generate(1);
        let mut flight = GalaxyFlight::new(glam::DVec3::new(0.0, 600.0, 0.0)); // planet frame
        flight.rotation = Quat::from_axis_angle(Vec3::X, -1.0);
        let t = 21.0;
        let cam = GalaxyCamera::from_flight_in_frame(
            &flight,
            crate::galaxy::FlightFrame::Captured(0),
            &g,
            t,
        );
        let p = &g.planets[0];
        assert!((cam.position - p.from_planet_frame(flight.position, t)).length() < 1e-3);
        assert!(
            cam.rotation
                .dot(p.rotation_from_planet_frame(flight.rotation, t))
                .abs()
                > 1.0 - 1e-6
        );
        assert_eq!(cam.fov_y, FOV_Y_RADIANS);
        let free =
            GalaxyCamera::from_flight_in_frame(&flight, crate::galaxy::FlightFrame::Free, &g, t);
        assert_eq!(free.position, flight.position);
    }

    #[test]
    fn content_selects_star_and_planets() {
        assert!(GalaxyContent::Everything.draws_planet(3));
        assert!(GalaxyContent::AllButPlanet(2).draws_planet(1));
        assert!(!GalaxyContent::AllButPlanet(2).draws_planet(2));
    }

    // standing on a galaxy planet, the backdrop camera sits where the planet frame puts the eye in
    // galaxy space, turned the same way
    #[test]
    fn camera_on_a_galaxy_planet_is_placed_in_galaxy_space() {
        let g = Galaxy::generate(1);
        let p = &g.planets[0];
        let eye = Vec3::new(0.0, 175.0, 0.0);
        let rot = Quat::from_axis_angle(Vec3::X, 0.4);
        let t = 42.0;
        let cam = GalaxyCamera::on_galaxy_planet(p, eye, rot, 1.2, t);
        assert!((cam.position - p.from_planet_frame(eye.as_dvec3(), t)).length() < 1e-6);
        assert!(cam.rotation.dot(p.rotation_from_planet_frame(rot, t)).abs() > 1.0 - 1e-6);
        assert_eq!(cam.fov_y, 1.2);
    }

    #[test]
    fn camera_from_flight_uses_galaxy_modes_fov() {
        let flight = GalaxyFlight::new(glam::DVec3::new(1.0, 2.0, 3.0));
        let cam = GalaxyCamera::from_flight(&flight);
        assert_eq!(cam.position, flight.position);
        assert_eq!(cam.fov_y, FOV_Y_RADIANS);
    }

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

    #[test]
    fn subdivision_is_coarsest_far_away() {
        assert_eq!(subdivision_for_distance(2000.0, 100.0), 2); // 20x radius
    }

    #[test]
    fn subdivision_steps_up_at_the_medium_threshold() {
        assert_eq!(subdivision_for_distance(1200.0, 100.0), 4); // 12x radius, just inside 13x
    }

    #[test]
    fn subdivision_is_finest_up_close() {
        assert_eq!(subdivision_for_distance(500.0, 100.0), 5); // 5x radius, inside 6x
    }

    #[test]
    fn subdivision_thresholds_scale_with_radius() {
        // same distance, bigger planet: a 250-radius planet at 2000 units is only 8x its own
        // radius (medium tier), while a 40-radius planet at the same distance is 50x (coarsest)
        assert_eq!(subdivision_for_distance(2000.0, 250.0), 4);
        assert_eq!(subdivision_for_distance(2000.0, 40.0), 2);
    }

    // composed regression: every link in noise_seed -> mesh -> cache -> LOD-tier has its own
    // isolated test, but nothing exercised Galaxy::generate's *real* planets end to end before
    // this — fixed tiers are used directly rather than going through distance mapping, since
    // subdivision_for_distance already has its own coverage above.
    #[test]
    fn seeded_planets_produce_distinct_geometry_across_tiers_and_planets() {
        let galaxy = Galaxy::generate(1);
        let p0 = &galaxy.planets[0];
        let p1 = &galaxy.planets[1];

        let (v2, _) = crate::galaxy_terrain::generate_planet_mesh(p0, 2);
        let (v4, _) = crate::galaxy_terrain::generate_planet_mesh(p0, 4);
        let (v5, _) = crate::galaxy_terrain::generate_planet_mesh(p0, 5);
        assert!(
            v2.len() < v4.len() && v4.len() < v5.len(),
            "tiers should have strictly increasing vertex counts"
        );

        let (v0, _) = crate::galaxy_terrain::generate_planet_mesh(p0, 2);
        let (v1, _) = crate::galaxy_terrain::generate_planet_mesh(p1, 2);
        let differs = v0.iter().zip(v1.iter()).any(|(a, b)| {
            let ra = Vec3::from_array(a.pos).length();
            let rb = Vec3::from_array(b.pos).length();
            (ra - rb).abs() > 1e-3
        });
        assert!(
            differs,
            "planets 0 and 1 (different noise seeds, different radii) should not produce identical geometry"
        );
    }
}
