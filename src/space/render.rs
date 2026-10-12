// render.rs
// Sub-project #1 (system foundation): a deliberately separate, minimal forward-rendering pass for
// Space mode (a starfield background plus flat-shaded icosphere placeholders for the star and
// each planet). Does not touch the existing deferred G-buffer/shadow pipeline in deferred.rs —
// sub-project #2 replaces these placeholder spheres with real terrain-shaped impostors.

use crate::render::deferred::DEPTH_FORMAT;
use crate::space::solar_system::{Planet, SolarSystem, SpaceFlight};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

// Space mode's shader: the shared atmosphere maths (atmosphere.wgsl) followed by space.wgsl
pub(crate) const SPACE_SHADER: &str = concat!(
    include_str!("../render/atmosphere.wgsl"),
    "\n",
    include_str!("../weather/weather.wgsl"),
    "\n",
    include_str!("../weather/weather_map.wgsl"),
    "\n",
    include_str!("../render/clouds.wgsl"),
    include_str!("../render/star.wgsl"),
    "\n",
    include_str!("space.wgsl")
);

const ICOSPHERE_SUBDIVISIONS: u32 = 2;
// the first-person planet camera's 80 degrees (Controller::fov_y): the landing/liftoff handover
// switches between the two cameras, and a different field of view would zoom the view right then
const FOV_Y_RADIANS: f32 = 80.0 * std::f32::consts::PI / 180.0;
const NEAR_PLANE: f32 = 1.0;

// reversed-Z with no far plane (depth 1 at the near plane, toward 0 at infinity, compare Greater, clear
// 0): /system add puts planets on orbits millions of units out, and a Depth32Float buffer keeps about
// the same relative precision at every distance this way, so every body is drawn at its real distance
// and size and far bodies still sort correctly. (A far plane with far bodies pulled in to it sorted
// them by their shrunk size instead, and couldn't tell a far planet's front from its back.)
fn projection(fov_y: f32, aspect: f32) -> Mat4 {
    glam::camera::rh::proj::directx::perspective_infinite_reverse(fov_y, aspect, NEAR_PLANE)
}
const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::Greater;
const DEPTH_CLEAR: f32 = 0.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpaceCameraUniform {
    view_proj: [f32; 16],
    screen: [f32; 4],
    ray_dirs: [[f32; 4]; 3],
    forward: [f32; 4], // the view direction (xyz), for the star's reversed-Z depth
    pixel: [f32; 4],   // x: radians per screen pixel (cloud map level of detail)
    fog: [f32; 4],     // xyz: the storm fog's scrolled offset (weather/fog.rs)
}

// the star's uniform for vs_star/fs_star (must match StarUniform in space.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StarUniform {
    offset: [f32; 4],  // camera-relative centre (xyz), radius (w)
    surface: [f32; 4], // StarTypeDef.surface_color
    limb: [f32; 4],    // StarTypeDef.limb_color
    corona: [f32; 4],  // rgb corona colour, w corona_size
    params: [f32; 4],  // granulation, granule_scale, sunspots, minimum angular radius (rad)
}

// a point `distance` along the view axis in reversed-Z (projection: NEAR_PLANE / distance)
fn star_depth(distance: f32) -> f32 {
    NEAR_PLANE / distance
}

// the star for the lens flare (flare.rs): its screen position (NDC), the reversed-Z depth of its nearest
// point as the star pipeline writes it, and its angular radius; None when it's behind the camera
pub(crate) fn star_on_screen(
    camera: &SpaceCamera,
    system: &SolarSystem,
    aspect: f32,
) -> Option<(glam::Vec2, f32, f32)> {
    let offset = (system.star.position() - camera.position).as_vec3();
    let view = Mat4::from_quat(camera.rotation).inverse();
    let clip = projection(camera.fov_y, aspect) * view * offset.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let dist = offset.length();
    // the radius fs_star draws (at least the minimum angular size); its nearest point along the view
    // axis is `along - r` for a star anywhere on screen (an off-axis star's edge is nearer than the
    // point on its centre's line of sight)
    let r = (system.star.radius as f32).max(dist * STAR_MIN_ANGULAR_RADIUS);
    let along = offset.dot(camera.rotation * Vec3::NEG_Z);
    let nearest = (along - r).max(NEAR_PLANE);
    Some((
        glam::Vec2::new(clip.x / clip.w, clip.y / clip.w),
        star_depth(nearest),
        (r / dist).min(1.0).asin(),
    ))
}

// the star is never drawn smaller than this angular radius, so a distant sun stays a visible point
const STAR_MIN_ANGULAR_RADIUS: f32 = 0.0025; // ~0.14 degrees, a few pixels at 1080p

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PlanetUniform {
    offset: [f32; 4], // xyz: camera-relative planet centre, w: planet radius (voxel_resolution / 2)
    light_dir: [f32; 4], // xyz: direction from this planet toward the star (space), w unused
    sky_zenith: [f32; 4], // the planet type's AtmosphereDef colours (rgb), as the engine's BiomeUniform
    sky_horizon: [f32; 4],
    space_color: [f32; 4],
    cloud_light: [f32; 4],
    cloud_dark: [f32; 4],
    sun_color: [f32; 4],     // the solar system star type's sunlight (rgb)
    glow: [f32; 4],          // the planet type's glowing liquid colour (rgb), w = 1 if it has one
    glow_emission: [f32; 4], // what it emits (rgb): lava_emission
    plastic: [f32; 4], // base roughness, specular strength, reflection weight (plastic.rs impostor_uniform)
    light: [f32; 4],   // ground bounce, wrap lighting (light.rs)
    weather_rule: [f32; 4], // the planet type's weather rule and cover (weather.rs WeatherKind)
    weather_cover: [f32; 4],
    atmosphere: [f32; 4], // x: atmosphere density (AtmosphereDef.density)
    fog_tint: [f32; 4], // rgb: the storm fog's tint, w: its peak extinction x the setting (0 = off)
    model: [[f32; 4]; 3], // planet frame -> space rotation (Planet::orientation), mat3 columns
}

// what a glowing liquid emits on an impostor: the engine's fs_water look (the body, darkening from
// the shallow toward the deep colour with depth, x 2.5, unlit, blended over the floor at the depth's
// opacity) for a typical depth, LAVA_TYPICAL_DEPTH (calibrated against the voxel world at the landing
// handover: lava lakes are deep, nearly the deep colour); the floor is dark basalt, so the blend
// mostly dims
const LAVA_TYPICAL_DEPTH: f32 = 20.0;

fn lava_emission(liquid: &crate::planet::biome::LiquidDef) -> [f32; 3] {
    let deep = 1.0 - (-LAVA_TYPICAL_DEPTH * 0.15f32).exp();
    let alpha = 1.0 - (-LAVA_TYPICAL_DEPTH * 0.35f32).exp();
    // fs_water blends after tone mapping: dimming by alpha there is about alpha^2.2 before it
    let dim = alpha.powf(2.2);
    [0, 1, 2]
        .map(|i| (liquid.shallow_color[i] * (1.0 - deep) + liquid.deep_color[i] * deep) * 2.5 * dim)
}

struct PlanetMesh {
    v_buf: wgpu::Buffer,
    i_buf: wgpu::Buffer,
    num_indices: u32,
}

use crate::space::solar_system::MAX_PLANETS; // solar_system.rs caps generated + added planets at this many

// starting points from the solar system-terrain-impostors design discussion's faceting estimate
// (~10-13x radius before individual facets become visually obvious at this project's FOV);
// tune after a visual check, same as every other distance-based constant in this codebase
const LOD_MEDIUM_DISTANCE_MULTIPLIER: f32 = 13.0;
const LOD_NEAR_DISTANCE_MULTIPLIER: f32 = 6.0;

// does the planet have an atmosphere to draw a shell for (density 0 = vacuum)
pub fn has_atmosphere(planet: &Planet) -> bool {
    planet.planet_type.def().atmosphere.density > 0.0
}

// draw order for the atmosphere shells: farthest first. They blend without writing depth, so
// two overlapping glows only composite correctly back to front
fn back_to_front(distances: &[(usize, f32)]) -> Vec<usize> {
    let mut sorted = distances.to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    sorted.into_iter().map(|(i, _)| i).collect()
}

fn subdivision_for_distance(distance: f32, radius: f32) -> u32 {
    if distance > LOD_MEDIUM_DISTANCE_MULTIPLIER * radius {
        2
    } else if distance > LOD_NEAR_DISTANCE_MULTIPLIER * radius {
        4
    } else {
        5
    }
}

// where the solar system is seen from: a position in space, an orientation (looking down local -Z)
// and a vertical field of view. Space mode uses the flight; planet mode uses the planet camera
// converted into space, so the backdrop behind the voxel world lines up with it exactly.
#[derive(Clone, Copy, Debug)]
pub struct SpaceCamera {
    pub position: DVec3,
    pub rotation: Quat,
    pub fov_y: f32,
}

impl SpaceCamera {
    pub fn from_flight(flight: &SpaceFlight) -> Self {
        Self {
            position: flight.position,
            rotation: flight.rotation,
            fov_y: FOV_Y_RADIANS,
        }
    }

    // Space mode's camera, whichever frame the flight is in (space, or captured by a planet)
    pub fn from_flight_in_frame(
        flight: &SpaceFlight,
        frame: crate::space::solar_system::FlightFrame,
        system: &SolarSystem,
        t: f64,
    ) -> Self {
        match frame {
            crate::space::solar_system::FlightFrame::Free => Self::from_flight(flight),
            crate::space::solar_system::FlightFrame::Captured(i) => Self::on_system_planet(
                &system.planets[i],
                flight.position.as_vec3(),
                flight.rotation,
                FOV_Y_RADIANS,
                t,
            ),
        }
    }

    // the planet camera (eye and orientation in the planet frame) on a planet
    pub fn on_system_planet(
        planet: &Planet,
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

// which parts of the solar system to draw: everything in Space mode; behind a planet's voxel world
// everything except that planet (the voxel engine draws it)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpaceContent {
    Everything,
    AllButPlanet(usize),
}

impl SpaceContent {
    pub fn draws_planet(self, i: usize) -> bool {
        match self {
            Self::Everything => true,
            Self::AllButPlanet(skip) => i != skip,
        }
    }
}

// what Renderer::render draws behind the voxel world before anything else
pub struct Backdrop<'a> {
    pub system: &'a SolarSystem,
    pub camera: SpaceCamera,
    pub content: SpaceContent,
    pub t: f64,
    // the landing handover's cross-fade: (planet index, opacity) of the near impostor drawn over the
    // voxel frame (SpaceRenderer::draw_handover_overlay), None once it has faded out
    pub handover_overlay: Option<(usize, f32)>,
}

// where the bodies are drawn: at their real positions, or on the solar-system map (space/map.rs).
// On the map every body's camera-relative offset is the mapped offset divided by the map's size factor
// k (MapLayout::render_offset): perspective then shows it at its mapped apparent size, while radii,
// light and every shading term stay in real units, so map planets look exactly like the real ones
pub enum Layout<'a> {
    Real,
    Map {
        map: &'a crate::space::map::MapLayout,
        eye: Vec3, // the map camera's position in map units (the star at the map origin)
    },
}

impl Layout<'_> {
    // planet i's offset from the camera, in real units
    pub fn planet_offset(
        &self,
        system: &SolarSystem,
        i: usize,
        camera: &SpaceCamera,
        t: f64,
    ) -> DVec3 {
        let pos = system.planets[i].position_at(t);
        match self {
            Layout::Real => pos - camera.position,
            Layout::Map { map, eye } => {
                map.render_offset(map.position(pos - system.star.position()), *eye)
            }
        }
    }

    // the star's offset from the camera and its drawn radius, in real units
    pub fn star(&self, system: &SolarSystem, camera: &SpaceCamera) -> (DVec3, f64) {
        match self {
            Layout::Real => (system.star.position() - camera.position, system.star.radius),
            Layout::Map { map, eye } => (
                map.render_offset(Vec3::ZERO, *eye),
                (crate::space::map::STAR_MAP_RADIUS / map.k) as f64,
            ),
        }
    }
}

// a vertex of the map's orbit rings and travel line (fs_ring): camera-relative position, premultiplied
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RingVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

// one star of the galaxy view (fs_galaxy_star)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GalaxyStarInstance {
    pub pos: [f32; 3],   // camera-relative, galaxy view units
    pub color: [f32; 3], // linear, normalised to its largest component
    pub size_px: f32,
}

const MAX_GALAXY_STARS: usize = 512;

// the map's line budget: every orbit ring (RING_SEGMENTS each), the travel line (TRAVEL_LINE_POINTS
// points) and three circles (selection, hover, ship; CIRCLE_SEGMENTS each), two vertices per segment
pub const RING_SEGMENTS: usize = 128;
pub const TRAVEL_LINE_POINTS: usize = 65;
pub const CIRCLE_SEGMENTS: usize = 48;
const MAX_RING_VERTS: usize = (crate::space::solar_system::MAX_PLANETS * RING_SEGMENTS
    + (TRAVEL_LINE_POINTS - 1)
    + 3 * CIRCLE_SEGMENTS)
    * 2;

// where a camera-relative offset lands on screen (pixels, y down); None behind the camera
pub fn project(camera: &SpaceCamera, offset: Vec3, screen: (f32, f32)) -> Option<glam::Vec2> {
    let view = Mat4::from_quat(camera.rotation).inverse();
    let clip = projection(camera.fov_y, screen.0 / screen.1) * view * offset.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(glam::Vec2::new(
        (ndc.x * 0.5 + 0.5) * screen.0,
        (0.5 - ndc.y * 0.5) * screen.1,
    ))
}

pub struct SpaceRenderer {
    pub fog_scale: f32, // the storm fog setting (0 = off), Renderer sets it every frame
    pub fog_wind: Vec3, // the storm fog's scrolled offset
    pub plastic: [f32; 4], // the voxel terrain's plastic look for the impostors (Renderer sets it every frame)
    pub light: [f32; 4],   // the voxel terrain's fill-light tuning, likewise (light.rs)
    camera_buf: wgpu::Buffer,
    camera_bind: wgpu::BindGroup,
    star_buf: wgpu::Buffer,
    star_bind: wgpu::BindGroup,
    star_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    ring_pipeline: wgpu::RenderPipeline, // the map's orbit rings and travel line (line list)
    galaxy_star_pipeline: wgpu::RenderPipeline, // the galaxy view's star sprites (instanced)
    warp_pipeline: wgpu::RenderPipeline, // the jump's light-speed streaks over the frame (space/warp.rs)
    warp_buf: wgpu::Buffer,
    warp_bind: wgpu::BindGroup,
    galaxy_star_buf: wgpu::Buffer,
    ring_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    i_buf: wgpu::Buffer,
    num_indices: u32,
    planet_pipeline: wgpu::RenderPipeline,
    overlay_depth_pipeline: wgpu::RenderPipeline, // the landing handover's cross-fade: depth pre-pass
    overlay_pipeline: wgpu::RenderPipeline,       // and its blended pass (draw_handover_overlay)
    atmosphere_pipeline: wgpu::RenderPipeline,
    planet_uniform_buf: wgpu::Buffer,
    planet_uniform_bind: wgpu::BindGroup,
    planet_uniform_stride: u64,
    planet_meshes: std::collections::HashMap<(usize, u32), PlanetMesh>, // (planet index, subdivision)
    near_impostor: Option<(usize, PlanetMesh)>, // (planet index, mesh) — see set_near_impostor
}

impl SpaceRenderer {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        cloud_map: &crate::render::cloud_map::CloudMap, // its bindings join the camera's group 0 (clouds.wgsl)
        weather_map: &crate::weather::WeatherMap, // so do the weather map's (weather_map.wgsl)
    ) -> Self {
        let (verts, indices) = crate::space::icosphere::generate(ICOSPHERE_SUBDIVISIONS);
        let vert_data: Vec<[f32; 3]> = verts.iter().map(|v| v.to_array()).collect();
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SolarSystem Icosphere Vertices"),
            contents: bytemuck::cast_slice(&vert_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SolarSystem Icosphere Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let cloud_layout = crate::render::cloud_map::CloudMap::layout_entries();
        let cloud_entries = cloud_map.bind_entries();
        let weather_layout = crate::weather::WeatherMap::layout_entries();
        let weather_entries = weather_map.bind_entries();
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_camera_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                cloud_layout[0],
                cloud_layout[1],
                cloud_layout[2],
                weather_layout[0],
                weather_layout[1],
            ],
        });
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SolarSystem Camera Uniform"),
            size: std::mem::size_of::<SpaceCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space_camera_bind"),
            layout: &camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                cloud_entries[0].clone(),
                cloud_entries[1].clone(),
                cloud_entries[2].clone(),
                weather_entries[0].clone(),
                weather_entries[1].clone(),
            ],
        });

        let star_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_star_layout"),
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
        let star_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SolarSystem Star Uniform"),
            size: std::mem::size_of::<StarUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let star_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space_star_bind"),
            layout: &star_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: star_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SPACE_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_pipeline_layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let star_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_star_pipeline_layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&star_layout)],
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
            depth_compare: Some(DEPTH_COMPARE),
            stencil: Default::default(),
            bias: Default::default(),
        };

        // the star: one camera-facing quad, ray-cast in fs_star (round at any size, its true sphere depth),
        // the corona blended around it (premultiplied)
        let star_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SolarSystem Star Pipeline"),
            layout: Some(&star_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_star"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_star"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil.clone()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let planet_uniform_stride = {
            let min_align = device.limits().min_uniform_buffer_offset_alignment as u64;
            let unaligned = std::mem::size_of::<PlanetUniform>() as u64;
            unaligned.div_ceil(min_align) * min_align
        };
        let planet_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("space_planet_uniform_layout"),
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
            label: Some("SolarSystem Planet Uniform"),
            size: planet_uniform_stride * MAX_PLANETS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let planet_uniform_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space_planet_uniform_bind"),
            layout: &planet_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &planet_uniform_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<PlanetUniform>() as u64),
                }),
            }],
        });

        let planet_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_planet_pipeline_layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&planet_uniform_layout)],
            immediate_size: 0,
        });

        let planet_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SolarSystem Planet Pipeline"),
            layout: Some(&planet_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_planet"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::planet::data::Vertex>() as u64,
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

        // the landing handover's cross-fade (draw_handover_overlay): the planet pipeline, alpha-blended
        // over the finished voxel frame. Blending with a plain depth test would also let through what is
        // drawn before the surface in front of it (the far side, back faces, the LOD skirts at node
        // seams): a depth-only pass first, then the blended pass keeps only the nearest surface
        // (GreaterEqual, reversed-Z)
        let overlay_depth_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SolarSystem Handover Overlay Depth Pipeline"),
                layout: Some(&planet_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_planet"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<crate::planet::data::Vertex>() as u64,
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
                    entry_point: Some("fs_planet_overlay"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    })],
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
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SolarSystem Handover Overlay Pipeline"),
            layout: Some(&planet_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_planet"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::planet::data::Vertex>() as u64,
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
                entry_point: Some("fs_planet_overlay"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // no culling, like the voxel engine's own geometry pass: the near impostor is the engine's LOD
                // meshes, whose triangle winding isn't consistent (back-face culling punched holes at
                // cube-face corners and seams); the depth test hides the far side of the closed sphere
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                ..depth_stencil.clone()
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // atmosphere shell: the star's unit icosphere scaled per planet in vs_atmosphere; back faces
        // only (so it also renders with the camera inside it), depth-tested against the planets but not
        // writing depth, premultiplied over whatever is behind
        let atmosphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SolarSystem Atmosphere Pipeline"),
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
                depth_compare: Some(DEPTH_COMPARE),
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
            label: Some("SolarSystem Background Pipeline"),
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
                ..depth_stencil.clone()
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // the map's orbit rings and travel line: thin lines behind the planets, premultiplied, no depth write
        let ring_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SolarSystem Ring Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ring"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RingVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ring"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(false),
                ..depth_stencil.clone()
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // the galaxy view's stars: instanced screen-sized sprites, added over the background, no depth
        let galaxy_star_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Galaxy Star Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_galaxy_star"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GalaxyStarInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_galaxy_star"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                ..depth_stencil.clone()
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // the jump's streak pass: its own shader and uniform; alpha (the scene fade) darkens what's there,
        // the streaks add on top
        let warp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Warp Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::space::warp::WARP_SHADER.into()),
        });
        let warp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp_layout"),
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
        let warp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Warp Uniform"),
            size: std::mem::size_of::<crate::space::warp::WarpUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let warp_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warp_bind"),
            layout: &warp_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: warp_buf.as_entire_binding(),
            }],
        });
        let warp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Warp Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("warp_pipeline_layout"),
                    bind_group_layouts: &[Some(&warp_layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &warp_shader,
                entry_point: Some("vs_warp"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &warp_shader,
                entry_point: Some("fs_warp"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let galaxy_star_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Galaxy Stars"),
            size: (MAX_GALAXY_STARS * std::mem::size_of::<GalaxyStarInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ring_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Map Rings"),
            size: (MAX_RING_VERTS * std::mem::size_of::<RingVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            plastic: crate::render::plastic::PlasticSettings::default().impostor_uniform(),
            fog_scale: 1.0,
            fog_wind: Vec3::ZERO,
            light: crate::render::light::LightSettings::default().uniform(),
            camera_buf,
            camera_bind,
            star_buf,
            star_bind,
            star_pipeline,
            background_pipeline,
            ring_pipeline,
            ring_buf,
            galaxy_star_pipeline,
            galaxy_star_buf,
            warp_pipeline,
            warp_buf,
            warp_bind,
            v_buf,
            i_buf,
            num_indices: indices.len() as u32,
            planet_pipeline,
            overlay_depth_pipeline,
            overlay_pipeline,
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
        camera: &SpaceCamera,
        content: SpaceContent,
        system: &SolarSystem,
        t: f64,
        screen: (f32, f32),
        layout: &Layout,
        rings: &[RingVertex], // the map's orbit rings and travel line (empty at true scale)
    ) {
        self.write_camera(queue, camera, t, screen, 0.0);
        debug_assert!(rings.len() <= MAX_RING_VERTS, "map lines over budget");
        if rings.len() > MAX_RING_VERTS {
            eprintln!(
                "map lines over budget: {} of {MAX_RING_VERTS} vertices drawn",
                rings.len()
            );
        }
        let rings = &rings[..rings.len().min(MAX_RING_VERTS)];
        if !rings.is_empty() {
            queue.write_buffer(&self.ring_buf, 0, bytemuck::cast_slice(rings));
        }

        // the star (always drawn: only planets can be left out)
        {
            let star = system.star.star_type.def();
            let (offset, radius) = layout.star(system, camera);
            let offset = offset.as_vec3();
            let v4 = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
            queue.write_buffer(
                &self.star_buf,
                0,
                bytemuck::cast_slice(&[StarUniform {
                    offset: [offset.x, offset.y, offset.z, radius as f32],
                    surface: v4(star.surface_color, 0.0),
                    limb: v4(star.limb_color, 0.0),
                    corona: v4(star.corona_color, star.corona_size),
                    params: [
                        star.granulation,
                        star.granule_scale,
                        star.sunspots,
                        STAR_MIN_ANGULAR_RADIUS,
                    ],
                }]),
            );
        }

        let mut planet_uniforms = Vec::with_capacity(system.planets.len());
        for (i, p) in system.planets.iter().enumerate() {
            let camera_relative = layout.planet_offset(system, i, camera, t).as_vec3();
            // every planet keeps its uniform slot (slot i = planet i); only drawn ones need a mesh
            if content.draws_planet(i) && !self.has_near_impostor(i) {
                let subdivision = subdivision_for_distance(camera_relative.length(), p.radius);
                self.ensure_planet_mesh(device, i, p, subdivision);
            }
            planet_uniforms.push(Self::planet_uniform(
                system,
                i,
                camera_relative,
                t,
                self.plastic,
                self.light,
                self.fog_scale,
            ));
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
                label: Some("SolarSystem Pass"),
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
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_bind_group(0, &self.camera_bind, &[]);

            pass.set_pipeline(&self.background_pipeline);
            pass.draw(0..3, 0..1);

            // the star right after the background: planets in front cover its disc and corona (depth),
            // and their atmosphere shells, drawn last, blend over it
            pass.set_pipeline(&self.star_pipeline);
            pass.set_bind_group(1, &self.star_bind, &[]);
            pass.draw(0..6, 0..1);

            pass.set_pipeline(&self.planet_pipeline);
            pass.set_bind_group(0, &self.camera_bind, &[]);
            for (i, p) in system.planets.iter().enumerate() {
                if !content.draws_planet(i) {
                    continue;
                }
                let camera_relative = layout.planet_offset(system, i, camera, t).as_vec3();
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

            // atmosphere shells last, over the planets and everything behind them, farthest first
            pass.set_pipeline(&self.atmosphere_pipeline);
            pass.set_vertex_buffer(0, self.v_buf.slice(..));
            pass.set_index_buffer(self.i_buf.slice(..), wgpu::IndexFormat::Uint32);
            let shells: Vec<(usize, f32)> = (0..system.planets.len())
                .filter(|&i| content.draws_planet(i))
                .filter(|&i| has_atmosphere(&system.planets[i]))
                .map(|i| {
                    (
                        i,
                        layout.planet_offset(system, i, camera, t).length() as f32,
                    )
                })
                .collect();
            for i in back_to_front(&shells) {
                pass.set_bind_group(
                    1,
                    &self.planet_uniform_bind,
                    &[(i as u64 * self.planet_uniform_stride) as u32],
                );
                pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }

            // the map's orbit rings and travel line, over the background, behind the planets (depth)
            if !rings.is_empty() {
                pass.set_pipeline(&self.ring_pipeline);
                pass.set_bind_group(0, &self.camera_bind, &[]);
                pass.set_vertex_buffer(0, self.ring_buf.slice(..));
                pass.draw(0..rings.len() as u32, 0..1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    fn write_camera(
        &self,
        queue: &wgpu::Queue,
        camera: &SpaceCamera,
        t: f64,
        screen: (f32, f32),
        overlay_opacity: f32,
    ) {
        let proj = projection(camera.fov_y, screen.0 / screen.1);
        let view = Mat4::from_quat(camera.rotation).inverse();
        let view_proj = proj * view;

        let ray_dirs = Self::ray_dirs(camera.rotation, camera.fov_y, screen.0 / screen.1);
        queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::cast_slice(&[SpaceCameraUniform {
                view_proj: view_proj.to_cols_array(),
                // z: cloud animation time, wrapped like the engine's GlobalUniform.screen.w;
                // w: the handover overlay's opacity (fs_planet_overlay)
                screen: [screen.0, screen.1, (t % 3600.0) as f32, overlay_opacity],
                ray_dirs,
                forward: (camera.rotation * Vec3::NEG_Z).extend(0.0).to_array(),
                pixel: [camera.fov_y / screen.1, 0.0, 0.0, 0.0],
                fog: [self.fog_wind.x, self.fog_wind.y, self.fog_wind.z, 0.0],
            }]),
        );
    }

    // planet i's uniform slot as seen from `camera`
    fn planet_uniform(
        system: &SolarSystem,
        i: usize,
        camera_relative: Vec3, // the planet's offset from the camera (Layout::planet_offset)
        t: f64,
        plastic: [f32; 4],
        light: [f32; 4],
        fog_scale: f32,
    ) -> PlanetUniform {
        let p = &system.planets[i];
        let body_pos = p.position_at(t);
        let light_dir = (-body_pos).normalize_or_zero().as_vec3();
        let atmosphere = p.planet_type.def().atmosphere;
        let model = glam::Mat3::from_quat(p.orientation(t));
        let v4 = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
        PlanetUniform {
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
            sun_color: v4(system.star.star_type.def().sunlight),
            glow: match p.planet_type.def().liquid {
                Some(l) if matches!(l.behavior, crate::planet::biome::LiquidBehavior::Glowing) => {
                    let c = l.shallow_color;
                    [c[0], c[1], c[2], 1.0]
                }
                _ => [0.0; 4],
            },
            glow_emission: match p.planet_type.def().liquid {
                Some(l) => v4(lava_emission(&l)),
                None => [0.0; 4],
            },
            plastic,
            light,
            weather_rule: crate::weather::WeatherKind::for_planet(p.planet_type).rule(),
            weather_cover: crate::weather::WeatherKind::for_planet(p.planet_type).cover(),
            // y: the storm fog's base, the planet's radius plus the relief the impostor mesh uses
            atmosphere: [atmosphere.density, super::impostor::storm_base(p), 0.0, 0.0],
            fog_tint: {
                let f = crate::weather::fog::storm_fog(crate::weather::WeatherKind::for_planet(
                    p.planet_type,
                ));
                [f.tint[0], f.tint[1], f.tint[2], f.peak * fog_scale]
            },
            model: [
                model.x_axis.extend(0.0).to_array(),
                model.y_axis.extend(0.0).to_array(),
                model.z_axis.extend(0.0).to_array(),
            ],
        }
    }

    // the landing handover's cross-fade: planet `index`'s near impostor alpha-blended at `opacity` over
    // the finished voxel frame in `color_view`, recorded into the frame's encoder. `depth_view` is
    // cleared and reused for the impostor's own depth, so it must not be needed after this pass. Does
    // nothing without a near impostor for that planet. Writes the camera and planet uniforms, so record it
    // after any render() this frame has already submitted.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_handover_overlay(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &SpaceCamera,
        system: &SolarSystem,
        index: usize,
        t: f64,
        screen: (f32, f32),
        opacity: f32,
    ) {
        let Some((_, mesh)) = self.near_impostor.as_ref().filter(|(n, _)| *n == index) else {
            return;
        };
        self.write_camera(queue, camera, t, screen, opacity);
        let slot = index as u64 * self.planet_uniform_stride;
        queue.write_buffer(
            &self.planet_uniform_buf,
            slot,
            bytemuck::cast_slice(&[Self::planet_uniform(
                system,
                index,
                Layout::Real
                    .planet_offset(system, index, camera, t)
                    .as_vec3(),
                t,
                self.plastic,
                self.light,
                self.fog_scale,
            )]),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Handover Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                depth_slice: None,
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.camera_bind, &[]);
        pass.set_bind_group(1, &self.planet_uniform_bind, &[slot as u32]);
        pass.set_vertex_buffer(0, mesh.v_buf.slice(..));
        pass.set_index_buffer(mesh.i_buf.slice(..), wgpu::IndexFormat::Uint32);
        // depth first, then only the nearest surface blends (overlay_pipeline)
        for pipeline in [&self.overlay_depth_pipeline, &self.overlay_pipeline] {
            pass.set_pipeline(pipeline);
            pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
        }
    }

    // the galaxy view: the systems as star sprites and the marker rings, on black
    #[allow(clippy::too_many_arguments)]
    pub fn render_galaxy_view(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &SpaceCamera,
        t: f64,
        screen: (f32, f32),
        stars: &[GalaxyStarInstance],
        rings: &[RingVertex],
    ) {
        self.write_camera(queue, camera, t, screen, 0.0);
        let stars = &stars[..stars.len().min(MAX_GALAXY_STARS)];
        let rings = &rings[..rings.len().min(MAX_RING_VERTS)];
        queue.write_buffer(&self.galaxy_star_buf, 0, bytemuck::cast_slice(stars));
        if !rings.is_empty() {
            queue.write_buffer(&self.ring_buf, 0, bytemuck::cast_slice(rings));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Galaxy View Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // no starfield behind the galaxy: its own stars are the picture
            pass.set_bind_group(0, &self.camera_bind, &[]);
            if !stars.is_empty() {
                pass.set_pipeline(&self.galaxy_star_pipeline);
                pass.set_vertex_buffer(0, self.galaxy_star_buf.slice(..));
                pass.draw(0..4, 0..stars.len() as u32);
            }
            if !rings.is_empty() {
                pass.set_pipeline(&self.ring_pipeline);
                pass.set_vertex_buffer(0, self.ring_buf.slice(..));
                pass.draw(0..rings.len() as u32, 0..1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // the jump's streaks over the finished frame in `color_view` (space/warp.rs), seen from `camera`
    #[allow(clippy::too_many_arguments)]
    pub fn draw_warp(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        camera: &SpaceCamera,
        screen: (f32, f32),
        speed: f32,
        fade: f32,
        travel_dir: Vec3,
        flow: f32,
    ) {
        let u = crate::space::warp::WarpUniform {
            ray_dirs: Self::ray_dirs(camera.rotation, camera.fov_y, screen.0 / screen.1),
            vanish: [travel_dir.x, travel_dir.y, travel_dir.z, flow],
            params: [speed, fade, screen.0, screen.1],
        };
        queue.write_buffer(&self.warp_buf, 0, bytemuck::cast_slice(&[u]));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Warp Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: color_view,
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
            pass.set_pipeline(&self.warp_pipeline);
            pass.set_bind_group(0, &self.warp_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // a jump to another solar system: the planet meshes (keyed by planet index) and the near impostor
    // belong to the old one
    pub fn clear_system_caches(&mut self) {
        self.planet_meshes.clear();
        self.near_impostor = None;
    }

    // replaces planet `planet_index`'s noise impostor with the voxel engine's own distant-terrain
    // mesh (impostor::near_impostor_mesh) once that planet is baked; one planet at a time
    pub fn set_near_impostor(
        &mut self,
        device: &wgpu::Device,
        planet_index: usize,
        verts: &[crate::planet::data::Vertex],
        indices: &[u32],
    ) {
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SolarSystem Near Impostor Vertices"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SolarSystem Near Impostor Indices"),
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
        planet: &crate::space::solar_system::Planet,
        subdivision: u32,
    ) {
        if self.planet_meshes.contains_key(&(index, subdivision)) {
            return;
        }
        let (verts, indices) = crate::space::impostor::generate_planet_mesh(planet, subdivision);
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SolarSystem Planet Vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SolarSystem Planet Indices"),
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

    #[test]
    fn atmosphere_shells_draw_farthest_first() {
        // planet index order 0, 1, 2 at distances 500, 3000, 1200: the farthest shell blends first
        let order = back_to_front(&[(0, 500.0), (1, 3000.0), (2, 1200.0)]);
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn only_planets_with_an_atmosphere_get_a_shell() {
        use crate::planet::biome::PlanetType;
        let system = SolarSystem::generate(1);
        for p in &system.planets {
            assert_eq!(has_atmosphere(p), p.planet_type != PlanetType::Cratered);
        }
    }

    #[test]
    fn a_cratered_planet_has_no_atmosphere_shell() {
        use crate::planet::biome::PlanetType;
        let mut system = SolarSystem::generate(1);
        system.add_planet(PlanetType::Cratered, 100.0).unwrap();
        assert!(!has_atmosphere(system.planets.last().unwrap()));
    }
    use crate::space::solar_system::SolarSystem;

    // PlanetUniform must mirror space.wgsl's PlanetUniform: 15 vec4s + a mat3x3 (3 × 16 bytes)
    #[test]
    fn planet_uniform_matches_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<PlanetUniform>(), 16 * 16 + 3 * 16);
        // + plastic, + atmosphere, + fog_tint
    }

    #[test]
    fn planet_uniform_carries_the_density_of_its_type() {
        let system = SolarSystem::generate(1);
        for (i, p) in system.planets.iter().enumerate() {
            let u =
                SpaceRenderer::planet_uniform(&system, i, Vec3::ZERO, 0.0, [0.0; 4], [0.0; 4], 1.0);
            assert_eq!(u.atmosphere[0], p.planet_type.def().atmosphere.density);
        }
    }

    // the impostor's storm fog uses the planet type's table and the setting; nothing when off
    #[test]
    fn planet_uniform_carries_the_storm_fog_of_its_type() {
        let system = SolarSystem::generate(1);
        let on =
            SpaceRenderer::planet_uniform(&system, 0, Vec3::ZERO, 0.0, [0.0; 4], [0.0; 4], 1.0);
        let off =
            SpaceRenderer::planet_uniform(&system, 0, Vec3::ZERO, 0.0, [0.0; 4], [0.0; 4], 0.0);
        let kind = crate::weather::WeatherKind::for_planet(system.planets[0].planet_type);
        let want = crate::weather::fog::storm_fog(kind);
        assert_eq!(on.fog_tint[3], want.peak);
        assert_eq!(&on.fog_tint[..3], &want.tint);
        assert_eq!(off.fog_tint[3], 0.0);
    }

    #[test]
    fn space_shader_is_valid_wgsl() {
        crate::render::renderer::tests::assert_valid_wgsl(SPACE_SHADER);
        // the impostors are plastic like the voxel terrain (atmosphere.wgsl atmo_plastic)
        assert!(
            SPACE_SHADER.matches("atmo_plastic(").count() >= 2,
            "defined and called"
        );
    }

    // the landing handover switches from Space mode's camera to the first-person planet camera:
    // a different field of view would zoom the planet visibly at that moment
    #[test]
    fn space_mode_uses_the_first_person_planet_cameras_field_of_view() {
        let mut controller = crate::player::controller::Controller::new();
        controller.first_person = true;
        assert!((FOV_Y_RADIANS - controller.fov_y()).abs() < 1e-6);
    }

    #[test]
    fn camera_from_a_captured_flight_is_placed_in_space() {
        let g = SolarSystem::generate(1);
        let mut flight = SpaceFlight::new(glam::DVec3::new(0.0, 600.0, 0.0)); // planet frame
        flight.rotation = Quat::from_axis_angle(Vec3::X, -1.0);
        let t = 21.0;
        let cam = SpaceCamera::from_flight_in_frame(
            &flight,
            crate::space::solar_system::FlightFrame::Captured(0),
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
        let free = SpaceCamera::from_flight_in_frame(
            &flight,
            crate::space::solar_system::FlightFrame::Free,
            &g,
            t,
        );
        assert_eq!(free.position, flight.position);
    }

    #[test]
    fn content_selects_star_and_planets() {
        assert!(SpaceContent::Everything.draws_planet(3));
        assert!(SpaceContent::AllButPlanet(2).draws_planet(1));
        assert!(!SpaceContent::AllButPlanet(2).draws_planet(2));
    }

    // standing on a planet, the backdrop camera sits where the planet frame puts the eye in
    // space, turned the same way
    #[test]
    fn camera_on_a_planet_is_placed_in_space() {
        let g = SolarSystem::generate(1);
        let p = &g.planets[0];
        let eye = Vec3::new(0.0, 175.0, 0.0);
        let rot = Quat::from_axis_angle(Vec3::X, 0.4);
        let t = 42.0;
        let cam = SpaceCamera::on_system_planet(p, eye, rot, 1.2, t);
        assert!((cam.position - p.from_planet_frame(eye.as_dvec3(), t)).length() < 1e-6);
        assert!(cam.rotation.dot(p.rotation_from_planet_frame(rot, t)).abs() > 1.0 - 1e-6);
        assert_eq!(cam.fov_y, 1.2);
    }

    #[test]
    fn camera_from_flight_uses_space_modes_fov() {
        let flight = SpaceFlight::new(glam::DVec3::new(1.0, 2.0, 3.0));
        let cam = SpaceCamera::from_flight(&flight);
        assert_eq!(cam.position, flight.position);
        assert_eq!(cam.fov_y, FOV_Y_RADIANS);
    }

    // reversed-Z depth of a camera-relative point, as the depth test sees it
    fn depth_of(offset: Vec3) -> f32 {
        let clip = projection(FOV_Y_RADIANS, 16.0 / 9.0) * offset.extend(1.0);
        clip.z / clip.w
    }

    // integration: a system filled to MAX_PLANETS (/system add puts planets on ever wider orbits,
    // millions of units out), seen from just above its outermost planet: every body, the star
    // included, is in the depth range at its real distance with its near side in front of its far
    // side, and the body buffer holds the star plus every planet
    #[test]
    fn a_full_solar_system_seen_from_its_outermost_planet_has_ordered_depths() {
        let mut system = SolarSystem::generate(1);
        while system.planets.len() < MAX_PLANETS {
            system
                .add_planet(crate::planet::biome::PlanetType::Ice, 100.0)
                .unwrap();
        }
        let outer = system.planets[MAX_PLANETS - 1];
        let t = 0.0;
        let camera = outer.position_at(t) + glam::DVec3::new(0.0, 3.0 * outer.radius as f64, 0.0);
        let mut bodies: Vec<(glam::DVec3, f32)> =
            vec![(system.star.position(), system.star.radius as f32)];
        bodies.extend(system.planets.iter().map(|p| (p.position_at(t), p.radius)));
        for (i, (pos, radius)) in bodies.into_iter().enumerate() {
            // looked at straight on: depth depends only on the distance along the view axis
            let distance = (pos - camera).length() as f32;
            let front = depth_of(Vec3::NEG_Z * (distance - radius));
            let back = depth_of(Vec3::NEG_Z * (distance + radius));
            assert!(
                front <= 1.0 && back > 0.0,
                "body {i} outside the depth range"
            );
            assert!(
                front > back,
                "body {i}: front {front} not in front of back {back}"
            );
        }
    }

    // two bodies on the same line of sight far out: the nearer one is in front, even when the
    // farther one is much bigger (pulling both in to a far plane sorted them by their shrunk size)
    #[test]
    fn far_bodies_sort_by_their_real_distance() {
        let dir = Vec3::new(0.3, 0.1, -1.0).normalize();
        let small_near_back = depth_of(dir * (1_000_000.0 + 40.0));
        let big_far_front = depth_of(dir * (3_000_000.0 - 5_000.0));
        assert!(small_near_back > big_far_front);
    }

    // an unobstructed sun off the view axis: the flare's expected depth must not be exceeded by any point
    // of the drawn star (its edge toward the screen border is nearer along the view axis than its centre's
    // line of sight), or the star occludes its own flare
    #[test]
    fn the_flare_depth_covers_an_off_axis_star() {
        let system = SolarSystem::generate(1);
        let r = system.star.radius;
        for (dist, yaw) in [(2.2 * r, 0.25f32), (6.0 * r, 0.35), (30.0 * r, 0.5)] {
            let camera = SpaceCamera {
                position: DVec3::new(0.0, 0.0, dist),
                rotation: Quat::from_rotation_y(yaw), // the star (at -Z) off to the side
                fov_y: FOV_Y_RADIANS,
            };
            let (_, expected, _) = star_on_screen(&camera, &system, 16.0 / 9.0).unwrap();
            let along = (dist as f32) * yaw.cos();
            let nearest_point_depth = star_depth(along - r as f32);
            assert!(
                expected >= nearest_point_depth * 0.999,
                "dist {dist}, yaw {yaw}: expected {expected} < the star's own {nearest_point_depth}"
            );
        }
    }

    #[test]
    fn star_uniform_matches_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<StarUniform>(), 5 * 16);
    }

    // the star writes its sphere's depth in reversed-Z, as the projection does for geometry
    #[test]
    fn star_depth_is_reversed_z() {
        for d in [10.0f32, 3_000.0, 250_000.0] {
            let clip = projection(FOV_Y_RADIANS, 16.0 / 9.0) * Vec3::new(0.0, 0.0, -d).extend(1.0);
            assert!((star_depth(d) - clip.z / clip.w).abs() < 1e-6 * (1.0 + 1.0 / d));
        }
    }

    // ray_dirs packs the camera frustum as a corner (ray_dirs[0]) plus two edge vectors
    // (ray_dirs[1]/[2]) so the shader can interpolate `ray_dirs[0] + uv.x*ray_dirs[1] +
    // uv.y*ray_dirs[2]` per pixel (same scheme as GlobalUniform.ray_dirs / fs_light). At the
    // screen center (uv = 0.5, 0.5) that interpolation must reduce to the forward direction.
    #[test]
    fn ray_dirs_center_matches_forward_at_identity_rotation() {
        let ray_dirs = SpaceRenderer::ray_dirs(Quat::IDENTITY, 1.0, 16.0 / 9.0);
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
        let ray_dirs = SpaceRenderer::ray_dirs(rotation, 1.0, 16.0 / 9.0);
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
    // isolated test, but nothing exercised SolarSystem::generate's *real* planets end to end before
    // this — fixed tiers are used directly rather than going through distance mapping, since
    // subdivision_for_distance already has its own coverage above.
    #[test]
    fn seeded_planets_produce_distinct_geometry_across_tiers_and_planets() {
        let system = SolarSystem::generate(1);
        let p0 = &system.planets[0];
        let p1 = &system.planets[1];

        let (v2, _) = crate::space::impostor::generate_planet_mesh(p0, 2);
        let (v4, _) = crate::space::impostor::generate_planet_mesh(p0, 4);
        let (v5, _) = crate::space::impostor::generate_planet_mesh(p0, 5);
        assert!(
            v2.len() < v4.len() && v4.len() < v5.len(),
            "tiers should have strictly increasing vertex counts"
        );

        let (v0, _) = crate::space::impostor::generate_planet_mesh(p0, 2);
        let (v1, _) = crate::space::impostor::generate_planet_mesh(p1, 2);
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
