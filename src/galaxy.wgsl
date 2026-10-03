// galaxy.wgsl
// Minimal forward-rendering shader for galaxy mode: a procedural starfield background plus
// icosphere bodies for the star and each planet. Planet impostors share the sky, cloud, fog and
// tone-mapping maths with the voxel engine through atmosphere.wgsl (prepended at compile time,
// galaxy_render.rs GALAXY_SHADER), so they look the same at the landing handover.

struct Camera {
    view_proj: mat4x4<f32>, // camera-relative: positions are pre-translated so the camera is at the origin
    screen: vec4<f32>,      // x, y: width/height in pixels, z: cloud animation time (seconds, wrapped at 3600)
    ray_dirs: array<vec4<f32>, 3>, // camera basis corners for the starfield background
}

struct Body {
    offset: vec4<f32>,    // camera-relative position (xyz), radius (w)
    color: vec4<f32>,     // rgb, w: 1.0 = emissive (the star), 0.0 = lit (a planet)
    light_dir: vec4<f32>, // direction from this body toward the star (xyz)
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> bodies: array<Body>;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>, // unit-sphere vertex position IS the normal, pre-scale
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) light_dir: vec3<f32>,
}

@vertex
fn vs_body(@location(0) pos: vec3<f32>, @builtin(instance_index) instance: u32) -> VertexOut {
    let body = bodies[instance];
    let world_pos = body.offset.xyz + pos * body.offset.w;
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = pos;
    out.color = body.color;
    out.light_dir = body.light_dir.xyz;
    return out;
}

@fragment
fn fs_body(in: VertexOut) -> @location(0) vec4<f32> {
    if (in.color.w > 0.5) {
        return vec4<f32>(in.color.rgb, 1.0); // emissive: the star, unlit
    }
    let n = normalize(in.world_normal);
    let ndotl = max(dot(n, in.light_dir), 0.05); // small ambient floor so the dark side isn't pure black
    return vec4<f32>(in.color.rgb * ndotl, 1.0);
}

struct PlanetUniform {
    offset: vec4<f32>,      // xyz: camera-relative planet centre, w: planet radius (voxel resolution / 2)
    light_dir: vec4<f32>,   // direction from this planet toward the star (galaxy space)
    sky_zenith: vec4<f32>,  // the planet type's atmosphere colours (rgb), as the engine's Biome uniform
    sky_horizon: vec4<f32>,
    space_color: vec4<f32>,
    cloud_light: vec4<f32>,
    cloud_dark: vec4<f32>,
    model: mat3x3<f32>,     // planet frame -> galaxy space rotation: the planet's spin at this time
}

@group(1) @binding(0) var<uniform> planet: PlanetUniform;

fn planet_atmosphere() -> Atmosphere {
    return Atmosphere(
        planet.offset.w,
        planet.sky_zenith.rgb,
        planet.sky_horizon.rgb,
        planet.space_color.rgb,
        planet.cloud_light.rgb,
        planet.cloud_dark.rgb,
    );
}

// the camera (at the galaxy-space origin, camera-relative rendering) and the sun, in this planet's
// own frame — where the voxel engine does all its lighting, clouds and sky
fn planet_frame_camera() -> vec3<f32> {
    return transpose(planet.model) * (-planet.offset.xyz);
}

fn planet_frame_light() -> vec3<f32> {
    return normalize(transpose(planet.model) * planet.light_dir.xyz);
}

struct PlanetVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec3<f32>,    // planet frame (the mesh is built there)
    @location(1) local_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

@vertex
fn vs_planet(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
) -> PlanetVertexOut {
    let world_pos = planet.offset.xyz + planet.model * pos;
    var out: PlanetVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.local_pos = pos;
    out.local_normal = normal;
    out.color = color;
    return out;
}

// the voxel engine's shade() (shader.wgsl) without what an impostor can't have: ray-traced shadows,
// caustics and the per-voxel grain — sun, sky ambient, rim, cloud shadow and air fog are the same
@fragment
fn fs_planet(in: PlanetVertexOut) -> @location(0) vec4<f32> {
    let a = planet_atmosphere();
    let cam = planet_frame_camera();
    let L = planet_frame_light();
    let t = camera.screen.z;
    let N = normalize(in.local_normal);
    let V = normalize(cam - in.local_pos);

    let albedo = pow(in.color, vec3<f32>(2.2));
    let NdotL = max(dot(N, L), 0.0);
    let direct = SUN_COLOR * NdotL * atmo_cloud_shadow(in.local_pos, L, t, a.planet_r);
    let hemi = dot(N, normalize(in.local_pos)) * 0.5 + 0.5;
    let ambient = mix(GROUND_COLOR, a.sky_zenith, hemi);
    let rim = a.sky_zenith * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.2;
    var color = atmo_air_fog(albedo * (direct + ambient + rim), in.local_pos, cam, L, a);

    // clouds in front of the surface, like the engine's shade_pixel
    let ray_dir = normalize(in.local_pos - cam);
    let cloud_t = sphere_hit(cam, ray_dir, a.planet_r * CLOUD_ALT);
    if (cloud_t > 0.0 && cloud_t < distance(cam, in.local_pos)) {
        let cl = atmo_cloud_shade(cam + ray_dir * cloud_t, ray_dir, t, L, a);
        color = mix(color, cl.rgb, cl.a);
    }
    return vec4<f32>(aces_and_gamma(color), 1.0);
}

// deterministic hash for the starfield, independent of shader.wgsl's hash31 (kept standalone).
// Integer-only (PCG3D, Jarzynski & Olano 2020), on purpose: the classic fract(sin(dot(p, k)) * big)
// hash this replaced fed sin() arguments up to ~200,000 (cell coords reach ±400), where GPU f32 sin
// has no precision left. It only stayed random for directions near the plane where that dot product
// is small, so the sky showed a band of stars, an empty hole on one side and a regular dot lattice
// on the other. Integer math has no such range limit.
fn star_hash(cell: vec3<i32>) -> f32 {
    var v = bitcast<vec3<u32>>(cell) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return f32(v.x >> 8u) / 16777216.0; // top 24 bits → [0, 1), exact in f32
}

@vertex
fn vs_background(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_background(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / camera.screen.xy;
    let ray_dir = normalize(camera.ray_dirs[0].xyz + uv.x * camera.ray_dirs[1].xyz + uv.y * camera.ray_dirs[2].xyz);

    let cell_scale = 400.0;
    let cell = vec3<i32>(floor(ray_dir * cell_scale));
    let h = star_hash(cell);
    let brightness = smoothstep(0.985, 1.0, h); // sparse: only the top ~1.5% of cells show a star
    return vec4<f32>(vec3<f32>(brightness), 1.0);
}
