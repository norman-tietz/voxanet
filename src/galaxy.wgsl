// galaxy.wgsl
// Minimal forward-rendering shader for galaxy mode: a procedural starfield background plus
// flat/Lambertian-shaded icosphere placeholders for the star and each planet. Deliberately
// standalone — does not share code with shader.wgsl (the deferred voxel pipeline's shader).

struct Camera {
    view_proj: mat4x4<f32>, // camera-relative: positions are pre-translated so the camera is at the origin
    screen: vec4<f32>,      // x, y: width/height in pixels
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
    offset: vec4<f32>,         // camera-relative position (xyz), unused (w)
    light_dir: vec4<f32>,      // direction from this planet toward the star (xyz), unused (w)
    atmosphere_color: vec4<f32>, // rgb: limb-glow tint (the planet type's sky_zenith), w: glow strength
}

@group(1) @binding(0) var<uniform> planet: PlanetUniform;

struct PlanetVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) light_dir: vec3<f32>,
    @location(4) @interpolate(flat) atmosphere_color: vec4<f32>,
}

@vertex
fn vs_planet(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
) -> PlanetVertexOut {
    let world_pos = planet.offset.xyz + pos;
    var out: PlanetVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normal;
    out.color = color;
    out.light_dir = planet.light_dir.xyz;
    out.atmosphere_color = planet.atmosphere_color;
    return out;
}

@fragment
fn fs_planet(in: PlanetVertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let ndotl = max(dot(n, in.light_dir), 0.05);
    var color = in.color * ndotl;

    // the camera sits at the origin in this camera-relative scheme (see Camera.view_proj's
    // comment), so the direction back to it is simply the negated world position
    let view_dir = normalize(-in.world_pos);
    let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0);
    color += in.atmosphere_color.rgb * fresnel * in.atmosphere_color.w;

    return vec4<f32>(color, 1.0);
}

// deterministic hash for the starfield, independent of shader.wgsl's hash31 (kept standalone)
fn star_hash(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
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
    let cell = floor(ray_dir * cell_scale);
    let h = star_hash(cell);
    let brightness = smoothstep(0.985, 1.0, h); // sparse: only the top ~1.5% of cells show a star
    return vec4<f32>(vec3<f32>(brightness), 1.0);
}
