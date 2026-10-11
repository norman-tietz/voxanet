// precip.wgsl
// Rain, snow, ash and dust around the camera (precipitation.rs): stateless instanced quads, particle i
// at hash(i) in a box around the camera plus the CPU's accumulated offset (fall, wind, minus the
// camera's movement), wrapped; hidden below the roof grid (the highest surface per column) and faded
// near the camera, toward the box faces and against the G-buffer distance. Group 1 bindings 1-2 (0 is
// the scene's Local, unused here); group 3 is the deferred G-buffer the lighting pass has bound.

const BOX_SIZE: vec3<f32> = vec3<f32>(40.0, 30.0, 40.0); // precipitation.rs BOX_SIZE
const ROOF_GRID: i32 = 64;                                // precipitation.rs GRID

struct Precip {
    east: vec4<f32>,   // xyz: the camera frame's east, w: time (seconds, wrapped)
    up: vec4<f32>,     // xyz: up, w: kind (0 rain, 1 snow, 2 ash, 3 dust)
    north: vec4<f32>,  // xyz: north, w: fall speed
    offset: vec4<f32>, // xyz: offset (east, up, north) inside the box, w: sway amplitude
    wind: vec4<f32>,   // xyz: wind (world, units/s), w: size (flake size / streak width)
    color: vec4<f32>,  // rgb: flake colour (linear), a: opacity
    grid_origin: vec4<f32>, // xyz: the roof grid's origin, w: cell size
    grid_east: vec4<f32>,
    grid_north: vec4<f32>,
}
@group(1) @binding(1) var<uniform> precip: Precip;
@group(1) @binding(2) var roof_grid: texture_2d<f32>; // R32Float radii, x along grid_east, y along grid_north

fn precip_hash(n: u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn precip_rand3(i: u32) -> vec3<f32> {
    let a = precip_hash(i);
    let b = precip_hash(a);
    let c = precip_hash(b);
    return vec3<f32>(f32(a), f32(b), f32(c)) / 4294967295.0;
}

struct PrecipOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,       // -1..1 across the quad
    @location(1) world: vec3<f32>,
    @location(2) fade: f32,           // near/box/roof fades and thin-streak opacity, multiplied
}

@vertex
fn vs_precip(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PrecipOut {
    var out: PrecipOut;
    let kind = u32(precip.up.w);
    let t = precip.east.w;
    let r = precip_rand3(ii);
    // position in the camera frame, centred on the camera
    var local = r * BOX_SIZE + precip.offset.xyz;
    local = local - BOX_SIZE * floor(local / BOX_SIZE) - BOX_SIZE * 0.5;
    // snow and ash sway sideways, each at its own phase
    let phase = r.x * 37.0 + r.z * 11.0;
    local += vec3<f32>(sin(t * 3.1 + phase), 0.0, cos(t * 2.7 + phase)) * precip.offset.w;
    let cam = global.camera_pos.xyz;
    let center = cam + precip.east.xyz * local.x + precip.up.xyz * local.y + precip.north.xyz * local.z;

    // fades: near the camera, toward the box faces
    let d = length(center - cam);
    var fade = smoothstep(0.5, 1.5, d);
    let edge = abs(local) / (BOX_SIZE * 0.5);
    fade *= 1.0 - smoothstep(0.85, 1.0, max(edge.x, max(edge.y, edge.z)));

    // under a roof (or below the water / ground): hidden
    let g = center - precip.grid_origin.xyz;
    let cell = vec2<i32>(floor(vec2<f32>(dot(g, precip.grid_east.xyz), dot(g, precip.grid_north.xyz)) / precip.grid_origin.w)) + ROOF_GRID / 2;
    if (all(cell >= vec2<i32>(0)) && all(cell < vec2<i32>(ROOF_GRID))) {
        let roof = textureLoad(roof_grid, cell, 0).r;
        if (length(center) < roof + 0.1) { fade = 0.0; }
    }

    // the quad: a camera-facing streak along the velocity (rain), else a camera-facing square
    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
    let to_cam = normalize(cam - center);
    let size = precip.wind.w;
    var p = center;
    // rain falls in streaks; dust is blown along in short streaks
    if (kind == 0u || kind == 3u) {
        let vel = precip.wind.xyz - precip.up.xyz * precip.north.w;
        let along = normalize(vel);
        let side = normalize(cross(along, to_cam));
        let len = length(vel) / 30.0;
        // at least one pixel wide: radians per pixel × distance; thinner streaks get fainter instead
        let px = global.motion.y * d;
        let w = max(size, px);
        fade *= size / w;
        p += along * (corner.y * len * 0.5) + side * (corner.x * w * 0.5);
    } else {
        // flakes stay at least ~2 px across; smaller ones get fainter by their real area instead
        let w = max(size, global.motion.y * d * 2.0);
        fade *= (size / w) * (size / w);
        let right = normalize(cross(precip.up.xyz, to_cam));
        let upq = cross(to_cam, right);
        p += (right * corner.x + upq * corner.y) * w * 0.5;
    }
    out.clip_pos = global.view_proj * vec4<f32>(p, 1.0);
    out.uv = corner;
    out.world = center;
    out.fade = fade;
    return out;
}

@fragment
fn fs_precip(in: PrecipOut) -> @location(0) vec4<f32> {
    if (in.fade <= 0.001) { discard; }
    let kind = u32(precip.up.w);
    // soft round flakes; rain streaks fade along their length
    var shape = 1.0 - smoothstep(0.4, 1.0, length(in.uv));
    if (kind == 0u || kind == 3u) { shape = (1.0 - abs(in.uv.y)) * (1.0 - abs(in.uv.x)); }
    // soft against the surface behind (G-buffer distance, 0 = sky)
    let behind = textureLoad(d_dist, vec2<i32>(in.clip_pos.xy), 0).r;
    let dist = distance(in.world, global.camera_pos.xyz);
    let soft = select(1.0, clamp((behind - dist) / 0.5, 0.0, 1.0), behind > 0.0);
    let a = precip.color.a * shape * in.fade * soft;
    if (a <= 0.001) { discard; }
    // lit by the sky and (a little) the sun, no shadows; rain takes the sky colour itself
    let L = normalize(global.sun_dir.xyz);
    let light = biome.sky_zenith.rgb * atmo_ambient_daylight(in.world, L)
        + biome.sun.rgb * cloud_shadow(in.world, L) * 0.5;
    let base = select(precip.color.rgb, vec3<f32>(0.6, 0.65, 0.7), kind == 0u);
    let lit = base * light;
    // premultiplied, tone-mapped like the rest of the frame
    return vec4<f32>(post_process(lit * a), a);
}
