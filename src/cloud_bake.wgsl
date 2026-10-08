// cloud_bake.wgsl
// Bakes fbm_clouds (atmosphere.wgsl, prepended) into the cloud map, one cube face per pass, and builds
// its mip chain (cloud_map.rs). The layer (slot * 6 + face) comes in as the instance index.

const CLOUD_MAP_SIZE: f32 = 512.0; // cloud_map.rs SIZE

struct BakeParams {
    times: vec4<f32>, // the snapshot time (cloud time, seconds) of each slot
}
@group(0) @binding(0) var<uniform> bake: BakeParams;
@group(0) @binding(1) var mip_src: texture_2d<f32>; // the previous mip level of the same layer

struct FaceOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) layer: u32,
}

@vertex
fn vs_cloud_face(@builtin(vertex_index) i: u32, @builtin(instance_index) layer: u32) -> FaceOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: FaceOut;
    out.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.layer = layer;
    return out;
}

// the direction through texel centre `px` of cube face `face` (+X, -X, +Y, -Y, +Z, -Z), as the
// hardware's cube sampling maps it (rows run downward, so t points down)
fn cube_dir(face: u32, px: vec2<f32>) -> vec3<f32> {
    let st = px / CLOUD_MAP_SIZE * 2.0 - 1.0;
    let s = st.x;
    let t = st.y;
    switch face {
        case 0u: { return vec3<f32>(1.0, -t, -s); }
        case 1u: { return vec3<f32>(-1.0, -t, s); }
        case 2u: { return vec3<f32>(s, 1.0, t); }
        case 3u: { return vec3<f32>(s, -1.0, -t); }
        case 4u: { return vec3<f32>(s, -t, 1.0); }
        default: { return vec3<f32>(-s, -t, -1.0); }
    }
}

@fragment
fn fs_cloud_bake(in: FaceOut) -> @location(0) vec4<f32> {
    let dir = normalize(cube_dir(in.layer % 6u, in.pos.xy));
    return vec4<f32>(fbm_clouds(dir, bake.times[in.layer / 6u]), 0.0, 0.0, 1.0);
}

@fragment
fn fs_cloud_mip(in: FaceOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy) * 2;
    let sum = textureLoad(mip_src, p, 0).r + textureLoad(mip_src, p + vec2<i32>(1, 0), 0).r
        + textureLoad(mip_src, p + vec2<i32>(0, 1), 0).r + textureLoad(mip_src, p + vec2<i32>(1, 1), 0).r;
    return vec4<f32>(sum * 0.25, 0.0, 0.0, 1.0);
}
