// weather_bake.wgsl
// Bakes weather_sample (weather.wgsl, prepended) into the weather map (weather.rs WeatherMap), one cube
// face per pass; the layer (slot * 6 + face) comes in as the instance index.

const WEATHER_MAP_SIZE: f32 = 64.0; // weather.rs MAP_SIZE

struct WeatherBakeParams {
    times: vec4<f32>,     // the snapshot time of each slot (clock seconds, wrapped at 3600)
    force: WeatherForce,
}
@group(0) @binding(0) var<uniform> wbake: WeatherBakeParams;

struct FaceOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) layer: u32,
}

@vertex
fn vs_weather_face(@builtin(vertex_index) i: u32, @builtin(instance_index) layer: u32) -> FaceOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: FaceOut;
    out.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.layer = layer;
    return out;
}

// as cloud_bake.wgsl's (a test keeps them identical)
fn cube_dir(face: u32, px: vec2<f32>) -> vec3<f32> {
    let st = px / WEATHER_MAP_SIZE * 2.0 - 1.0;
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
fn fs_weather_bake(in: FaceOut) -> @location(0) vec4<f32> {
    let dir = normalize(cube_dir(in.layer % 6u, in.pos.xy));
    return weather_sample(dir, wbake.times[in.layer / 6u], wbake.force);
}
