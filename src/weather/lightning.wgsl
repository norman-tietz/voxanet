// lightning.wgsl
// A lightning bolt (lightning.rs, bolt.rs): one instanced camera-facing strip per segment, additive and
// HDR, depth-tested against the G-buffer distance (group 3's d_dist, like precip.wgsl). Appended to the scene
// shader.

struct BoltOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,   // x across the strip -1..1, y along 0..1
    @location(1) world: vec3<f32>,
    @location(2) bright: f32,
}

@vertex
fn vs_bolt(@builtin(vertex_index) vi: u32, @location(0) a: vec4<f32>, @location(1) b: vec4<f32>) -> BoltOut {
    var out: BoltOut;
    let along = f32(vi >> 1u);                 // 0 at a, 1 at b
    let side = f32(vi & 1u) * 2.0 - 1.0;       // -1 / +1
    let p = mix(a.xyz, b.xyz, along);
    // widen with distance so a distant bolt stays about a pixel wide
    let dist = distance(p, global.camera_pos.xyz);
    let width = max(a.w, dist * global.motion.y * 2.0);
    let dir = normalize(b.xyz - a.xyz);
    let to_cam = normalize(global.camera_pos.xyz - p);
    var cr = cross(dir, to_cam);
    if (length(cr) < 1e-6) {
        // seen straight on: any perpendicular will do
        cr = cross(dir, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.9));
    }
    let across = normalize(cr);
    let world = p + across * side * width * 0.5;
    out.world = world;
    out.clip_pos = global.view_proj * vec4<f32>(world, 1.0);
    out.uv = vec2<f32>(side, along);
    out.bright = b.w;
    return out;
}

@fragment
fn fs_bolt(in: BoltOut) -> @location(0) vec4<f32> {
    // hidden behind the surface in front (G-buffer distance, 0 = sky)
    let behind = textureLoad(d_dist, vec2<i32>(in.clip_pos.xy), 0).r;
    let dist = distance(in.world, global.camera_pos.xyz);
    if (behind > 0.0 && behind < dist - 0.5) { discard; }
    // white-hot core, blue-violet edge; HDR, so the tone mapping saturates the core
    let core = pow(1.0 - abs(in.uv.x), 2.0);
    let edge = 1.0 - abs(in.uv.x);
    let hdr = (vec3<f32>(1.0, 0.97, 0.95) * core * 6.0 + LIGHTNING_COLOR * edge * 1.5) * in.bright;
    // additive: tone-mapped like the rest of the frame, alpha 0
    return vec4<f32>(post_process(hdr), 0.0);
}
