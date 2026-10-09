// rt_hw.wgsl
// Hardware ray-traced shadow term (see hw_rt.rs). cs_gbuf_down (shader.wgsl) has written each shadow texel's world
// position, camera distance and normal; this compute shader casts one ray per texel toward the sun and a
// few short ambient-occlusion rays (ao.rs), and writes (shadow, distance, ao) like the ray march (cs_march,
// which writes ao = 1), so the blur and upsampling are shared.
// Kept in its own small module: ray-query code in the big scene shader made every fragment much slower.

enable wgpu_ray_query;

struct HwParams {
    sun_dir: vec4<f32>,
    ao: vec4<f32>, // x: radius in world units, y: ray count (0 = no AO, ao = 1), z: strength, w: fade end (ao.rs)
}

@group(0) @binding(0) var g_pos: texture_2d<f32>;      // xyz world position, w camera distance (0 = sky)
@group(0) @binding(1) var g_nrm: texture_2d<f32>;      // xyz world normal
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var scene_tlas: acceleration_structure;
@group(0) @binding(4) var<uniform> hw: HwParams;

const HW_RT_RANGE: f32 = 4000.0; // world units; rays leave the planet long before that

const GOLDEN_ANGLE: f32 = 2.39996323;
const TAU: f32 = 6.28318531;

// a fixed per-pixel rotation (interleaved gradient noise): stable from frame to frame, different
// between neighbouring pixels, so the blur averages neighbours' rays
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// ambient occlusion at surface point `pos` with normal N: `n` cosine-weighted rays (a Fibonacci spiral
// over the hemisphere, rotated per pixel), each the closest hit within `radius`, weighted
// (1 - t / radius)^2 so occlusion fades out toward the radius; returns 1 - strength * mean
fn ambient_occlusion(pos: vec3<f32>, N: vec3<f32>, px: vec2<f32>) -> f32 {
    let n = u32(hw.ao.y);
    if (n == 0u) {
        return 1.0;
    }
    let radius = hw.ao.x;
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(N.x) > 0.9);
    let t = normalize(cross(a, N));
    let b = cross(N, t);
    let rot = ign(px) * TAU;
    var occ = 0.0;
    // declared once, outside the loop: Mesa's Intel driver lost the device compiling a ray_query
    // variable declared inside it (the whole shader module, even with this function unused)
    var rq: ray_query;
    for (var k = 0u; k < n; k++) {
        let u = (f32(k) + 0.5) / f32(n);
        let phi = f32(k) * GOLDEN_ANGLE + rot;
        let r = sqrt(u);
        let dir = t * (r * cos(phi)) + b * (r * sin(phi)) + N * sqrt(1.0 - u);
        rayQueryInitialize(&rq, scene_tlas, RayDesc(0u, 0xFFu, 0.001, radius, pos + N * 0.002, dir));
        rayQueryProceed(&rq);
        let hit = rayQueryGetCommittedIntersection(&rq);
        if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
            let f = 1.0 - hit.t / radius;
            occ += f * f;
        }
    }
    return 1.0 - hw.ao.z * occ / f32(n);
}

@compute @workgroup_size(8, 8)
fn cs_shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let p = vec2<i32>(id.xy);
    let g = textureLoad(g_pos, p, 0);
    if (g.w <= 0.0) {
        textureStore(out_tex, p, vec4<f32>(1.0, 0.0, 0.0, 0.0)); // sky, same as the ray march's clear
        return;
    }
    let N = normalize(textureLoad(g_nrm, p, 0).xyz);
    let L = normalize(hw.sun_dir.xyz);
    var s = 0.0; // faces turned away from the sun are in their own shadow
    if (dot(N, L) > 0.0) {
        var rq: ray_query;
        // start just off the surface so the ray can't hit the face it starts on
        rayQueryInitialize(&rq, scene_tlas, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, 0.001, HW_RT_RANGE, g.xyz + N * 0.002, L));
        rayQueryProceed(&rq);
        s = select(1.0, 0.0, rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE);
    }
    // AO fades out with camera distance, no rays beyond hw.ao.w (ao.rs FADE_END)
    var ao = 1.0;
    if (g.w < hw.ao.w) {
        let fade = smoothstep(0.375 * hw.ao.w, hw.ao.w, g.w);
        ao = mix(ambient_occlusion(g.xyz, N, vec2<f32>(id.xy)), 1.0, fade);
    }
    textureStore(out_tex, p, vec4<f32>(s, g.w, ao, 1.0));
}
