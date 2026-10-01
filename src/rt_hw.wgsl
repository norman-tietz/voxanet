// rt_hw.wgsl
// Hardware ray-traced shadow term (see hw_rt.rs). fs_gbuf (shader.wgsl) has written each shadow texel's world
// position, camera distance and normal; this compute shader casts one ray per texel toward the sun and
// writes the same (shadow, distance) texel as the ray march (cs_march), so the blur and upsampling are shared.
// Kept in its own small module: ray-query code in the big scene shader made every fragment much slower.

enable wgpu_ray_query;

struct HwParams {
    sun_dir: vec4<f32>,
}

@group(0) @binding(0) var g_pos: texture_2d<f32>;      // xyz world position, w camera distance (0 = sky)
@group(0) @binding(1) var g_nrm: texture_2d<f32>;      // xyz world normal
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var scene_tlas: acceleration_structure;
@group(0) @binding(4) var<uniform> hw: HwParams;

const HW_RT_RANGE: f32 = 4000.0; // world units; rays leave the planet long before that

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
    textureStore(out_tex, p, vec4<f32>(s, g.w, 0.0, 1.0));
}
