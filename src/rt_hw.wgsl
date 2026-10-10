// rt_hw.wgsl
// Hardware ray-traced shadow term and ambient occlusion (see hw_rt.rs). cs_gbuf_down (shader.wgsl) has
// written each shadow texel's world position, camera distance and normal. cs_shadow casts one ray per
// texel toward the sun and writes (shadow, distance, 1) like the ray march (cs_march). The progressive AO
// (ao.rs) runs in four more dispatches: cs_ao_sample casts AO rays for one texel per BLOCK x BLOCK block
// (Bayer-rotated per frame), cs_ao_accumulate adds them to a running mean per texel (reset when the
// camera or the meshes change), cs_ao_fill_h/_v fill and blur that sparse accumulation into the AO
// texture the lighting upsamples (RtBlur::ao_out, group 2 binding 1 of the scene).
// Kept in its own small module: ray-query code in the big scene shader made every fragment much slower.

enable wgpu_ray_query;

struct HwParams {
    sun_dir: vec4<f32>,
}

// hw_rt.rs AoFrameParams
struct AoParams {
    ao: vec4<f32>,     // x: radius in world units, y: rays per sample, z: strength, w: fade end (ao.rs)
    frame: u32,        // running frame index: the ray rotation
    frames: u32,       // frames since the accumulation was reset (0: start over)
    cap: u32,          // accumulation cap per texel
    block: u32,        // one sample per block x block texels
    offset_x: u32,     // this frame's sample texel inside each block (ao::sample_offset)
    offset_y: u32,
    fill_radius: f32,  // texels (ao::fill_radius_texels)
    focal: f32,        // shadow-target pixels per world unit at distance 1
}

@group(0) @binding(0) var g_pos: texture_2d<f32>;      // xyz world position, w camera distance (0 = sky)
@group(0) @binding(1) var g_nrm: texture_2d<f32>;      // xyz world normal
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var scene_tlas: acceleration_structure;
@group(0) @binding(4) var<uniform> hw: HwParams;

// group 1: one bind group per AO stage (hw_rt.rs), distinct bindings, so no texture is read and written
// in the same dispatch
@group(1) @binding(0) var ao_new_w: texture_storage_2d<rgba16float, write>; // sample: (ao, dist) per block
@group(1) @binding(1) var ao_new_r: texture_2d<f32>;                        // accumulate
@group(1) @binding(2) var acc_prev: texture_2d<f32>;                        // accumulate: (mean, count, dist)
@group(1) @binding(3) var acc_cur: texture_storage_2d<rgba16float, write>;  // accumulate
@group(1) @binding(4) var acc_r: texture_2d<f32>;                           // fill_h
@group(1) @binding(5) var fill_tmp_w: texture_storage_2d<rgba16float, write>; // fill_h: (sum, weight, dist)
@group(1) @binding(6) var fill_tmp_r: texture_2d<f32>;                      // fill_v
@group(1) @binding(7) var ao_out_w: texture_storage_2d<rgba16float, write>; // fill_v: (ao, dist)
@group(1) @binding(8) var<uniform> aop: AoParams;

const HW_RT_RANGE: f32 = 4000.0; // world units; rays leave the planet long before that

const GOLDEN_ANGLE: f32 = 2.39996323;
const TAU: f32 = 6.28318531;

// a per-pixel rotation (interleaved gradient noise); the AO sample shifts it every frame, so samples
// accumulated at a texel over frames look in different directions
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// ambient occlusion at surface point `pos` with normal N: `n` cosine-weighted rays (a Fibonacci spiral
// over the hemisphere, rotated per pixel), each the closest hit within `radius`, weighted
// (1 - t / radius)^2 so occlusion fades out toward the radius; returns 1 - strength * mean
fn ambient_occlusion(pos: vec3<f32>, N: vec3<f32>, px: vec2<f32>) -> f32 {
    let n = u32(aop.ao.y);
    if (n == 0u) {
        return 1.0;
    }
    let radius = aop.ao.x;
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
    return 1.0 - aop.ao.z * occ / f32(n);
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
    textureStore(out_tex, p, vec4<f32>(s, g.w, 1.0, 1.0)); // AO: its own texture (cs_ao_*)
}

// --- progressive AO (ao.rs) ---

// one AO sample per block: at texel block * BLOCK + offset, `aop.ao.y` rays, faded out with camera
// distance (none beyond aop.ao.w: under a texel there, and on large planets the morphed LOD terrain the
// TLAS holds unmorphed). Out: (ao, distance), distance 0 for sky or outside the targets.
@compute @workgroup_size(8, 8)
fn cs_ao_sample(@builtin(global_invocation_id) id: vec3<u32>) {
    let bsize = textureDimensions(ao_new_w);
    if (id.x >= bsize.x || id.y >= bsize.y) { return; }
    let size = textureDimensions(g_pos);
    let px = id.xy * aop.block + vec2<u32>(aop.offset_x, aop.offset_y);
    var r = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    if (px.x < size.x && px.y < size.y) {
        let g = textureLoad(g_pos, vec2<i32>(px), 0);
        if (g.w > 0.0) {
            r = vec4<f32>(1.0, g.w, 0.0, 1.0);
            if (g.w < aop.ao.w) {
                let N = normalize(textureLoad(g_nrm, vec2<i32>(px), 0).xyz);
                let fade = smoothstep(0.375 * aop.ao.w, aop.ao.w, g.w);
                let rot = vec2<f32>(px) + f32(aop.frame % 251u) * 5.588;
                r.x = mix(ambient_occlusion(g.xyz, N, rot), 1.0, fade);
            }
        }
    }
    textureStore(ao_new_w, vec2<i32>(id.xy), r);
}

// the running mean per texel: (mean, sample count, distance). Starts over when aop.frames is 0; the
// texel that got this frame's sample adds it, its count capped at aop.cap (a moving average beyond).
@compute @workgroup_size(8, 8)
fn cs_ao_accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(acc_cur);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let p = vec2<i32>(id.xy);
    let g = textureLoad(g_pos, p, 0);
    if (g.w <= 0.0) {
        textureStore(acc_cur, p, vec4<f32>(1.0, 0.0, 0.0, 1.0)); // sky
        return;
    }
    var acc = vec4<f32>(1.0, 0.0, g.w, 1.0);
    if (aop.frames > 0u) {
        let prev = textureLoad(acc_prev, p, 0);
        acc = vec4<f32>(prev.x, prev.y, g.w, 1.0);
    }
    if (all(id.xy % aop.block == vec2<u32>(aop.offset_x, aop.offset_y))) {
        let s = textureLoad(ao_new_r, vec2<i32>(id.xy / aop.block), 0);
        if (s.y > 0.0) {
            let count = min(acc.y + 1.0, f32(aop.cap));
            acc.x = acc.x + (s.x - acc.x) / count;
            acc.y = count;
        }
    }
    textureStore(acc_cur, p, acc);
}

const FILL_MAX_TAPS: i32 = 12;     // each side
const FILL_MAX_RADIUS: f32 = 24.0; // texels
const FILL_WORLD_WIDTH: f32 = 0.3; // world units, like the shadow blur (rt_blur.rs PENUMBRA_WIDTH)

// the fill's radius at camera distance `dist`: the progressive radius (wide right after a reset, to
// bridge the gaps between the sparse samples), at least the shadow blur's world-space width
fn fill_radius(dist: f32) -> f32 {
    return min(max(aop.fill_radius, FILL_WORLD_WIDTH * aop.focal / dist), FILL_MAX_RADIUS);
}

// taps about one texel apart, so the sparse samples right after a reset (one per 4x4 block) are never
// stepped over (a coarser step aliased into a dot pattern)
fn fill_taps(radius: f32) -> i32 {
    return clamp(i32(ceil(radius)), 1, FILL_MAX_TAPS);
}

// taps on other surfaces (sky, silhouettes) don't count; the allowed depth difference grows with the
// offset so slanted floors seen at grazing angles still blur (blur.wgsl's rule); `t` = offset / radius
fn fill_depth_ok(d: f32, center: f32, t: f32) -> bool {
    return d > 0.0 && abs(d - center) <= 0.05 + 0.1 * center * t;
}

// weight of a tap `x` texels from the centre of a fill `radius` texels wide: a flat Gaussian (sigma =
// radius / sqrt 2), so right after a reset a texel at a sample position isn't dominated by that one
// noisy sample (a peaked kernel showed the 4x4 sample grid as dots)
fn fill_weight(x: f32, radius: f32) -> f32 {
    return exp(-x * x / (radius * radius));
}

// horizontal: over the texels that have samples, (sum of weighted means, sum of weights, distance)
@compute @workgroup_size(8, 8)
fn cs_ao_fill_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(fill_tmp_w);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let p = vec2<i32>(id.xy);
    let c = textureLoad(acc_r, p, 0);
    if (c.z <= 0.0) {
        textureStore(fill_tmp_w, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let radius = fill_radius(c.z);
    let taps = fill_taps(radius);
    let step = radius / f32(taps);
    var sum = 0.0;
    var wsum = 0.0;
    for (var k = -taps; k <= taps; k++) {
        let x = round(f32(k) * step);
        let q = clamp(p + vec2<i32>(i32(x), 0), vec2<i32>(0), vec2<i32>(size) - 1);
        let s = textureLoad(acc_r, q, 0);
        if (s.y <= 0.0 || !fill_depth_ok(s.z, c.z, abs(x) / radius)) { continue; }
        let w = fill_weight(x, radius);
        sum += s.x * w;
        wsum += w;
    }
    textureStore(fill_tmp_w, p, vec4<f32>(sum, wsum, c.z, 1.0));
}

// vertical: combines the rows' sums into the AO, (ao, distance); 1 where no sample is in reach
@compute @workgroup_size(8, 8)
fn cs_ao_fill_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(ao_out_w);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let p = vec2<i32>(id.xy);
    let c = textureLoad(fill_tmp_r, p, 0);
    if (c.z <= 0.0) {
        textureStore(ao_out_w, p, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let radius = fill_radius(c.z);
    let taps = fill_taps(radius);
    let step = radius / f32(taps);
    var sum = 0.0;
    var wsum = 0.0;
    for (var k = -taps; k <= taps; k++) {
        let x = round(f32(k) * step);
        let q = clamp(p + vec2<i32>(0, i32(x)), vec2<i32>(0), vec2<i32>(size) - 1);
        let s = textureLoad(fill_tmp_r, q, 0);
        if (!fill_depth_ok(s.z, c.z, abs(x) / radius)) { continue; }
        let w = fill_weight(x, radius);
        sum += s.x * w;
        wsum += s.y * w;
    }
    let ao = select(1.0, sum / wsum, wsum > 1e-6);
    textureStore(ao_out_w, p, vec4<f32>(ao, c.z, 0.0, 1.0));
}
