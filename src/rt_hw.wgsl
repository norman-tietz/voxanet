// rt_hw.wgsl
// Hardware ray-traced shadow term and ambient occlusion (see hw_rt.rs). cs_gbuf_down (shader.wgsl) has
// written each shadow texel's world position, camera distance and normal. cs_shadow casts one ray per
// texel toward the sun and writes (shadow, distance, blocker distance, 1) (cs_march: the same with 1 for the
// third; the blocker distance feeds the soft shadows, blur.wgsl and shadow.rs). The progressive AO
// (ao.rs) runs in four more dispatches: cs_ao_sample casts AO rays for one texel per BLOCK x BLOCK block
// (Bayer-rotated per frame), cs_ao_accumulate adds them to a running mean per texel (its history
// reprojected from the last frame; reset on settings, size and block edits), cs_ao_fill_h/_v fill and
// blur that sparse accumulation into the AO
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
    slow_cap: f32,     // accumulation caps while the history moves on screen (ao::motion_cap)
    fast_cap: f32,
    prev_cam: vec4<f32>,          // xyz: last frame's camera (planet frame)
    prev_view_proj: mat4x4<f32>,  // last frame's view-projection: the history's reprojection
    view_proj: mat4x4<f32>,       // this frame's: the history offset is the projection difference
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
@group(1) @binding(5) var fill_tmp_w: texture_storage_2d<rgba16float, write>; // fill_h: (sum, weight, dist, radius)
@group(1) @binding(6) var fill_tmp_r: texture_2d<f32>;                      // fill_v
@group(1) @binding(7) var ao_out_w: texture_storage_2d<rgba16float, write>; // fill_v: (ao, dist)
@group(1) @binding(8) var<uniform> aop: AoParams;

const HW_RT_RANGE: f32 = 4000.0; // world units; rays leave the planet long before that

const GOLDEN_ANGLE: f32 = 2.39996323;
const TAU: f32 = 6.28318531;

// PCG hash: independent values for every (texel, frame), so the samples accumulated at a texel look
// in uncorrelated directions. (Interleaved gradient noise shifted per frame gave texels along its
// diagonals the same direction sequences, which accumulated into diagonal screen-space stripes.)
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn unit(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0; // [0, 1)
}

// ambient occlusion at surface point `pos` with normal N: `n` cosine-weighted rays (a Fibonacci spiral
// over the hemisphere, rotated per pixel), each the closest hit within `radius`, weighted
// (1 - t / radius)^2 so occlusion fades out toward the radius; returns 1 - strength * mean
// `seed`: this sample's hash of (texel, frame)
fn ambient_occlusion(pos: vec3<f32>, N: vec3<f32>, seed: u32) -> f32 {
    let n = u32(aop.ao.y);
    if (n == 0u) {
        return 1.0;
    }
    let radius = aop.ao.x;
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(N.x) > 0.9);
    let t = normalize(cross(a, N));
    let b = cross(N, t);
    let rot = unit(seed) * TAU;
    // the elevation stratum's position, jittered like the rotation, so the accumulated samples cover
    // the whole hemisphere, not one fixed ring per ray
    let jitter = unit(pcg(seed ^ 0x9e3779b9u));
    var occ = 0.0;
    // declared once, outside the loop: Mesa's Intel driver lost the device compiling a ray_query
    // variable declared inside it (the whole shader module, even with this function unused)
    var rq: ray_query;
    for (var k = 0u; k < n; k++) {
        let u = (f32(k) + jitter) / f32(n);
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
    var blocker = 0.0; // distance to what casts the shadow (0: lit, or facing away): soft shadows (shadow.rs)
    if (dot(N, L) > 0.0) {
        var rq: ray_query;
        // start just off the surface so the ray can't hit the face it starts on; no early termination,
        // so the committed hit is the nearest blocker (a far first hit would soften contact shadows)
        rayQueryInitialize(&rq, scene_tlas, RayDesc(0u, 0xFFu, 0.001, HW_RT_RANGE, g.xyz + N * 0.002, L));
        rayQueryProceed(&rq);
        let hit = rayQueryGetCommittedIntersection(&rq);
        if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
            blocker = hit.t;
        } else {
            s = 1.0;
        }
    }
    textureStore(out_tex, p, vec4<f32>(s, g.w, blocker, 1.0)); // AO: its own texture (cs_ao_*)
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
                let seed = pcg(px.x ^ pcg(px.y ^ pcg(aop.frame)));
                r.x = mix(ambient_occlusion(g.xyz, N, seed), 1.0, fade);
            }
        }
    }
    textureStore(ao_new_w, vec2<i32>(id.xy), r);
}

// the accumulation cap for a texel whose history moved `m` texels on screen (mirror: ao::motion_cap):
// the full cap while still, toward slow_cap up to half a texel, fast_cap from 8 texels on
fn ao_motion_cap(m: f32) -> f32 {
    let still_to_slow = mix(f32(aop.cap), aop.slow_cap, smoothstep(0.05, 0.5, m));
    return mix(still_to_slow, aop.fast_cap, smoothstep(0.5, 8.0, m));
}

// how much this texel's camera distance changes per texel across its surface: the smaller of the
// forward and backward differences per axis (the larger one may cross a silhouette), summed over both
// axes since a bilinear tap lies up to a texel away in each. Grazing surfaces change by several percent
// per texel, more than the history's plain depth tolerance; capped so slivers don't accept anything.
fn depth_slope(p: vec2<i32>, size: vec2<u32>, d: f32) -> f32 {
    let hi = vec2<i32>(size) - 1;
    let l = textureLoad(g_pos, clamp(p - vec2<i32>(1, 0), vec2<i32>(0), hi), 0).w;
    let r = textureLoad(g_pos, clamp(p + vec2<i32>(1, 0), vec2<i32>(0), hi), 0).w;
    let u = textureLoad(g_pos, clamp(p - vec2<i32>(0, 1), vec2<i32>(0), hi), 0).w;
    let b = textureLoad(g_pos, clamp(p + vec2<i32>(0, 1), vec2<i32>(0), hi), 0).w;
    let sx = min(abs(l - d), abs(r - d));
    let sy = min(abs(u - d), abs(b - d));
    return min(sx + sy, 0.1 * d);
}

// the running mean per texel: (mean, sample count, distance). The history is read reprojected: this
// texel's world point through last frame's view-projection (mirror: ao::reproject), bilinear over the
// four history texels whose stored distance matches the point's distance from last frame's camera
// (others are other surfaces: silhouettes, disocclusions); its count is capped by how far it moved on
// screen, so moving views stay responsive and still ones converge. aop.frames 0 (settings, size or a
// block edit changed): start over. The texel that got this frame's sample adds it.
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
    var mean = 1.0;
    var count = 0.0;
    var cap = f32(aop.cap);
    if (aop.frames > 0u) {
        // where last frame's and this frame's view-projection put this texel's world point (mirror:
        // ao::history_offset): as a difference of the same formula, a still camera reads exactly its
        // own texel, also where the shadow targets are smaller than the screen (their world points
        // lie at screen-pixel centres, which don't reproject onto texel centres)
        let prev = aop.prev_view_proj * vec4<f32>(g.xyz, 1.0);
        let cur = aop.view_proj * vec4<f32>(g.xyz, 1.0);
        if (prev.w > 0.0 && cur.w > 0.0) {
            let texel = vec2<f32>(0.5, -0.5) * vec2<f32>(size);
            let offset = (prev.xy / prev.w - cur.xy / cur.w) * texel;
            let q = vec2<f32>(id.xy) + offset;
            let expected = distance(aop.prev_cam.xyz, g.xyz);
            var tol = 0.02 * expected + 0.02;
            var widened = false;
            let base = vec2<i32>(floor(q));
            let f = q - floor(q);
            var msum = 0.0;
            var csum = 0.0;
            var wsum = 0.0;
            for (var i = 0; i < 4; i++) {
                let o = vec2<i32>(i & 1, i >> 1u);
                let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
                // a still camera's offset is exactly 0: three of the taps weigh nothing, skip their loads
                if (w <= 0.0) { continue; }
                let t = base + o;
                if (any(t < vec2<i32>(0)) || any(t >= vec2<i32>(size))) { continue; }
                let h = textureLoad(acc_prev, t, 0);
                if (h.y <= 0.0) { continue; }
                // off by more than the plain tolerance: maybe a grazing surface (depth_slope, read
                // only then: most texels never need it)
                if (abs(h.z - expected) > tol && !widened) {
                    tol += depth_slope(p, size, g.w);
                    widened = true;
                }
                if (abs(h.z - expected) > tol) { continue; }
                msum += h.x * w;
                csum += h.y * w;
                wsum += w;
            }
            if (wsum > 1e-4) {
                cap = ao_motion_cap(length(offset));
                mean = msum / wsum;
                // not renormalised: a history only partly backed by valid taps counts for that share,
                // so edges and disocclusions fill wider and take new samples faster
                count = min(csum, cap);
            }
        }
    }
    if (all(id.xy % aop.block == vec2<u32>(aop.offset_x, aop.offset_y))) {
        let s = textureLoad(ao_new_r, vec2<i32>(id.xy / aop.block), 0);
        if (s.y > 0.0) {
            count = min(count + 1.0, cap);
            mean = mean + (s.x - mean) / count;
        }
    }
    textureStore(acc_cur, p, vec4<f32>(mean, count, g.w, 1.0));
}

const FILL_MAX_TAPS: i32 = 8;      // each side: at least FILL_MAX_RADIUS, so the step stays <= 1 texel
const FILL_MAX_RADIUS: f32 = 8.0;  // texels: bridges the 4x4 sample gaps right after a reset

// the fill's radius from the centre texel's sample count (mirror: ao::fill_radius_for_count): wide where
// a surface just appeared, to bridge the gaps between the sparse samples, narrow where the history
// carries many. No world-space minimum like the shadow blur's: converged AO needs none, and close up it
// forced the widest fill over the whole screen (~5 ms)
fn fill_radius(count: f32) -> f32 {
    return clamp(2.0 * f32(aop.block) / sqrt(count + 1.0), 1.5, FILL_MAX_RADIUS);
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

// horizontal: over the texels that have samples, (sum of weighted means, sum of weights, distance,
// radius); the vertical pass uses the same radius
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
    let radius = fill_radius(c.y);
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
    textureStore(fill_tmp_w, p, vec4<f32>(sum, wsum, c.z, radius));
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
    let radius = c.w;
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
