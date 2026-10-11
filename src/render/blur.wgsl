// blur.wgsl
// The shadow term's soft edges (rt_blur.rs, shadow.rs). Texel layout of the input `target`: r = shadow
// (1 lit, 0 shadowed), g = camera distance (0 = sky), b = blocker distance (hardware rays; 0 = none).
// Soft mode (bp.mode 1): fs_search_h/_v spread the mean blocker distance of the shadowed texels around
// each texel (same surface: depth test) and write the edge width, penumbra_width (mirror: shadow.rs);
// fs_blur then blurs with that width per texel. Fixed mode (0: /shadow off, the ray-march path): fs_blur
// reads `target` directly with bp.width. fs_blur's output: (shadow, distance, width, 1).

struct BlurParams {
    dir: vec2<f32>,    // (1, 0) horizontal pass, (0, 1) vertical pass
    focal: f32,        // pixels per world unit at distance 1
    width: f32,        // fixed mode's edge width in world units
    tan_sun: f32,      // tan of the sun's angular radius
    min_width: f32,    // shadow.rs MIN_WIDTH
    max_blocker: f32,  // shadow.rs MAX_BLOCKER
    mode: u32,         // 1: width per texel (soft), 0: fixed
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bp: BlurParams;

const TAPS: i32 = 8;          // samples on each side (shadow.rs TAPS)
const SEARCH_TAPS: i32 = 4;   // the search's: it only averages distances (shadow.rs SEARCH_TAPS)
const MAX_RADIUS: f32 = 32.0; // pixels (shadow.rs MAX_RADIUS_PX)

// one triangle covering the screen
@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// shadow.rs penumbra_width
fn penumbra_width(t: f32) -> f32 {
    return min(t, bp.max_blocker) * bp.tan_sun + bp.min_width;
}

// whether neighbour `d` (camera distance) lies on the same surface as the centre `c` at tap `k`: the
// allowed difference grows with the offset, so slanted floors seen at grazing angles still count
fn same_surface(c: f32, d: f32, k: i32, taps: i32) -> bool {
    let tol = 0.05 + 0.1 * c * f32(abs(k)) / f32(taps);
    return d > 0.0 && abs(d - c) <= tol;
}

// the search radius in pixels at camera distance `dist`: the widest edge possible there
fn search_radius(dist: f32) -> f32 {
    return min(penumbra_width(bp.max_blocker) * bp.focal / dist, MAX_RADIUS);
}

// horizontal search: (Σ blocker, count, distance, shadow) of the shadowed texels along the row
@fragment
fn fs_search_h(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(src));
    let pc = vec2<i32>(pos.xy);
    let c = textureLoad(src, pc, 0);
    if (c.g <= 0.0) { return vec4<f32>(0.0, 0.0, 0.0, c.r); }
    let step = search_radius(c.g) / f32(SEARCH_TAPS);
    var sum = 0.0;
    var n = 0.0;
    for (var k = -SEARCH_TAPS; k <= SEARCH_TAPS; k++) {
        let q = clamp(pc + vec2<i32>(i32(round(f32(k) * step)), 0), vec2<i32>(0), size - 1);
        let s = textureLoad(src, q, 0);
        if (s.b > 0.0 && same_surface(c.g, s.g, k, SEARCH_TAPS)) {
            // capped per sample: one far blocker (a distant ridge) mustn't wash out a contact edge
            sum += min(s.b, bp.max_blocker);
            n += 1.0;
        }
    }
    return vec4<f32>(sum, n, c.g, c.r);
}

// vertical search over the horizontal sums: the mean blocker distance of the shadowed texels around,
// turned into the edge width; none around (lit far from any shadow, or sky): MIN_WIDTH
@fragment
fn fs_search_v(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(src));
    let pc = vec2<i32>(pos.xy);
    let c = textureLoad(src, pc, 0); // (Σ blocker, count, distance, shadow)
    if (c.b <= 0.0) { return vec4<f32>(c.a, 0.0, bp.min_width, 1.0); } // sky
    let step = search_radius(c.b) / f32(SEARCH_TAPS);
    var sum = 0.0;
    var n = 0.0;
    for (var k = -SEARCH_TAPS; k <= SEARCH_TAPS; k++) {
        let q = clamp(pc + vec2<i32>(0, i32(round(f32(k) * step))), vec2<i32>(0), size - 1);
        let s = textureLoad(src, q, 0);
        if (same_surface(c.b, s.b, k, SEARCH_TAPS)) {
            sum += s.r;
            n += s.g;
        }
    }
    let width = select(bp.min_width, penumbra_width(sum / max(n, 1.0)), n > 0.0);
    return vec4<f32>(c.a, c.b, width, 1.0);
}

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(src));
    let pc = vec2<i32>(pos.xy);
    let c = textureLoad(src, pc, 0);
    // the edge width: the search's per texel (soft), else the fixed one
    let width = select(bp.width, c.b, bp.mode == 1u);
    if (c.g <= 0.0) { return vec4<f32>(c.r, c.g, width, 1.0); }

    // a width in the world, so the radius in pixels shrinks with distance
    let radius = min(width * bp.focal / c.g, MAX_RADIUS);
    if (radius < 0.5) { return vec4<f32>(c.r, c.g, width, 1.0); }
    let step = radius / f32(TAPS);

    var sum = 0.0;
    var wsum = 0.0;
    for (var k = -TAPS; k <= TAPS; k++) {
        let q = clamp(pc + vec2<i32>(round(bp.dir * (f32(k) * step))), vec2<i32>(0), size - 1);
        let s = textureLoad(src, q, 0);
        // skip samples from other surfaces (sky, silhouettes)
        if (!same_surface(c.g, s.g, k, TAPS)) { continue; }
        let wgt = exp(-f32(k * k) / f32(TAPS * TAPS / 2));
        sum += s.r * wgt;
        wsum += wgt;
    }
    // the centre is always on its own surface, so wsum > 0
    return vec4<f32>(sum / wsum, c.g, width, 1.0);
}
