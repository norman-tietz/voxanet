// blur.wgsl
// Separable, depth-aware blur of the ray-marched shadow term (see rt_blur.rs).
// Input/output texel: r = shadow (1 lit, 0 shadowed), g = distance to the camera (0 = sky),
// b = ambient occlusion (1 open; rt_hw.wgsl), blurred with the same weights as the shadow.

struct BlurParams {
    dir: vec2<f32>, // (1, 0) horizontal pass, (0, 1) vertical pass
    focal: f32,     // pixels per world unit at distance 1
    width: f32,     // penumbra width in world units
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bp: BlurParams;

const TAPS: i32 = 6;          // samples on each side
const MAX_RADIUS: f32 = 24.0; // pixels

// one triangle covering the screen
@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(src));
    let pc = vec2<i32>(pos.xy);
    let c = textureLoad(src, pc, 0);
    if (c.g <= 0.0) { return c; }

    // constant width in the world, so the radius in pixels shrinks with distance
    let radius = min(bp.width * bp.focal / c.g, MAX_RADIUS);
    if (radius < 0.5) { return c; }
    let step = radius / f32(TAPS);

    var sum = vec2<f32>(0.0); // shadow, ambient occlusion
    var wsum = 0.0;
    for (var k = -TAPS; k <= TAPS; k++) {
        let q = clamp(pc + vec2<i32>(round(bp.dir * (f32(k) * step))), vec2<i32>(0), size - 1);
        let s = textureLoad(src, q, 0);
        // skip samples from other surfaces (sky, silhouettes); the allowed depth difference grows
        // with the offset so slanted floors seen at grazing angles still blur
        let tol = 0.05 + 0.1 * c.g * f32(abs(k)) / f32(TAPS);
        if (s.g <= 0.0 || abs(s.g - c.g) > tol) { continue; }
        let wgt = exp(-f32(k * k) / f32(TAPS * TAPS / 2));
        sum += s.rb * wgt;
        wsum += wgt;
    }
    let r = sum / wsum;
    return vec4<f32>(r.x, c.g, r.y, 1.0);
}
