

// basic shading (IMPROVE THIS LATER)
struct Global {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    screen: vec4<f32>, // width, height in pixels
}

@group(0) @binding(0) var<uniform> global: Global;

// ray-marched shadows: per cube face a window of solid/air bits near the player, see rt_shadow.rs
struct FaceWindow {
    origin_u: i32,
    origin_v: i32,
    size: u32, // 0 = no data for this face
    base_layer: i32,
    words_per_column: u32,
    offset: u32,
    max_layer: i32,
    tile_offset: u32, // highest solid layer per RT_TILE x RT_TILE columns
}
struct RtParams {
    faces: array<FaceWindow, 6>,
    resolution: u32,
    enabled: u32,
    max_layer: i32,
    _pad: u32,
}
@group(0) @binding(1) var<uniform> rt: RtParams;
@group(0) @binding(2) var<storage, read> rt_bits: array<u32>;
// blurred shadow term written by fs_rt + blur.wgsl (r = shadow), see rt_blur.rs
@group(2) @binding(0) var rt_blurred: texture_2d<f32>;

struct Local {
    model: mat4x4<f32>,
    params: vec4<f32>, // x = opacity
}
@group(1) @binding(0) var<uniform> local: Local;

// --- CONSTANTS ---
// Natural, physical light values
const SUN_COLOR       = vec3<f32>(1.6, 1.5, 1.3);    // High intensity warm sun
const SKY_COLOR       = vec3<f32>(0.15, 0.3, 0.6);   // Deep blue ambient sky
const GROUND_COLOR    = vec3<f32>(0.05, 0.04, 0.03); // Dark earth ambient bounce
const SHADOW_OPACITY  = 0.85;                        // Shadows are not pitch black
const SRGB_TO_P3 = mat3x3<f32>(                     // linear sRGB -> linear Display P3 (column-major)
    vec3<f32>(0.8225, 0.0332, 0.0171),
    vec3<f32>(0.1774, 0.9669, 0.0724),
    vec3<f32>(0.0000, 0.0000, 0.9108),
);

// --- VERTEX SHADER ---

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) view_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    
    // World Position
    let world_pos = local.model * vec4<f32>(in.pos, 1.0);
    out.world_pos = world_pos.xyz;
    
    // Clip Position (Main Camera)
    out.clip_pos = global.view_proj * world_pos;
    
    // Normal Transformation
    let normal_mat = mat3x3<f32>(
        local.model[0].xyz,
        local.model[1].xyz,
        local.model[2].xyz
    );
    out.world_normal = normalize(normal_mat * in.normal);
    
    // Color (Vertex Color + Baked AO)
    out.color = in.color;
    out.view_pos = global.camera_pos.xyz;

    return out;
}

// --- RAY-MARCHED SHADOWS ---
// Walks the window's cells from the fragment toward the sun. Block space (u, v, layer) is curved
// in world space, so the ray is split into short segments and the direction is re-linearised per segment.

const LAYER_K: f32 = 0.85;     // CoordSystem::K in gen.rs
const RT_RANGE: f32 = 128.0;   // max ray length in world units

struct BlockPos {
    face: u32,
    p: vec3<f32>, // continuous (u, v, layer)
}

// inverse of CoordSystem::cube_to_sphere for one face's in-plane pair (x, z) of a unit vector.
// Same as cubize_point in gen.rs, rewritten so it doesn't cancel catastrophically in f32 near face centres.
fn cube_pair(x: f32, z: f32) -> vec2<f32> {
    let a2 = 2.0 * x * x;
    let b2 = 2.0 * z * z;
    let d = sqrt(max((a2 - b2) * (a2 - b2) - 6.0 * (a2 + b2) + 9.0, 0.0));
    let cx = x * sqrt(12.0 / (a2 - b2 + 3.0 + d));
    let cz = z * sqrt(12.0 / (b2 - a2 + 3.0 + d));
    return clamp(vec2<f32>(cx, cz), vec2<f32>(-1.0), vec2<f32>(1.0));
}

// world position -> face + continuous block coordinates (CoordSystem::get_local_coords)
fn to_block(world: vec3<f32>) -> BlockPos {
    let res = f32(rt.resolution);
    let s = res * 0.5;
    let dist = length(world);
    let n = world / dist;
    let a = abs(n);

    var face: u32;
    var c: vec2<f32>;
    if (a.y >= a.x && a.y >= a.z) {
        face = select(1u, 0u, n.y > 0.0);
        c = cube_pair(n.x, n.z);
    } else if (a.x >= a.y && a.x >= a.z) {
        face = select(3u, 2u, n.x > 0.0);
        c = cube_pair(n.y, n.z);
    } else {
        face = select(5u, 4u, n.z > 0.0);
        c = cube_pair(n.x, n.y);
    }

    let layer = s * (1.0 + log(dist / s) / LAYER_K);
    return BlockPos(face, vec3<f32>((c * res + res) * 0.5, layer));
}

const RT_AIR: i32 = 0;
const RT_SOLID: i32 = 1;
const RT_NO_DATA: i32 = 2;  // outside every window: stop and treat as lit
const RT_CONTINUE: i32 = 3;
const RT_CROSSES_FACE: i32 = 4;
const RT_TILE: i32 = 8;  // rt_shadow.rs TILE

// true if the segment a..b (same face, ray climbing) stays above every tile it passes over
fn rt_above_tiles(a: BlockPos, b: BlockPos) -> bool {
    let w = rt.faces[a.face];
    let size = i32(w.size);
    let lo = vec2<i32>(floor(min(a.p.xy, b.p.xy))) - vec2<i32>(w.origin_u, w.origin_v);
    let hi = vec2<i32>(floor(max(a.p.xy, b.p.xy))) - vec2<i32>(w.origin_u, w.origin_v);
    if (any(lo < vec2<i32>(0)) || any(hi >= vec2<i32>(size))) { return false; }
    let tiles = (size + RT_TILE - 1) / RT_TILE;
    let lowest = min(a.p.z, b.p.z);
    for (var tv = lo.y / RT_TILE; tv <= hi.y / RT_TILE; tv++) {
        for (var tu = lo.x / RT_TILE; tu <= hi.x / RT_TILE; tu++) {
            let top = rt_bits[w.tile_offset + u32(tv * tiles + tu)];
            if (lowest < f32(top + 1u)) { return false; }
        }
    }
    return true;
}

fn rt_cell(face: u32, cell: vec3<i32>) -> i32 {
    return rt_cell_in(rt.faces[face], cell);
}

fn rt_cell_in(w: FaceWindow, cell: vec3<i32>) -> i32 {
    let lu = cell.x - w.origin_u;
    let lv = cell.y - w.origin_v;
    if (w.size == 0u || lu < 0 || lv < 0 || lu >= i32(w.size) || lv >= i32(w.size)) { return RT_NO_DATA; }
    let l = cell.z - w.base_layer;
    if (l < 0) { return RT_SOLID; }                                // below the window: solid ground
    if (l >= i32(w.words_per_column * 32u)) { return RT_AIR; }     // above the window: air
    let col = u32(lv) * w.size + u32(lu);
    let word = rt_bits[w.offset + col * w.words_per_column + u32(l) / 32u];
    return select(RT_AIR, RT_SOLID, ((word >> (u32(l) & 31u)) & 1u) == 1u);
}

// exact 3D DDA over the cells between a and b on one face (the cell containing a was checked already)
fn rt_walk(a: BlockPos, b: BlockPos) -> i32 {
    let res = i32(rt.resolution);
    let w = rt.faces[a.face]; // load once: indexing the uniform array per cell is slow
    let d = b.p - a.p;
    let step = vec3<i32>(sign(d));
    let inv = 1.0 / max(abs(d), vec3<f32>(1e-6));
    var cell = vec3<i32>(floor(a.p));
    let cell_f = vec3<f32>(cell);
    var t_max = select(a.p - cell_f, cell_f + 1.0 - a.p, d > vec3<f32>(0.0)) * inv;

    loop {
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            if (t_max.x > 1.0) { break; }
            cell.x += step.x;
            t_max.x += inv.x;
        } else if (t_max.y < t_max.z) {
            if (t_max.y > 1.0) { break; }
            cell.y += step.y;
            t_max.y += inv.y;
        } else {
            if (t_max.z > 1.0) { break; }
            cell.z += step.z;
            t_max.z += inv.z;
        }
        if (cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res) { return RT_CROSSES_FACE; }
        let c = rt_cell_in(w, cell);
        if (c != RT_AIR) { return c; }
    }
    return RT_CONTINUE;
}

// block coordinates of two faces don't line up: split the segment at the face edge (found by bisection)
// and walk each side on its own face. Point sampling is the fallback, e.g. near cube corners.
fn rt_cross(a: BlockPos, wa: vec3<f32>, wb: vec3<f32>) -> i32 {
    var lo = 0.0;
    var hi = 1.0;
    for (var i = 0; i < 10; i++) {
        let mid = 0.5 * (lo + hi);
        if (to_block(mix(wa, wb, mid)).face == a.face) { lo = mid; } else { hi = mid; }
    }
    var r = rt_walk(a, to_block(mix(wa, wb, lo)));
    if (r != RT_CONTINUE) { return r; }

    let c = to_block(mix(wa, wb, hi));
    let b = to_block(wb);
    if (c.face != b.face) { return rt_sample(wa, wb); }
    r = rt_cell(c.face, vec3<i32>(floor(c.p))); // rt_walk skips its start cell
    if (r != RT_AIR) { return r; }
    return rt_walk(c, b);
}

fn rt_sample(wa: vec3<f32>, wb: vec3<f32>) -> i32 {
    let n = max(i32(ceil(length(wb - wa) / 0.1)), 1);
    for (var k = 1; k <= n; k++) {
        let s = to_block(mix(wa, wb, f32(k) / f32(n)));
        let c = rt_cell(s.face, vec3<i32>(floor(s.p)));
        if (c != RT_AIR) { return c; }
    }
    return RT_CONTINUE;
}

// 1.0 = lit, 0.0 = shadowed
fn rt_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    var a = to_block(world_pos);
    if (rt.faces[a.face].size == 0u) { return 1.0; }

    // Faces are drawn as flat quads but are curved in block space, so the fragment can sit slightly
    // inside its own block. Snap the coordinate along the face normal onto the cell boundary,
    // just on the air side, so the ray can't clip the neighbouring block on the same line.
    let an = to_block(world_pos + N * 0.05);
    if (an.face == a.face) {
        let dn = an.p - a.p;
        let adn = abs(dn);
        if (adn.x >= adn.y && adn.x >= adn.z) {
            a.p.x = round(a.p.x) + sign(dn.x) * 1e-3;
        } else if (adn.y >= adn.z) {
            a.p.y = round(a.p.y) + sign(dn.y) * 1e-3;
        } else {
            a.p.z = round(a.p.z) + sign(dn.z) * 1e-3;
        }
    } else {
        a = to_block(world_pos + N * 0.01); // right at a face edge
    }

    // segments short enough that the planet's curvature over one segment stays below ~0.01 units
    let seg_len = clamp(sqrt(0.02 * f32(rt.resolution) * 0.5), 0.25, 4.0);
    let segments = min(i32(ceil(RT_RANGE / seg_len)), 512);
    let top = f32(rt.max_layer + 1);
    var wa = world_pos;

    for (var seg = 1; seg <= segments; seg++) {
        if (a.p.z >= top) { return 1.0; } // above every solid cell, and the ray only climbs
        let wb = world_pos + L * (seg_len * f32(seg));
        let b = to_block(wb);

        var r = RT_CROSSES_FACE;
        if (b.face == a.face) {
            r = RT_CONTINUE;
            if (!rt_above_tiles(a, b)) { r = rt_walk(a, b); }
        } else {
            r = rt_cross(a, wa, wb);
        }
        if (r == RT_CROSSES_FACE) { r = rt_sample(wa, wb); }
        if (r == RT_SOLID) { return 0.0; }
        if (r == RT_NO_DATA) { return 1.0; }
        a = b;
        wa = wb;
    }
    return 1.0;
}

// blurred ray-marched shadow term at this pixel (1 = lit); the UI bind group has rt.enabled = 0.
// The shadow targets can be smaller than the screen (rt_blur.rs MAX_RT_PIXELS): upsample from the
// four nearest texels, weighted bilinearly and by how well their camera distance matches this pixel's,
// so shadows don't bleed across silhouettes. At full resolution this is a single exact tap.
fn shadow_factor(in: VertexOut) -> f32 {
    if (rt.enabled != 1u) { return 1.0; }
    let dims = vec2<i32>(textureDimensions(rt_blurred));
    let p = in.clip_pos.xy * vec2<f32>(dims) / global.screen.xy - 0.5;
    let base = vec2<i32>(floor(p));
    let f = fract(p);
    let dist = distance(global.camera_pos.xyz, in.world_pos);

    var sum = 0.0;
    var wsum = 0.0;
    var closest = 1.0;
    var closest_err = 1e9;
    for (var i = 0; i < 4; i++) {
        let o = vec2<i32>(i & 1, i >> 1u);
        let t = textureLoad(rt_blurred, clamp(base + o, vec2<i32>(0), dims - 1), 0);
        if (t.g <= 0.0) { continue; } // sky
        let err = abs(t.g - dist) / dist;
        if (err < closest_err) { closest_err = err; closest = t.r; }
        let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let w = bilinear / (1.0 + (err / 0.02) * (err / 0.02));
        sum += t.r * w;
        wsum += w;
    }
    if (wsum < 1e-3) { return closest; }
    return sum / wsum;
}

// sharp ray-marched shadow term + camera distance, blurred afterwards (rt_blur.rs)
@fragment
fn fs_rt(in: VertexOut) -> @location(0) vec4<f32> {
    if (local.params.x < 1.0 && dither_opacity(in.clip_pos, local.params.x)) {
        discard;
    }
    let N = normalize(in.world_normal);
    let L = normalize(global.sun_dir.xyz);
    var s = 0.0; // faces turned away from the sun are in their own shadow
    if (dot(N, L) > 0.0) { s = rt_shadow(in.world_pos, N, L); }
    return vec4<f32>(s, distance(global.camera_pos.xyz, in.world_pos), 0.0, 1.0);
}

// world position, camera distance and normal per shadow texel for the hardware ray-traced shadows
// (hw_rt.rs, rt_hw.wgsl casts the rays in a compute pass)
struct GBufOut {
    @location(0) pos: vec4<f32>,
    @location(1) nrm: vec4<f32>,
}

@fragment
fn fs_gbuf(in: VertexOut) -> GBufOut {
    if (local.params.x < 1.0 && dither_opacity(in.clip_pos, local.params.x)) {
        discard;
    }
    var out: GBufOut;
    out.pos = vec4<f32>(in.world_pos, distance(global.camera_pos.xyz, in.world_pos));
    out.nrm = vec4<f32>(normalize(in.world_normal), 0.0);
    return out;
}

// --- UTILS ---

fn dither_opacity(pos: vec4<f32>, alpha: f32) -> bool {
    // 4x4 Ordered Dithering Matrix
    let dither_threshold = dot(vec2<f32>(171.0, 231.0), pos.xy);
    return fract(dither_threshold / 71.0) > alpha;
}

fn triplanar_detail(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    // Adds subtle grain to voxels so they don't look like plastic
    let p = pos * 2.0;
    let n = abs(normal);
    // Tight blend
    let w = pow(n, vec3<f32>(16.0)); 
    let weights = w / (w.x + w.y + w.z);
    
    // Fast hash noise
    let hx = fract(sin(dot(p.yz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let hy = fract(sin(dot(p.zx, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let hz = fract(sin(dot(p.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);

    return (hx * weights.x + hy * weights.y + hz * weights.z) * 2.0 - 1.0;
}

// --- TONE MAPPING (ACES) ---
// Industry standard for realistic color reproduction
fn aces_approx(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- FRAGMENT SHADER ---

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // 1. Transparency Dithering
    if (local.params.x < 1.0 && dither_opacity(in.clip_pos, local.params.x)) {
        discard;
    }

    let N = normalize(in.world_normal);
    let L = normalize(global.sun_dir.xyz);
    let V = normalize(global.camera_pos.xyz - in.world_pos);

    // 2. Material Setup
    // De-Gamma the vertex color to Linear Space for math
    let vert_color_linear = pow(in.color, vec3<f32>(2.2));
    
    // Apply Detail Noise (Grain)
    let noise = triplanar_detail(in.world_pos, N);
    let albedo = vert_color_linear * (1.0 + 0.03 * noise);

    // 3. Lighting Math
    let NdotL = max(dot(N, L), 0.0);
    
    // Shadow (ray-marched, see rt_shadow() / fs_rt); faces turned away from the sun are in their own shadow
    let shadow_raw = select(shadow_factor(in), 0.0, NdotL <= 0.0);
    // Smooth transition shadow
    let shadow = mix(1.0 - SHADOW_OPACITY, 1.0, shadow_raw);

    // A. Direct Sun Light
    let direct_light = SUN_COLOR * NdotL * shadow;

    // B. Hemispheric Ambient
    // Top of objects gets Sky Color, Bottom gets Ground Bounce
    let up_dot = dot(N, normalize(in.world_pos)); // Relative Up for sphere
    let hemi_factor = up_dot * 0.5 + 0.5;
    let ambient_light = mix(GROUND_COLOR, SKY_COLOR, hemi_factor);

    // C. Fresnel Rim
    // Adds a subtle glow at grazing angles (atmosphere dust effect)
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    let rim_light = SKY_COLOR * fresnel * 0.2 * shadow;

    // Combine
    // Note: Ambient is multiplied by albedo (diffuse reflection)
    var final_color = albedo * (direct_light + ambient_light + rim_light);

    // 4. Fog (Atmospheric Scattering)
    let dist = distance(global.camera_pos.xyz, in.world_pos);
    // Fog density tuned for the scale defined in gen.rs
    let fog_density = 0.0015; 
    let fog_factor = 1.0 - exp(-(dist * fog_density) * (dist * fog_density * 0.5)); // Exp2 fog
    
    // Horizon Fog Color blends into Sky
    let fog_col = mix(SKY_COLOR * 0.8, vec3<f32>(0.7, 0.8, 0.9), 0.2); 
    final_color = mix(final_color, fog_col, clamp(fog_factor, 0.0, 1.0));

    // 5. Post Processing
    // Tone Mapping (HDR -> LDR)
    final_color = aces_approx(final_color);
    
    // Gamma Correction (Linear -> sRGB)
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    // when the surface is tagged Display P3 (sun_dir.w = 1, see renderer.rs) macOS colour-manages it;
    // convert the sRGB primaries to P3 (both share the sRGB transfer curve the Srgb format applies)
    if (global.sun_dir.w > 0.5) {
        final_color = clamp(SRGB_TO_P3 * final_color, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return vec4<f32>(final_color, 1.0);
}