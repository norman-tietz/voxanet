

// basic shading (IMPROVE THIS LATER)
struct Global {
    view_proj: mat4x4<f32>,
    ray_dirs: mat4x4<f32>, // camera rays: col0 through the top-left corner, col1/col2 per screen width/height
    camera_pos: vec4<f32>, // w: planet radius (rt.resolution is 0 in hardware shadow mode, see hw_rt.rs)
    sun_dir: vec4<f32>,
    screen: vec4<f32>, // width, height in pixels, z: sea surface radius (0 = no water), w: time in seconds
    motion: vec4<f32>, // x: radial motion blur strength 0..1, from the player's current speed; y: radians per screen pixel (clouds.wgsl)
    bevel: vec4<f32>, // x: block bevel width in world units (0 = off), y: cavity darkening (fs_geom, bevel.rs)
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
struct Biome {
    liquid_shallow: vec4<f32>, // w: 0 = reflective, 1 = glowing
    liquid_deep: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    cloud_light: vec4<f32>,
    cloud_dark: vec4<f32>,
    space_color: vec4<f32>,
    sun: vec4<f32>, // rgb: sunlight (the star type's), w: star angular radius
    star_surface: vec4<f32>, // the star type's look, for the sky's sun disc (atmo_sun_disc)
    star_limb: vec4<f32>,
    star_corona: vec4<f32>,  // rgb, w: corona_size
    star_params: vec4<f32>,  // granulation, granule_scale, sunspots, unused
    star_frame: vec4<f32>,   // quaternion: planet frame -> galaxy space (GalaxyPlanet::orientation)
}
@group(0) @binding(3) var<uniform> biome: Biome;
// blurred shadow term written by cs_march or rt_hw.wgsl + blur.wgsl (r = shadow), see rt_blur.rs
@group(2) @binding(0) var rt_blurred: texture_2d<f32>;

struct Local {
    model: mat4x4<f32>,
    params: vec4<f32>, // x = opacity, y = 1 while fading out, z = LOD geomorph factor (vs_lod)
}
@group(1) @binding(0) var<uniform> local: Local;

// --- CONSTANTS ---
// Natural, physical light values
const FOAM_COLOR      = vec3<f32>(0.80, 0.85, 0.88);  // shore foam (linear albedo)
const CAUSTIC_FOCUS   = 0.4;                         // how strongly the ripples focus sunlight below them
const CAUSTIC_STEEP   = 0.08;                        // slope of the caustic ripples
const CAUSTIC_BASE    = 0.7;                         // plain sunlight under water, between the caustic lines
const CAUSTIC_MAX     = 3.0;                         // brightest caustic, times the plain sunlight
const WATER_DEPTH_RANGE: f32 = 16.0;                   // layers of water depth the G-buffer's albedo alpha spans
const FOAM_DEPTH      = 2.0;                         // vertical water depth below which foam forms
const SHADOW_OPACITY  = 0.85;                        // Shadows are not pitch black
const MOTION_BLUR_SAMPLES    = 6;    // extra taps of the shaded image per pixel when global.motion.x > 0
const MOTION_BLUR_MAX_PIXELS = 40.0; // blur radius in pixels at the screen edge, at full strength
const SRGB_TO_P3 = mat3x3<f32>(                     // linear sRGB -> linear Display P3 (column-major)
    vec3<f32>(0.8225, 0.0332, 0.0171),
    vec3<f32>(0.1774, 0.9669, 0.0724),
    vec3<f32>(0.0000, 0.0000, 0.9108),
);

// cloud shell: a thin band of coverage noise at CLOUD_ALT * planet radius, well above the terrain
// (which caps at 1.2x the radius, see CLAUDE.md). Shaped as a direction-space FBM so it has no seams
// at cube faces, the same reason ripple_curvature works on world positions instead of a 2D texture.

// --- VERTEX SHADER ---

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) water: f32, // water surface radius over the cell the face looks into, 0 = dry
    @location(7) edge0: vec4<f32>, // f16 distances to up to eight bevelled edges (bevel.rs), 1e4 = none
    @location(8) edge1: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) view_pos: vec3<f32>,
    @location(4) water: f32,
    @location(5) @interpolate(flat) flat_color: vec3<f32>, // low-poly: one colour per facet (first vertex)
    @location(6) edge0: vec4<f32>, // bevel edge distances, affine on a face, so interpolated exactly
    @location(7) edge1: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    return transform(in);
}

// a LOD mesh vertex's geomorph target (common.rs LodMorph): the parent LOD level's surface here
struct LodMorphIn {
    @location(4) height: f32,         // parent radius minus this vertex's, along its own up
    @location(5) normal: vec4<f32>,
    @location(6) color: vec4<f32>,
};

// LOD meshes: blended toward their parent's shape, normal and colour by local.params.z (1 = the
// parent's), so LOD levels hand over without a visible change and sharpen as the camera closes in
@vertex
fn vs_lod(in: VertexIn, morph: LodMorphIn) -> VertexOut {
    let t = local.params.z;
    var v = in;
    v.pos = in.pos + normalize(in.pos) * morph.height * t;
    v.normal = normalize(mix(in.normal, morph.normal.xyz, t));
    v.color = mix(in.color, morph.color.rgb, t);
    return transform(v);
}

fn transform(in: VertexIn) -> VertexOut {
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
    out.flat_color = in.color;
    out.view_pos = global.camera_pos.xyz;
    out.water = in.water;
    out.edge0 = in.edge0;
    out.edge1 = in.edge1;

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
const RT_MAX_STRIDE: f32 = 8.0; // longest ray step, in base segments

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
    let top = f32(rt.max_layer + 1);
    var wa = world_pos;
    var t = 0.0;
    var len = seg_len;

    // Adaptive steps: while the ray climbs above every height tile it passes, the step doubles (up to
    // RT_MAX_STRIDE segments). A long step that isn't clearly above the terrain is retried as a short one,
    // so cells are only ever walked on short, curvature-accurate segments. Once a straight ray climbs
    // relative to the planet it keeps climbing, so a climbing segment's lowest point is its start.
    for (var i = 0; i < 512; i++) {
        if (t >= RT_RANGE) { break; }
        if (a.p.z >= top) { return 1.0; } // above every solid cell, and the ray only climbs
        let t1 = min(t + len, RT_RANGE);
        let wb = world_pos + L * t1;
        let b = to_block(wb);

        if (b.face == a.face && rt_above_tiles(a, b)) {
            if (b.p.z > a.p.z) { len = min(len * 2.0, seg_len * RT_MAX_STRIDE); }
            a = b;
            wa = wb;
            t = t1;
            continue;
        }
        if (len > seg_len) {
            len = seg_len; // retry from `a` with a short segment
            continue;
        }

        var r = RT_CROSSES_FACE;
        if (b.face == a.face) {
            r = rt_walk(a, b);
        } else {
            r = rt_cross(a, wa, wb);
        }
        if (r == RT_CROSSES_FACE) { r = rt_sample(wa, wb); }
        if (r == RT_SOLID) { return 0.0; }
        if (r == RT_NO_DATA) { return 1.0; }
        a = b;
        wa = wb;
        t = t1;
    }
    return 1.0;
}

// blurred ray-traced shadow term at screen pixel frag_xy (1 = lit); the UI bind group has rt.enabled = 0.
// The shadow targets can be smaller than the screen (rt_blur.rs MAX_RT_PIXELS): upsample from the
// four nearest texels, weighted bilinearly and by how well their camera distance matches this pixel's,
// so shadows don't bleed across silhouettes. At full resolution this is a single exact tap.
fn shadow_at(frag_xy: vec2<f32>, world_pos: vec3<f32>) -> f32 {
    if (rt.enabled != 1u) { return 1.0; }
    let dims = vec2<i32>(textureDimensions(rt_blurred));
    let p = frag_xy * vec2<f32>(dims) / global.screen.xy - 0.5;
    let base = vec2<i32>(floor(p));
    let f = fract(p);
    let dist = distance(global.camera_pos.xyz, world_pos);

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

// ray-marched shadow term, once per shadow texel of the shadow G-buffer (cs_gbuf_down, rt_blur.rs)
@group(3) @binding(0) var g_pos: texture_2d<f32>;    // xyz world position, w camera distance (0 = sky)
@group(3) @binding(1) var g_nrm: texture_2d<f32>;    // xyz world normal
@group(3) @binding(2) var shadow_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_march(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(shadow_out);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let p = vec2<i32>(id.xy);
    let g = textureLoad(g_pos, p, 0);
    if (g.w <= 0.0) {
        textureStore(shadow_out, p, vec4<f32>(1.0, 0.0, 0.0, 0.0)); // sky
        return;
    }
    let N = normalize(textureLoad(g_nrm, p, 0).xyz);
    let L = normalize(global.sun_dir.xyz);
    var s = 0.0; // faces turned away from the sun are in their own shadow
    if (dot(N, L) > 0.0) { s = rt_shadow(g.xyz, N, L); }
    textureStore(shadow_out, p, vec4<f32>(s, g.w, 0.0, 1.0));
}

// --- UTILS ---

// screen-door transparency for LOD/chunk fades: true = discard this pixel. A mesh fading in keeps the
// pixels whose threshold is below its opacity, a mesh fading out (`fading_out`, params.y = 1) those at
// or above 1 - opacity, so a swap whose two fades start together covers every pixel exactly once
// (the same test for both left the pixels above max(a, 1 - a) empty: half the area see-through mid-fade)
fn dither_discard(pos: vec4<f32>, alpha: f32, fading_out: bool) -> bool {
    let threshold = fract(dot(vec2<f32>(171.0, 231.0), pos.xy) / 71.0);
    return select(threshold >= alpha, threshold < 1.0 - alpha, fading_out);
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

// --- ATMOSPHERE & CLOUDS ---
// The maths lives in atmosphere.wgsl (shared with the galaxy planet impostors); these wrappers feed it
// the engine's globals so every call site below stays as it was. A thin cloud shell wraps the planet at
// CLOUD_ALT * planet radius, shaded procedurally (no mesh): every call site solves for where a ray crosses
// that shell, then shades it — from the ground, at the horizon, reflected in water, or from orbit.

fn engine_atmosphere() -> Atmosphere {
    return Atmosphere(
        global.camera_pos.w,
        biome.sky_zenith.rgb,
        biome.sky_horizon.rgb,
        biome.space_color.rgb,
        biome.cloud_light.rgb,
        biome.cloud_dark.rgb,
        biome.sun.rgb,
        StarLook(biome.star_surface.rgb, biome.star_limb.rgb, biome.star_corona.rgb, biome.star_params.x,
                 biome.star_params.y, biome.star_params.z, biome.star_corona.w),
        biome.sun.w,
        biome.star_frame,
        global.screen.w,
    );
}

fn cloud_shadow(world_pos: vec3<f32>, L: vec3<f32>, t: f32) -> f32 {
    return atmo_cloud_shadow(world_pos, L, t, global.camera_pos.w);
}

// `dist`: from the camera to `hit` (picks the cloud map's level of detail)
fn cloud_shade(hit: vec3<f32>, ray_dir: vec3<f32>, dist: f32, L: vec3<f32>) -> vec4<f32> {
    return atmo_cloud_shade(hit, ray_dir, global.motion.y * dist, L, engine_atmosphere());
}

fn clouds(origin: vec3<f32>, ray_dir: vec3<f32>, L: vec3<f32>) -> vec4<f32> {
    return atmo_clouds(origin, ray_dir, global.motion.y, L, engine_atmosphere());
}

fn sky_opacity(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>) -> f32 {
    return atmo_sky_opacity(ray_dir, cam_pos, L, global.camera_pos.w);
}

fn sky_gradient(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>) -> vec3<f32> {
    return atmo_sky_gradient(ray_dir, cam_pos, L, engine_atmosphere());
}

fn sky_over_black(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>) -> vec3<f32> {
    return atmo_sky_over_black(ray_dir, cam_pos, L, engine_atmosphere());
}

// --- FRAGMENT SHADER ---

// lighting, fog, tone mapping and output colour space of one surface point (vertex colour `color`,
// unit normal N); used per pixel by fs_light (deferred) and by the forward-drawn overlays (fs_main)
fn shade(color: vec3<f32>, N: vec3<f32>, world_pos: vec3<f32>, frag_xy: vec2<f32>, water_depth: f32) -> vec3<f32> {
    let L = normalize(global.sun_dir.xyz);
    let V = normalize(global.camera_pos.xyz - world_pos);

    // 2. Material Setup
    // De-Gamma the vertex color to Linear Space for math
    let vert_color_linear = pow(color, vec3<f32>(2.2));
    
    // Apply Detail Noise (Grain)
    let noise = triplanar_detail(world_pos, N);
    let albedo = vert_color_linear * (1.0 + 0.03 * noise);

    // 3. Lighting Math
    let NdotL = max(dot(N, L), 0.0);
    
    // Shadow (ray-traced, see rt_blur.rs); faces turned away from the sun are in their own shadow
    let shadow_raw = select(shadow_at(frag_xy, world_pos), 0.0, NdotL <= 0.0);
    // Smooth transition shadow
    let shadow = mix(1.0 - SHADOW_OPACITY, 1.0, shadow_raw);

    // A. Direct Sun Light, dimmed under the cloud shell and focused into caustics below a water surface:
    // `water_depth` is how deep the point lies below its own water (a lake's or the sea's, through the
    // G-buffer), 0 when dry — so lake floors get caustics and tunnels sealed off from water don't
    var direct_light = biome.sun.rgb * NdotL * shadow * cloud_shadow(world_pos, L, global.screen.w);
    // caustics are a refraction effect of clear reflective liquid; skip them for glowing lava,
    // which doesn't focus light the same way (matches fs_water's reflective-vs-glowing branch)
    if (water_depth > 0.0 && biome.liquid_shallow.w < 0.5 && shadow_raw > 0.0) {
        direct_light *= mix(1.0, caustics(world_pos, L, water_depth), shadow_raw);
    }

    // B. Hemispheric Ambient
    // Top of objects gets Sky Color, Bottom gets Ground Bounce
    let up_dot = dot(N, normalize(world_pos)); // Relative Up for sphere
    let hemi_factor = up_dot * 0.5 + 0.5;
    // dims to moonlight on the night side, like the sky itself
    let sky_light = atmo_ambient_daylight(world_pos, L);
    let ambient_light = mix(GROUND_COLOR, biome.sky_zenith.rgb, hemi_factor) * sky_light;

    // C. Fresnel Rim
    // Adds a subtle glow at grazing angles (atmosphere dust effect)
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    let rim_light = biome.sky_zenith.rgb * fresnel * 0.2 * shadow * sky_light;

    // Combine
    // Note: Ambient is multiplied by albedo (diffuse reflection)
    let final_color = albedo * (direct_light + ambient_light + rim_light);
    return fog(final_color, world_pos);
}

// fog and underwater tint of a linear HDR colour at world_pos; stays in linear space (no tone mapping)
// so callers can still composite clouds on top before post_process() runs once at the very end
fn fog(lit: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    var final_color = lit;

    // 4. Fog (Atmospheric Scattering)
    let dist = distance(global.camera_pos.xyz, world_pos);

    // seen from below the sea surface, everything fades into deep water instead
    if (length(global.camera_pos.xyz) < global.screen.z) {
        final_color = mix(final_color * vec3<f32>(0.4, 0.75, 0.9), biome.liquid_deep.rgb * 2.0, 1.0 - exp(-dist * 0.12));
    }
    // Horizon fog blends into the same sky the camera would see in that direction (atmosphere.wgsl)
    let L = normalize(global.sun_dir.xyz);
    final_color = atmo_air_fog(final_color, world_pos, global.camera_pos.xyz, L, engine_atmosphere());

    return final_color;
}

// tone mapping and output colour space of a linear HDR colour; the final step of every fragment shader
fn post_process(lit: vec3<f32>) -> vec3<f32> {
    var final_color = aces_and_gamma(lit); // tone mapping + gamma (atmosphere.wgsl)

    // when the surface is tagged Display P3 (sun_dir.w = 1, see renderer.rs) macOS colour-manages it;
    // convert the sRGB primaries to P3 (both share the sRGB transfer curve the Srgb format applies)
    if (global.sun_dir.w > 0.5) {
        final_color = clamp(SRGB_TO_P3 * final_color, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return final_color;
}

// forward shading, for overlays drawn after the deferred lighting (cursor box, player in third person...)
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // 1. Transparency Dithering
    if (local.params.x < 1.0 && dither_discard(in.clip_pos, local.params.x, local.params.y > 0.5)) {
        discard;
    }
    return vec4<f32>(post_process(shade(in.color, normalize(in.world_normal), in.world_pos, in.clip_pos.xy, 0.0)), 1.0);
}

// block cursor: lit like fs_main, but translucent (renderer.rs draws a depth pre-pass first, so only
// its nearest surface blends)
const CURSOR_OPACITY: f32 = 0.55;

@fragment
fn fs_cursor(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(post_process(shade(in.color, normalize(in.world_normal), in.world_pos, in.clip_pos.xy, 0.0)), CURSOR_OPACITY);
}

// --- DEFERRED SHADING (deferred.rs) ---
// The geometry pass writes vertex colour, normal and camera distance per screen pixel; fs_light shades
// each pixel once and cs_gbuf_down derives the shadow-resolution G-buffer for the shadow passes.

struct GeomOut {
    @location(0) albedo: vec4<f32>,   // vertex colour (with baked AO)
    @location(1) normal: vec4<f32>,   // world normal * 0.5 + 0.5
    @location(2) dist: f32,           // camera distance, 0 = sky
}

// block bevels (bevel.rs): edge0/edge1 hold the fragment's distances to up to eight edges of its face
// (1e4 = none), ex*/ey* their screen derivatives, px/py world_pos's. Each distance is affine on the face,
// so its world gradient (in the face plane, pointing away from the edge) is solved exactly from the
// derivatives. Within global.bevel.x of an edge the normal tilts outward on a quarter circle (45° at the
// edge, so two faces meeting at 90° agree there) and the albedo darkens by up to global.bevel.y; both
// fade out where the band is under ~2 px wide. Returns the normal and the albedo factor.
fn bevel_normal(N: vec3<f32>, edge0: vec4<f32>, edge1: vec4<f32>, ex0: vec4<f32>, ey0: vec4<f32>,
                ex1: vec4<f32>, ey1: vec4<f32>, px: vec3<f32>, py: vec3<f32>) -> vec4<f32> {
    let w = global.bevel.x;
    let near = min(min(edge0, edge1).xy, min(edge0, edge1).zw);
    if (w <= 0.0 || min(near.x, near.y) >= w) {
        return vec4<f32>(N, 1.0);
    }
    let gxx = dot(px, px);
    let gxy = dot(px, py);
    let gyy = dot(py, py);
    let det = gxx * gyy - gxy * gxy;
    // edge-on (or degenerate) triangle: no usable gradient
    if (det <= 1e-6 * gxx * gyy) {
        return vec4<f32>(N, 1.0);
    }
    var tilt = vec3<f32>(0.0);
    var albedo = 1.0;
    for (var i = 0; i < 8; i++) {
        let j = i & 3;
        let lo = i < 4;
        let d = max(select(edge1[j], edge0[j], lo), 0.0);
        if (d >= w) {
            continue;
        }
        let dx = select(ex1[j], ex0[j], lo);
        let dy = select(ey1[j], ey0[j], lo);
        let fade = smoothstep(1.0, 3.0, w / max(length(vec2<f32>(dx, dy)), 1e-9));
        let g = ((gyy * dx - gxy * dy) * px + (gxx * dy - gxy * dx) * py) / det;
        let gl = length(g);
        if (gl <= 0.0) {
            continue;
        }
        let s = (w - d) / (w * 1.41421356); // sin θ, 0 … sin 45°
        tilt -= g / gl * (s / sqrt(1.0 - s * s)) * fade;
        let c = 1.0 - d / w;
        albedo = min(albedo, 1.0 - global.bevel.y * c * c * fade);
    }
    return vec4<f32>(normalize(N + tilt), albedo);
}

@fragment
fn fs_geom(in: VertexOut) -> GeomOut {
    // derivatives first, while every fragment of the quad still runs (the dither discard below)
    let px = dpdx(in.world_pos);
    let py = dpdy(in.world_pos);
    let facet = cross(px, py);
    let ex0 = dpdx(in.edge0);
    let ey0 = dpdy(in.edge0);
    let ex1 = dpdx(in.edge1);
    let ey1 = dpdy(in.edge1);
    if (local.params.x < 1.0 && dither_discard(in.clip_pos, local.params.x, local.params.y > 0.5)) {
        discard;
    }
    var out: GeomOut;
    // alpha: depth below the face's own water surface (caustics in fs_light), 0 = dry
    let water_depth = select(0.0, clamp((in.water - length(in.world_pos)) / WATER_DEPTH_RANGE, 0.0, 1.0), in.water > 0.0);
    var normal = normalize(in.world_normal);
    var color = in.color;
    // low-poly terrain (LocalUniform.params.w, lowpoly.rs): the facet's flat colour and its own normal
    // from screen-space derivatives, oriented like the vertex normal
    if (local.params.w > 0.5) {
        color = in.flat_color;
        if (dot(facet, facet) > 0.0) {
            let flat_n = normalize(facet);
            normal = select(-flat_n, flat_n, dot(flat_n, normal) >= 0.0);
        }
    }
    // block bevels, after the low-poly branch: its facets carry no edges, its placed cubes do
    let bev = bevel_normal(normal, in.edge0, in.edge1, ex0, ey0, ex1, ey1, px, py);
    normal = bev.xyz;
    color = color * bev.w;
    out.albedo = vec4<f32>(color, water_depth);
    out.normal = vec4<f32>(normal * 0.5 + 0.5, 0.0);
    out.dist = distance(global.camera_pos.xyz, in.world_pos);
    return out;
}

@group(3) @binding(3) var d_albedo: texture_2d<f32>;
@group(3) @binding(4) var d_normal: texture_2d<f32>;
@group(3) @binding(5) var d_dist: texture_2d<f32>;

// world position of the surface `dist` units from the camera behind screen pixel `frag_xy`
fn world_from_pixel(frag_xy: vec2<f32>, dist: f32) -> vec3<f32> {
    let uv = frag_xy / global.screen.xy;
    let dir = normalize(global.ray_dirs[0].xyz + uv.x * global.ray_dirs[1].xyz + uv.y * global.ray_dirs[2].xyz);
    return global.camera_pos.xyz + dir * dist;
}

// one triangle covering the screen
@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// shades one pixel (G-buffer sample, sky/cloud compositing), without post-processing (fs_light, and
// fs_shade for the motion blur, which post-processes after averaging).
// Returns premultiplied colour and coverage: terrain is opaque, the sky only as opaque as
// sky_opacity, so the galaxy backdrop shows through where the atmosphere is thin or dark.
fn shade_pixel(px: vec2<i32>, cam_pos: vec3<f32>, L: vec3<f32>, t: f32) -> vec4<f32> {
    let dist = textureLoad(d_dist, px, 0).r;
    let uv = (vec2<f32>(px) + 0.5) / global.screen.xy;
    let ray_dir = normalize(global.ray_dirs[0].xyz + uv.x * global.ray_dirs[1].xyz + uv.y * global.ray_dirs[2].xyz);

    var color: vec3<f32>;
    var alpha = 1.0;
    if (dist <= 0.0) {
        // sky: no geometry behind the cloud shell
        alpha = sky_opacity(ray_dir, cam_pos, L);
        color = sky_gradient(ray_dir, cam_pos, L) * alpha;
    } else {
        let N = normalize(textureLoad(d_normal, px, 0).xyz * 2.0 - 1.0);
        let world_pos = world_from_pixel(vec2<f32>(px) + 0.5, dist);
        let albedo = textureLoad(d_albedo, px, 0);
        color = shade(albedo.rgb, N, world_pos, vec2<f32>(px) + 0.5, albedo.a * WATER_DEPTH_RANGE);
    }

    // clouds, wherever the view ray crosses the shell before it reaches any terrain (always, for sky
    // pixels; also from orbit, looking down at the cloud layer over the ground); "over" in
    // premultiplied form, which for opaque terrain is the same mix as before
    let cloud_r = global.camera_pos.w * CLOUD_ALT;
    let cloud_t = sphere_hit(cam_pos, ray_dir, cloud_r);
    if (cloud_t > 0.0 && (dist <= 0.0 || cloud_t < dist)) {
        let cl = cloud_shade(cam_pos + ray_dir * cloud_t, ray_dir, cloud_t, L);
        color = cl.rgb * cl.a + color * (1.0 - cl.a);
        alpha = cl.a + alpha * (1.0 - cl.a);
    }
    return vec4<f32>(color, alpha);
}

// premultiplied all the way out: the lighting pass blends with PREMULTIPLIED_ALPHA_BLENDING, so a
// sky pixel lands as post(sky * a) + backdrop * (1 - a) — the same post(sky * a) that fog() fades
// distant terrain toward (sky_over_black). Un-premultiplying before post_process and blending
// post(sky) * a instead made fogged terrain visibly brighter than the sky next to it whenever the
// sky was partly transparent (twilight, and the space-fade band seen from altitude).
@fragment
fn fs_light(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let color = shade_pixel(vec2<i32>(pos.xy), global.camera_pos.xyz, normalize(global.sun_dir.xyz), global.screen.w);
    return vec4<f32>(post_process(color.rgb), color.a);
}

// --- RADIAL MOTION BLUR (Deferred::shade_pipeline + blur_pipeline, only while it is active) ---
// Each pixel is shaded once by fs_shade into Deferred::shaded (premultiplied colour + coverage, before
// post-processing), then fs_motion_blur averages taps of that image along the line to the screen centre
// and post-processes the average once (tonemapping already-tonemapped samples would double-compress
// them). Shading every tap again in place, as fs_light once did, cost up to 7 full shades per pixel:
// ~120 ms a frame at 4K, sky and clouds included.
@group(3) @binding(6) var shaded: texture_2d<f32>;

@fragment
fn fs_shade(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return shade_pixel(vec2<i32>(pos.xy), global.camera_pos.xyz, normalize(global.sun_dir.xyz), global.screen.w);
}

// strength ramps from the screen center (none) to the edges (full), scaled by the player's current
// speed (global.motion.x, set in Renderer::render): the sides blur while running/flying fast, the
// center of view stays sharp
@fragment
fn fs_motion_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var color = textureLoad(shaded, vec2<i32>(pos.xy), 0);
    let center = global.screen.xy * 0.5;
    let offset = pos.xy - center;
    let edge_dist = length(offset) / length(center); // 0 at center, 1 at the corners
    let strength = global.motion.x * edge_dist;
    if (strength > 0.01) {
        let dir = offset / max(length(offset), 0.001);
        var sum = color;
        var count = 1.0;
        for (var i = 1; i <= MOTION_BLUR_SAMPLES; i++) {
            let reach = strength * (f32(i) / f32(MOTION_BLUR_SAMPLES)) * MOTION_BLUR_MAX_PIXELS;
            let sample_px = vec2<i32>(pos.xy - dir * reach);
            if (sample_px.x >= 0 && sample_px.y >= 0
                && sample_px.x < i32(global.screen.x) && sample_px.y < i32(global.screen.y)) {
                sum += textureLoad(shaded, sample_px, 0);
                count += 1.0;
            }
        }
        color = sum / count;
    }
    return vec4<f32>(post_process(color.rgb), color.a);
}

// full-screen white flash drawn over everything (text included) right after F2/`/screenshot` captures a
// frame, so the capture itself is reliably invisible: it only fires once the swapchain image already
// has the un-flashed frame copied out. `local.params.x` is the flash's current opacity (Renderer::render),
// `params.yzw` its colour (white for the flash; the galaxy's star heat glow uses the same pass).
@fragment
fn fs_flash() -> @location(0) vec4<f32> {
    return vec4<f32>(local.params.yzw, local.params.x);
}

// film look (film.rs): a vignette and animated grain over the finished frame, drawn right after the
// lens flare (under the HUD) in both render paths. local.params = (grain, vignette, time in seconds,
// width / height). fs_film returns one factor that the "2x multiply" blend applies (out = 2 · src · dst),
// so grain brightens as much as it darkens and the frame keeps its mean brightness.
struct FilmOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_film(@builtin(vertex_index) i: u32) -> FilmOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return FilmOut(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), p);
}

// PCG hash: well spread for neighbouring pixels and frames, unlike sin-based hashes at 4K coordinates
fn pcg_hash(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

@fragment
fn fs_film(in: FilmOut) -> @location(0) vec4<f32> {
    let grain = local.params.x;
    let vignette = local.params.y;
    let aspect = local.params.w;
    // a new grain pattern every millisecond of clock, i.e. every frame
    let frame = u32(local.params.z * 1000.0);
    let h = pcg_hash(u32(in.pos.x) + pcg_hash(u32(in.pos.y) + pcg_hash(frame)));
    let noise = f32(h) / 4294967295.0 * 2.0 - 1.0;
    // 0 at the centre, 1 in the corners, round on screen
    let d = (in.uv - 0.5) * vec2<f32>(aspect, 1.0);
    let r = length(d) / length(vec2<f32>(aspect, 1.0) * 0.5);
    let v = 1.0 - vignette * smoothstep(0.4, 1.0, r);
    return vec4<f32>(vec3<f32>(0.5 * v * (1.0 + grain * noise)), 1.0);
}

// translucent water surface (MeshGen::build_water), drawn after the lighting over the lit sea floor.
// Opacity grows with the depth of water along the view ray (G-buffer distance behind the surface).
// one travelling sine wave in world space (a plane wave crossing the sphere, so there are no seams at cube
// faces); `steep` is its maximum slope. Returns the height gradient (xyz) and the height (w).
// Waves fade out with distance, before they shrink below a pixel and alias.
fn water_wave(p: vec3<f32>, dir: vec3<f32>, len: f32, steep: f32, phase: f32, t: f32, dist: f32) -> vec4<f32> {
    let k = 6.2832 / len;
    let w = sqrt(9.81 * k) * 0.5;                          // deep-water dispersion, slowed down
    let fade = clamp(1.0 - dist / (len * 60.0), 0.0, 1.0);
    let ph = k * dot(p, normalize(dir)) - w * t + phase;
    return vec4<f32>(normalize(dir) * (steep * fade * cos(ph)), steep * fade * sin(ph) / k);
}

fn water_waves(p: vec3<f32>, t: f32, dist: f32) -> vec4<f32> {
    var s = water_wave(p, vec3<f32>(1.0, 0.2, 0.4), 7.3, 0.07, 0.0, t, dist);
    s += water_wave(p, vec3<f32>(-0.3, 1.0, 0.6), 5.1, 0.07, 1.3, t, dist);
    s += water_wave(p, vec3<f32>(0.5, -0.4, 1.0), 3.7, 0.06, 2.9, t, dist);
    s += water_wave(p, vec3<f32>(-1.0, -0.6, 0.2), 2.3, 0.06, 4.1, t, dist);
    s += water_wave(p, vec3<f32>(0.2, 0.9, -1.0), 1.6, 0.05, 5.7, t, dist);
    s += water_wave(p, vec3<f32>(0.9, -1.0, -0.5), 1.1, 0.05, 0.6, t, dist);
    return s;
}

// one ripple that only drives the caustics (too small to matter for the surface shading): its curvature
// (Laplacian of the height within the sea surface, whose normal is `up`) at p, negative under crests, which
// focus the sunlight below them, positive under troughs, which spread it. Faded out with camera distance.
// A 3D plane wave crosses the surface at an angle, so it shows a longer wavelength there: k * |dir in plane|.
fn ripple(p: vec3<f32>, up: vec3<f32>, dir: vec3<f32>, len: f32, phase: f32, t: f32, dist: f32) -> f32 {
    let k = 6.2832 / len;
    let along = 1.0 - dot(dir, up) * dot(dir, up); // squared share of dir within the surface
    let fade = clamp(1.0 - dist / (len * 150.0), 0.0, 1.0);
    return -CAUSTIC_STEEP * k * along * fade * sin(k * dot(p, dir) - sqrt(9.81 * k) * 0.5 * t + phase);
}

// six ripples, unit directions spread evenly over a hemisphere (so the pattern looks alike everywhere on the
// planet) and wavelengths without common multiples. The lookup point is warped by a slow large-scale
// distortion first, which bends the caustic lines differently from place to place instead of tiling.
fn ripple_curvature(p: vec3<f32>, t: f32, dist: f32) -> f32 {
    let up = normalize(p);
    let q = p + vec3<f32>(
        0.9 * sin(p.z * 0.43 + p.y * 0.17 + 0.07 * t) + 0.5 * sin(p.y * 0.29 - p.x * 0.37 + 2.0),
        0.9 * sin(p.x * 0.39 - p.z * 0.21 - 0.05 * t) + 0.5 * sin(p.z * 0.31 + p.y * 0.33 + 4.0),
        0.9 * sin(p.y * 0.41 + p.x * 0.23 + 0.06 * t) + 0.5 * sin(p.x * 0.27 - p.z * 0.35 + 1.0));
    var c = ripple(q, up, vec3<f32>(0.3997, 0.9167, 0.0000), 1.93, 0.0, t, dist);
    c += ripple(q, up, vec3<f32>(-0.4877, 0.7500, 0.4468), 1.64, 1.7, t, dist);
    c += ripple(q, up, vec3<f32>(0.0710, 0.5833, -0.8091), 1.36, 3.4, t, dist);
    c += ripple(q, up, vec3<f32>(0.5531, 0.4167, 0.7214), 1.16, 5.1, t, dist);
    c += ripple(q, up, vec3<f32>(-0.9534, 0.2500, -0.1687), 0.95, 6.8, t, dist);
    c += ripple(q, up, vec3<f32>(0.8408, 0.0833, -0.5349), 0.80, 8.5, t, dist);
    return c;
}

// caustics: sunlight on a surface `depth` units below the sea surface, focused (> 1) or spread (< 1) by the
// ripples where the sun ray entered the water. 1 / |1 + depth * curvature * CAUSTIC_FOCUS| is the light
// density of a refracted beam; bright lines form where it focuses. The plain sunlight is dimmed
// (CAUSTIC_BASE) so the lines stand out; faded out right at the surface and with depth (absorption).
fn caustics(world_pos: vec3<f32>, L: vec3<f32>, depth: f32) -> f32 {
    let up = normalize(world_pos);
    let entry = world_pos + L * (depth / max(dot(up, L), 0.2));
    let dist = distance(global.camera_pos.xyz, world_pos);
    let curvature = ripple_curvature(entry, global.screen.w, dist);
    let focus = min(1.0 / max(abs(1.0 + depth * curvature * CAUSTIC_FOCUS), 0.01), CAUSTIC_MAX);
    let strength = smoothstep(0.0, 0.6, depth) * exp(-depth * 0.12);
    return mix(1.0, focus * CAUSTIC_BASE, strength);
}

@fragment
fn fs_water(in: VertexOut) -> @location(0) vec4<f32> {
    let up = normalize(in.world_pos);
    let t = global.screen.w;
    let L = normalize(global.sun_dir.xyz);
    let to_cam = global.camera_pos.xyz - in.world_pos;
    let water_dist = length(to_cam);
    let V = to_cam / water_dist;
    let underwater = length(global.camera_pos.xyz) < global.screen.z;
    // from above the sea, a grazing view ray can pass under the water without reaching the sea floor
    // and come back up through the surface further on; that far surface, seen from below, would be
    // drawn too (no depth write, no culling) and, depending on draw order, cover the near one
    if (!underwater && dot(up, V) < 0.0) {
        discard;
    }

    // ripples: tilt the normal by the waves' slope along the surface (the mesh itself stays flat)
    let waves = water_waves(in.world_pos, t, water_dist);
    let N = normalize(up - (waves.xyz - dot(waves.xyz, up) * up));

    let floor_dist = textureLoad(d_dist, vec2<i32>(in.clip_pos.xy), 0).r;
    var depth = 30.0;
    if (floor_dist > 0.0) { depth = max(floor_dist - water_dist, 0.0); }
    var alpha = 1.0 - exp(-depth * 0.35);

    // the sea floor's shadow (no shadow texel belongs to the surface itself), dimmed under cloud cover too
    let shadow = mix(1.0 - SHADOW_OPACITY, 1.0, shadow_at(in.clip_pos.xy, in.world_pos)) * cloud_shadow(in.world_pos, L, t);
    let NdotL = max(dot(up, L), 0.0);
    let glowing = biome.liquid_shallow.w > 0.5;
    let body = mix(biome.liquid_shallow.rgb, biome.liquid_deep.rgb, 1.0 - exp(-depth * 0.15));
    let sky_fill = biome.sky_zenith.rgb * 2.0 * atmo_ambient_daylight(in.world_pos, L);
    var color = body * (biome.sun.rgb * NdotL * shadow + sky_fill);
    if (glowing) {
        color = body * 2.5; // emissive: pushed above 1.0 so ACES gives it a hot, blown-out look
    }

    // sky + cloud reflection at grazing angles, and a sun glint — skipped for glowing liquids (lava
    // doesn't reflect the sky, it just glows)
    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    var spec = 0.0;
    if (!glowing) {
        var refl_dir = reflect(-V, N);
        // at grazing angles a ripple can tilt the normal past the view direction, reflecting the ray
        // into the ground, where the sky and cloud lookups return what lies beyond the planet (space,
        // its far side's clouds): holes into space. Fold such rays back above the horizon, like the
        // far side of the wave; seen from below the surface, reflecting downward is right
        if (!underwater) {
            refl_dir -= up * (2.0 * min(dot(refl_dir, up), 0.0));
        }
        let refl_cloud = clouds(global.camera_pos.xyz, refl_dir, L);
        let refl = mix(sky_over_black(refl_dir, global.camera_pos.xyz, L), refl_cloud.rgb, refl_cloud.a);
        color = mix(color, refl * 1.2, fresnel);
        spec = pow(max(dot(N, normalize(L + V)), 0.0), 300.0) * shadow;
        color += biome.sun.rgb * spec * 3.0;
    }
    alpha = clamp(max(alpha, fresnel) + spec, 0.25, 0.95);
    // nothing behind the surface (the view ray leaves the water again without reaching the sea floor):
    // there's no sea floor to show through, only the backdrop's stars
    if (floor_dist <= 0.0) {
        alpha = 1.0;
    }

    // shore foam: a solid line where the water meets land, plus bands that run in toward the shore,
    // broken up by the waves. `depth` is along the view ray; the vertical depth decides the shore.
    // Skipped for glowing liquids (lava doesn't froth like water at the shoreline either).
    if (!glowing && floor_dist > 0.0) {
        let vdepth = depth * max(dot(up, V), 0.1);
        let shore = 1.0 - smoothstep(0.0, FOAM_DEPTH, vdepth);
        let bands = 0.5 + 0.5 * sin(vdepth * 6.0 + t * 1.6);
        // drifting patches, so foam also varies across the (blocky) shallows of one depth
        let q = in.world_pos + waves.xyz * 2.0;
        let patches = 0.5 + 0.5 * sin(dot(q, vec3<f32>(0.9, 0.3, -0.5)) * 1.1 + t * 0.7)
                                * sin(dot(q, vec3<f32>(-0.2, 0.8, 0.7)) * 1.4 - t * 0.5);
        let band_fade = clamp(1.0 - water_dist / 150.0, 0.0, 1.0); // thin bands alias far away
        let edge = 1.0 - smoothstep(0.05, 0.3, vdepth);
        let froth = shore * (0.55 * bands + 0.45 * patches) + shore * 0.3;
        let foam = max(edge, smoothstep(0.45, 0.8, froth) * band_fade) * 0.9;
        let foam_col = FOAM_COLOR * (biome.sun.rgb * NdotL * shadow + sky_fill);
        color = mix(color, foam_col, foam);
        alpha = max(alpha, foam);
    }
    if (underwater) {
        if (glowing) {
            color = body * 2.5; // submerged in lava: same hot emissive look as the surface, not a sky refraction
        } else {
            color = biome.liquid_deep.rgb * biome.sky_zenith.rgb * 4.0; // looking up at the surface from below
        }
        alpha = 0.6;
    }
    return vec4<f32>(post_process(fog(color, in.world_pos)), alpha * local.params.x);
}

// shadow-resolution G-buffer (rt_blur.rs g_pos / g_nrm) from the full-resolution one, nearest pixel
@group(1) @binding(1) var down_pos: texture_storage_2d<rgba32float, write>;
@group(1) @binding(2) var down_nrm: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_gbuf_down(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(down_pos);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let screen = vec2<i32>(textureDimensions(d_dist));
    let src = min(vec2<i32>((vec2<f32>(id.xy) + 0.5) * vec2<f32>(screen) / vec2<f32>(size)), screen - 1);
    let dist = textureLoad(d_dist, src, 0).r;
    if (dist <= 0.0) {
        textureStore(down_pos, vec2<i32>(id.xy), vec4<f32>(0.0));
        textureStore(down_nrm, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }
    let world_pos = world_from_pixel(vec2<f32>(src) + 0.5, dist);
    textureStore(down_pos, vec2<i32>(id.xy), vec4<f32>(world_pos, dist));
    textureStore(down_nrm, vec2<i32>(id.xy), vec4<f32>(textureLoad(d_normal, src, 0).xyz * 2.0 - 1.0, 0.0));
}
