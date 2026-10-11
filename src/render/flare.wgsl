// flare.wgsl
// Screen-space lens flares for the visible sun (flare.rs): a starburst at the sun, six ghosts along the
// line from the sun through the screen centre and a faint halo ring, drawn additively over the finished
// frame before the HUD text. Compiled with atmosphere.wgsl and star.wgsl prepended (for
// atmo_cloud_shadow). Occlusion: 9 taps around the sun in the vertex shader — the galaxy depth buffer
// (galaxy mode) or the G-buffer distance (planet mode).

struct Flare {
    sun: vec4<f32>,    // xy: sun NDC, z: galaxy mode: the star's expected reversed-Z depth, w: intensity
    tint: vec4<f32>,   // corona colour
    tint2: vec4<f32>,  // surface colour
    mode: vec4<f32>,   // x: 0 galaxy depth taps, 1 G-buffer taps; y: tap radius px; zw: screen size
    camera: vec4<f32>, // planet mode: camera position (planet frame), planet radius
    light: vec4<f32>,  // planet mode: direction to the sun, cloud time
}
@group(0) @binding(0) var<uniform> flare: Flare;
@group(0) @binding(1) var g_dist: texture_2d<f32>;
@group(0) @binding(2) var g_depth: texture_depth_2d;

// (position along the sun->centre line, size in screen heights, softness, ring) — flare.rs GHOSTS
const GHOSTS = array<vec4<f32>, 6>(
    vec4<f32>(0.35, 0.05, 0.6, 0.0),
    vec4<f32>(0.6, 0.025, 0.4, 0.0),
    vec4<f32>(0.8, 0.09, 0.85, 1.0),
    vec4<f32>(1.15, 0.04, 0.5, 0.0),
    vec4<f32>(1.4, 0.12, 0.9, 1.0),
    vec4<f32>(1.75, 0.06, 0.7, 0.0),
);

// share of 9 taps around the sun that see it: planet mode, open sky in the G-buffer (distance 0);
// galaxy mode, nothing nearer than the star (reversed-Z: a tap's depth not above the star's expected
// depth, with a small tolerance)
fn sun_visibility() -> f32 {
    let size = flare.mode.zw;
    let centre = (flare.sun.xy * vec2<f32>(0.5, -0.5) + 0.5) * size;
    var seen = 0.0;
    for (var i = 0; i < 9; i++) {
        let o = vec2<f32>(f32(i % 3) - 1.0, f32(i / 3) - 1.0) * flare.mode.y;
        let px = vec2<i32>(clamp(centre + o, vec2<f32>(0.0), size - 1.0));
        if (flare.mode.x > 0.5) {
            seen += select(0.0, 1.0, textureLoad(g_dist, px, 0).r <= 0.0);
        } else {
            seen += select(0.0, 1.0, textureLoad(g_depth, px, 0) <= flare.sun.z * 1.001);
        }
    }
    return seen / 9.0;
}

// planet mode: clouds between the camera and the sun dim the flare, as they dim sunlight on the ground
fn cloud_dimming() -> f32 {
    if (flare.mode.x < 0.5) {
        return 1.0;
    }
    return atmo_cloud_shadow(flare.camera.xyz, flare.light.xyz, flare.light.w, flare.camera.w);
}

struct FlareOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,    // -1..1 across the element
    @location(1) color: vec3<f32>, // rgb * strength
    @location(2) shape: vec2<f32>, // softness, ring
}

// instance 0: the starburst at the sun; 1..6: the ghosts; 7: the halo ring around the centre
@vertex
fn vs_flare(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FlareOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var ghosts = GHOSTS;
    let aspect = flare.mode.z / flare.mode.w;
    let strength = flare.sun.w * sun_visibility() * cloud_dimming();
    var centre = flare.sun.xy;
    var size = 0.15;
    var shape = vec2<f32>(1.0, 0.0);
    var color = flare.tint.rgb * 0.6;
    if (ii >= 1u && ii <= 6u) {
        let g = ghosts[ii - 1u];
        centre = flare.sun.xy * (1.0 - 2.0 * g.x);
        size = g.y;
        shape = g.zw;
        color = mix(flare.tint.rgb, flare.tint2.rgb, f32(ii % 2u)) * 0.3;
    } else if (ii == 7u) {
        centre = vec2<f32>(0.0);
        size = 0.6;
        shape = vec2<f32>(0.3, 1.0);
        color = flare.tint.rgb * 0.06;
    }
    let c = corners[vi];
    var out: FlareOut;
    out.pos = vec4<f32>(centre + vec2<f32>(c.x / aspect, c.y) * size, 0.0, 1.0);
    out.uv = c;
    out.color = color * strength;
    out.shape = shape;
    return out;
}

@fragment
fn fs_flare(in: FlareOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    var a = 1.0 - smoothstep(1.0 - in.shape.x, 1.0, r);
    if (in.shape.y > 0.5) {
        a *= smoothstep(0.55, 0.85, r); // a ring: hollow centre
    }
    return vec4<f32>(in.color * a, 0.0); // additive: premultiplied with zero coverage
}
