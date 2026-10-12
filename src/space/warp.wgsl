// warp.wgsl
// The jump's light-speed streaks (space/warp.rs): a full-screen pass over the finished space frame. Each
// pixel's view direction becomes (angle from the vanishing point, angle around it); stars sit in cells of
// those coordinates and stream outward, each drawn as a radial streak whose length grows with the speed and
// with the distance from the vanishing point. The scene underneath is darkened by the scene fade.

struct Warp {
    ray_dirs: array<vec4<f32>, 3>, // top-left ray, per screen width, per screen height
    vanish: vec4<f32>,             // xyz: travel direction, w: time
    params: vec4<f32>,             // x: speed, y: scene fade, zw: screen size
}
@group(0) @binding(0) var<uniform> warp: Warp;

const PI: f32 = 3.14159265;
const SECTORS: f32 = 220.0;   // cells around the vanishing point
const RINGS: f32 = 0.12;      // cell size along log(angle from the vanishing point)
const MAX_CELLS: i32 = 12;    // cells a streak can span

fn warp_hash(p: vec2<f32>) -> vec2<f32> {
    var q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

@vertex
fn vs_warp(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_warp(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let fade = warp.params.y;
    let speed = warp.params.x;
    // the scene underneath fades to black (premultiplied: dst × (1 − fade)), streaks add on top
    if (warp.params.x <= 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, fade);
    }
    let uv = pos.xy / warp.params.zw;
    let dir = normalize(warp.ray_dirs[0].xyz + uv.x * warp.ray_dirs[1].xyz + uv.y * warp.ray_dirs[2].xyz);
    let v = normalize(warp.vanish.xyz);
    let theta = acos(clamp(dot(dir, v), -1.0, 1.0)); // angle from the vanishing point
    var ref_up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(v.y) > 0.99) { ref_up = vec3<f32>(1.0, 0.0, 0.0); }
    let a = normalize(cross(v, ref_up));
    let b = cross(v, a);
    let phi = atan2(dot(dir, b), dot(dir, a)); // angle around it
    let sector = floor((phi + PI) / (2.0 * PI) * SECTORS);
    let sector_frac = fract((phi + PI) / (2.0 * PI) * SECTORS);
    // radial coordinate: log of the angle, streaming outward with time × speed
    let lt = log(max(theta, 0.002));
    let flow = warp.vanish.w * (0.4 + 2.6 * speed);
    let len = 0.05 + speed * 1.1 * (0.6 + theta); // streak length in log-angle units
    var light = vec3<f32>(0.0);
    let cell0 = floor((lt - flow) / RINGS);
    for (var k = 0; k < MAX_CELLS; k++) {
        let cell = cell0 - f32(k);
        let h = warp_hash(vec2<f32>(sector, cell));
        if (h.x > 0.35) { continue; } // most cells empty
        // the star's head and the streak trailing behind it (toward the vanishing point)
        let head = (cell + h.y) * RINGS + flow;
        let along = (head - lt) / len; // 0 at the head, 1 at the tail
        if (along < 0.0 || along > 1.0) { continue; }
        let across = abs(sector_frac - 0.5) * 2.0; // 0 on the streak's axis
        let width = 1.0 - smoothstep(0.15, 0.6, across);
        let tint = mix(vec3<f32>(0.75, 0.85, 1.0), vec3<f32>(1.0, 0.85, 0.7), h.x / 0.35);
        light += tint * width * (1.0 - along) * (1.0 - along) * 1.6;
    }
    // a faint blue glow around the vanishing point at speed, red-shifted toward the edges
    let glow = exp(-theta * 6.0) * 0.25 * speed * speed;
    light = light * mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.8, 0.75), smoothstep(0.6, 1.4, theta) * speed)
        + vec3<f32>(0.3, 0.5, 1.0) * glow;
    light *= smoothstep(0.0, 0.15, speed);
    return vec4<f32>(light, fade);
}
