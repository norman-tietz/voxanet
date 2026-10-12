// warp.wgsl
// The jump's light-speed streaks (space/warp.rs): a full-screen pass over the finished space frame. Each
// pixel's view direction becomes (angle from the vanishing point, angle around it); stars sit in cells of
// those coordinates and stream outward, each drawn as a radial streak whose length grows with the speed and
// with the distance from the vanishing point. The scene underneath is darkened by the scene fade.

struct Warp {
    ray_dirs: array<vec4<f32>, 3>, // top-left ray, per screen width, per screen height
    vanish: vec4<f32>,             // xyz: travel direction, w: flow (warp.rs advance_flow)
    params: vec4<f32>,             // x: speed, y: scene fade, zw: screen size
}
@group(0) @binding(0) var<uniform> warp: Warp;

const PI: f32 = 3.14159265;
const SECTORS: f32 = 220.0;   // cells around the vanishing point
const RINGS: f32 = 0.12;      // cell size along log(angle from the vanishing point)
const MAX_LEN: f32 = 0.9;     // longest streak in log-angle units
const LEN_GAIN: f32 = 0.8;    // streak length per speed and per radian from the vanishing point
const MAX_CELLS: i32 = 9;     // cells a streak can span: ceil(MAX_LEN / RINGS) + 1
const DENSITY: f32 = 0.04;    // share of cells holding a star
const BRIGHTNESS: f32 = 2.0;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// two uniform values in [0, 1) for the integer cell (x, y), varied by `salt`; an integer hash, since the
// sin() hash degenerates for large cell numbers
fn warp_hash(x: f32, y: f32, salt: u32) -> vec2<f32> {
    let h1 = pcg(bitcast<u32>(i32(x)) ^ pcg(bitcast<u32>(i32(y)) ^ pcg(salt)));
    let h2 = pcg(h1);
    return vec2<f32>(f32(h1 >> 8u), f32(h2 >> 8u)) / 16777216.0;
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
    // each sector streams at its own pace and phase, so the stars don't line up in rings
    let lane = warp_hash(sector, 0.0, 1u);
    // radial coordinate: log of the angle, streaming outward with the flow
    let lt = log(max(theta, 0.002)) + lane.x * RINGS * 7.0;
    let flow = warp.vanish.w * (0.7 + 0.6 * lane.y);
    // streak length in log-angle units: longer with speed and away from the vanishing point, capped so
    // the loop below covers it (MAX_CELLS × RINGS ≥ MAX_LEN + RINGS)
    let len = min(0.04 + speed * LEN_GAIN * (0.5 + theta), MAX_LEN);
    // the streak's half-width as an angle on screen, the same everywhere (a sector widens with theta)
    let across = abs(sector_frac - 0.5) * (2.0 * PI / SECTORS) * max(sin(theta), 0.02);
    let width = 1.0 - smoothstep(0.0005, 0.0018, across);
    var light = vec3<f32>(0.0);
    // the star in cell c has its head at (c + h.y)·RINGS + flow and its tail len behind it: this pixel can
    // lie on the streaks of cells cell0 … cell0 + ceil(len / RINGS)
    let cell0 = floor((lt - flow) / RINGS);
    let cells = i32(ceil(len / RINGS)) + 1;
    for (var k = 0; k < MAX_CELLS; k++) {
        if (k >= cells) { break; }
        let cell = cell0 + f32(k);
        let h = warp_hash(sector, cell, 2u);
        if (h.x > DENSITY) { continue; } // most cells empty
        // the star's head and the streak trailing behind it (toward the vanishing point)
        let head = (cell + h.y) * RINGS + flow;
        let g = warp_hash(sector, cell, 3u); // its own length and brightness
        let along = (head - lt) / (len * (0.5 + 0.5 * g.x)); // 0 at the head, 1 at the tail
        if (along < 0.0 || along > 1.0) { continue; }
        let tint = mix(vec3<f32>(0.75, 0.85, 1.0), vec3<f32>(1.0, 0.85, 0.7), h.x / DENSITY);
        let fall = 1.0 - along;
        light += tint * width * fall * fall * BRIGHTNESS * (0.3 + 0.7 * g.y * g.y);
    }
    // thinned out right at the vanishing point, where the sectors are narrower than a streak
    light *= smoothstep(0.02, 0.2, theta);
    // a faint blue glow around the vanishing point at speed, red-shifted toward the edges
    let glow = exp(-theta * 6.0) * 0.25 * speed * speed;
    light = light * mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.8, 0.75), smoothstep(0.6, 1.4, theta) * speed)
        + vec3<f32>(0.3, 0.5, 1.0) * glow;
    light *= smoothstep(0.0, 0.15, speed);
    return vec4<f32>(light, fade);
}
