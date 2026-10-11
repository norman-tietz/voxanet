// weather.wgsl
// The deterministic weather (weather.rs mirrors it line by line; keep them identical): a function of the
// direction from the planet centre (planet frame) and the shared clock (seconds, wrapped at 3600). Prepended
// after atmosphere.wgsl to the weather bake, the cloud bake, the scene and the galaxy shaders.

const WEATHER_PERIOD: f32 = 3600.0;          // the clock's wrap: every rate is whole cycles per period
const WEATHER_SCALE: f32 = 1.6;              // noise frequency over the unit direction (~3-5 fronts around)
const WEATHER_SPIN_CYCLES: f32 = 2.0;        // turns of the field about the spin axis per period
const WEATHER_TILT_CYCLES: f32 = 1.0;        // turns about a horizontal axis (moves the poles' weather)
const WEATHER_EVOLVE_CYCLES: f32 = 3.0;      // laps of the shape-changing offset per period
const WEATHER_EVOLVE_RADIUS: f32 = 0.6;      // that offset's circle radius in noise space
const STORM_LO: f32 = 0.46;                   // noise -> storm smoothstep (tuned: ~30 % with storm > 0.5)
const STORM_HI: f32 = 0.62;
const RAIN_LO: f32 = 0.3;                    // storm -> precipitation smoothstep
const RAIN_HI: f32 = 0.8;
const SNOW_LAT_LO: f32 = 0.6;                // |dir.y| band where precipitation turns to snow (Rain kind)
const SNOW_LAT_HI: f32 = 0.8;
const WEATHER_AGE_LOOKBACK: f32 = 200.0;
const FORCE_FADE_SECS: f32 = 4.0;            // /weather force fade in/out
const FORCE_EDGE_LO: f32 = 0.35;             // cos of the forced region's soft edge (~60° around the player)
const FORCE_EDGE_HI: f32 = 0.65;
const STORM_COVER: f32 = 0.35;               // added to the raw cloud noise at storm 1 (cloud_bake.wgsl)

// x: start, y: end (1e9 while active), z: forced snow share (< 0: none), w: 1 = weather on (0: storm 0)
struct WeatherForce {
    dir_value: vec4<f32>, // xyz: the forced region's centre direction, w: forced storm value
    timing: vec4<f32>,
}

fn weather_rotate(v: vec3<f32>, k: vec3<f32>, a: f32) -> vec3<f32> {
    return v * cos(a) + cross(k, v) * sin(a) + k * dot(k, v) * (1.0 - cos(a));
}

fn weather_noise(dir: vec3<f32>, t: f32) -> f32 {
    let w = 6.2831853 / WEATHER_PERIOD;
    var d = weather_rotate(dir, vec3<f32>(0.0, 1.0, 0.0), t * w * WEATHER_SPIN_CYCLES);
    d = weather_rotate(d, vec3<f32>(0.6, 0.0, 0.8), t * w * WEATHER_TILT_CYCLES);
    let e = t * w * WEATHER_EVOLVE_CYCLES;
    let p = d * WEATHER_SCALE + vec3<f32>(cos(e), sin(e), 0.0) * WEATHER_EVOLVE_RADIUS + vec3<f32>(11.3, 4.7, 8.1);
    return 0.65 * value_noise3(p) + 0.35 * value_noise3(p * 2.03 + vec3<f32>(17.1, 3.3, 9.7));
}

// seconds since `since`, across the clock's wrap; a time up to the history's reach (180 s) before `since`
// stays negative (the histories look back from now: not started / not yet ended)
fn weather_age(t: f32, since: f32) -> f32 {
    let a = t - since;
    return select(a, a + WEATHER_PERIOD, a < -WEATHER_AGE_LOOKBACK);
}

fn weather_force_weight(dir: vec3<f32>, t: f32, f: WeatherForce) -> f32 {
    let region = smoothstep(FORCE_EDGE_LO, FORCE_EDGE_HI, dot(dir, f.dir_value.xyz));
    let fade_in = clamp(weather_age(t, f.timing.x) / FORCE_FADE_SECS, 0.0, 1.0);
    let fade_out = select(1.0, 1.0 - clamp(weather_age(t, f.timing.y) / FORCE_FADE_SECS, 0.0, 1.0), f.timing.y < 1e8);
    return region * fade_in * fade_out;
}

fn weather_storm(dir: vec3<f32>, t: f32, f: WeatherForce) -> f32 {
    if (f.timing.w < 0.5) { return 0.0; }
    let s = smoothstep(STORM_LO, STORM_HI, weather_noise(dir, t));
    return mix(s, f.dir_value.w, weather_force_weight(dir, t, f));
}

fn weather_rain(dir: vec3<f32>, t: f32, f: WeatherForce) -> f32 {
    return smoothstep(RAIN_LO, RAIN_HI, weather_storm(dir, t, f));
}

// (storm, rain history, cover history, snow share): the weather map's texel
fn weather_sample(dir: vec3<f32>, t: f32, f: WeatherForce) -> vec4<f32> {
    let s0 = weather_storm(dir, t, f);
    let r0 = smoothstep(RAIN_LO, RAIN_HI, s0);
    let rain = max(max(r0, 0.7 * weather_rain(dir, t - 30.0, f)),
                   max(0.45 * weather_rain(dir, t - 60.0, f), 0.2 * weather_rain(dir, t - 90.0, f)));
    let cover = max(max(r0, 0.7 * weather_rain(dir, t - 60.0, f)),
                    max(0.45 * weather_rain(dir, t - 120.0, f), 0.2 * weather_rain(dir, t - 180.0, f)));
    var share = smoothstep(SNOW_LAT_LO, SNOW_LAT_HI, abs(dir.y));
    if (f.timing.z >= 0.0) {
        share = mix(share, f.timing.z, weather_force_weight(dir, t, f));
    }
    return vec4<f32>(s0, rain, cover, share);
}

// the planet type's rule (weather.rs WeatherKind::rule): x wet on, y split by snow share, z cover on,
// w puddles on. Returns (wet, cover).
fn weather_surface(s: vec4<f32>, rule: vec4<f32>) -> vec2<f32> {
    if (rule.y > 0.5) {
        return vec2<f32>(s.y * (1.0 - s.w) * rule.x, s.z * s.w * rule.z);
    }
    return vec2<f32>(s.y * rule.x, s.z * rule.z);
}
