// clouds.wgsl
// The visible clouds, sampled from the baked cloud map (cloud_map.rs) instead of evaluating the noise per
// pixel. Prepended after atmosphere.wgsl to the scene and space shaders, whose group 0 both carry the
// map at these bindings.

const CLOUD_MAP_SIZE: f32 = 512.0; // cloud_map.rs SIZE
const CLOUD_MAP_MIPS: f32 = 10.0;  // cloud_map.rs MIPS
// one texel's angle at a face centre (2 / SIZE in face coordinates)
const CLOUD_MAP_TEXEL: f32 = 2.0 / CLOUD_MAP_SIZE;

struct CloudMapParams {
    slots: vec4<f32>, // x: the older snapshot's slot, y: the newer one's, z: blend 0..1 toward y
}
@group(0) @binding(8) var cloud_map: texture_cube_array<f32>;
@group(0) @binding(9) var cloud_sampler: sampler;
@group(0) @binding(10) var<uniform> cloud_params: CloudMapParams;

// coverage at a direction on the cloud shell: the noise blended between the two snapshots around now,
// then thresholded (after filtering, so edges stay crisp)
fn cloud_coverage(dir: vec3<f32>, lod: f32) -> f32 {
    let a = textureSampleLevel(cloud_map, cloud_sampler, dir, i32(cloud_params.slots.x), lod).r;
    let b = textureSampleLevel(cloud_map, cloud_sampler, dir, i32(cloud_params.slots.y), lod).r;
    return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, mix(a, b, cloud_params.slots.z));
}

// the glow strikes give the clouds around them: a Gaussian blob on the cloud shell around each strike's
// point, in the lightning's colour (exactly 0 without strikes)
fn lightning_cloud_glow(hit: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    var g = 0.0;
    for (var i = 0; i < 6; i++) {
        let s = a.strikes[i];
        if (s.w > 0.0) {
            let p = normalize(s.xyz) * length(hit);
            let d = distance(hit, p) / (a.planet_r * LIGHTNING_GLOW_RADIUS * CLOUD_ALT);
            g += s.w * exp(-d * d);
        }
    }
    return LIGHTNING_COLOR * g * LIGHTNING_CLOUD_GAIN;
}

// cloud colour and coverage at `hit` (a point already known to be on the shell). Raymarches a short span
// around it along the view ray instead of taking one sample: at a grazing/horizon angle that span covers
// far more of the shell's thickness, so clouds naturally thicken and brighten near the horizon.
// `pixel_size` is a screen pixel's width at the hit (radians per pixel x distance): it picks the mip
// level, explicitly, since this runs in non-uniform control flow where derivatives aren't allowed.
fn atmo_cloud_shade(hit: vec3<f32>, ray_dir: vec3<f32>, pixel_size: f32, L: vec3<f32>, a: Atmosphere) -> vec4<f32> {
    if (a.density <= 0.0) { return vec4<f32>(0.0); }
    let up = normalize(hit);
    let thickness = a.planet_r * CLOUD_THICKNESS;
    let radial = max(abs(dot(ray_dir, up)), 0.05);
    let span = min(thickness / radial, thickness * 10.0);

    // the pixel's footprint on the shell, in radians of direction; stretched along the view at a grazing
    // angle, so take the geometric mean of the two axes
    let footprint = pixel_size / length(hit) / sqrt(radial);
    let lod = clamp(log2(footprint / CLOUD_MAP_TEXEL), 0.0, CLOUD_MAP_MIPS - 1.0);

    var cover = 0.0;
    for (var i = 0; i < 4; i++) {
        let s = (f32(i) + 0.5) / 4.0 - 0.5;
        cover += cloud_coverage(normalize(hit + ray_dir * (s * span)), lod);
    }
    cover *= 0.25;
    // storm clouds: darker, little silver lining (the weather map at the hit)
    let storm = weather_at(up).x * a.density;

    let ndotl = clamp(dot(up, L) * 0.5 + 0.5, 0.15, 1.0);
    let lit = mix(mix(a.cloud_dark, a.cloud_light, ndotl), a.cloud_dark * STORM_DARK_LEVEL, storm * STORM_DARKEN) * a.sun_color * 0.55 * a.sun_vis;
    let silver = pow(max(dot(ray_dir, L), 0.0), 6.0) * CLOUD_SILVER * a.cloud_light * (1.0 - STORM_SILVER_LOSS * storm);
    // on the night side clouds are barely lit (same twilight band as the sky opacity): with the night
    // sky transparent over the stars, the ndotl floor above alone left them glowing grey on black
    let daylight = mix(CLOUD_NIGHT_BRIGHTNESS, 1.0, smoothstep(SKY_NIGHT_ELEVATION, SKY_DAY_ELEVATION, dot(up, L)));
    let glow = lightning_cloud_glow(hit, a) * (0.3 + 0.7 * clamp(cover, 0.0, 1.0));
    return vec4<f32>((lit + silver) * daylight + glow, clamp(cover, 0.0, 1.0) * a.density);
}

// clouds along an arbitrary ray (camera view ray, or a water reflection ray); empty if it misses the
// shell. `pixel_angle`: radians per screen pixel.
fn atmo_clouds(origin: vec3<f32>, ray_dir: vec3<f32>, pixel_angle: f32, L: vec3<f32>, a: Atmosphere) -> vec4<f32> {
    let hit_t = sphere_hit(origin, ray_dir, a.planet_r * CLOUD_ALT);
    if (hit_t < 0.0) { return vec4<f32>(0.0); }
    return atmo_cloud_shade(origin + ray_dir * hit_t, ray_dir, pixel_angle * hit_t, L, a);
}

const CLOUD_SHADOW_LOD: f32 = 4.0; // a coarse level: the shadow is soft and one read
const STORM_DARKEN: f32 = 0.8;     // how far storm clouds go toward STORM_DARK_LEVEL of cloud_dark
const STORM_DARK_LEVEL: f32 = 0.4;
const STORM_SILVER_LOSS: f32 = 0.7; // silver lining lost in a full storm
const STORM_SHADOW: f32 = 0.25;    // extra sunlight blocked under storm clouds

// how much direct sunlight survives the visible clouds (the cloud map, storms included) on the way to
// the sun; the flare shader has no cloud map and keeps atmo_cloud_shadow's own noise
fn atmo_cloud_shadow_map(world_pos: vec3<f32>, L: vec3<f32>, planet_r: f32, density: f32) -> f32 {
    let hit_t = sphere_hit(world_pos, L, planet_r * CLOUD_ALT);
    if (hit_t < 0.0) { return 1.0; }
    let dir = normalize(world_pos + L * hit_t);
    let coverage = cloud_coverage(dir, CLOUD_SHADOW_LOD);
    return 1.0 - density * coverage * (CLOUD_SHADOW_STRENGTH + weather_at(dir).x * STORM_SHADOW);
}
