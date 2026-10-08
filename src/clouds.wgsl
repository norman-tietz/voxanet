// clouds.wgsl
// The visible clouds, sampled from the baked cloud map (cloud_map.rs) instead of evaluating the noise per
// pixel. Prepended after atmosphere.wgsl to the scene and galaxy shaders, whose group 0 both carry the
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

// cloud colour and coverage at `hit` (a point already known to be on the shell). Raymarches a short span
// around it along the view ray instead of taking one sample: at a grazing/horizon angle that span covers
// far more of the shell's thickness, so clouds naturally thicken and brighten near the horizon.
// `pixel_size` is a screen pixel's width at the hit (radians per pixel x distance): it picks the mip
// level, explicitly, since this runs in non-uniform control flow where derivatives aren't allowed.
fn atmo_cloud_shade(hit: vec3<f32>, ray_dir: vec3<f32>, pixel_size: f32, L: vec3<f32>, a: Atmosphere) -> vec4<f32> {
    let up = normalize(hit);
    let thickness = a.planet_r * CLOUD_THICKNESS;
    let radial = max(abs(dot(ray_dir, up)), 0.05);
    let span = min(thickness / radial, thickness * 10.0);

    // the pixel's footprint on the shell, in radians of direction; stretched along the view at a grazing
    // angle, so take the geometric mean of the two axes
    let footprint = pixel_size / length(hit) / sqrt(radial);
    let lod = clamp(log2(footprint / CLOUD_MAP_TEXEL), 0.0, CLOUD_MAP_MIPS - 1.0);

    var density = 0.0;
    for (var i = 0; i < 4; i++) {
        let s = (f32(i) + 0.5) / 4.0 - 0.5;
        density += cloud_coverage(normalize(hit + ray_dir * (s * span)), lod);
    }
    density *= 0.25;

    let ndotl = clamp(dot(up, L) * 0.5 + 0.5, 0.15, 1.0);
    let lit = mix(a.cloud_dark, a.cloud_light, ndotl) * a.sun_color * 0.55;
    let silver = pow(max(dot(ray_dir, L), 0.0), 6.0) * CLOUD_SILVER * a.cloud_light;
    // on the night side clouds are barely lit (same twilight band as the sky opacity): with the night
    // sky transparent over the stars, the ndotl floor above alone left them glowing grey on black
    let daylight = mix(CLOUD_NIGHT_BRIGHTNESS, 1.0, smoothstep(SKY_NIGHT_ELEVATION, SKY_DAY_ELEVATION, dot(up, L)));
    return vec4<f32>((lit + silver) * daylight, clamp(density, 0.0, 1.0));
}

// clouds along an arbitrary ray (camera view ray, or a water reflection ray); empty if it misses the
// shell. `pixel_angle`: radians per screen pixel.
fn atmo_clouds(origin: vec3<f32>, ray_dir: vec3<f32>, pixel_angle: f32, L: vec3<f32>, a: Atmosphere) -> vec4<f32> {
    let hit_t = sphere_hit(origin, ray_dir, a.planet_r * CLOUD_ALT);
    if (hit_t < 0.0) { return vec4<f32>(0.0); }
    return atmo_cloud_shade(origin + ray_dir * hit_t, ray_dir, pixel_angle * hit_t, L, a);
}
