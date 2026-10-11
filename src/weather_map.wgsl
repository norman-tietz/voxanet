// weather_map.wgsl
// The baked weather (weather.rs WeatherMap): (storm, rain history, cover history, snow share) per
// direction, blended between the two snapshots around now. Prepended to the scene and galaxy shaders,
// whose group 0 both carry the map at these bindings (sampled with the cloud map's sampler).

struct WeatherMapParams {
    slots: vec4<f32>, // x: the older snapshot's slot, y: the newer one's, z: blend 0..1 toward y
}
@group(0) @binding(13) var weather_map: texture_cube_array<f32>;
@group(0) @binding(14) var<uniform> weather_params: WeatherMapParams;

fn weather_at(dir: vec3<f32>) -> vec4<f32> {
    let a = textureSampleLevel(weather_map, cloud_sampler, dir, i32(weather_params.slots.x), 0.0);
    let b = textureSampleLevel(weather_map, cloud_sampler, dir, i32(weather_params.slots.y), 0.0);
    return mix(a, b, weather_params.slots.z);
}

const STORM_FILL_DROP: f32 = 0.3; // fill light lost under a full storm (shade() and planet_color)

struct Weathered {
    albedo: vec3<f32>,
    roughness: f32,
    reflection: f32, // the plastic's sky reflection weight
}

// wet plastic and snow/ash cover at `pos` (planet frame), shared by shade() and the impostors.
// `s`: weather_at(up); `rule`, `cover`: the planet type's (weather.rs WeatherKind); `shelter` 0..1 keeps
// rain and cover out from under overhangs (the ray-traced AO; 1 without it)
fn weather_material(albedo: vec3<f32>, roughness: f32, reflection: f32, N: vec3<f32>, pos: vec3<f32>,
                    shelter: f32, s: vec4<f32>, rule: vec4<f32>, cover: vec4<f32>) -> Weathered {
    let wc = weather_surface(s, rule);
    let facing = dot(N, normalize(pos));
    var out = Weathered(albedo, roughness, reflection);
    // wet: a water film, glossy and slightly deeper in colour; tops fully, walls a little
    let wet = wc.x * mix(0.3, 1.0, smoothstep(0.0, 0.8, facing)) * shelter;
    out.roughness = mix(out.roughness, 0.05, wet);
    out.albedo = out.albedo * mix(1.0, 0.8, wet);
    out.reflection = mix(out.reflection, 0.35, wet);
    // puddles on flat tops once soaked, growing with the wetness to about a third of them
    let puddle = smoothstep(-0.05, 0.05, (wet - 0.6) - value_noise3(pos * 0.25) * 0.9)
        * step(0.9, facing) * rule.w;
    out.roughness = mix(out.roughness, 0.02, puddle);
    out.reflection = mix(out.reflection, 1.0, puddle);
    // cover: grows in patches (against noise), closing as it thickens; replaces colour and gloss
    let raw = wc.y * smoothstep(0.3, 0.8, facing) * shelter;
    let n = value_noise3(pos * 0.35) * 0.6 + value_noise3(pos * 1.3) * 0.4;
    let c = smoothstep(-0.08, 0.08, raw - n * 0.9);
    out.albedo = mix(out.albedo, cover.rgb, c);
    out.roughness = mix(out.roughness, cover.w, c);
    out.reflection = mix(out.reflection, reflection, c);
    return out;
}
