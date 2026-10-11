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
