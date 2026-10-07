// galaxy.wgsl
// Minimal forward-rendering shader for galaxy mode: a procedural starfield background plus
// icosphere bodies for the star and each planet. Planet impostors share the sky, cloud, fog and
// tone-mapping maths with the voxel engine through atmosphere.wgsl (prepended at compile time,
// galaxy_render.rs GALAXY_SHADER), so they look the same at the landing handover.

struct Camera {
    view_proj: mat4x4<f32>, // camera-relative: positions are pre-translated so the camera is at the origin
    screen: vec4<f32>,      // x, y: width/height in pixels, z: cloud animation time (seconds, wrapped at 3600), w: handover overlay opacity (fs_planet_overlay)
    ray_dirs: array<vec4<f32>, 3>, // camera basis corners for the starfield background
    forward: vec4<f32>,     // the view direction (xyz), for the star's reversed-Z depth
}

@group(0) @binding(0) var<uniform> camera: Camera;

// --- the star: one camera-facing quad, ray-cast here (star.wgsl shades it) ---

struct StarUniform {
    offset: vec4<f32>,  // camera-relative centre (xyz), radius (w)
    surface: vec4<f32>,
    limb: vec4<f32>,
    corona: vec4<f32>,  // rgb, w: corona_size
    params: vec4<f32>,  // granulation, granule_scale, sunspots, minimum angular radius (rad)
}
@group(1) @binding(0) var<uniform> star: StarUniform;

// just in front of the cleared far plane (0 in reversed-Z): the corona passes over the background
// and fails behind any planet, so a planet in front hides it
const STAR_CORONA_DEPTH: f32 = 1e-9;

fn star_look() -> StarLook {
    return StarLook(star.surface.rgb, star.limb.rgb, star.corona.rgb, star.params.x, star.params.y, star.params.z, star.corona.w);
}

// the drawn radius: the real one, or larger so the star subtends at least the minimum angular radius
fn star_drawn_radius() -> f32 {
    return max(star.offset.w, length(star.offset.xyz) * star.params.w);
}

struct StarOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) view_ray: vec3<f32>, // camera-relative point on the quad (the ray through it)
    @location(1) quad: vec2<f32>,     // position within the quad, -1..1
}

@vertex
fn vs_star(@builtin(vertex_index) vi: u32) -> StarOut {
    // a camera-facing quad around the star covering its corona (3 corona sizes beyond the limb)
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = star.offset.xyz;
    let to_star = normalize(c);
    var side = cross(to_star, vec3<f32>(0.0, 1.0, 0.0));
    if (length(side) < 1e-4) {
        side = vec3<f32>(1.0, 0.0, 0.0);
    }
    side = normalize(side);
    let up = cross(side, to_star);
    let half_size = star_drawn_radius() * (1.0 + 3.0 * star.corona.w);
    let p = c + (side * corners[vi].x + up * corners[vi].y) * half_size;
    var out: StarOut;
    out.clip_pos = camera.view_proj * vec4<f32>(p, 1.0);
    out.view_ray = p;
    out.quad = corners[vi];
    return out;
}

struct StarFragOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_star(in: StarOut) -> StarFragOut {
    let ray = normalize(in.view_ray);
    let c = star.offset.xyz;
    let r = star_drawn_radius();
    let along = dot(c, ray);
    let closest = length(c - ray * along); // the ray's closest approach to the centre
    var out: StarFragOut;
    if (closest < r && along > 0.0) {
        let hit = along - sqrt(r * r - closest * closest);
        let n = normalize(ray * hit - c);
        let mu = max(dot(n, -ray), 0.0);
        let col = star_surface(n, mu, camera.screen.z, star_look());
        out.color = vec4<f32>(aces_and_gamma(col), 1.0);
        // reversed-Z: NEAR_PLANE (1) / distance along the view axis
        out.depth = 1.0 / max(hit * dot(ray, camera.forward.xyz), 1.0);
    } else {
        // fades to nothing before the quad's edge, or the faint outskirts, lifted by tone mapping, would
        // show the quad's square outline (measured within the quad: up close, rays toward its edge pass
        // the star well inside the corona)
        let glow = star_corona((closest - r) / r, star_look()) * (1.0 - smoothstep(0.55, 0.95, length(in.quad)));
        let coverage = clamp(max(glow.r, max(glow.g, glow.b)), 0.0, 1.0);
        out.color = vec4<f32>(aces_and_gamma(glow), coverage); // premultiplied: the glow adds, dimming what it covers
        out.depth = STAR_CORONA_DEPTH;
    }
    return out;
}

struct PlanetUniform {
    offset: vec4<f32>,      // xyz: camera-relative planet centre, w: planet radius (voxel resolution / 2)
    light_dir: vec4<f32>,   // xyz: direction from this planet toward the star (galaxy space), w unused
    sky_zenith: vec4<f32>,  // the planet type's atmosphere colours (rgb), as the engine's Biome uniform
    sky_horizon: vec4<f32>,
    space_color: vec4<f32>,
    cloud_light: vec4<f32>,
    cloud_dark: vec4<f32>,
    sun_color: vec4<f32>,   // sunlight (rgb): the galaxy star type's
    glow: vec4<f32>,        // rgb: the planet type's glowing liquid colour (lava), w: 1 if it has one
    glow_emission: vec4<f32>, // rgb: what that liquid emits (galaxy_render.rs lava_emission)
    model: mat3x3<f32>,     // planet frame -> galaxy space rotation: the planet's spin at this time
}

@group(1) @binding(0) var<uniform> planet: PlanetUniform;

fn planet_atmosphere() -> Atmosphere {
    return Atmosphere(
        planet.offset.w,
        planet.sky_zenith.rgb,
        planet.sky_horizon.rgb,
        planet.space_color.rgb,
        planet.cloud_light.rgb,
        planet.cloud_dark.rgb,
        planet.sun_color.rgb,
        // impostor shells draw the sky dome only: the galaxy draws the star itself
        StarLook(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 1.0, 0.0, 1.0),
        0.0,
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        camera.screen.z,
    );
}

// the camera (at the galaxy-space origin, camera-relative rendering) and the sun, in this planet's
// own frame — where the voxel engine does all its lighting, clouds and sky
fn planet_frame_camera() -> vec3<f32> {
    return transpose(planet.model) * -planet.offset.xyz;
}

fn planet_frame_light() -> vec3<f32> {
    return normalize(transpose(planet.model) * planet.light_dir.xyz);
}

struct PlanetVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec3<f32>,    // planet frame (the mesh is built there)
    @location(1) local_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) flat_color: vec3<f32>,
}

@vertex
fn vs_planet(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
) -> PlanetVertexOut {
    let world_pos = planet.offset.xyz + planet.model * pos;
    var out: PlanetVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.local_pos = pos;
    out.local_normal = normal;
    out.color = color;
    out.flat_color = color;
    return out;
}

// low-poly impostors: the facet's own normal from screen-space derivatives, oriented like the vertex
// normal (called first thing in the fragment entry points, in uniform control flow)
fn facet_normal(in: PlanetVertexOut) -> vec3<f32> {
    let n = normalize(in.local_normal);
    let f = cross(dpdx(in.local_pos), dpdy(in.local_pos));
    if (dot(f, f) <= 0.0) {
        return n;
    }
    let fl = normalize(f);
    return select(-fl, fl, dot(fl, n) >= 0.0);
}

// the voxel engine's shade() (shader.wgsl) without what an impostor can't have: ray-traced shadows,
// caustics and the per-voxel grain — sun, sky ambient, rim, cloud shadow and air fog are the same
@fragment
fn fs_planet(in: PlanetVertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(planet_color(in, facet_normal(in)), 1.0);
}

// the landing handover's cross-fade (GalaxyRenderer::draw_handover_overlay): the near impostor blended
// over the finished voxel frame at camera.screen.w opacity, fading out
@fragment
fn fs_planet_overlay(in: PlanetVertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(planet_color(in, facet_normal(in)), camera.screen.w);
}

fn planet_color(in: PlanetVertexOut, N: vec3<f32>) -> vec3<f32> {
    let a = planet_atmosphere();
    let cam = planet_frame_camera();
    let L = planet_frame_light();
    let t = camera.screen.z;
    let V = normalize(cam - in.local_pos);

    let albedo = pow(in.flat_color, vec3<f32>(2.2));
    let NdotL = max(dot(N, L), 0.0);
    let direct = a.sun_color * NdotL * atmo_cloud_shadow(in.local_pos, L, t, a.planet_r);
    let hemi = dot(N, normalize(in.local_pos)) * 0.5 + 0.5;
    let sky_light = atmo_ambient_daylight(in.local_pos, L);
    let ambient = mix(GROUND_COLOR, a.sky_zenith, hemi) * sky_light;
    let rim = a.sky_zenith * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.2 * sky_light;
    var lit = albedo * (direct + ambient + rim);
    // a glowing liquid (lava) is emissive like the engine's (fs_water); impostor vertices carry the
    // liquid colour itself, so match on it, fading over shore blends
    let glowing = planet.glow.w * (1.0 - smoothstep(0.0, 0.3, distance(in.flat_color, planet.glow.rgb)));
    lit = mix(lit, planet.glow_emission.rgb, glowing);
    var color = atmo_air_fog(lit, in.local_pos, cam, L, a);

    // clouds in front of the surface, like the engine's shade_pixel
    let ray_dir = normalize(in.local_pos - cam);
    let cloud_t = sphere_hit(cam, ray_dir, a.planet_r * CLOUD_ALT);
    if (cloud_t > 0.0 && cloud_t < distance(cam, in.local_pos)) {
        let cl = atmo_cloud_shade(cam + ray_dir * cloud_t, ray_dir, t, L, a);
        color = mix(color, cl.rgb, cl.a);
    }
    return aces_and_gamma(color);
}

// the atmosphere shell around each planet: a sphere this many planet radii out, comfortably past where
// the sky opacity reaches 0 (~2.1 radii, atmosphere.wgsl), drawn after the planets
const ATMOSPHERE_SHELL_RADII: f32 = 2.3;

struct AtmosphereVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
}

@vertex
fn vs_atmosphere(@location(0) pos: vec3<f32>) -> AtmosphereVertexOut {
    let local = pos * planet.offset.w * ATMOSPHERE_SHELL_RADII;
    var out: AtmosphereVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(planet.offset.xyz + planet.model * local, 1.0);
    out.local_pos = local;
    return out;
}

// the sky the voxel engine would show along this ray (atmosphere.wgsl), premultiplied: the glow where
// the ray grazes the planet's lit limb, clouds where it crosses the cloud shell. Only rays that miss the
// planet reach this: the shell's back faces are drawn depth-tested behind the planet, so where the
// planet is in front, fs_planet has already drawn the surface with its clouds.
@fragment
fn fs_atmosphere(in: AtmosphereVertexOut) -> @location(0) vec4<f32> {
    let a = planet_atmosphere();
    let cam = planet_frame_camera();
    let L = planet_frame_light();
    let t = camera.screen.z;
    let ray_dir = normalize(in.local_pos - cam);

    var alpha = atmo_sky_opacity(ray_dir, cam, L, a.planet_r);
    // no sun disc: the galaxy draws the star itself (atmo_sky_dome)
    var color = atmo_sky_dome(ray_dir, cam, L, a) * alpha;
    let cloud_t = sphere_hit(cam, ray_dir, a.planet_r * CLOUD_ALT);
    if (cloud_t > 0.0) {
        let cl = atmo_cloud_shade(cam + ray_dir * cloud_t, ray_dir, t, L, a);
        color = cl.rgb * cl.a + color * (1.0 - cl.a);
        alpha = cl.a + alpha * (1.0 - cl.a);
    }
    // the engine blends post(sky * a) over its backdrop (fs_light), so the same here
    return vec4<f32>(aces_and_gamma(color), alpha);
}

// deterministic hash for the starfield, independent of shader.wgsl's hash31 (kept standalone).
// Integer-only (PCG3D, Jarzynski & Olano 2020), on purpose: the classic fract(sin(dot(p, k)) * big)
// hash this replaced fed sin() arguments up to ~200,000 (cell coords reach ±400), where GPU f32 sin
// has no precision left. It only stayed random for directions near the plane where that dot product
// is small, so the sky showed a band of stars, an empty hole on one side and a regular dot lattice
// on the other. Integer math has no such range limit.
fn star_hash(cell: vec3<i32>) -> f32 {
    var v = bitcast<vec3<u32>>(cell) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return f32(v.x >> 8u) / 16777216.0; // top 24 bits → [0, 1), exact in f32
}

@vertex
fn vs_background(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_background(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / camera.screen.xy;
    let ray_dir = normalize(camera.ray_dirs[0].xyz + uv.x * camera.ray_dirs[1].xyz + uv.y * camera.ray_dirs[2].xyz);

    let cell_scale = 400.0;
    let cell = vec3<i32>(floor(ray_dir * cell_scale));
    let h = star_hash(cell);
    let brightness = smoothstep(0.985, 1.0, h); // sparse: only the top ~1.5% of cells show a star
    return vec4<f32>(vec3<f32>(brightness), 1.0);
}
