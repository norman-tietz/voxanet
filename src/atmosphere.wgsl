// atmosphere.wgsl
// Sky, atmosphere limb, clouds (the noise; clouds.wgsl samples it from the baked cloud map), air fog and tone mapping shared by the voxel engine (shader.wgsl) and
// the galaxy planet impostors (galaxy.wgsl). Both are compiled with this file prepended (renderer.rs
// SCENE_SHADER, galaxy_render.rs GALAXY_SHADER), so a planet looks the same from galaxy flight and from
// the voxel engine at the landing handover. Everything here is in the planet's own frame (centre at
// the origin) and takes the planet's radius and colours as an `Atmosphere`, never engine globals.

const GROUND_COLOR    = vec3<f32>(0.05, 0.04, 0.03); // Dark earth ambient bounce

const CLOUD_ALT           = 1.32;  // cloud shell radius, as a multiple of the planet radius
const CLOUD_THICKNESS     = 0.05;  // shell thickness, as a fraction of the planet radius
const CLOUD_SCALE         = 3.2;   // noise frequency over the unit sphere direction
const CLOUD_COVERAGE      = 0.52;  // threshold: higher = less sky covered
const CLOUD_SOFTNESS      = 0.28;  // smoothstep band around the threshold (soft cloud edges)
const CLOUD_WIND_SPEED    = 0.012; // drift speed of the noise field
const CLOUD_SHADOW_STRENGTH = 0.6; // max fraction of sunlight a thick cloud blocks
const CLOUD_SILVER = 1.5;          // backlit edge glow strength, looking toward the sun

// sun elevation (sine) at which the sky is fully transparent / fully opaque; the band between is twilight
const SKY_NIGHT_ELEVATION: f32 = -0.25;
const SKY_DAY_ELEVATION: f32 = 0.15;
// atmosphere limb values between which the sky goes from fully transparent (space) to fully opaque.
// limb is ~0.7-0.9 for a camera standing on terrain (0.1-0.2 planet radii up), so it can't be used
// as the opacity directly — stars would show through the daytime sky. 0.6 ~ 0.3 radii up, 0.15 ~ 1.1.
const SKY_SPACE_LIMB: f32 = 0.15;
const SKY_OPAQUE_LIMB: f32 = 0.6;
// how bright clouds stay on the night side, relative to daylight (atmo_cloud_shade)
const CLOUD_NIGHT_BRIGHTNESS: f32 = 0.06;
// sky light (ambient, ground bounce, rim) left on the night side, relative to daylight: dim and cool,
// like moonlight, so terrain stays readable at night (atmo_ambient_daylight)
const NIGHT_AMBIENT = vec3<f32>(0.10, 0.12, 0.17);
// exp² air fog density, tuned for the scale defined in gen.rs
const FOG_DENSITY: f32 = 0.0015;

// a planet's atmosphere: radius (world units, = voxel resolution / 2) and its planet type's colours
struct Atmosphere {
    planet_r: f32,
    sky_zenith: vec3<f32>,
    sky_horizon: vec3<f32>,
    space_color: vec3<f32>,
    cloud_light: vec3<f32>,
    cloud_dark: vec3<f32>,
    sun_color: vec3<f32>, // sunlight: the star type's (galaxy.rs StarTypeDef.sunlight)
    star: StarLook,       // the star's look (star.wgsl), for the sun disc in the sky
    star_angle: f32,      // the star's angular radius from this planet; 0 = no disc (impostor shells)
    star_frame: vec4<f32>, // quaternion: planet frame -> galaxy space, so the disc shows the star's own surface
    time: f32,            // seconds, for the star's boiling surface
}

// --- TONE MAPPING (ACES) ---
// Industry standard for realistic color reproduction
fn aces_approx(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// linear HDR -> display: ACES, then gamma (the engine's post_process adds its P3 conversion after this)
fn aces_and_gamma(lit: vec3<f32>) -> vec3<f32> {
    return pow(aces_approx(lit), vec3<f32>(1.0 / 2.2));
}

// --- CLOUD NOISE ---

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// trilinear value noise, in [0, 1]
fn value_noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash31(i), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x);
    let x10 = mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x);
    let x01 = mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x);
    let x11 = mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// cumulus-like coverage at a direction on the cloud shell: a domain-warped FBM; baked into the cloud map
// (cloud_map.rs, cloud_bake.wgsl) and sampled from there (clouds.wgsl) (the warp is what keeps
// it from reading as a single tiling noise field, the same trick ripple_curvature uses for caustics)
fn fbm_clouds(dir: vec3<f32>, t: f32) -> f32 {
    let wind = vec3<f32>(t * CLOUD_WIND_SPEED, 0.0, t * CLOUD_WIND_SPEED * 0.6);
    let warp = vec3<f32>(
        value_noise3(dir * 0.8 + wind * 0.5 + vec3<f32>(5.2, 1.3, 0.0)) - 0.5,
        0.0,
        value_noise3(dir * 0.8 + wind * 0.5 + vec3<f32>(3.4, 0.0, 9.6)) - 0.5,
    );
    let p = dir * CLOUD_SCALE + warp * 1.1 + wind;

    var f = 0.0;
    var amp = 0.55;
    var freq = 1.0;
    for (var i = 0; i < 3; i++) {
        f += amp * value_noise3(p * freq);
        freq *= 2.07; // off power-of-two so octaves don't align periodically
        amp *= 0.5;
    }
    return f;
}

// single-octave, unwarped: cheap enough to sample once per shaded pixel for cloud shadows
fn cloud_coverage_fast(dir: vec3<f32>, t: f32) -> f32 {
    let wind = vec3<f32>(t * CLOUD_WIND_SPEED, 0.0, t * CLOUD_WIND_SPEED * 0.6);
    let n = value_noise3(dir * CLOUD_SCALE + wind);
    return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, n);
}

// nearest intersection in front of `origin` with a sphere of `radius` centred on the planet; < 0 if it misses
fn sphere_hit(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) { return -1.0; }
    let s = sqrt(disc);
    let t_in = -b - s;
    let t_out = -b + s;
    if (t_out < 0.0) { return -1.0; }
    return select(t_out, t_in, t_in > 0.0);
}

// --- CLOUDS ---

// how much direct sunlight at `world_pos` survives the cloud shell on the way to the sun
fn atmo_cloud_shadow(world_pos: vec3<f32>, L: vec3<f32>, t: f32, planet_r: f32) -> f32 {
    let hit_t = sphere_hit(world_pos, L, planet_r * CLOUD_ALT);
    if (hit_t < 0.0) { return 1.0; }
    let coverage = cloud_coverage_fast(normalize(world_pos + L * hit_t), t);
    return 1.0 - coverage * CLOUD_SHADOW_STRENGTH;
}

// --- SKY ---

// how much atmosphere the ray crosses: 1 when it skims the ground, falling to 0 out in space
fn atmo_limb(ray_dir: vec3<f32>, cam_pos: vec3<f32>, planet_r: f32) -> f32 {
    let atmo_r = planet_r * CLOUD_ALT * 1.2;
    let s_star = max(-dot(cam_pos, ray_dir), 0.0);
    let closest = length(cam_pos + ray_dir * s_star);
    return clamp(exp(-max(closest - planet_r, 0.0) / max(atmo_r - planet_r, 1.0)), 0.0, 1.0);
}

// how much of the sky's fill light (ambient, ground bounce, rim) reaches a surface point at `pos`:
// white by day, NIGHT_AMBIENT on the night side, over the same twilight band as the sky's opacity
fn atmo_ambient_daylight(pos: vec3<f32>, L: vec3<f32>) -> vec3<f32> {
    let up = pos / max(length(pos), 1e-4);
    return mix(NIGHT_AMBIENT, vec3<f32>(1.0), smoothstep(SKY_NIGHT_ELEVATION, SKY_DAY_ELEVATION, dot(up, L)));
}

// how opaque the sky is along the ray, for blending it over what lies behind (the galaxy): thick,
// sunlit atmosphere = 1; open space and the night side = 0. Daylight is the sun's elevation at the
// ray's closest point to the planet (for a camera on the ground looking up: the camera itself).
fn atmo_sky_opacity(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, planet_r: f32) -> f32 {
    let s_star = max(-dot(cam_pos, ray_dir), 0.0);
    let closest_point = cam_pos + ray_dir * s_star;
    let up = closest_point / max(length(closest_point), 1e-4);
    let day = smoothstep(SKY_NIGHT_ELEVATION, SKY_DAY_ELEVATION, dot(up, L));
    let thick = smoothstep(SKY_SPACE_LIMB, SKY_OPAQUE_LIMB, atmo_limb(ray_dir, cam_pos, planet_r));
    return thick * day;
}

// atmosphere seen beyond the clouds: a stylised gradient from deep space into a blue dome that warms
// toward the sun, plus a glow where the view ray grazes the planet's limb (seen from orbit/third person)
fn atmo_sky_gradient(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    return atmo_sky_dome(ray_dir, cam_pos, L, a) + atmo_sun_disc(ray_dir, L, a);
}

// the sky without the sun disc: for the galaxy impostors' atmosphere shell, where the galaxy draws
// the star itself — and where L (planet -> star) is not the camera's direction to the star, so the
// engine's disc would show up as a second sun beside the real one
fn atmo_sky_dome(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    let limb = atmo_limb(ray_dir, cam_pos, a.planet_r);
    let sun_glow = pow(max(dot(ray_dir, L), 0.0), 8.0);
    let dome = mix(a.sky_zenith * 0.7, a.sky_horizon, sun_glow);
    return mix(a.space_color, dome, limb);
}

// the sun itself: a tight HDR-bright core plus a softer glare halo, additive so ACES blows it out
// white-hot. Only ever drawn where the ray truly reaches deep space (never for a ray that hit terrain
// first), so it's automatically hidden on the planet's own night side and correctly occluded wherever
// callers composite clouds on top afterwards.
// the star in the sky at its real apparent size (a.star_angle), shaded like the galaxy's star (star.wgsl):
// limb-darkened disc, then its corona, so the sky's sun and the galaxy backdrop's star match at the
// handover. It fades with the sky's opacity like the rest of the sky (atmo_sky_over_black, fs_light).
fn atmo_sun_disc(ray_dir: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    if (a.star_angle <= 0.0) {
        return vec3<f32>(0.0);
    }
    // the star scaled to distance 1: centre at L, radius sin(star_angle); the near side's hit point, as
    // fs_star finds it, turned into galaxy space so the same granules and spots show (star_shading.rs
    // sky_disc_normal mirrors this)
    let r = sin(a.star_angle);
    let along = dot(ray_dir, L);
    let closest = length(L - ray_dir * along);
    if (closest < r && along > 0.0) {
        let hit = along - sqrt(r * r - closest * closest);
        let n_planet = (ray_dir * hit - L) / r;
        let mu = max(dot(n_planet, -ray_dir), 0.0);
        let q = a.star_frame;
        let n = n_planet + 2.0 * cross(q.xyz, cross(q.xyz, n_planet) + q.w * n_planet);
        return star_surface(normalize(n), mu, a.time, a.star);
    }
    // the corona around the ray's closest point to the star that lies ahead of the camera: measured
    // against the whole line, a ray pointing straight away from the sun passed it at distance 0 and
    // drew a corona-only second sun opposite the real one (star_shading.rs sky_corona_distance)
    let closest_ahead = length(L - ray_dir * max(along, 0.0));
    return star_corona((closest_ahead - r) / r, a.star);
}

// the sky as it looks over a black background — for fog and water reflections, which can't see what
// lies behind: at night and in space they fade toward dark like the sky itself, not toward blue
fn atmo_sky_over_black(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    return atmo_sky_gradient(ray_dir, cam_pos, L, a) * atmo_sky_opacity(ray_dir, cam_pos, L, a.planet_r);
}

// length of the segment from the camera to `world_pos` that lies inside the atmosphere (the sphere at
// the limb scale, planet_r * CLOUD_ALT * 1.2): a camera inside it gets the whole distance
fn atmo_fog_distance(cam_pos: vec3<f32>, world_pos: vec3<f32>, planet_r: f32) -> f32 {
    let dist = distance(cam_pos, world_pos);
    let atmo_r = planet_r * CLOUD_ALT * 1.2;
    if (dot(cam_pos, cam_pos) <= atmo_r * atmo_r) {
        return dist;
    }
    let entry = sphere_hit(cam_pos, (world_pos - cam_pos) / max(dist, 1e-6), atmo_r);
    if (entry < 0.0) {
        return 0.0;
    }
    return max(dist - entry, 0.0);
}

// exp² air fog toward the sky the camera would see in that direction, so there's no seam between
// distant terrain and open sky; stays in linear space (no tone mapping)
fn atmo_air_fog(lit: vec3<f32>, world_pos: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    // only the stretch of the view ray inside the atmosphere fogs: on the ground that's all of it (as
    // before); from orbit the empty space in between adds no haze — otherwise a planet seen from a few
    // radii out faded into a flat sky-coloured disc
    let dist = atmo_fog_distance(cam_pos, world_pos, a.planet_r);
    let fog_factor = 1.0 - exp(-(dist * FOG_DENSITY) * (dist * FOG_DENSITY * 0.5));
    let ray_dir = normalize(world_pos - cam_pos);
    let fog_col = atmo_sky_dome(ray_dir, cam_pos, L, a) * atmo_sky_opacity(ray_dir, cam_pos, L, a.planet_r);
    return mix(lit, fog_col, clamp(fog_factor, 0.0, 1.0));
}
