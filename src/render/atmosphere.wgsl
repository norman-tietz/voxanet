// atmosphere.wgsl
// Sky, atmosphere limb, clouds (the noise; clouds.wgsl samples it from the baked cloud map), air fog and tone mapping shared by the voxel engine (shader.wgsl) and
// the planet impostors (space.wgsl). Both are compiled with this file prepended (renderer.rs
// SCENE_SHADER, render.rs SPACE_SHADER), so a planet looks the same from space flight and from
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
// lightning (weather/lightning.rs mirrors the radius)
const LIGHTNING_GLOW_RADIUS: f32 = 0.13;  // cloud glow radius at the shell, in planet radii
const LIGHTNING_COLOR = vec3<f32>(0.55, 0.65, 1.0); // cool blue-white
const LIGHTNING_CLOUD_GAIN: f32 = 2.0;    // how bright a strike makes the clouds around it
const LIGHTNING_AMBIENT_GAIN: f32 = 0.5;  // sky fill added at full scene flash
const LIGHTNING_DOME_GAIN: f32 = 0.25;     // the dome's brightening at full scene flash
// exp² air fog density, tuned for the scale defined in gen.rs
const FOG_DENSITY: f32 = 0.0015;
// storm and mist fog (weather/fog.rs mirrors these)
const FOG_STORM_LO: f32 = 0.25;        // storm value where the haze starts
const FOG_STRUCTURE: f32 = 0.6;        // the storm density swings by this share around its mean
const FOG_STRUCT_SCALE: f32 = 0.02;    // structure noise cells per unit
const FOG_STRUCT_MAX_DIST: f32 = 300.0; // the structure taps lie within this of the camera
const MIST_HEIGHT: f32 = 10.0;         // mist e-folding height, blocks
const MIST_EXTINCTION: f32 = 0.02;     // per unit at mist 1, at the surface
const STORM_HEIGHT: f32 = 30.0;        // storm haze e-folding height above the highest peak, blocks
const MIST_REGION_HEIGHT: f32 = 150.0; // the camera's mist fades out with its height over this
const MIST_COLOR = vec3<f32>(0.75, 0.78, 0.82);

// a planet's atmosphere: radius (world units, = voxel resolution / 2) and its planet type's colours
struct Atmosphere {
    planet_r: f32,
    sky_zenith: vec3<f32>,
    sky_horizon: vec3<f32>,
    space_color: vec3<f32>,
    cloud_light: vec3<f32>,
    cloud_dark: vec3<f32>,
    sun_color: vec3<f32>, // sunlight: the star type's (solar_system.rs StarTypeDef.sunlight)
    star: StarLook,       // the star's look (star.wgsl), for the sun disc in the sky
    star_angle: f32,      // the star's angular radius from this planet; 0 = no disc (impostor shells)
    star_frame: vec4<f32>, // quaternion: planet frame -> space, so the disc shows the star's own surface
    time: f32,            // seconds, for the star's boiling surface
    overcast: f32,        // 0..1 storm overhead (weather.rs overcast): greys and dims the dome
    density: f32,         // 1 = full atmosphere, 0 = vacuum (planet/biome.rs AtmosphereDef.density)
    flash: f32,           // 0..1 scene flash at the camera (weather/lightning.rs flash_at)
    strikes: array<vec4<f32>, 6>, // active strikes: xyz planet-frame position on the surface, w intensity
    sky_dim: f32,         // eclipse: the sky dome's dimming (1 = none, eclipse.rs)
    sun_vis: f32,         // eclipse: share of the star's light reaching the camera (1 = none); dims the clouds' sun term
}

// the weather's fog at the camera (weather/fog.rs FogParams)
struct FogState {
    storm: f32,      // the storm over the camera, 0..1
    peak: f32,       // extinction per unit at storm 1 (the planet type's, x the setting); 0 = no storm fog
    mist: f32,       // 0..1 ground mist at the camera
    mist_base: f32,  // radius the mist's height is measured from (the sea surface)
    storm_base: f32, // radius of the planet's highest natural peak: the storm haze thins above it
    tint: vec3<f32>, // the storm haze's colour at full daylight
    wind: vec3<f32>, // the structure's scroll offset (planet frame, units)
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
fn atmo_cloud_shadow(world_pos: vec3<f32>, L: vec3<f32>, t: f32, planet_r: f32, density: f32) -> f32 {
    let hit_t = sphere_hit(world_pos, L, planet_r * CLOUD_ALT);
    if (hit_t < 0.0) { return 1.0; }
    let coverage = cloud_coverage_fast(normalize(world_pos + L * hit_t), t);
    return 1.0 - density * coverage * CLOUD_SHADOW_STRENGTH;
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

// how opaque the sky is along the ray, for blending it over what lies behind (the solar system): thick,
// sunlit atmosphere = 1; open space and the night side = 0. Daylight is the sun's elevation at the
// ray's closest point to the planet (for a camera on the ground looking up: the camera itself).
fn atmo_sky_opacity(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, planet_r: f32, density: f32) -> f32 {
    let s_star = max(-dot(cam_pos, ray_dir), 0.0);
    let closest_point = cam_pos + ray_dir * s_star;
    let up = closest_point / max(length(closest_point), 1e-4);
    let day = smoothstep(SKY_NIGHT_ELEVATION, SKY_DAY_ELEVATION, dot(up, L));
    let thick = smoothstep(SKY_SPACE_LIMB, SKY_OPAQUE_LIMB, atmo_limb(ray_dir, cam_pos, planet_r));
    return thick * day * density;
}

// atmosphere seen beyond the clouds: a stylised gradient from deep space into a blue dome that warms
// toward the sun, plus a glow where the view ray grazes the planet's limb (seen from orbit/third person)
fn atmo_sky_gradient(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    return atmo_sky_dome(ray_dir, cam_pos, L, a) + atmo_sun_disc(ray_dir, L, a);
}

// the sky without the sun disc: for the planet impostors' atmosphere shell, where the solar system draws
// the star itself — and where L (planet -> star) is not the camera's direction to the star, so the
// engine's disc would show up as a second sun beside the real one
const OVERCAST_GREY: f32 = 0.6; // an overcast dome goes this far toward grey ...
const OVERCAST_DIM: f32 = 0.4;  // ... and loses this much brightness

fn atmo_sky_dome(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    let limb = atmo_limb(ray_dir, cam_pos, a.planet_r);
    let sun_glow = pow(max(dot(ray_dir, L), 0.0), 8.0);
    let dome = mix(a.sky_zenith * 0.7, a.sky_horizon, sun_glow);
    // an overcast sky: greyer and darker under a storm (up to 40 %)
    let grey = vec3<f32>(dot(dome, vec3<f32>(0.3, 0.59, 0.11)));
    let dull = mix(dome, grey, OVERCAST_GREY * a.overcast) * (1.0 - OVERCAST_DIM * a.overcast);
    return mix(a.space_color, dull + LIGHTNING_COLOR * a.flash * LIGHTNING_DOME_GAIN, limb) * a.sky_dim;
}

// the sun itself: a tight HDR-bright core plus a softer glare halo, additive so ACES blows it out
// white-hot. Only ever drawn where the ray truly reaches deep space (never for a ray that hit terrain
// first), so it's automatically hidden on the planet's own night side and correctly occluded wherever
// callers composite clouds on top afterwards.
// the star in the sky at its real apparent size (a.star_angle), shaded like the solar system's star (star.wgsl):
// limb-darkened disc, then its corona, so the sky's sun and the space backdrop's star match at the
// handover. It fades with the sky's opacity like the rest of the sky (atmo_sky_over_black, fs_light).
fn atmo_sun_disc(ray_dir: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    if (a.star_angle <= 0.0) {
        return vec3<f32>(0.0);
    }
    // the star scaled to distance 1: centre at L, radius sin(star_angle); the near side's hit point, as
    // fs_star finds it, turned into space so the same granules and spots show (star_shading.rs
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

// ---- eclipses and moons in the sky (space/eclipse.rs mirrors the maths) ----
// The family is a short list of bodies (a planet's moons; a moon's parent and siblings) as two vec4 each:
// xyz centre + w radius in the camera body's frame, rgb colour + w 1. With count 0 every function exits at
// once, returning exactly 1.0 / 0.0, so a family-less frame is bit-identical to one without them.
const MAX_OCCLUDERS: i32 = 4;
const ECLIPSE_SKY_DIM_MIN: f32 = 0.25;  // sky and fill light at a total eclipse (mirrors space/eclipse.rs)

// the share of a disc of angular radius rs covered by a disc of radius ro, centres d apart (eclipse.rs
// disc_overlap_fraction)
fn atmo_disc_overlap(rs: f32, ro: f32, d: f32) -> f32 {
    if (rs <= 0.0 || ro <= 0.0 || d >= rs + ro) { return 0.0; }
    if (d <= abs(rs - ro)) { return select((ro / rs) * (ro / rs), 1.0, ro >= rs); }
    let a = rs * rs * acos(clamp((d * d + rs * rs - ro * ro) / (2.0 * d * rs), -1.0, 1.0));
    let b = ro * ro * acos(clamp((d * d + ro * ro - rs * rs) / (2.0 * d * ro), -1.0, 1.0));
    let k = sqrt(max((-d + rs + ro) * (d + rs - ro) * (d - rs + ro) * (d + rs + ro), 0.0));
    return clamp((a + b - 0.5 * k) / (3.14159265 * rs * rs), 0.0, 1.0);
}

// 1 minus the share of the star's disc the family bodies cover as seen from `p` (L: unit vector to the star,
// star_angle: its angular radius there; eclipse.rs sun_visibility)
fn atmo_sun_visibility(p: vec3<f32>, L: vec3<f32>, star_angle: f32, family: array<vec4<f32>, 12>, count: f32) -> f32 {
    if (count <= 0.0) { return 1.0; }
    var covered = 0.0;
    for (var i = 0; i < MAX_OCCLUDERS; i++) {
        if (f32(i) >= count) { break; }
        let c = family[i * 2];
        let v = c.xyz - p;
        let dist = length(v);
        if (dist <= c.w || dist < 1e-4) { continue; }
        let cosd = dot(v / dist, L);
        if (cosd <= 0.0) { continue; }
        covered += atmo_disc_overlap(star_angle, asin(min(c.w / dist, 1.0)), acos(clamp(cosd, -1.0, 1.0)));
    }
    return clamp(1.0 - covered, 0.0, 1.0);
}

// how much of the family covers the sky along `ray_dir` from `cam` (0..1, the largest of the discs'),
// for hiding the sun's disc behind a moon
fn atmo_family_cover(ray_dir: vec3<f32>, cam: vec3<f32>, family: array<vec4<f32>, 12>, count: f32) -> f32 {
    if (count <= 0.0) { return 0.0; }
    var best = 0.0;
    for (var i = 0; i < 6; i++) {
        if (f32(i) >= count) { break; }
        let c = family[i * 2];
        let v = c.xyz - cam;
        let dist = length(v);
        if (dist <= c.w || dist < 1e-4) { continue; }
        let ro = asin(min(c.w / dist, 1.0));
        let d = acos(clamp(dot(v / dist, ray_dir), -1.0, 1.0));
        let edge = max(ro * 0.04, 1e-5);
        best = max(best, 1.0 - smoothstep(ro - edge, ro + edge, d));
    }
    return best;
}

// the family bodies as discs in the sky along `ray_dir` from `cam`: phase-lit by the star (L: direction to
// the star), a faint floor on the dark side; rgb and the coverage as alpha. Used where the sky is opaque
// (by day), added to the sky (the unlit side shows the sky through it); the backdrop's real impostor takes over
// where it turns transparent. Overlapping discs: the nearest in front.
fn atmo_sky_bodies(ray_dir: vec3<f32>, cam: vec3<f32>, L: vec3<f32>, family: array<vec4<f32>, 12>, count: f32, sun: vec3<f32>) -> vec4<f32> {
    if (count <= 0.0) { return vec4<f32>(0.0); }
    var prem = vec3<f32>(0.0); // premultiplied colour of the discs so far
    var alpha = 0.0;
    var nearest = 1e30;
    for (var i = 0; i < 6; i++) {
        if (f32(i) >= count) { break; }
        let c = family[i * 2];
        let col = family[i * 2 + 1];
        let v = c.xyz - cam;
        let dist = length(v);
        if (dist <= c.w || dist < 1e-4) { continue; }
        let ro = asin(min(c.w / dist, 1.0));
        let d = acos(clamp(dot(v / dist, ray_dir), -1.0, 1.0));
        let edge = max(ro * 0.04, 1e-5);
        let cover = 1.0 - smoothstep(ro - edge, ro + edge, d);
        if (cover <= 0.0) { continue; }
        // the sphere's normal under this ray: the near hit point, relative to the body's centre
        let along = dot(v, ray_dir);
        let off = length(v - ray_dir * along);
        let hit = along - sqrt(max(c.w * c.w - off * off, 0.0));
        let n = normalize(ray_dir * hit - v);
        let lit = max(dot(n, L), 0.0);
        let colour = col.rgb * sun * (lit * 0.9 + 0.02);
        // the nearest disc wins where discs overlap (the family list is sorted by apparent size, not distance)
        if (dist < nearest) {
            prem = colour * cover + prem * (1.0 - cover);
            alpha = cover + alpha * (1.0 - cover);
            nearest = dist;
        } else {
            prem = prem + colour * cover * (1.0 - alpha);
            alpha = alpha + cover * (1.0 - alpha);
        }
    }
    return vec4<f32>(prem / max(alpha, 1e-6), alpha);
}

// the sky as it looks over a black background — for fog and water reflections, which can't see what
// lies behind: at night and in space they fade toward dark like the sky itself, not toward blue
fn atmo_sky_over_black(ray_dir: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere) -> vec3<f32> {
    return atmo_sky_gradient(ray_dir, cam_pos, L, a) * atmo_sky_opacity(ray_dir, cam_pos, L, a.planet_r, a.density);
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

fn fog_storm_weight(storm: f32) -> f32 {
    return smoothstep(FOG_STORM_LO, 1.0, storm);
}

// the storm haze's optical depth over `dist`: the mean of the storm at the camera and at the shaded
// point, times 3 noise taps along the ray (world-anchored, scrolling with the wind) so it comes in drifting
// walls instead of an even grey. Exactly 0 without a storm.
fn atmo_storm_optical_depth(f: FogState, storm_there: f32, cam: vec3<f32>, pos: vec3<f32>, dist: f32) -> f32 {
    // heights above the highest peak (0 below it, so on the ground the falloff is exactly 1); from altitude
    // only the storm at the shaded point counts (fog.rs storm_optical_depth)
    let h0 = max(length(cam) - f.storm_base, 0.0);
    let h1 = max(length(pos) - f.storm_base, 0.0);
    let storm_mix = mix(storm_there, 0.5 * (f.storm + storm_there), exp(-h0 / STORM_HEIGHT));
    let w = fog_storm_weight(storm_mix);
    if (w * f.peak <= 0.0 || dist <= 0.0) { return 0.0; }
    let dir = (pos - cam) / max(distance(pos, cam), 1e-6);
    let reach = min(dist, FOG_STRUCT_MAX_DIST);
    // the structure is anchored in the world and drifts with the wind: sample behind the offset
    let o = f.wind;
    let n = (value_noise3((cam + dir * reach * 0.2 - o) * FOG_STRUCT_SCALE)
           + value_noise3((cam + dir * reach * 0.5 - o) * FOG_STRUCT_SCALE)
           + value_noise3((cam + dir * reach * 0.8 - o) * FOG_STRUCT_SCALE)) / 3.0;
    let structure = 1.0 + FOG_STRUCTURE * (n - 0.5) * 2.0;
    let e0 = exp(-h0 / STORM_HEIGHT);
    let e1 = exp(-h1 / STORM_HEIGHT);
    let dh = h1 - h0;
    let falloff = select(e0, STORM_HEIGHT * (e0 - e1) / dh, abs(dh) > 1e-3);
    return f.peak * w * structure * falloff * dist;
}

// the mist's optical depth: density mist * MIST_EXTINCTION * exp(-h / MIST_HEIGHT), h the height above the
// sea surface, integrated in closed form along the path (fog.rs mist_optical_depth)
fn atmo_mist_optical_depth(f: FogState, cam: vec3<f32>, pos: vec3<f32>, dist: f32) -> f32 {
    if (f.mist <= 0.0) { return 0.0; }
    let h0 = max(length(cam) - f.mist_base, 0.0);
    let h1 = max(length(pos) - f.mist_base, 0.0);
    let e0 = exp(-h0 / MIST_HEIGHT);
    let e1 = exp(-h1 / MIST_HEIGHT);
    let dh = h1 - h0;
    let mean = select(e0, MIST_HEIGHT * (e0 - e1) / dh, abs(dh) > 1e-3);
    // the mist at the camera belongs to the ground: fades out with the camera's height
    return f.mist * exp(-h0 / MIST_REGION_HEIGHT) * MIST_EXTINCTION * mean * dist;
}

// exp² air fog toward the sky the camera would see in that direction, so there's no seam between
// distant terrain and open sky; stays in linear space (no tone mapping). Storm and mist add optical
// depth to the exponent and tint the colour; with both zero this is the plain exp² fog.
fn atmo_air_fog(lit: vec3<f32>, world_pos: vec3<f32>, cam_pos: vec3<f32>, L: vec3<f32>, a: Atmosphere,
                f: FogState, storm_there: f32) -> vec3<f32> {
    // only the stretch of the view ray inside the atmosphere fogs: on the ground that's all of it (as
    // before); from orbit the empty space in between adds no haze — otherwise a planet seen from a few
    // radii out faded into a flat sky-coloured disc
    let dist = atmo_fog_distance(cam_pos, world_pos, a.planet_r);
    let ts = atmo_storm_optical_depth(f, storm_there, cam_pos, world_pos, dist);
    let tm = atmo_mist_optical_depth(f, cam_pos, world_pos, dist);
    let extra = ts + tm;
    let fog_factor = (1.0 - exp(-(dist * FOG_DENSITY) * (dist * FOG_DENSITY * 0.5) - extra)) * a.density;
    let ray_dir = normalize(world_pos - cam_pos);
    var fog_col = atmo_sky_dome(ray_dir, cam_pos, L, a) * atmo_sky_opacity(ray_dir, cam_pos, L, a.planet_r, a.density);
    if (extra > 0.0) {
        // the haze is lit by the day: sand tan, ash grey, snow white by day, dark at night
        let day = atmo_ambient_daylight(world_pos, L);
        let storm_col = f.tint * day * (1.0 - OVERCAST_DIM * a.overcast);
        let mist_col = mix(fog_col, MIST_COLOR * day, 0.5);
        let base = (dist * FOG_DENSITY) * (dist * FOG_DENSITY * 0.5);
        fog_col = mix(fog_col, mix(mist_col, storm_col, ts / extra), extra / (base + extra));
    }
    return mix(lit, fog_col, clamp(fog_factor, 0.0, 1.0));
}

// --- plastic shading (plastic.rs, docs/superpowers/specs/2026-10-10-plastic-shading-design.md) ---

const PLASTIC_F0: f32 = 0.04; // Fresnel reflectance at normal incidence: dielectric plastic (plastic.rs)

// wrap lighting (light.rs): the diffuse sun term reaching `wrap` past the terminator,
// (N·L + w) / (1 + w); 0 = plain Lambert. Mirrored in light.rs.
fn atmo_wrap_diffuse(n_l: f32, wrap: f32) -> f32 {
    return max((n_l + wrap) / (1.0 + wrap), 0.0);
}

// the ground's light at `pos`, the lower half of the hemispheric sky light (light.rs): GROUND_COLOR's
// dark floor under the sky's fill light, plus sunlit ground in the surface's own colour (`albedo`,
// linear) — the cheap stand-in for one bounce: what lies around a block is mostly its colour — lit at
// the sun's elevation there by `sun` (sun colour x cloud shadow), times `bounce`
fn atmo_ground_bounce(albedo: vec3<f32>, pos: vec3<f32>, L: vec3<f32>, sun: vec3<f32>,
                      bounce: f32) -> vec3<f32> {
    let up = pos / max(length(pos), 1e-4);
    return GROUND_COLOR * atmo_ambient_daylight(pos, L) + albedo * sun * (max(dot(up, L), 0.0) * bounce);
}

struct Plastic {
    spec: vec3<f32>, // linear radiance: sun highlight + sky reflection
    diffuse_w: f32,  // weight of the diffuse terms: 1 - strength * F (energy split)
}

// the plastic's environment without the map (the impostors, and /plastic env off): the sky dome along
// the reflection (no sun disc: the sun is the GGX highlight), the ground colour below the local horizon,
// blended toward `ambient` with roughness as a cheap stand-in for pre-filtering
fn atmo_env_analytic(R: vec3<f32>, pos: vec3<f32>, L: vec3<f32>, ground: vec3<f32>,
                     ambient: vec3<f32>, roughness: f32, a: Atmosphere) -> vec3<f32> {
    let up = normalize(pos);
    let sky = atmo_sky_dome(R, pos, L, a) * atmo_sky_opacity(R, pos, L, a.planet_r, a.density);
    let env = mix(ground, sky, smoothstep(-0.1, 0.1, dot(R, up)));
    return mix(env, ambient, clamp(roughness, 0.05, 1.0));
}

// glossy plastic at a surface point (planet frame): a GGX sun highlight (Trowbridge-Reitz distribution,
// Schlick Fresnel, Smith-Schlick visibility) lit by `sun` (sun colour x shadow x cloud shadow), and the
// environment `env` along the reflection (shade(): the pre-filtered map, env_map.rs; the impostors and
// /plastic env off: atmo_env_analytic), times `ao` and `reflection` (its own weight: at the physical ~13 % a bright sky washed block colours out).
// strength 0: no specular and diffuse_w 1, the purely diffuse look. Mirrored in plastic.rs.
fn atmo_plastic(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, sun: vec3<f32>, env: vec3<f32>,
                roughness: f32, strength: f32, reflection: f32, ao: f32) -> Plastic {
    if (strength <= 0.0) {
        return Plastic(vec3<f32>(0.0), 1.0);
    }
    let r = clamp(roughness, 0.05, 1.0);
    // bevel-tilted or deferred normals can face slightly away from the camera: clamp, no NaN
    let n_v = max(dot(N, V), 1e-4);
    let n_l = max(dot(N, L), 0.0);
    let H = normalize(L + V);
    let n_h = max(dot(N, H), 0.0);
    let al = r * r;
    let a2 = al * al;
    let dd = n_h * n_h * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159265 * dd * dd);
    let F = PLASTIC_F0 + (1.0 - PLASTIC_F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    let k = (r + 1.0) * (r + 1.0) / 8.0;
    let vis = 1.0 / (4.0 * (n_v * (1.0 - k) + k) * (n_l * (1.0 - k) + k));
    let highlight = sun * (D * F * vis * n_l);

    let F_env = PLASTIC_F0 + (max(1.0 - r, PLASTIC_F0) - PLASTIC_F0) * pow(1.0 - n_v, 5.0);
    let F_refl = F_env * reflection;
    // capped: at the lowest roughness the GGX peak overflowed half-float targets (motion blur's
    // shaded image) into Inf, and ACES turned that into black NaN pixels
    let spec = min(strength * (highlight + F_refl * env * ao), vec3<f32>(64.0));
    return Plastic(spec, 1.0 - strength * F_refl);
}
