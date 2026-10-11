// star.wgsl
// The star's surface and corona, shared by the galaxy renderer (the star in space, galaxy.wgsl) and the
// voxel engine's sky (atmo_sun_disc, atmosphere.wgsl), so both show the same sun. Prepended after
// atmosphere.wgsl. Mirrored for tests in star_shading.rs.

const STAR_LIMB_DARKENING: f32 = 0.6;

// a star type's look (galaxy.rs StarTypeDef)
struct StarLook {
    surface: vec3<f32>,
    limb: vec3<f32>,
    corona: vec3<f32>,
    granulation: f32,
    granule_scale: f32,
    sunspots: f32,
    corona_size: f32,
}

fn star_limb(mu: f32) -> f32 {
    return 1.0 - STAR_LIMB_DARKENING * (1.0 - clamp(mu, 0.0, 1.0));
}

// PCG (as shader.wgsl's pcg_hash; this file is also prepended where that one isn't)
fn star_pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// a lattice point's value in [0, 1]. Integer hashing on purpose: the classic fract(sin(dot(p, k)) * big)
// loses its randomness once the dot product is large (GPU f32 sin, see galaxy.wgsl's star_hash)
fn sun_hash(p: vec3<f32>) -> f32 {
    let c = bitcast<vec3<u32>>(vec3<i32>(p));
    return f32(star_pcg(c.x ^ star_pcg(c.y ^ star_pcg(c.z)))) / 4294967295.0;
}

fn star_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = sun_hash(i);
    let n100 = sun_hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = sun_hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = sun_hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = sun_hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = sun_hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = sun_hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = sun_hash(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z,
    );
}

// surface colour at `dir` (unit, on the star's sphere) seen at cos `mu` from the view ray, time `t`:
// limb darkening (dimmer and warmer toward the edge), boiling granulation, slowly drifting sunspots
fn star_surface(dir: vec3<f32>, mu: f32, t: f32, s: StarLook) -> vec3<f32> {
    let p = dir * s.granule_scale + vec3<f32>(0.0, t * 0.05, 0.0);
    let cells = star_noise(p) * 0.6 + star_noise(p * 2.3 + vec3<f32>(t * 0.08)) * 0.4;
    let grain = 1.0 + s.granulation * (cells - 0.5);
    let spot_field = star_noise(dir * 3.0 + vec3<f32>(t * 0.01, 0.0, 0.0));
    let spot = smoothstep(1.0 - s.sunspots - 0.05, 1.0 - s.sunspots + 0.05, spot_field) * step(0.001, s.sunspots);
    let color = mix(s.limb, s.surface, clamp(mu, 0.0, 1.0));
    return color * star_limb(mu) * grain * (1.0 - 0.7 * spot);
}

// the corona `d` star radii outside the limb (premultiplied, additive)
fn star_corona(d: f32, s: StarLook) -> vec3<f32> {
    let dd = max(d, 0.0);
    return s.corona * exp(-dd / s.corona_size) / (1.0 + 8.0 * dd);
}
