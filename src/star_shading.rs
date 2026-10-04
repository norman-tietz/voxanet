// star_shading.rs
// Rust mirror of star.wgsl's closed-form shading terms, for tests (the WGSL is what renders).
#![cfg_attr(not(test), allow(dead_code))]

pub const LIMB_DARKENING: f32 = 0.6; // star.wgsl STAR_LIMB_DARKENING

// limb darkening: brightness at a point whose normal makes cos `mu` with the view ray
pub fn limb(mu: f32) -> f32 {
    1.0 - LIMB_DARKENING * (1.0 - mu.clamp(0.0, 1.0))
}

// corona brightness `d` star radii outside the limb, for a corona_size `size`
pub fn corona_falloff(d: f32, size: f32) -> f32 {
    let d = d.max(0.0);
    (-d / size).exp() / (1.0 + 8.0 * d)
}

// how far outside the sun's limb a ray passes, in star radii (mirror of atmo_sun_disc's corona input):
// with `l` the direction to the star and `angle` its angular radius
pub fn sky_corona_distance(ray: glam::Vec3, l: glam::Vec3, angle: f32) -> f32 {
    let r = angle.sin();
    let along = ray.dot(l);
    // the ray's closest point to the star, never behind the camera: measured against the whole line,
    // the point straight away from the sun passed it at distance 0 (a second, corona-only sun)
    let closest = (l - ray * along.max(0.0)).length();
    (closest - r) / r
}

// the star's surface normal the sky's sun disc samples for a ray (planet frame), with `l` the direction
// to the star and `angle` its angular radius, turned into galaxy space by `planet_to_galaxy` — the same
// point fs_star shades (mirror of atmosphere.wgsl atmo_sun_disc). None outside the disc.
pub fn sky_disc_normal(
    ray: glam::Vec3,
    l: glam::Vec3,
    angle: f32,
    planet_to_galaxy: glam::Quat,
) -> Option<glam::Vec3> {
    // the star scaled to distance 1: centre at `l`, radius sin(angle); the near side's hit, as fs_star
    let r = angle.sin();
    let along = ray.dot(l);
    let closest = (l - ray * along).length();
    if closest >= r || along <= 0.0 {
        return None;
    }
    let hit = along - (r * r - closest * closest).sqrt();
    Some(planet_to_galaxy * ((ray * hit - l) / r).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    // the sky's corona (atmo_sun_disc) glows only around the sun: a ray pointing away from it passes
    // the sun's line at distance 0, but never comes near the sun itself
    #[test]
    fn sky_corona_is_only_around_the_sun() {
        let l = glam::Vec3::new(0.3, 0.8, -0.52).normalize();
        let angle = 1.5f32.to_radians();
        let size = 1.0;
        let toward = corona_falloff(sky_corona_distance(l, l, angle), size);
        let away = corona_falloff(sky_corona_distance(-l, l, angle), size);
        let sideways = corona_falloff(
            sky_corona_distance(l.any_orthonormal_vector(), l, angle),
            size,
        );
        assert!(toward > 0.9, "{toward}");
        assert!(away < 1e-6, "anti-sun corona {away}");
        assert!(away <= sideways, "away {away} > sideways {sideways}");
    }

    // the sky's sun disc (atmo_sun_disc) shows the same point of the star as the galaxy's star (fs_star):
    // seen from a planet whose frame is turned by `q` (planet -> galaxy), a ray hits the star at the same
    // galaxy-space surface normal
    #[test]
    fn the_sky_disc_samples_the_galaxy_stars_surface() {
        use glam::{Quat, Vec3};
        let (centre, radius) = (Vec3::ZERO, 3000.0f32);
        let cam = Vec3::new(2000.0, 500.0, 8000.0);
        let q = Quat::from_rotation_y(1.1) * Quat::from_rotation_x(0.3); // the planet's spin
        let to_star = centre - cam;
        for (dx, dy) in [(0.0f32, 0.0f32), (0.15, 0.05), (-0.2, 0.25)] {
            let ray = (to_star.normalize() + Vec3::new(dx, dy, 0.0)).normalize();
            // galaxy (fs_star): ray-sphere hit, outward normal
            let along = (centre - cam).dot(ray);
            let closest = (centre - cam - ray * along).length();
            if closest >= radius {
                continue;
            }
            let hit = along - (radius * radius - closest * closest).sqrt();
            let expected = (cam + ray * hit - centre).normalize();
            // the sky, in the planet frame
            let l_planet = q.inverse() * to_star.normalize();
            let ray_planet = q.inverse() * ray;
            let angle = (radius / to_star.length()).asin();
            let n = sky_disc_normal(ray_planet, l_planet, angle, q).unwrap();
            assert!(
                (n - expected).length() < 1e-3,
                "{dx},{dy}: {n:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn limb_darkening_is_darker_at_the_edge() {
        assert_eq!(limb(1.0), 1.0);
        assert!((limb(0.0) - 0.4).abs() < 1e-6);
        assert!(limb(0.3) < limb(0.7));
    }

    #[test]
    fn corona_falls_off_from_the_limb() {
        assert_eq!(corona_falloff(0.0, 0.45), 1.0);
        let (a, b, c) = (
            corona_falloff(0.1, 0.45),
            corona_falloff(0.5, 0.45),
            corona_falloff(2.0, 0.45),
        );
        assert!(1.0 > a && a > b && b > c && c >= 0.0);
        assert!(c < 0.02, "still bright 2 radii out: {c}");
    }
}
