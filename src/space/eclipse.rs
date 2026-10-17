// Eclipses and phases (spec docs/superpowers/specs/2026-10-12-moons-design.md): how much of the star's disc
// the family bodies (a planet's moons; for a moon its parent and siblings) cover as seen from a point, the lit
// fraction of a body, and the family list for the GPU. atmosphere.wgsl mirrors the maths; CONSTANTS and the
// formulas here are the reference for its tests.
// Everything here is used by the renderer only from Task 8 on.
#![cfg_attr(not(test), allow(dead_code))] // until Task 8 wires it in

use glam::{DVec3, Vec3};

use crate::space::solar_system::SolarSystem;

pub const MAX_FAMILY: usize = 6;
pub const MAX_OCCLUDERS: usize = 4;
pub const SKY_DIM_MIN: f32 = 0.25; // the sky and fill light at total eclipse, relative to full sun
pub const CONSTANTS: [(&str, f32); 1] = [("ECLIPSE_SKY_DIM_MIN", SKY_DIM_MIN)];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FamilyBody {
    pub centre: Vec3, // in the camera body's frame
    pub radius: f32,
    pub color: [f32; 3],
}

// the share (0..1) of a disc of angular radius `rs` (the star) that a disc of radius `ro` covers when their
// centres are `d` apart (radians, small angles): the standard lens area
pub fn disc_overlap_fraction(rs: f32, ro: f32, d: f32) -> f32 {
    if rs <= 0.0 || ro <= 0.0 || d >= rs + ro {
        return 0.0;
    }
    if d <= (rs - ro).abs() {
        return if ro >= rs { 1.0 } else { (ro / rs) * (ro / rs) };
    }
    let a = rs
        * rs
        * ((d * d + rs * rs - ro * ro) / (2.0 * d * rs))
            .clamp(-1.0, 1.0)
            .acos();
    let b = ro
        * ro
        * ((d * d + ro * ro - rs * rs) / (2.0 * d * ro))
            .clamp(-1.0, 1.0)
            .acos();
    let k = ((-d + rs + ro) * (d + rs - ro) * (d - rs + ro) * (d + rs + ro))
        .max(0.0)
        .sqrt();
    ((a + b - 0.5 * k) / (std::f32::consts::PI * rs * rs)).clamp(0.0, 1.0)
}

// 1 - the star's disc covered by up to MAX_OCCLUDERS bodies, seen from `point` (sun_dir: unit vector from the
// point to the star, star_angle: the star's angular radius there)
pub fn sun_visibility(point: Vec3, sun_dir: Vec3, star_angle: f32, bodies: &[FamilyBody]) -> f32 {
    let mut covered = 0.0;
    for b in bodies.iter().take(MAX_OCCLUDERS) {
        let v = b.centre - point;
        let dist = v.length();
        if dist <= b.radius || dist < 1e-4 {
            continue; // inside the body: the point is not in the sky
        }
        let dir = v / dist;
        let cos = dir.dot(sun_dir);
        if cos <= 0.0 {
            continue; // behind the point
        }
        let ro = (b.radius / dist).min(1.0).asin();
        covered += disc_overlap_fraction(star_angle, ro, cos.clamp(-1.0, 1.0).acos());
    }
    (1.0 - covered).clamp(0.0, 1.0)
}

// the lit share of a body seen from an observer: (1 + cos a) / 2 with a the angle at the body between the
// observer and the star (observer_to_body: from the observer to the body, body_to_star: from the body to the star)
pub fn lit_fraction(observer_to_body: Vec3, body_to_star: Vec3) -> f32 {
    let to_observer = (-observer_to_body).normalize_or_zero();
    let to_star = body_to_star.normalize_or_zero();
    0.5 * (1.0 + to_observer.dot(to_star))
}

// the bodies that can eclipse or hang in the sky of `body`: a planet's moons; a moon's parent and siblings.
// In body's own frame, nearest by apparent size first, at most MAX_FAMILY.
pub fn family_of(system: &SolarSystem, body: usize, t: f64) -> Vec<FamilyBody> {
    let me = &system.planets[body];
    let members: Vec<usize> = match me.parent {
        Some(p) => std::iter::once(p)
            .chain(system.moons_of(p).filter(|&m| m != body))
            .collect(),
        None => system.moons_of(body).collect(),
    };
    let mut out: Vec<(f32, FamilyBody)> = members
        .into_iter()
        .map(|i| {
            let o = &system.planets[i];
            let centre = me.to_planet_frame(o.position_at(t), t).as_vec3();
            let c = o.planet_type.def().palette.ground.color();
            let color = [c[0] * 0.8, c[1] * 0.8, c[2] * 0.8];
            let size = o.radius / centre.length().max(1e-3);
            (
                size,
                FamilyBody {
                    centre,
                    radius: o.radius,
                    color,
                },
            )
        })
        .collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.into_iter().take(MAX_FAMILY).map(|(_, b)| b).collect()
}

// two vec4 per body: xyz centre + w radius; rgb colour + w 1 (present); zeros after the count
pub fn pack_family(f: &[FamilyBody]) -> [[f32; 4]; MAX_FAMILY * 2] {
    let mut out = [[0.0; 4]; MAX_FAMILY * 2];
    for (i, b) in f.iter().take(MAX_FAMILY).enumerate() {
        out[2 * i] = [b.centre.x, b.centre.y, b.centre.z, b.radius];
        out[2 * i + 1] = [b.color[0], b.color[1], b.color[2], 1.0];
    }
    out
}

// sun visibility at `camera` (in `body`'s frame) at time t, and the sky/fill dim factor derived from it
pub fn camera_visibility(system: &SolarSystem, body: usize, camera: Vec3, t: f64) -> (f32, f32) {
    let p = &system.planets[body];
    let sun_dir = p.sun_dir_in_planet_frame(DVec3::ZERO, t);
    let v = sun_visibility(
        camera,
        sun_dir,
        p.star_angular_radius(&system.star),
        &family_of(system, body, t),
    );
    (v, SKY_DIM_MIN + (1.0 - SKY_DIM_MIN) * v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric(rs: f32, ro: f32, d: f32) -> f32 {
        let n = 800;
        let (mut inside, mut covered) = (0, 0);
        for i in 0..n {
            for j in 0..n {
                let x = -rs + 2.0 * rs * (i as f32 + 0.5) / n as f32;
                let y = -rs + 2.0 * rs * (j as f32 + 0.5) / n as f32;
                if x * x + y * y <= rs * rs {
                    inside += 1;
                    if (x - d) * (x - d) + y * y <= ro * ro {
                        covered += 1;
                    }
                }
            }
        }
        covered as f32 / inside as f32
    }

    #[test]
    fn overlap_is_none_partial_annular_or_total() {
        assert_eq!(disc_overlap_fraction(0.2, 0.05, 0.5), 0.0, "apart");
        assert_eq!(disc_overlap_fraction(0.2, 0.3, 0.0), 1.0, "total");
        assert_eq!(
            disc_overlap_fraction(0.2, 0.3, 0.05),
            1.0,
            "total, off centre"
        );
        let annular = disc_overlap_fraction(0.2, 0.1, 0.0);
        assert!((annular - 0.25).abs() < 1e-6, "annular: {annular}");
        for (rs, ro, d) in [
            (0.2, 0.1, 0.25),
            (0.2, 0.2, 0.15),
            (0.2, 0.35, 0.3),
            (0.05, 0.04, 0.06),
        ] {
            let (a, b) = (disc_overlap_fraction(rs, ro, d), numeric(rs, ro, d));
            assert!((a - b).abs() < 0.01, "{rs} {ro} {d}: {a} vs {b}");
        }
        for (rs, ro, d) in [
            (0.2, 0.2, 0.0),
            (0.2, 0.2, 0.4),
            (0.2, 0.0, 0.1),
            (0.0, 0.1, 0.1),
            (0.2, 0.1, 0.3),
        ] {
            let v = disc_overlap_fraction(rs, ro, d);
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "{rs} {ro} {d}: {v}"
            );
        }
    }

    #[test]
    fn the_sun_is_visible_without_occluders_and_gone_in_a_total_eclipse() {
        let sun = Vec3::X;
        let star = 0.2_f32;
        assert_eq!(sun_visibility(Vec3::ZERO, sun, star, &[]), 1.0);
        let big = FamilyBody {
            centre: Vec3::new(10.0, 0.0, 0.0),
            radius: 3.0,
            color: [1.0; 3],
        };
        assert_eq!(sun_visibility(Vec3::ZERO, sun, star, &[big]), 0.0, "total");
        let small = FamilyBody {
            centre: Vec3::new(10.0, 0.0, 0.0),
            radius: 1.0,
            color: [1.0; 3],
        };
        let v = sun_visibility(Vec3::ZERO, sun, star, &[small]);
        assert!(v > 0.7 && v < 0.9, "annular {v}");
        let behind = FamilyBody {
            centre: Vec3::new(-10.0, 0.0, 0.0),
            radius: 3.0,
            color: [1.0; 3],
        };
        let side = FamilyBody {
            centre: Vec3::new(0.0, 10.0, 0.0),
            radius: 3.0,
            color: [1.0; 3],
        };
        let at = FamilyBody {
            centre: Vec3::ZERO,
            radius: 3.0,
            color: [1.0; 3],
        };
        assert_eq!(sun_visibility(Vec3::ZERO, sun, star, &[behind]), 1.0);
        assert_eq!(sun_visibility(Vec3::ZERO, sun, star, &[side]), 1.0);
        assert!(sun_visibility(Vec3::ZERO, sun, star, &[at]).is_finite());
        let many = vec![small; MAX_FAMILY];
        let v = sun_visibility(Vec3::ZERO, sun, star, &many);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn phases_follow_the_dot_product() {
        // observer at the origin, body 5 units along +X
        let obs = Vec3::X * 5.0;
        let quarter = lit_fraction(obs, Vec3::Y * 5.0);
        assert!((quarter - 0.5).abs() < 0.01, "{quarter}");
        // star on the observer's side of the body (body_to_star toward the observer): full
        assert!(lit_fraction(obs, Vec3::NEG_X) > 0.99);
        // star behind the body as seen from the observer: new
        assert!(lit_fraction(obs, Vec3::X) < 0.01);
    }

    #[test]
    fn the_family_is_the_camera_bodys_moons_or_its_parent_and_siblings() {
        let s = crate::space::solar_system::tests::with_moon();
        let m = s.planets.len() - 1;
        let of_parent = family_of(&s, 2, 5.0);
        assert_eq!(of_parent.len(), 1);
        assert_eq!(of_parent[0].radius, 40.0);
        let of_moon = family_of(&s, m, 5.0);
        assert_eq!(of_moon.len(), 1, "its parent (no siblings here)");
        assert_eq!(of_moon[0].radius, s.planets[2].radius);
        assert!(
            family_of(&s, 0, 5.0).is_empty(),
            "a planet without moons has none"
        );
        assert!((of_parent[0].centre.length() - 3.0 * s.planets[2].radius).abs() < 1.0);
    }

    #[test]
    fn packing_pads_with_zeros_and_flags_present_bodies() {
        let b = FamilyBody {
            centre: Vec3::new(1.0, 2.0, 3.0),
            radius: 4.0,
            color: [0.1, 0.2, 0.3],
        };
        let p = pack_family(&[b]);
        assert_eq!(p[0], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p[1], [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(p[2], [0.0; 4]);
        assert_eq!(p[3], [0.0; 4]);
    }

    #[test]
    fn camera_visibility_is_full_without_a_family_and_dims_in_eclipse() {
        let s = crate::space::solar_system::tests::with_moon();
        assert_eq!(camera_visibility(&s, 0, Vec3::ZERO, 5.0), (1.0, 1.0));
        let (v, dim) = camera_visibility(&s, 2, Vec3::ZERO, 5.0);
        assert!((0.0..=1.0).contains(&v) && (SKY_DIM_MIN..=1.0).contains(&dim));
    }

    #[test]
    fn wgsl_constants_match_the_rust_mirror() {
        let w = include_str!("../render/atmosphere.wgsl");
        crate::weather::tests::assert_wgsl_constants(w, &CONSTANTS);
    }
}
