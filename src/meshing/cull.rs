// cull.rs: back-face group culling for the closed voxel styles (cubes, hex).
//
// A chunk's triangles are reordered into groups by face direction (top, bottom, the wall directions
// around the chunk's up, and an always-drawn "other" group); each group keeps a normal cone. The
// geometry pass skips a group when the camera is behind every plane its cone allows over the chunk's
// bounding sphere (`group_backfacing`). The meshes are closed, so a back face is always hidden by
// front faces; only a camera clipped inside geometry can notice. The BLAS uses the full index buffer.

use crate::planet::data::Vertex;
use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct IndexGroup {
    pub first: u32,
    pub count: u32,
    pub axis: Vec3,
    pub half_angle: f32, // >= PI/2: always drawn
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Closed {
    Cubes,
    Hex,
}

const TOP_COS: f32 = 0.82; // within ~35 degrees of up
const WALL_TILT: f32 = 0.5; // |n . up| below this: a wall
const MARGIN: f32 = 0.003; // radians added to every cone

// true when every triangle with a normal within `half_angle` of `axis`, anywhere in the sphere, faces
// away from `cam` (conservative: false when in doubt)
pub fn group_backfacing(center: Vec3, radius: f32, cam: Vec3, axis: Vec3, half_angle: f32) -> bool {
    let d = center - cam;
    let l = d.length();
    if l <= radius * 1.001 + 1e-3 {
        return false; // the camera is inside the sphere
    }
    let a = half_angle + (radius / l).asin();
    if a >= std::f32::consts::FRAC_PI_2 - 0.01 {
        return false;
    }
    d.dot(axis) / l > a.sin() + 1e-4
}

// reorders `inds` (triangles) into direction groups; returns the new index list and the groups
// (empty groups dropped, the always-drawn rest last). Triangle order inside a group is kept.
pub fn group_indices(verts: &[Vertex], inds: &[u32], kind: Closed) -> (Vec<u32>, Vec<IndexGroup>) {
    if inds.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let p = |i: u32| Vec3::from_array(verts[i as usize].pos);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for v in verts {
        lo = lo.min(Vec3::from_array(v.pos));
        hi = hi.max(Vec3::from_array(v.pos));
    }
    let up = ((lo + hi) * 0.5).normalize_or_zero();
    if up == Vec3::ZERO {
        return (inds.to_vec(), Vec::new());
    }
    let e1 = up
        .cross(if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y })
        .normalize();
    let e2 = up.cross(e1);
    let k = if kind == Closed::Hex { 6usize } else { 4 };
    let step = std::f32::consts::TAU / k as f32;

    // each triangle's outward plane normal (None: degenerate or ambiguous)
    let normals: Vec<Option<Vec3>> = inds
        .chunks_exact(3)
        .map(|t| {
            let g = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
            let len = g.length();
            if len < 1e-9 {
                return None;
            }
            let vn: Vec3 = t
                .iter()
                .map(|&i| Vec3::from_array(verts[i as usize].normal))
                .sum();
            let s = g.dot(vn);
            if s.abs() < 1e-6 * len {
                return None;
            }
            Some(if s < 0.0 { -g } else { g } / len)
        })
        .collect();

    // the walls' common azimuth offset (circular mean of k * azimuth)
    let azimuth = |n: Vec3| n.dot(e2).atan2(n.dot(e1));
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for n in normals.iter().flatten() {
        if n.dot(up).abs() < WALL_TILT {
            let a = azimuth(*n) * k as f32;
            sx += a.cos();
            sy += a.sin();
        }
    }
    let phi0 = sy.atan2(sx) / k as f32;

    // group index: 0 top, 1 bottom, 2.. walls, k + 2 other
    let other = k + 2;
    let group_of = |n: Option<Vec3>| -> usize {
        let Some(n) = n else { return other };
        let tu = n.dot(up);
        if tu > TOP_COS {
            0
        } else if tu < -TOP_COS {
            1
        } else if tu.abs() < WALL_TILT {
            let rel = azimuth(n) - phi0;
            let m = (rel / step).round();
            if (rel - m * step).abs() < 0.4 * step {
                2 + (m as i64).rem_euclid(k as i64) as usize
            } else {
                other
            }
        } else {
            other
        }
    };
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); other + 1];
    for (t, n) in normals.iter().enumerate() {
        members[group_of(*n)].push(t);
    }

    let mut out = Vec::with_capacity(inds.len());
    let mut groups = Vec::new();
    for (g, tris) in members.iter().enumerate() {
        if tris.is_empty() {
            continue;
        }
        let first = out.len() as u32;
        for &t in tris {
            out.extend_from_slice(&inds[t * 3..t * 3 + 3]);
        }
        let count = out.len() as u32 - first;
        if g == other {
            groups.push(IndexGroup {
                first,
                count,
                axis: Vec3::Y,
                half_angle: std::f32::consts::PI,
            });
            continue;
        }
        let axis = tris
            .iter()
            .filter_map(|&t| normals[t])
            .sum::<Vec3>()
            .normalize_or_zero();
        let half = tris
            .iter()
            .filter_map(|&t| normals[t])
            .map(|n| n.dot(axis).clamp(-1.0, 1.0).acos())
            .fold(0.0f32, f32::max)
            + MARGIN;
        groups.push(IndexGroup {
            first,
            count,
            axis,
            half_angle: half,
        });
    }
    (out, groups)
}

// the index ranges to draw: groups not facing away, adjacent ranges merged
pub fn visible_ranges(
    groups: &[IndexGroup],
    center: Vec3,
    radius: f32,
    cam: Vec3,
    mut emit: impl FnMut(u32, u32),
) {
    let mut cur: Option<(u32, u32)> = None;
    for g in groups {
        if group_backfacing(center, radius, cam, g.axis, g.half_angle) {
            continue;
        }
        match cur {
            Some((f, c)) if f + c == g.first => cur = Some((f, c + g.count)),
            _ => {
                if let Some((f, c)) = cur {
                    emit(f, f + c);
                }
                cur = Some((g.first, g.count));
            }
        }
    }
    if let Some((f, c)) = cur {
        emit(f, f + c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_side_is_never_culled() {
        let c = Vec3::new(0.0, 100.0, 0.0);
        // looking down at an upward face from above
        assert!(!group_backfacing(
            c,
            10.0,
            Vec3::new(0.0, 300.0, 0.0),
            Vec3::Y,
            0.1
        ));
        // from the side of a tilted cone
        assert!(!group_backfacing(
            c,
            10.0,
            Vec3::new(50.0, 101.0, 0.0),
            Vec3::Y,
            0.1
        ));
    }

    #[test]
    fn far_behind_is_culled() {
        let c = Vec3::new(0.0, 100.0, 0.0);
        assert!(group_backfacing(
            c,
            10.0,
            Vec3::new(0.0, -300.0, 0.0),
            Vec3::Y,
            0.1
        ));
    }

    #[test]
    fn inside_the_sphere_never_culls() {
        let c = Vec3::new(0.0, 100.0, 0.0);
        for cam in [
            c,
            c + Vec3::new(0.0, -5.0, 0.0),
            c + Vec3::new(9.0, 0.0, 0.0),
        ] {
            assert!(!group_backfacing(c, 10.0, cam, Vec3::Y, 0.0));
        }
    }

    #[test]
    fn in_the_plane_is_not_culled() {
        let c = Vec3::ZERO;
        assert!(!group_backfacing(
            c,
            1.0,
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::Y,
            0.0
        ));
        assert!(!group_backfacing(
            c,
            1.0,
            Vec3::new(100.0, -0.5, 0.0),
            Vec3::Y,
            0.0
        ));
    }

    #[test]
    fn matches_brute_force() {
        // random cones, spheres and cameras: whenever it culls, no sampled point/normal faces the camera
        let mut s = 12345u32;
        let mut r = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..2000 {
            let c = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5) * 100.0;
            let rad = 1.0 + r() * 20.0;
            let cam = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5) * 300.0;
            let axis = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
            let half = r() * 1.0;
            if !group_backfacing(c, rad, cam, axis, half) {
                continue;
            }
            for _ in 0..50 {
                let p = c + Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize() * rad * r();
                let t = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
                let perp = (t - axis * t.dot(axis)).normalize_or_zero();
                let ang = half * r();
                let n = axis * ang.cos() + perp * ang.sin();
                assert!(n.dot(p - cam) > 0.0, "culled a visible face");
            }
        }
    }
}
