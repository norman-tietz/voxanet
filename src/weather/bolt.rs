// A lightning bolt's jagged polyline (spec docs/superpowers/specs/2026-10-12-lightning-design.md): midpoint
// displacement from the cloud down to the ground, plus a few thinner branches, all from the strike's seed,
// so the shape is the same on every visit. lightning.wgsl draws the segments as camera-facing strips.

use glam::Vec3;

use super::lightning::{pcg, unit};

pub const LEVELS: u32 = 4; // 2^4 = 16 segments in the main channel
const BRANCHES: usize = 2;
const BRANCH_SEGMENTS: usize = 4;
pub const MAX_SEGMENTS_PER_BOLT: usize = (1 << LEVELS) + BRANCHES * BRANCH_SEGMENTS;
const MAIN_WIDTH: f32 = 1.2; // world units
const OFFSET_SHARE: f32 = 0.18; // the first displacement, of the bolt's length (halves per level)

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: Vec3,
    pub b: Vec3,
    pub width: f32, // world units
    pub glow: f32,  // 1 on the main channel, less on branches
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = pcg(self.0);
        unit(self.0)
    }
    fn signed(&mut self) -> f32 {
        self.next() * 2.0 - 1.0
    }
}

fn perpendicular(axis: Vec3) -> (Vec3, Vec3) {
    let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = axis.cross(helper).normalize_or_zero();
    (u, axis.cross(u).normalize_or_zero())
}

pub fn build(seed: u32, top: Vec3, bottom: Vec3) -> Vec<Segment> {
    let mut rng = Rng(pcg(seed ^ 0xb01_7ed));
    let axis = (bottom - top).normalize_or_zero();
    let len = (bottom - top).length();
    let (u, v) = perpendicular(axis);
    // midpoint displacement of the main channel
    let mut pts = vec![top, bottom];
    let mut amp = len * OFFSET_SHARE;
    for _ in 0..LEVELS {
        let mut next = Vec::with_capacity(pts.len() * 2 - 1);
        for w in pts.windows(2) {
            next.push(w[0]);
            let mid = (w[0] + w[1]) * 0.5 + (u * rng.signed() + v * rng.signed()) * amp;
            next.push(mid);
        }
        next.push(*pts.last().unwrap());
        pts = next;
        amp *= 0.5;
    }
    let mut segs: Vec<Segment> = pts
        .windows(2)
        .map(|w| Segment {
            a: w[0],
            b: w[1],
            width: MAIN_WIDTH,
            glow: 1.0,
        })
        .collect();
    // branches: from interior vertices of the upper two thirds, heading down and out
    for _ in 0..BRANCHES {
        let i = 1 + (rng.next() * (pts.len() as f32 * 0.6)) as usize;
        let mut p = pts[i.min(pts.len() - 2)];
        let out = (u * rng.signed() + v * rng.signed()).normalize_or_zero();
        let step = len * 0.05;
        for _ in 0..BRANCH_SEGMENTS {
            let dir = (axis * 0.7 + out * 0.5 + (u * rng.signed() + v * rng.signed()) * 0.25)
                .normalize_or_zero();
            let q = p + dir * step;
            segs.push(Segment {
                a: p,
                b: q,
                width: MAIN_WIDTH * 0.45,
                glow: 0.5,
            });
            p = q;
        }
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    const TOP: Vec3 = Vec3::new(0.0, 400.0, 0.0);
    const BOTTOM: Vec3 = Vec3::new(30.0, 250.0, -20.0);

    #[test]
    fn the_bolt_is_deterministic() {
        assert_eq!(build(7, TOP, BOTTOM), build(7, TOP, BOTTOM));
        assert_ne!(build(7, TOP, BOTTOM), build(8, TOP, BOTTOM));
    }

    #[test]
    fn the_main_channel_runs_from_the_cloud_to_the_ground() {
        let segs = build(3, TOP, BOTTOM);
        assert!(segs.len() <= MAX_SEGMENTS_PER_BOLT);
        assert_eq!(segs[0].a, TOP);
        // the main channel is the first 2^LEVELS segments, chained end to end
        let main = &segs[..1 << LEVELS];
        for w in main.windows(2) {
            assert_eq!(w[0].b, w[1].a);
        }
        assert_eq!(main.last().unwrap().b, BOTTOM);
    }

    #[test]
    fn the_bolt_is_jagged_but_stays_near_the_line() {
        let segs = build(5, TOP, BOTTOM);
        let axis = (BOTTOM - TOP).normalize();
        let len = (BOTTOM - TOP).length();
        let mut max_off = 0.0_f32;
        for s in &segs[..1 << LEVELS] {
            let off = (s.a - TOP) - axis * (s.a - TOP).dot(axis);
            max_off = max_off.max(off.length());
        }
        assert!(max_off > 0.01 * len, "jagged: {max_off}");
        assert!(max_off < 0.3 * len, "stays near the line: {max_off}");
    }

    #[test]
    fn branches_are_thinner_and_start_on_the_main_channel() {
        let segs = build(11, TOP, BOTTOM);
        let main = 1 << LEVELS;
        assert!(segs.len() > main, "has branches");
        let main_w = segs[0].width;
        for b in &segs[main..] {
            assert!(b.width < main_w);
        }
        let first_branch = &segs[main];
        assert!(segs[..main]
            .iter()
            .any(|m| m.a == first_branch.a || m.b == first_branch.a));
    }

    #[test]
    fn degenerate_bolts_stay_finite() {
        for (top, bottom) in [
            (TOP, TOP),
            (TOP, TOP + Vec3::Y * 1e-4),
            (Vec3::ZERO, Vec3::Y),
        ] {
            for s in build(1, top, bottom) {
                assert!(s.a.is_finite() && s.b.is_finite() && s.width.is_finite());
            }
        }
    }
}
