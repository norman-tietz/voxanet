// lowpoly.rs
// Low-poly terrain: marching cubes over a smoothed height field (PlanetData::smooth), flat colour per
// facet, normals from screen-space derivatives. Blocks stay the game's data; this only draws them.

use crate::gen::CoordSystem;
use std::sync::atomic::{AtomicBool, Ordering};

// how terrain is drawn: low-poly (the game's look) or the original cubes (development comparison only,
// console /terrain_style). Read once per mesh build at the entry points (MeshGen::build_chunk,
// generate_lod_mesh, generate_lod_morph) and passed down, so tests can pick a style explicitly
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainStyle {
    Cubes,
    LowPoly,
}

static CUBES: AtomicBool = AtomicBool::new(false);

pub fn style() -> TerrainStyle {
    if CUBES.load(Ordering::Relaxed) {
        TerrainStyle::Cubes
    } else {
        TerrainStyle::LowPoly
    }
}

pub fn set_style(s: TerrainStyle) {
    CUBES.store(s == TerrainStyle::Cubes, Ordering::Relaxed);
}

// a cell is solid above this density
pub const ISO: f32 = 0.5;

// the density of a natural (unmined) cell at `layer` in a column of smooth height `smooth_height`:
// 1 up to the smooth height's layer, 0 two layers above, linear between — so flat ground (an integer
// smooth height) crosses ISO exactly at the block tops
pub fn natural_density(smooth_height: f32, layer: u32) -> f32 {
    (smooth_height - layer as f32 + 1.0).clamp(0.0, 1.0)
}

// where the surface crosses a column of smooth height `smooth_height`: the solid cell's layer and the
// fraction (0..1) of the way to the centre of the empty cell above it
pub fn crossing(smooth_height: f32) -> (u32, f32) {
    let h = smooth_height.max(0.0);
    let base = h.floor();
    let frac = h - base;
    if frac <= ISO {
        // between `base` (density 1) and base + 1 (density frac, not solid: solid is > ISO)
        (base as u32, (1.0 - ISO) / (1.0 - frac))
    } else {
        // between base + 1 (density frac) and base + 2 (density 0)
        (base as u32 + 1, (frac - ISO) / frac)
    }
}

// the radius of the low-poly surface over a column of smooth height `smooth_height`: the marching-cubes
// vertex on the column's vertical edge (interpolated between the two block centres), also what LOD
// meshes use so they meet the voxel surface
pub fn surface_radius(smooth_height: f32, res: u32) -> f32 {
    let (layer, t) = crossing(smooth_height);
    let below = CoordSystem::get_layer_radius_f(layer as f32 + 0.5, res);
    let above = CoordSystem::get_layer_radius_f(layer as f32 + 1.5, res);
    below + (above - below) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    // flat smooth ground lands exactly on the block tops: half-way between the centres of the top block
    // and the cell above
    #[test]
    fn integer_smooth_height_crosses_at_the_block_top() {
        assert_eq!(crossing(17.0), (17, 0.5));
        let r = surface_radius(17.0, 32);
        let mid = (CoordSystem::get_layer_radius_f(17.5, 32)
            + CoordSystem::get_layer_radius_f(18.5, 32))
            / 2.0;
        assert!((r - mid).abs() < 1e-5);
        assert!(
            (r - CoordSystem::get_layer_radius(18, 32)).abs() < 0.02,
            "not at the block top"
        );
    }

    // the surface rises steadily with the smooth height, also across the half-layer switch
    #[test]
    fn surface_radius_is_continuous_and_rising() {
        let mut last = surface_radius(10.0, 32);
        for i in 1..=200 {
            let r = surface_radius(10.0 + i as f32 * 0.01, 32);
            assert!(r > last, "not rising at {}", 10.0 + i as f32 * 0.01);
            assert!(r - last < 0.05, "jump at {}", 10.0 + i as f32 * 0.01);
            last = r;
        }
    }

    // density: solid at and below the smooth height's layer, empty two layers above
    #[test]
    fn natural_density_brackets_the_smooth_height() {
        assert_eq!(natural_density(17.3, 17), 1.0);
        assert!((natural_density(17.3, 18) - 0.3).abs() < 1e-6);
        assert_eq!(natural_density(17.3, 19), 0.0);
        assert_eq!(natural_density(17.3, 3), 1.0);
    }

    // the crossing agrees with the density: interpolating the two cells' densities gives ISO
    #[test]
    fn crossing_matches_the_density() {
        for h in [5.0f32, 5.2, 5.49, 5.5, 5.51, 5.9] {
            let (l, t) = crossing(h);
            let (a, b) = (natural_density(h, l), natural_density(h, l + 1));
            assert!(a > ISO && b <= ISO, "{h}: {a} {b}");
            assert!((a + (b - a) * t - ISO).abs() < 1e-5, "{h}");
        }
    }
}
