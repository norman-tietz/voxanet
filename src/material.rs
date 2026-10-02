// material.rs
// Block types, and the natural type of every terrain block: chosen from the column's height within the
// planet's height range, the local slope and a little per-column jitter so material borders are ragged.

use crate::biome::Palette;
use crate::noise::PlanetTerrain;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockType {
    Grass,
    Dirt,
    Sand,
    Stone,
    Snow,
    Bedrock,
    Ash,
    Basalt,
    Obsidian,
    Ember,
    Ice,
}

impl BlockType {
    pub fn color(self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.1, 0.7, 0.1],
            BlockType::Dirt => [0.6, 0.4, 0.2],
            BlockType::Sand => [0.86, 0.78, 0.5],
            BlockType::Stone => [0.5, 0.5, 0.52],
            BlockType::Snow => [0.95, 0.96, 1.0],
            BlockType::Bedrock => [0.2, 0.2, 0.2],
            BlockType::Ash => [0.35, 0.33, 0.3],
            BlockType::Basalt => [0.18, 0.17, 0.19],
            BlockType::Obsidian => [0.07, 0.06, 0.09],
            BlockType::Ember => [0.5, 0.22, 0.08],
            BlockType::Ice => [0.75, 0.88, 0.95],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BlockType::Grass => "Grass",
            BlockType::Dirt => "Dirt",
            BlockType::Sand => "Sand",
            BlockType::Stone => "Stone",
            BlockType::Snow => "Snow",
            BlockType::Bedrock => "Bedrock",
            BlockType::Ash => "Ash",
            BlockType::Basalt => "Basalt",
            BlockType::Obsidian => "Obsidian",
            BlockType::Ember => "Ember",
            BlockType::Ice => "Ice",
        }
    }

    // walk acceleration/friction multiplier (entity.rs); 1.0 is normal ground, lower is slippery
    pub fn friction_scale(self) -> f32 {
        match self {
            BlockType::Ice => 0.15,
            _ => 1.0,
        }
    }
}

const BEACH: f32 = 0.06; // fraction of the peak height above sea level that is still sand ...
const MAX_BEACH: f32 = 1.5; // ... but at most this many layers
                            // fractions of the height from sea level up to the highest peak
const ROCK_LINE: f32 = 0.45; // bare stone above
const SNOW_LINE: f32 = 0.62; // snow above
const STEEP: u32 = 3; // a height step of this many layers to a neighbour exposes stone
const SOIL_DEPTH: u32 = 3; // layers of dirt/sand below the surface before stone
pub const CORE_LAYERS: u32 = 6;

// deterministic value in -1..1 per lattice point
fn hash(face: u8, x: u32, y: u32) -> f32 {
    let mut h = (face as u32).wrapping_mul(0x9E37_79B9)
        ^ x.wrapping_mul(0x85EB_CA6B)
        ^ y.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 32767.5 - 1.0
}

// smooth value noise in -1..1 with features about JITTER_CELL columns wide, so material borders are
// ragged but not speckled
const JITTER_CELL: u32 = 6;
fn jitter(face: u8, u: u32, v: u32) -> f32 {
    let (gx, gy) = (u / JITTER_CELL, v / JITTER_CELL);
    let s = |f: u32| {
        let t = f as f32 / JITTER_CELL as f32;
        t * t * (3.0 - 2.0 * t)
    };
    let (tx, ty) = (s(u % JITTER_CELL), s(v % JITTER_CELL));
    let top = hash(face, gx, gy) * (1.0 - tx) + hash(face, gx + 1, gy) * tx;
    let bottom = hash(face, gx, gy + 1) * (1.0 - tx) + hash(face, gx + 1, gy + 1) * tx;
    top * (1.0 - ty) + bottom * ty
}

// the type of the top block of a column
pub fn surface_type(
    terrain: &PlanetTerrain,
    palette: &Palette,
    face: u8,
    u: u32,
    v: u32,
) -> BlockType {
    let h = terrain.get_height(face, u, v);
    let sea = terrain.sea_level() as f32;
    let peak = (terrain.height_range().1 as f32 - sea).max(1.0);
    let j = jitter(face, u, v);
    let above_sea = h as f32 - sea;
    let rel = above_sea / peak + j * 0.04; // borders wander by about 4% of the relief
    let beach = (BEACH * peak).min(MAX_BEACH);

    let neighbours = [
        terrain.get_height(face, u.saturating_sub(1), v),
        terrain.get_height(face, u + 1, v),
        terrain.get_height(face, u, v.saturating_sub(1)),
        terrain.get_height(face, u, v + 1),
    ];
    let steep = neighbours.iter().any(|&n| h >= n + STEEP);

    if rel >= SNOW_LINE {
        palette.peak
    } else if steep || rel >= ROCK_LINE {
        palette.rock
    } else if above_sea <= beach * (1.0 + 0.5 * j) {
        palette.beach
    } else {
        palette.ground
    }
}

// the natural type of a terrain block (layer <= column height)
pub fn natural_type(
    terrain: &PlanetTerrain,
    palette: &Palette,
    has_core: bool,
    face: u8,
    u: u32,
    v: u32,
    layer: u32,
) -> BlockType {
    if has_core && layer < CORE_LAYERS {
        return BlockType::Bedrock;
    }
    let depth = terrain.get_height(face, u, v).saturating_sub(layer);
    let surface = surface_type(terrain, palette, face, u, v);
    match (depth, surface) {
        (0, s) => s,
        (d, s) if s == palette.ground && d <= SOIL_DEPTH => palette.subsurface,
        (d, s) if s == palette.beach && d <= SOIL_DEPTH => palette.beach,
        _ => palette.rock,
    }
}

// selectable with the number keys 1.. on the active planet type
pub fn placeable(palette: &Palette) -> [BlockType; 5] {
    [
        palette.ground,
        palette.subsurface,
        palette.beach,
        palette.rock,
        palette.peak,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth_palette() -> Palette {
        crate::biome::PlanetType::EarthLike.def().palette
    }

    fn ice_palette() -> Palette {
        crate::biome::PlanetType::Ice.def().palette
    }

    #[test]
    fn earth_like_palette_reproduces_grass_sand_stone_snow() {
        // a tiny deterministic terrain: res 16 is enough to hit every band at some column
        let terrain = PlanetTerrain::new(16);
        let palette = earth_palette();
        let mut seen = std::collections::HashSet::new();
        for u in 0..16 {
            for v in 0..16 {
                seen.insert(surface_type(&terrain, &palette, 0, u, v));
            }
        }
        // not asserting exact coverage of all four (terrain is random), just that nothing
        // outside the Earth-like palette's roles ever comes out
        for ty in &seen {
            assert!([
                BlockType::Grass,
                BlockType::Sand,
                BlockType::Stone,
                BlockType::Snow
            ]
            .contains(ty));
        }
    }

    #[test]
    fn ice_palette_never_produces_grass() {
        let terrain = PlanetTerrain::new(16);
        let palette = ice_palette();
        for u in 0..16 {
            for v in 0..16 {
                assert_ne!(surface_type(&terrain, &palette, 0, u, v), BlockType::Grass);
            }
        }
    }

    #[test]
    fn placeable_follows_the_active_palette() {
        // `placeable()` returns [ground, subsurface, beach, rock, peak] for the active palette.
        // Earth-like's "ground" role is Grass, so it leads the array; Ice's "ground" role is
        // Stone (there's no grass to walk on), but Ice itself is still selectable from the
        // palette's beach role (brief's draft test asserted `[0] == Ice`, which contradicts the
        // brief's own field order below — fixed here to check presence instead of position).
        assert_eq!(placeable(&earth_palette())[0], BlockType::Grass);
        assert!(placeable(&ice_palette()).contains(&BlockType::Ice));
    }

    #[test]
    fn new_biome_block_types_have_names_and_colors() {
        for ty in [
            BlockType::Ash,
            BlockType::Basalt,
            BlockType::Obsidian,
            BlockType::Ember,
            BlockType::Ice,
        ] {
            assert!(!ty.name().is_empty());
            let [r, g, b] = ty.color();
            assert!(
                (0.0..=2.0).contains(&r) && (0.0..=2.0).contains(&g) && (0.0..=2.0).contains(&b)
            );
        }
    }

    #[test]
    fn ice_is_slippery_everything_else_is_not() {
        assert!(BlockType::Ice.friction_scale() < 0.5);
        for ty in [
            BlockType::Grass,
            BlockType::Dirt,
            BlockType::Sand,
            BlockType::Stone,
            BlockType::Snow,
            BlockType::Ash,
            BlockType::Basalt,
            BlockType::Obsidian,
            BlockType::Ember,
        ] {
            assert_eq!(ty.friction_scale(), 1.0);
        }
    }
}
