// material.rs
// Block types, and the natural type of every terrain block: chosen from the column's height within the
// planet's height range, the local slope and a little per-column jitter so material borders are ragged.

use crate::noise::PlanetTerrain;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockType {
    Grass,
    Dirt,
    Sand,
    Stone,
    Snow,
    Bedrock,
}

impl BlockType {
    // selectable with the number keys 1..
    pub const PLACEABLE: [BlockType; 5] = [BlockType::Grass, BlockType::Dirt, BlockType::Sand, BlockType::Stone, BlockType::Snow];

    pub fn color(self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.1, 0.7, 0.1],
            BlockType::Dirt => [0.6, 0.4, 0.2],
            BlockType::Sand => [0.86, 0.78, 0.5],
            BlockType::Stone => [0.5, 0.5, 0.52],
            BlockType::Snow => [0.95, 0.96, 1.0],
            BlockType::Bedrock => [0.2, 0.2, 0.2],
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
        }
    }
}

const BEACH: f32 = 0.06;     // fraction of the peak height above sea level that is still sand ...
const MAX_BEACH: f32 = 1.5;  // ... but at most this many layers
// fractions of the height from sea level up to the highest peak
const ROCK_LINE: f32 = 0.45; // bare stone above
const SNOW_LINE: f32 = 0.62; // snow above
const STEEP: u32 = 3;        // a height step of this many layers to a neighbour exposes stone
const SOIL_DEPTH: u32 = 3;   // layers of dirt/sand below the surface before stone
pub const CORE_LAYERS: u32 = 6;

// colour of the ocean in the distant LOD meshes (near water is drawn translucent by fs_water)
pub const WATER_COLOR: [f32; 3] = [0.12, 0.32, 0.55];

// deterministic value in -1..1 per lattice point
fn hash(face: u8, x: u32, y: u32) -> f32 {
    let mut h = (face as u32).wrapping_mul(0x9E37_79B9) ^ x.wrapping_mul(0x85EB_CA6B) ^ y.wrapping_mul(0xC2B2_AE35);
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
    let s = |f: u32| { let t = f as f32 / JITTER_CELL as f32; t * t * (3.0 - 2.0 * t) };
    let (tx, ty) = (s(u % JITTER_CELL), s(v % JITTER_CELL));
    let top = hash(face, gx, gy) * (1.0 - tx) + hash(face, gx + 1, gy) * tx;
    let bottom = hash(face, gx, gy + 1) * (1.0 - tx) + hash(face, gx + 1, gy + 1) * tx;
    top * (1.0 - ty) + bottom * ty
}

// the type of the top block of a column
pub fn surface_type(terrain: &PlanetTerrain, face: u8, u: u32, v: u32) -> BlockType {
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
        BlockType::Snow
    } else if steep || rel >= ROCK_LINE {
        BlockType::Stone
    } else if above_sea <= beach * (1.0 + 0.5 * j) {
        BlockType::Sand
    } else {
        BlockType::Grass
    }
}

// the natural type of a terrain block (layer <= column height)
pub fn natural_type(terrain: &PlanetTerrain, has_core: bool, face: u8, u: u32, v: u32, layer: u32) -> BlockType {
    if has_core && layer < CORE_LAYERS {
        return BlockType::Bedrock;
    }
    let depth = terrain.get_height(face, u, v).saturating_sub(layer);
    let surface = surface_type(terrain, face, u, v);
    match (depth, surface) {
        (0, s) => s,
        (d, BlockType::Grass) if d <= SOIL_DEPTH => BlockType::Dirt,
        (d, BlockType::Sand) if d <= SOIL_DEPTH => BlockType::Sand,
        _ => BlockType::Stone,
    }
}
