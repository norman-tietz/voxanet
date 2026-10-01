// rt_shadow.rs
// CPU side of the ray-marched sun shadows: per cube face, a window of solid/air bits near the player.
// The fragment shader walks these cells toward the sun (see rt_shadow() in shader.wgsl).
//
// Planets up to WINDOW_SIZE columns per face are stored completely (all six faces). On larger planets
// each face turned toward the player gets a window placed as close to the player as the face allows,
// so near a face edge the neighbouring face's window covers the strip along that edge.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::common::{BlockId, PlanetData};

pub const WINDOW_SIZE: u32 = 256;         // columns per side
pub const MAX_WORDS_PER_COLUMN: u32 = 16; // 32 layers per word -> at most 512 layers
const PIT_MARGIN: u32 = 32;               // layers kept below the lowest column so mined pits still count
pub const TILE: u32 = 16;                 // columns per side of a max-height tile (lets rays skip open air)

// must match FaceWindow in shader.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FaceWindow {
    pub origin_u: i32,
    pub origin_v: i32,
    pub size: u32, // 0 = no data for this face
    pub base_layer: i32,
    pub words_per_column: u32,
    pub offset: u32, // first word of this face in the bits buffer
    pub max_layer: i32,
    pub tile_offset: u32, // first word of this face's tile maxima: highest solid layer per TILE x TILE columns
}

// must match RtParams in shader.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RtParams {
    pub faces: [FaceWindow; 6],
    pub resolution: u32,
    pub enabled: u32,
    pub max_layer: i32, // highest solid layer over all windows
    pub _pad: u32,
}

pub struct ShadowWindow {
    pub params: RtParams,
    pub bits: Vec<u32>, // per face: (v * size + u) * words_per_column + layer / 32
}

impl ShadowWindow {
    pub fn max_words() -> u64 {
        let tiles = WINDOW_SIZE.div_ceil(TILE);
        6 * (WINDOW_SIZE * WINDOW_SIZE * MAX_WORDS_PER_COLUMN + tiles * tiles) as u64
    }

    pub fn build(planet: &PlanetData, center: BlockId, player_dir: Vec3) -> Self {
        let res = planet.resolution;
        let size = res.min(WINDOW_SIZE);
        let mut params = RtParams { resolution: res, enabled: 1, max_layer: 0, ..Zeroable::zeroed() };
        let mut bits = Vec::new();

        for face in 0..6u8 {
            let origin = if res <= WINDOW_SIZE {
                Some((0, 0))
            } else if face == center.face {
                Some((center.u as i32, center.v as i32))
            } else {
                face_coords(face, player_dir).map(|(x, y)| (((x + 1.0) * 0.5 * res as f32) as i32, ((y + 1.0) * 0.5 * res as f32) as i32))
            }
            .map(|(cu, cv)| {
                let max_origin = (res - size) as i32;
                ((cu - size as i32 / 2).clamp(0, max_origin), (cv - size as i32 / 2).clamp(0, max_origin))
            });

            let Some((origin_u, origin_v)) = origin else { continue };
            let window = fill_face(planet, face, origin_u, origin_v, size, bits.len() as u32, &mut bits);
            params.max_layer = params.max_layer.max(window.max_layer);
            params.faces[face as usize] = window;
        }
        Self { params, bits }
    }
}

// cube face coordinates (-1..1) of a direction, for the faces it points toward. A straight projection
// onto the cube rather than the exact inverse of cube_to_sphere, which is close enough to place a window.
fn face_coords(face: u8, d: Vec3) -> Option<(f32, f32)> {
    let (axis, x, y) = match face {
        0 => (d.y, d.x, d.z),
        1 => (-d.y, d.x, d.z),
        2 => (d.x, d.y, d.z),
        3 => (-d.x, d.y, d.z),
        4 => (d.z, d.x, d.y),
        _ => (-d.z, d.x, d.y),
    };
    if axis <= 0.0 { return None; }
    Some(((x / axis).clamp(-1.0, 1.0), (y / axis).clamp(-1.0, 1.0)))
}

fn fill_face(planet: &PlanetData, face: u8, origin_u: i32, origin_v: i32, size: u32, offset: u32, bits: &mut Vec<u32>) -> FaceWindow {
    let size_i = size as i32;
    let column = |lu: i32, lv: i32| planet.terrain.get_height(face, (origin_u + lu) as u32, (origin_v + lv) as u32);

    // layer range the bits have to cover
    let mut min_h = u32::MAX;
    let mut max_h = 0;
    for lv in 0..size_i {
        for lu in 0..size_i {
            let h = column(lu, lv);
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    // placed blocks can stick out above the terrain
    for mods in planet.chunks.values() {
        for id in mods.placed.keys() {
            if id.face == face { max_h = max_h.max(id.layer); }
        }
    }

    let base = min_h.saturating_sub(PIT_MARGIN);
    let words = ((max_h - base + 1 + 31) / 32).min(MAX_WORDS_PER_COLUMN);
    let top = base + words * 32; // first layer not stored (treated as air)
    let start = bits.len();
    bits.resize(start + (size * size * words) as usize, 0);
    let face_bits = &mut bits[start..];

    // natural terrain: layers base..=h are solid
    for lv in 0..size_i {
        for lu in 0..size_i {
            let solid = (column(lu, lv) + 1).min(top) - base;
            let col = (lv * size_i + lu) as usize * words as usize;
            for w in 0..words {
                let n = solid.saturating_sub(w * 32).min(32);
                face_bits[col + w as usize] = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
            }
        }
    }

    // player edits
    let mut set = |id: &BlockId, solid: bool| {
        let (lu, lv) = (id.u as i32 - origin_u, id.v as i32 - origin_v);
        if id.face != face || lu < 0 || lv < 0 || lu >= size_i || lv >= size_i { return; }
        if id.layer < base || id.layer >= top { return; }
        let l = id.layer - base;
        let idx = (lv * size_i + lu) as usize * words as usize + (l / 32) as usize;
        if solid { face_bits[idx] |= 1 << (l % 32); } else { face_bits[idx] &= !(1 << (l % 32)); }
    };
    for mods in planet.chunks.values() {
        for id in mods.placed.keys() { set(id, true); }
        for id in &mods.mined { set(id, false); }
    }

    // tile maxima; mined blocks are ignored, which only makes the maximum conservative
    let tiles = size.div_ceil(TILE);
    let mut tile_max = vec![0u32; (tiles * tiles) as usize];
    for lv in 0..size_i {
        for lu in 0..size_i {
            let t = &mut tile_max[(lv as u32 / TILE * tiles + lu as u32 / TILE) as usize];
            *t = (*t).max(column(lu, lv));
        }
    }
    for mods in planet.chunks.values() {
        for id in mods.placed.keys() {
            let (lu, lv) = (id.u as i32 - origin_u, id.v as i32 - origin_v);
            if id.face != face || lu < 0 || lv < 0 || lu >= size_i || lv >= size_i { continue; }
            let t = &mut tile_max[(lv as u32 / TILE * tiles + lu as u32 / TILE) as usize];
            *t = (*t).max(id.layer);
        }
    }
    let tile_offset = bits.len() as u32;
    bits.extend_from_slice(&tile_max);

    FaceWindow {
        origin_u,
        origin_v,
        size,
        base_layer: base as i32,
        words_per_column: words,
        offset,
        max_layer: max_h.min(top - 1) as i32,
        tile_offset,
    }
}
