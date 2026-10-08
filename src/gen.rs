//gen.rs

use crate::common::*;
use crate::material::BlockType;
use glam::Vec3;
use std::collections::HashSet;

pub struct CoordSystem;

impl CoordSystem {
    // k = 0.85 balances the shape.
    const K: f64 = 0.85;

    // forward Mapping: Unit Cube -> Sphere
    fn cube_to_sphere(x: f64, y: f64, z: f64) -> Vec3 {
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;

        let sx = x * (1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0).sqrt();
        let sy = y * (1.0 - z2 * 0.5 - x2 * 0.5 + z2 * x2 / 3.0).sqrt();
        let sz = z * (1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0).sqrt();

        Vec3::new(sx as f32, sy as f32, sz as f32)
    }

    // inverse Mapping: Sphere -> Unit Cube

    fn cubize_point(pos: Vec3) -> Vec3 {
        let mut x = pos.x as f64;
        let mut y = pos.y as f64;
        let mut z = pos.z as f64;

        let fx = x.abs();
        let fy = y.abs();
        let fz = z.abs();

        const INVERSE_SQRT_2: f64 = 0.70710676908493042;

        if fy >= fx && fy >= fz {
            let a2 = x * x * 2.0;
            let b2 = z * z * 2.0;
            let inner = -a2 + b2 - 3.0;
            let inner_sqrt = -((inner * inner) - 12.0 * a2).sqrt();

            if x == 0.0 {
                x = 0.0;
            } else {
                x = (inner_sqrt + a2 - b2 + 3.0).sqrt() * INVERSE_SQRT_2;
            }

            if z == 0.0 {
                z = 0.0;
            } else {
                z = (inner_sqrt - a2 + b2 + 3.0).sqrt() * INVERSE_SQRT_2;
            }

            if x > 1.0 {
                x = 1.0;
            }
            if z > 1.0 {
                z = 1.0;
            }

            if pos.x < 0.0 {
                x = -x;
            }
            if pos.z < 0.0 {
                z = -z;
            }

            y = if pos.y > 0.0 { 1.0 } else { -1.0 };
        } else if fx >= fy && fx >= fz {
            let a2 = y * y * 2.0;
            let b2 = z * z * 2.0;
            let inner = -a2 + b2 - 3.0;
            let inner_sqrt = -((inner * inner) - 12.0 * a2).sqrt();

            if y == 0.0 {
                y = 0.0;
            } else {
                y = (inner_sqrt + a2 - b2 + 3.0).sqrt() * INVERSE_SQRT_2;
            }

            if z == 0.0 {
                z = 0.0;
            } else {
                z = (inner_sqrt - a2 + b2 + 3.0).sqrt() * INVERSE_SQRT_2;
            }

            if y > 1.0 {
                y = 1.0;
            }
            if z > 1.0 {
                z = 1.0;
            }

            if pos.y < 0.0 {
                y = -y;
            }
            if pos.z < 0.0 {
                z = -z;
            }

            x = if pos.x > 0.0 { 1.0 } else { -1.0 };
        } else {
            let a2 = x * x * 2.0;
            let b2 = y * y * 2.0;
            let inner = -a2 + b2 - 3.0;
            let inner_sqrt = -((inner * inner) - 12.0 * a2).sqrt();

            if x == 0.0 {
                x = 0.0;
            } else {
                x = (inner_sqrt + a2 - b2 + 3.0).sqrt() * INVERSE_SQRT_2;
            }

            if y == 0.0 {
                y = 0.0;
            } else {
                y = (inner_sqrt - a2 + b2 + 3.0).sqrt() * INVERSE_SQRT_2;
            }

            if x > 1.0 {
                x = 1.0;
            }
            if y > 1.0 {
                y = 1.0;
            }

            if pos.x < 0.0 {
                x = -x;
            }
            if pos.y < 0.0 {
                y = -y;
            }

            z = if pos.z > 0.0 { 1.0 } else { -1.0 };
        }
        Vec3::new(x as f32, y as f32, z as f32)
    }

    pub fn get_local_coords(pos: Vec3, res: u32) -> Option<(BlockId, Vec3)> {
        let dist = pos.length() as f64;
        let s = res as f64 / 2.0;

        let min_r = s * (-Self::K).exp();
        if dist < min_r {
            return None;
        }

        let layer_f = s * (1.0 + (dist / s).ln() / Self::K);
        let layer = layer_f.floor() as i32;

        // no upper limit: small planets' terrain can rise above layer `res`
        if layer < 0 {
            return None;
        }

        // local Layer Coordinate (0.0 to 1.0)
        let f_layer = (layer_f - layer as f64) as f32;

        // map sphere point back to Unit Cube
        let cube_pos = Self::cubize_point(pos.normalize());
        let abs = cube_pos.abs();

        let (face, u_local, v_local) = if abs.y >= abs.x && abs.y >= abs.z {
            if cube_pos.y > 0.0 {
                (0, cube_pos.x, cube_pos.z)
            } else {
                (1, cube_pos.x, cube_pos.z)
            }
        } else if abs.x >= abs.y && abs.x >= abs.z {
            if cube_pos.x > 0.0 {
                (2, cube_pos.y, cube_pos.z)
            } else {
                (3, cube_pos.y, cube_pos.z)
            }
        } else {
            if cube_pos.z > 0.0 {
                (4, cube_pos.x, cube_pos.y)
            } else {
                (5, cube_pos.x, cube_pos.y)
            }
        };

        let rf = res as f64;

        // calculate raw grid coordinates
        let u_raw = (u_local as f64 * rf + rf) / 2.0;
        let v_raw = (v_local as f64 * rf + rf) / 2.0;

        let u = u_raw.floor() as i32;
        let v = v_raw.floor() as i32;

        // local UV Coordinates (0.0 to 1.0)
        let f_u = (u_raw - u as f64) as f32;
        let f_v = (v_raw - v as f64) as f32;

        let u = u.clamp(0, res as i32 - 1) as u32;
        let v = v.clamp(0, res as i32 - 1) as u32;

        Some((
            BlockId {
                face: face as u8,
                layer: layer as u32,
                u,
                v,
            },
            Vec3::new(f_u, f_v, f_layer), // x=u, y=v, z=layer
        ))
    }

    pub fn get_layer_radius(layer: u32, res: u32) -> f32 {
        let s = res as f64 / 2.0;
        let r = s * (Self::K * ((layer as f64 / s) - 1.0)).exp();
        r as f32
    }

    // get_layer_radius for a fractional layer (galaxy impostors sample the terrain noise continuously)
    pub fn get_layer_radius_f(layer: f32, res: u32) -> f32 {
        let s = res as f64 / 2.0;
        (s * (Self::K * ((layer as f64 / s) - 1.0)).exp()) as f32
    }

    // the fractional layer at radius `r`: get_layer_radius_f's inverse
    pub fn layer_of_radius(r: f32, res: u32) -> f32 {
        let s = res as f64 / 2.0;
        (s * ((r as f64 / s).ln() / Self::K + 1.0)) as f32
    }

    // the up to eight columns around (face, u, v): the four edge neighbours and the diagonals, which
    // across a cube-face edge are stepped from the in-face neighbour (neighbor_column's own diagonals
    // are unreliable where only one axis crosses an edge); seven at a three-face cube corner
    pub fn neighbour_columns(face: u8, u: u32, v: u32, res: u32) -> Vec<(u8, u32, u32)> {
        let column = |f, cu, cv, du, dv| Self::neighbor_column(f, cu, cv, du, dv, res);
        let mut out = Vec::with_capacity(8);
        for (su, sv) in [(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)] {
            let (c1, c3) = (column(face, u, v, su, 0), column(face, u, v, 0, sv));
            out.extend(c1.into_iter().chain(c3));
            let (Some(c1), Some(c3)) = (c1, c3) else {
                continue;
            };
            let diagonal = match (c1.0 == face, c3.0 == face) {
                (true, true) => column(face, u, v, su, sv),
                (false, true) => column(c3.0, c3.1, c3.2, su, 0),
                (true, false) => column(c1.0, c1.1, c1.2, 0, sv),
                (false, false) => None,
            };
            out.extend(diagonal);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    // the column next to (face, u, v) in direction (du, dv) as (face, u, v); across a cube-face edge that
    // is a column of the neighbouring face, found by continuing the line from the inner neighbour outward
    pub fn neighbor_column(
        face: u8,
        u: u32,
        v: u32,
        du: i32,
        dv: i32,
        res: u32,
    ) -> Option<(u8, u32, u32)> {
        let (nu, nv) = (u as i32 + du, v as i32 + dv);
        if nu >= 0 && nv >= 0 && nu < res as i32 && nv < res as i32 {
            return Some((face, nu as u32, nv as u32));
        }
        let mid = res / 2;
        let here = Self::get_block_center(face, u, v, mid, res);
        let inner = Self::get_block_center(
            face,
            (u as i32 - du) as u32,
            (v as i32 - dv) as u32,
            mid,
            res,
        );
        Self::pos_to_id(here * 2.0 - inner, res).map(|id| (id.face, id.u, id.v))
    }

    pub fn get_direction(face: u8, u: u32, v: u32, res: u32) -> Vec3 {
        let rf = res as f64;

        let x_local = if u == 0 {
            -1.0
        } else if u == res {
            1.0
        } else {
            (u as f64 * 2.0 - rf) / rf
        };

        let y_local = if v == 0 {
            -1.0
        } else if v == res {
            1.0
        } else {
            (v as f64 * 2.0 - rf) / rf
        };

        let (cx, cy, cz) = match face {
            0 => (x_local, 1.0, y_local),
            1 => (x_local, -1.0, y_local),
            2 => (1.0, x_local, y_local),
            3 => (-1.0, x_local, y_local),
            4 => (x_local, y_local, 1.0),
            _ => (x_local, y_local, -1.0),
        };

        Self::cube_to_sphere(cx, cy, cz).normalize()
    }

    pub fn get_vertex_pos(face: u8, u: u32, v: u32, layer: u32, res: u32) -> Vec3 {
        let dir = Self::get_direction(face, u, v, res);
        let radius = Self::get_layer_radius(layer, res);
        dir * radius
    }

    pub fn get_block_center(face: u8, u: u32, v: u32, layer: u32, res: u32) -> Vec3 {
        let rf = res as f64;
        // center is at index + 0.5
        let uf = u as f64 + 0.5;
        let vf = v as f64 + 0.5;

        let x_local = (uf * 2.0 - rf) / rf;
        let y_local = (vf * 2.0 - rf) / rf;

        let (cx, cy, cz) = match face {
            0 => (x_local, 1.0, y_local),
            1 => (x_local, -1.0, y_local),
            2 => (1.0, x_local, y_local),
            3 => (-1.0, x_local, y_local),
            4 => (x_local, y_local, 1.0),
            _ => (x_local, y_local, -1.0),
        };

        let dir = Self::cube_to_sphere(cx, cy, cz).normalize();

        let s = rf / 2.0;
        let radius = s * (Self::K * (((layer as f64 + 0.5) / s) - 1.0)).exp();

        dir * (radius as f32)
    }

    pub fn pos_to_id(pos: Vec3, res: u32) -> Option<BlockId> {
        let dist = pos.length() as f64;
        let s = res as f64 / 2.0;

        let min_r = s * (-Self::K).exp();
        if dist < min_r {
            return None;
        }

        let layer_f = s * (1.0 + (dist / s).ln() / Self::K);
        let layer = layer_f.floor() as i32;

        if layer < 0 {
            return None;
        }
        let layer = layer as u32;

        // map sphere point back to unit cube surface
        // normalize 'pos' first to project it onto the unit sphere required for the math
        let cube_pos = Self::cubize_point(pos.normalize());

        // determine Face based on which component is 1.0 or -1.0
        // use a small epsilon for float comparison safety, though logic forces exactly 1.0
        let abs = cube_pos.abs();
        let (face, u_local, v_local) = if abs.y >= abs.x && abs.y >= abs.z {
            if cube_pos.y > 0.0 {
                (0, cube_pos.x, cube_pos.z)
            } else {
                (1, cube_pos.x, cube_pos.z)
            }
        } else if abs.x >= abs.y && abs.x >= abs.z {
            if cube_pos.x > 0.0 {
                (2, cube_pos.y, cube_pos.z)
            } else {
                (3, cube_pos.y, cube_pos.z)
            }
        } else {
            if cube_pos.z > 0.0 {
                (4, cube_pos.x, cube_pos.y)
            } else {
                (5, cube_pos.x, cube_pos.y)
            }
        };

        // convert Local [-1, 1] coords to grid indices
        let rf = res as f64;
        // x = (u * 2 - res) / res  =>  u = (x * res + res) / 2
        let u_raw = ((u_local as f64 * rf + rf) / 2.0).floor() as i32;
        let v_raw = ((v_local as f64 * rf + rf) / 2.0).floor() as i32;

        let u = u_raw.clamp(0, res as i32 - 1) as u32;
        let v = v_raw.clamp(0, res as i32 - 1) as u32;

        Some(BlockId {
            face: face as u8,
            layer,
            u,
            v,
        })
    }

    // pos as continuous face coordinates: the cube face, u and v in columns (0..=res) and the fractional
    // layer; None inside the core. pos_to_id's mapping without the rounding to a cell (hex cells,
    // PlanetData::cell_at, need the position within the face)
    pub fn face_coords(pos: Vec3, res: u32) -> Option<(u8, f64, f64, f64)> {
        let dist = pos.length() as f64;
        let s = res as f64 / 2.0;
        if dist < s * (-Self::K).exp() {
            return None;
        }
        let layer_f = s * (1.0 + (dist / s).ln() / Self::K);
        if layer_f < 0.0 {
            return None;
        }
        let cube_pos = Self::cubize_point(pos.normalize());
        let abs = cube_pos.abs();
        let (face, u_local, v_local) = if abs.y >= abs.x && abs.y >= abs.z {
            (if cube_pos.y > 0.0 { 0 } else { 1 }, cube_pos.x, cube_pos.z)
        } else if abs.x >= abs.y && abs.x >= abs.z {
            (if cube_pos.x > 0.0 { 2 } else { 3 }, cube_pos.y, cube_pos.z)
        } else {
            (if cube_pos.z > 0.0 { 4 } else { 5 }, cube_pos.x, cube_pos.y)
        };
        let rf = res as f64;
        let u = ((u_local as f64 * rf + rf) / 2.0).clamp(0.0, rf);
        let v = ((v_local as f64 * rf + rf) / 2.0).clamp(0.0, rf);
        Some((face, u, v, layer_f))
    }

    // get_direction at a fractional face position (columns); the face border is exactly ±1, like
    // get_direction's, so neighbouring faces share their border points
    pub fn get_direction_f(face: u8, u: f64, v: f64, res: u32) -> Vec3 {
        let rf = res as f64;
        let local = |c: f64| {
            if c <= 0.0 {
                -1.0
            } else if c >= rf {
                1.0
            } else {
                (c * 2.0 - rf) / rf
            }
        };
        let (x_local, y_local) = (local(u), local(v));
        let (cx, cy, cz) = match face {
            0 => (x_local, 1.0, y_local),
            1 => (x_local, -1.0, y_local),
            2 => (1.0, x_local, y_local),
            3 => (-1.0, x_local, y_local),
            4 => (x_local, y_local, 1.0),
            _ => (x_local, y_local, -1.0),
        };
        Self::cube_to_sphere(cx, cy, cz).normalize()
    }

    // a hex corner (face space in sixths of a column, hex.rs) at the bottom of `layer`
    pub fn get_vertex_pos_six(face: u8, u6: i64, v6: i64, layer: u32, res: u32) -> Vec3 {
        Self::get_direction_f(face, u6 as f64 / 6.0, v6 as f64 / 6.0, res)
            * Self::get_layer_radius(layer, res)
    }
}

// fs_water's opacity and colour rates per unit of water depth (shader.wgsl), for LOD water seen from above
const WATER_OPACITY_RATE: f32 = 0.35;
const WATER_DEEPENING_RATE: f32 = 0.15;

// how much darker voxel terrain renders on average than a smooth surface of the same colours, from
// its ambient occlusion and terrain shadows (linear; calibrated against the voxel world at the landing
// handover)
const LOD_OCCLUSION: f32 = 0.8;

// the share of terrace step faces in a LOD vertex's colour per unit of slope (1 - n.up), up to
// LOD_MAX_STEP_FACES, and their brightness (they face sideways, with ambient occlusion)
const LOD_STEP_FACE_SHARE: f32 = 1.5;
const LOD_MAX_STEP_FACES: f32 = 0.3;
const LOD_STEP_FACE_SHADE: f32 = 0.75;

// a dry LOD vertex: `top` is its column's top block, `below` the one under it, `slope` the normal's
// up component. Slopes are voxel terraces from close up, whose step faces show the block below the
// top (dirt under grass, rock under snow), in about the share the slope tilts. (A vertex past the face
// edge can have a degenerate normal, NaN: no step faces there.) Darkened by LOD_OCCLUSION.
pub(crate) fn lod_land_color(top: BlockType, below: BlockType, slope: f32) -> [f32; 3] {
    let steps = ((1.0 - slope) * LOD_STEP_FACE_SHARE).clamp(0.0, LOD_MAX_STEP_FACES);
    let steps = if steps.is_nan() { 0.0 } else { steps };
    let color = mix_linear(
        top.color(),
        below.color().map(|c| c * LOD_STEP_FACE_SHADE),
        steps,
    );
    color.map(|c| c * LOD_OCCLUSION.powf(1.0 / 2.2))
}

// a LOD vertex under `depth` layers of a reflective liquid: the voxel engine's water seen from above
// (shader.wgsl fs_water), the floor through a body that darkens from the shallow to the deep colour.
// fs_water lights the body with the sun plus twice the sky zenith (`sky_zenith`, the planet type's),
// the terrain shading of LOD meshes and impostors with the sun plus the sky once, so the body is
// brightened per channel by (sun + 2 sky) / (sun + sky) for a typical daylight sun (TYPICAL_SUN);
// LOD_WATER_GAIN and LOD_WATER_REFLECTION (fs_water's fresnel sky reflection) bring it to what
// fs_water shows, calibrated against the voxel world at the landing handover (Earth-like planet,
// Yellow star)
pub(crate) fn lod_water_color(
    floor: BlockType,
    liquid: &crate::biome::LiquidDef,
    depth: f32,
    sky_zenith: [f32; 3],
) -> [f32; 3] {
    let alpha = (1.0 - (-depth * WATER_OPACITY_RATE).exp()).clamp(0.25, 0.95);
    let deep = 1.0 - (-depth * WATER_DEEPENING_RATE).exp();
    let lin = |c: f32| c.max(0.0).powf(2.2);
    [0, 1, 2].map(|i| {
        let body = lin(liquid.shallow_color[i]) * (1.0 - deep) + lin(liquid.deep_color[i]) * deep;
        let sky_boost = (TYPICAL_SUN[i] + 2.0 * sky_zenith[i]) / (TYPICAL_SUN[i] + sky_zenith[i]);
        let surface = (body * sky_boost * LOD_WATER_GAIN + LOD_WATER_REFLECTION).powf(1.0 / 2.2);
        // fs_water blends over the floor after tone mapping (display space): mix the encoded colours
        surface * alpha + floor.color()[i] * (1.0 - alpha)
    })
}

const LOD_WATER_GAIN: f32 = 9.0;
const LOD_WATER_REFLECTION: f32 = 0.01;

// a Yellow star's sunlight (galaxy.rs) at a typical daytime incidence, for lod_water_color
const TYPICAL_SUN: [f32; 3] = [1.6 * 0.7, 1.5 * 0.7, 1.3 * 0.7];

// mixes two sRGB-ish vertex colours in linear space (the shaders decode them with pow 2.2)
fn mix_linear(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| {
        let lin = |c: f32| c.max(0.0).powf(2.2);
        (lin(a[i]) * (1.0 - t) + lin(b[i]) * t).powf(1.0 / 2.2)
    })
}

pub struct MeshGen;

impl MeshGen {
    // blocks a mined block exposes, including its neighbours across a cube-face edge
    fn add_mined_candidates(
        mods: &ChunkMods,
        candidates: &mut HashSet<BlockId>,
        data: &PlanetData,
    ) {
        for &id in &mods.mined {
            candidates.insert(BlockId {
                layer: id.layer + 1,
                ..id
            });
            if id.layer > 0 {
                candidates.insert(BlockId {
                    layer: id.layer - 1,
                    ..id
                });
            }
            for (du, dv) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                if let Some((face, u, v)) = data.neighbor_column(id.face, id.u, id.v, du, dv) {
                    candidates.insert(BlockId {
                        face,
                        u,
                        v,
                        layer: id.layer,
                    });
                }
            }
        }
    }

    // a voxel chunk's terrain mesh in the current terrain style (lowpoly.rs)
    pub fn build_chunk(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
        match crate::lowpoly::style_for(data) {
            crate::lowpoly::TerrainStyle::LowPoly => crate::lowpoly::build_chunk_lowpoly(key, data),
            crate::lowpoly::TerrainStyle::Cubes => Self::build_chunk_cubes(key, data),
            crate::lowpoly::TerrainStyle::Hex => Self::build_chunk_hex(key, data),
        }
    }

    pub fn build_chunk_cubes(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        let mut idx = 0u32;
        let res = data.resolution;
        let mut candidates = HashSet::new();

        let u_start = key.u_idx * CHUNK_SIZE;
        let v_start = key.v_idx * CHUNK_SIZE;
        // Ensure we don't iterate past resolution even if key exists
        let u_end = (u_start + CHUNK_SIZE).min(res);
        let v_end = (v_start + CHUNK_SIZE).min(res);

        // natural Surface (with slope filling)
        // need to check neighbors to see how far down the cliff goes.
        // if a neighbor is lower than us, we must generate the blocks between our height and theirs.

        for u in u_start..u_end {
            for v in v_start..v_end {
                let h = data.effective_height(key.face, u, v);
                if h == 0 {
                    continue;
                }

                // always add the top surface block
                candidates.insert(BlockId {
                    face: key.face,
                    layer: h,
                    u,
                    v,
                });

                // check immediate neighbors to find the lowest exposed point; at a cube-face edge the
                // neighbour is a column of the next face, whose wall would otherwise be missing
                let min_h = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    .iter()
                    .map(|&(du, dv)| data.neighbor_height(key.face, u, v, du, dv))
                    .fold(h, u32::min);

                if min_h < h {
                    for l in (min_h + 1)..h {
                        candidates.insert(BlockId {
                            face: key.face,
                            layer: l,
                            u,
                            v,
                        });
                    }
                }
            }
        }

        // current Chunk Modifications
        if let Some(mods) = data.edits.chunks.get(&key) {
            for &id in mods.placed.keys() {
                candidates.insert(id);
            }
            Self::add_mined_candidates(mods, &mut candidates, data);
        }

        // neighbor Chunks Modifications, including chunks of the neighbouring face along a face edge
        let mut neighbor_keys: HashSet<ChunkKey> = [
            ChunkKey {
                u_idx: key.u_idx.wrapping_sub(1),
                ..key
            },
            ChunkKey {
                u_idx: key.u_idx + 1,
                ..key
            },
            ChunkKey {
                v_idx: key.v_idx.wrapping_sub(1),
                ..key
            },
            ChunkKey {
                v_idx: key.v_idx + 1,
                ..key
            },
        ]
        .into_iter()
        .collect();
        let on_edge = |c: u32| c == 0 || c == res - 1;
        for u in u_start..u_end {
            for v in v_start..v_end {
                if !on_edge(u) && !on_edge(v) {
                    continue;
                }
                for (du, dv) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    if let Some((face, nu, nv)) = data.neighbor_column(key.face, u, v, du, dv) {
                        if face != key.face {
                            neighbor_keys.insert(PlanetData::chunk_key(BlockId {
                                face,
                                u: nu,
                                v: nv,
                                layer: 0,
                            }));
                        }
                    }
                }
            }
        }

        for n_key in neighbor_keys {
            if let Some(mods) = data.edits.chunks.get(&n_key) {
                Self::add_mined_candidates(mods, &mut candidates, data);
            }
        }

        // generate Mesh
        for id in candidates {
            if id.face == key.face
                && id.u >= u_start
                && id.u < u_end
                && id.v >= v_start
                && id.v < v_end
            {
                if data.exists(id) {
                    Self::add_voxel(id, data, &mut verts, &mut inds, &mut idx);
                }
            }
        }
        (verts, inds)
    }

    // a voxel chunk in hex columns (hex.rs, PlanetData::cells = Hex): build_chunk_cubes' candidate
    // rules over the hex neighbours, each block a hex prism (add_voxel_hex)
    pub fn build_chunk_hex(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
        let (mut verts, mut inds, mut idx) = (Vec::new(), Vec::new(), 0u32);
        let res = data.resolution;
        let u_start = key.u_idx * CHUNK_SIZE;
        let v_start = key.v_idx * CHUNK_SIZE;
        let u_end = (u_start + CHUNK_SIZE).min(res);
        let v_end = (v_start + CHUNK_SIZE).min(res);
        let mut candidates = HashSet::new();
        // the chunks whose mined blocks can expose blocks of this one: its own and its rim cells'
        // neighbours' (diagonal chunks too, odd rows reach into them; the next face's at a face edge)
        let mut keys = HashSet::from([key]);
        for u in u_start..u_end {
            for v in v_start..v_end {
                let around = data.column_neighbors(key.face, u, v);
                let h = data.effective_height(key.face, u, v);
                let rim = u == u_start || u + 1 == u_end || v == v_start || v + 1 == v_end;
                if rim {
                    keys.extend(around.iter().map(|&(face, u, v)| {
                        PlanetData::chunk_key(BlockId {
                            face,
                            u,
                            v,
                            layer: 0,
                        })
                    }));
                }
                if h == 0 {
                    continue;
                }
                // the top block and the cliff below it down to the lowest neighbour
                let min_h = around
                    .iter()
                    .map(|&(f, nu, nv)| data.effective_height(f, nu, nv))
                    .fold(h, u32::min);
                for layer in (min_h + 1).min(h)..=h {
                    candidates.insert(BlockId {
                        face: key.face,
                        layer,
                        u,
                        v,
                    });
                }
            }
        }
        for k in keys {
            let Some(mods) = data.edits.chunks.get(&k) else {
                continue;
            };
            if k == key {
                candidates.extend(mods.placed.keys().copied());
            }
            for &id in &mods.mined {
                candidates.insert(BlockId {
                    layer: id.layer + 1,
                    ..id
                });
                if id.layer > 0 {
                    candidates.insert(BlockId {
                        layer: id.layer - 1,
                        ..id
                    });
                }
                for (face, u, v) in data.column_neighbors(id.face, id.u, id.v) {
                    candidates.insert(BlockId {
                        face,
                        u,
                        v,
                        layer: id.layer,
                    });
                }
            }
        }
        for id in candidates {
            let mine = id.face == key.face
                && (u_start..u_end).contains(&id.u)
                && (v_start..v_end).contains(&id.v);
            if mine && data.exists(id) {
                Self::add_voxel_hex(id, data, &mut verts, &mut inds, &mut idx);
            }
        }
        (verts, inds)
    }

    // one hex prism (hex columns): a fan on top and bottom over the outline's corners, a quad per open
    // side wall. The cube voxel's shading: per-corner AO on top (an interior hex corner touches two other
    // columns, no diagonals), 0.8 × sky cover on the walls, 0.4 underneath
    fn add_voxel_hex(
        id: BlockId,
        data: &PlanetData,
        verts: &mut Vec<Vertex>,
        inds: &mut Vec<u32>,
        idx: &mut u32,
    ) {
        let res = data.resolution;
        let solid = |(face, u, v): (u8, u32, u32), layer: i64| {
            layer < 0
                || data.exists(BlockId {
                    face,
                    layer: layer as u32,
                    u,
                    v,
                })
        };
        let own = (id.face, id.u, id.v);
        let l = id.layer as i64;
        let walls = data.hex_walls(id.face, id.u, id.v);
        // which straight side of the cell each wall piece belongs to (border sides come in one-sixth
        // pieces): bevels round off the sides' corners, not the joints between pieces
        let side_of =
            crate::bevel::polygon_lines(&walls.iter().map(|w| (w.a, w.b)).collect::<Vec<_>>());
        let open: Vec<bool> = walls
            .iter()
            .map(|w| w.across.is_none_or(|c| !solid(c, l)))
            .collect();
        let (has_top, has_btm) = (solid(own, l + 1), solid(own, l - 1));
        if has_top && has_btm && !open.contains(&true) {
            return;
        }
        let sky = |c: (u8, u32, u32)| {
            if (1..=8).any(|i| solid(c, l + i)) {
                0.15
            } else {
                1.0
            }
        };
        let base = data.block_type(id).unwrap_or(BlockType::Dirt).color();
        let shade = |k: f32| [base[0] * k, base[1] * k, base[2] * k];
        let corner = |p: (i64, i64), layer: u32| {
            CoordSystem::get_vertex_pos_six(id.face, p.0, p.1, layer, res)
        };
        let (cu, cv) = crate::hex::centroid(&crate::hex::outline(id.u, id.v, res));
        let centre_dir = CoordSystem::get_direction_f(id.face, cu, cv, res);
        let block_center = centre_dir * CoordSystem::get_layer_radius_f(id.layer as f32 + 0.5, res);
        let water_here = data.water_surface_radius(id.face, id.u, id.v);

        let fan = |verts: &mut Vec<Vertex>,
                   inds: &mut Vec<u32>,
                   idx: &mut u32,
                   layer: u32,
                   colors: &[[f32; 3]],
                   centre_color: [f32; 3]| {
            let normal = if layer > id.layer {
                centre_dir
            } else {
                -centre_dir
            };
            let centre = centre_dir * CoordSystem::get_layer_radius(layer, res);
            let pieces: Vec<_> = walls
                .iter()
                .map(|w| (corner(w.a, layer), corner(w.b, layer)))
                .collect();
            // every vertex measures every side of the cell (slot = side), so the fan shares them
            let vertex = |p: Vec3, color: [f32; 3]| Vertex {
                pos: p.to_array(),
                color,
                normal: normal.to_array(),
                water: water_here,
                edge: crate::bevel::pack(crate::bevel::polygon_edges(p, &side_of, &pieces)),
            };
            let base_idx = *idx;
            verts.push(vertex(centre, centre_color));
            for (&(a, _), &color) in pieces.iter().zip(colors) {
                verts.push(vertex(a, color));
            }
            let n = walls.len() as u32;
            for i in 0..n {
                inds.extend_from_slice(&[base_idx, base_idx + 1 + i, base_idx + 1 + (i + 1) % n]);
            }
            *idx += n + 1;
        };

        if !has_top {
            // corner i (the start of wall i) touches the columns across walls i − 1 and i
            let top_sky = sky(own);
            let n = walls.len();
            let ao: Vec<f32> = (0..n)
                .map(|i| {
                    let (prev, next) = (walls[(i + n - 1) % n].across, walls[i].across);
                    let a = prev.is_some_and(|c| solid(c, l + 1));
                    let b = next != prev && next.is_some_and(|c| solid(c, l + 1));
                    Self::calculate_ao(a, b, false)
                })
                .collect();
            let mean = ao.iter().sum::<f32>() / n as f32;
            let colors: Vec<_> = ao.iter().map(|&a| shade(a * top_sky)).collect();
            fan(
                verts,
                inds,
                idx,
                id.layer + 1,
                &colors,
                shade(mean * top_sky),
            );
        }
        if !has_btm {
            let colors = vec![shade(0.4); walls.len()];
            fan(verts, inds, idx, id.layer, &colors, shade(0.4));
        }
        for (i, w) in walls.iter().enumerate().filter(|&(i, _)| open[i]) {
            let c = shade(0.8 * w.across.map_or(1.0, sky));
            let water = w
                .across
                .map_or(0.0, |(f, u, v)| data.water_surface_radius(f, u, v));
            Self::quad(
                verts,
                inds,
                idx,
                [
                    corner(w.a, id.layer),
                    corner(w.b, id.layer),
                    corner(w.b, id.layer + 1),
                    corner(w.a, id.layer + 1),
                ],
                [c; 4],
                false,
                block_center,
                water,
                crate::bevel::wall_mask(&side_of, i),
            );
        }
    }

    // side1, side2: the two blocks flanking the vertex
    // corner: the block diagonally connecting the vertex
    fn calculate_ao(side1: bool, side2: bool, corner: bool) -> f32 {
        let mut occ = 0;
        if side1 {
            occ += 1;
        }
        if side2 {
            occ += 1;
        }
        if corner && (side1 || side2) {
            occ += 1;
        }

        // 0=Bright, 1=Dim, 2=Dark, 3=Very Dark
        match occ {
            0 => 1.0,
            1 => 0.8,
            2 => 0.6,
            _ => 0.4,
        }
    }

    // Generates wireframe boxes for collision detection debugging
    pub fn generate_collision_debug(
        player_pos: Vec3,
        planet: &PlanetData,
    ) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        let res = planet.resolution;
        let color = [1.0, 0.0, 0.0]; // red
        let normal = [0.0, 1.0, 0.0];

        // check a 3x3x3 area around the player
        let range = 2;
        if planet.cells == crate::common::CellShape::Hex {
            return Self::hex_collision_debug(player_pos, planet, range, color, normal);
        }

        if let Some((center_id, _)) = CoordSystem::get_local_coords(player_pos, res) {
            let start_u = (center_id.u as i32 - range).max(0);
            let end_u = (center_id.u as i32 + range).min(res as i32 - 1);
            let start_v = (center_id.v as i32 - range).max(0);
            let end_v = (center_id.v as i32 + range).min(res as i32 - 1);
            let start_l = (center_id.layer as i32 - range).max(0);
            let end_l = (center_id.layer as i32 + range).min(res as i32 - 1);

            let mut idx = 0;

            for l in start_l..=end_l {
                for v in start_v..=end_v {
                    for u in start_u..=end_u {
                        let id = crate::common::BlockId {
                            face: center_id.face,
                            layer: l as u32,
                            u: u as u32,
                            v: v as u32,
                        };

                        let block_pos =
                            CoordSystem::get_block_center(id.face, id.u, id.v, id.layer, res);

                        if crate::physics::Physics::is_solid(block_pos, planet) {
                            // visualize the "Core" of the block that triggers collision
                            let get_p = |uu, vv, ll| {
                                CoordSystem::get_vertex_pos(
                                    id.face,
                                    id.u + uu,
                                    id.v + vv,
                                    id.layer + ll,
                                    res,
                                )
                            };

                            // get corners of the voxel
                            let c000 = get_p(0, 0, 0);
                            let c100 = get_p(1, 0, 0);
                            let c010 = get_p(0, 1, 0);
                            let c110 = get_p(1, 1, 0);
                            let c001 = get_p(0, 0, 1);
                            let c101 = get_p(1, 0, 1);
                            let c011 = get_p(0, 1, 1);
                            let c111 = get_p(1, 1, 1);

                            // shrink corners towards center by margin (visualize the "shave")
                            let center =
                                (c000 + c100 + c010 + c110 + c001 + c101 + c011 + c111) * 0.125;
                            let shrink = 0.90; // Exaggerate the shrink slightly so we can see it inside the block

                            let v = |p: Vec3| Vertex {
                                pos: (center + (p - center) * shrink).to_array(),
                                color,
                                normal,
                                water: 0.0,
                                edge: Vertex::NO_EDGE,
                            };

                            let corners = [
                                v(c000),
                                v(c100),
                                v(c110),
                                v(c010), // Bottom
                                v(c001),
                                v(c101),
                                v(c111),
                                v(c011), // Top
                            ];

                            // add vertices
                            for c in &corners {
                                verts.push(*c);
                            }

                            // add line indices (Cube wireframe)
                            let base = idx;
                            let lines = [
                                (0, 1),
                                (1, 2),
                                (2, 3),
                                (3, 0), // Bottom ring
                                (4, 5),
                                (5, 6),
                                (6, 7),
                                (7, 4), // Top ring
                                (0, 4),
                                (1, 5),
                                (2, 6),
                                (3, 7), // Pillars
                            ];

                            for (s, e) in lines {
                                inds.push(base + s);
                                inds.push(base + e);
                            }
                            idx += 8;
                        }
                    }
                }
            }
        }
        (verts, inds)
    }

    // generate_collision_debug for hex columns: the outline prisms of the solid cells around the player,
    // shrunk toward their centres like the cube boxes
    fn hex_collision_debug(
        player_pos: Vec3,
        planet: &PlanetData,
        range: i32,
        color: [f32; 3],
        normal: [f32; 3],
    ) -> (Vec<Vertex>, Vec<u32>) {
        let (mut verts, mut inds) = (Vec::new(), Vec::new());
        let res = planet.resolution;
        let Some(c) = planet.cell_at(player_pos) else {
            return (verts, inds);
        };
        let clamp = |x: i32, hi: u32| x.clamp(0, hi as i32 - 1) as u32;
        for layer in clamp(c.layer as i32 - range, res * 2)..=c.layer + range as u32 {
            for v in clamp(c.v as i32 - range, res)..=clamp(c.v as i32 + range, res) {
                for u in clamp(c.u as i32 - range, res)..=clamp(c.u as i32 + range, res) {
                    let pts = crate::hex::outline(u, v, res);
                    let (cu, cv) = crate::hex::centroid(&pts);
                    let centre = CoordSystem::get_direction_f(c.face, cu, cv, res)
                        * CoordSystem::get_layer_radius_f(layer as f32 + 0.5, res);
                    if !crate::physics::Physics::is_solid(centre, planet) {
                        continue;
                    }
                    let base = verts.len() as u32;
                    let n = pts.len() as u32;
                    for l in 0..2 {
                        for &(u6, v6) in &pts {
                            let p = CoordSystem::get_vertex_pos_six(c.face, u6, v6, layer + l, res);
                            verts.push(Vertex {
                                pos: (centre + (p - centre) * 0.9).to_array(),
                                color,
                                normal,
                                water: 0.0,
                                edge: Vertex::NO_EDGE,
                            });
                        }
                    }
                    for i in 0..n {
                        let j = (i + 1) % n;
                        inds.extend_from_slice(&[base + i, base + j]);
                        inds.extend_from_slice(&[base + n + i, base + n + j]);
                        inds.extend_from_slice(&[base + i, base + n + i]);
                    }
                }
            }
        }
        (verts, inds)
    }

    // generates a simplified heightmap mesh for distant terrain
    // the water surface of a chunk: one quad at the top of each column's water level (its lake's, else
    // sea level; flush with the beaches) over every column that holds water (PlanetData::holds_water:
    // natural ocean and lakes, holes dug below their level alike) and whose cell above is open — under a ceiling (a tunnel dug at
    // sea level) the surface would lie against the solid face above and flicker; that water has none.
    // the water level a dry column on a low-poly shore is covered at: the highest level of the water
    // columns around it, if its own cell at that level is solid (a column dug out beside a lake keeps
    // its wall of water: no flow, see PlanetData::holds_water)
    fn shore_level(data: &PlanetData, face: u8, u: u32, v: u32) -> Option<u32> {
        let level = data
            .neighbour_columns(face, u, v)
            .into_iter()
            .filter(|&(f, cu, cv)| data.holds_water(f, cu, cv))
            .map(|(f, cu, cv)| data.terrain.water_level(f, cu, cv))
            .max()?;
        data.exists(BlockId {
            face,
            layer: level,
            u,
            v,
        })
        .then_some(level)
    }

    // a chunk's water surface in the current terrain style (lowpoly.rs)
    pub fn build_water(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
        Self::build_water_styled(key, data, crate::lowpoly::style_for(data))
    }

    pub fn build_water_styled(
        key: ChunkKey,
        data: &PlanetData,
        style: crate::lowpoly::TerrainStyle,
    ) -> (Vec<Vertex>, Vec<u32>) {
        if data.planet_type.def().liquid.is_none() {
            return (Vec::new(), Vec::new());
        }
        let (mut verts, mut inds, mut idx) = (Vec::new(), Vec::new(), 0u32);
        let res = data.resolution;
        let u_start = key.u_idx * CHUNK_SIZE;
        let v_start = key.v_idx * CHUNK_SIZE;
        for u in u_start..(u_start + CHUNK_SIZE).min(res) {
            for v in v_start..(v_start + CHUNK_SIZE).min(res) {
                let (level, water) = if data.holds_water(key.face, u, v) {
                    let level = data.terrain.water_level(key.face, u, v);
                    let above = BlockId {
                        face: key.face,
                        layer: level + 1,
                        u,
                        v,
                    };
                    if data.exists(above) {
                        continue;
                    }
                    (level, data.water_surface_radius(key.face, u, v))
                } else if style == crate::lowpoly::TerrainStyle::LowPoly {
                    // low-poly shore: the drawn terrain crosses the water level between column
                    // centres, so the water plane reaches one column onto the land beside the water
                    // (hidden where the terrain is higher) and meets the terrain at its own coastline
                    // instead of ending in column steps over unflooded dips
                    match Self::shore_level(data, key.face, u, v) {
                        Some(level) => (level, 0.0),
                        None => continue,
                    }
                } else {
                    continue;
                };
                // liquid is Some here (checked above); color is unused by fs_water (which shades
                // purely from the biome uniform) but should still match the active planet type
                let color = data.planet_type.def().liquid.unwrap().shallow_color;
                let mut push = |c: Vec3| {
                    verts.push(Vertex {
                        pos: c.to_array(),
                        color,
                        normal: c.normalize().to_array(),
                        water,
                        edge: Vertex::NO_EDGE,
                    })
                };
                if style == crate::lowpoly::TerrainStyle::Hex {
                    // the hex cell's outline (seam pieces included, so neighbouring faces' surfaces
                    // share their border vertices), a fan around its centre
                    let pts: Vec<_> = data.hex_walls(key.face, u, v).iter().map(|w| w.a).collect();
                    let (cu, cv) = crate::hex::centroid(&crate::hex::outline(u, v, res));
                    let radius = CoordSystem::get_layer_radius(level + 1, res);
                    push(CoordSystem::get_direction_f(key.face, cu, cv, res) * radius);
                    for &(u6, v6) in &pts {
                        push(CoordSystem::get_vertex_pos_six(
                            key.face,
                            u6,
                            v6,
                            level + 1,
                            res,
                        ));
                    }
                    let n = pts.len() as u32;
                    for i in 0..n {
                        inds.extend_from_slice(&[idx, idx + 1 + i, idx + 1 + (i + 1) % n]);
                    }
                    idx += n + 1;
                    continue;
                }
                let p =
                    |du, dv| CoordSystem::get_vertex_pos(key.face, u + du, v + dv, level + 1, res);
                for c in [p(0, 0), p(1, 0), p(1, 1), p(0, 1)] {
                    push(c);
                }
                inds.extend_from_slice(&[idx, idx + 1, idx + 2, idx + 2, idx + 3, idx]);
                idx += 4;
            }
        }
        (verts, inds)
    }

    // LOD meshes are a LOD_GRID x LOD_GRID quad grid over their node, each vertex point-sampling a column
    // (so a child node's even grid vertices lie exactly on its parent's)
    const LOD_GRID: u32 = 64;

    // a LOD node's grid vertex (gx, gy may lie outside 0..=LOD_GRID, for normals): position, normal, colour
    fn lod_sample(
        key: crate::common::LodKey,
        data: &PlanetData,
        gx: i32,
        gy: i32,
        style: crate::lowpoly::TerrainStyle,
    ) -> (glam::Vec3, glam::Vec3, [f32; 3]) {
        let def = data.planet_type.def();
        let grid_res = Self::LOD_GRID;

        // calculate global pos for any grid index (even outside this chunk)
        // this allows us to "peek" into neighbor chunks for perfect normals.
        let get_sample_pos = |gx: i32, gy: i32| -> glam::Vec3 {
            let step_u = (gx as i64 * key.size as i64) / grid_res as i64;
            let step_v = (gy as i64 * key.size as i64) / grid_res as i64;

            // calculate absolute U/V
            let abs_u = (key.x as i64 + step_u).clamp(0, data.resolution as i64) as u32;
            let abs_v = (key.y as i64 + step_v).clamp(0, data.resolution as i64) as u32;

            if style == crate::lowpoly::TerrainStyle::LowPoly {
                // the low-poly surface at this column corner, or the water surface above it. Edits
                // shift the smooth surface by how far they moved the column's top (0 where unedited,
                // and for columns of an edited chunk whose top is unchanged), so edited chunks sit
                // where the voxel low-poly surface does
                let (cu, cv) = (
                    abs_u.min(data.resolution - 1),
                    abs_v.min(data.resolution - 1),
                );
                let shift = if data.column_edited(key.face, cu, cv) {
                    data.surface(key.face, cu, cv) as f32
                        - data.effective_height(key.face, cu, cv) as f32
                } else {
                    0.0
                };
                let land = crate::lowpoly::surface_radius(
                    data.smooth_corner(key.face, abs_u, abs_v) + shift,
                    data.resolution,
                );
                let water = match def.liquid {
                    Some(_) => CoordSystem::get_layer_radius(
                        data.terrain.water_level(key.face, abs_u, abs_v) + 1,
                        data.resolution,
                    ),
                    None => 0.0,
                };
                return CoordSystem::get_direction(key.face, abs_u, abs_v, data.resolution)
                    * land.max(water);
            }

            // the edited surface (natural height where unedited, LOD surfaces sit at layer h like the
            // land); oceans, lakes and liquid-less basins are flat at their water level from afar
            let h = data
                .surface(key.face, abs_u, abs_v)
                .max(data.terrain.water_level(key.face, abs_u, abs_v));
            CoordSystem::get_vertex_pos(key.face, abs_u, abs_v, h, data.resolution)
        };

        let pos = get_sample_pos(gx, gy);

        // seamless normal fix
        // instead of clamping to grid edges, we look -1 and +1 in global grid Space
        // this ensures the normal at the chunk edge matches the neighbor's normal perfectly

        let p_right = get_sample_pos(gx + 1, gy);
        let p_left = get_sample_pos(gx - 1, gy);
        let p_down = get_sample_pos(gx, gy + 1);
        let p_up = get_sample_pos(gx, gy - 1);

        // central Difference
        let tangent_u = p_right - p_left;
        let tangent_v = p_down - p_up;

        let mut normal = tangent_u.cross(tangent_v).normalize();
        if normal.dot(pos.normalize()) < 0.0 {
            normal = -normal;
        }

        // --- COLORING ---
        let slope = normal.dot(pos.normalize()).abs();

        // same material as the voxel surface here
        let offset_u = (gx.max(0) as u32 * key.size) / grid_res;
        let offset_v = (gy.max(0) as u32 * key.size) / grid_res;
        let (su, sv) = (
            (key.x + offset_u).min(data.resolution - 1),
            (key.y + offset_v).min(data.resolution - 1),
        );
        let h = data.surface(key.face, su, sv);
        // the block on top, edits included (the natural material where unedited)
        let surface = data
            .block_type(BlockId {
                face: key.face,
                layer: h,
                u: su,
                v: sv,
            })
            .unwrap_or(def.palette.beach);
        let sea = data.terrain.sea_level();
        // a liquid-less planet's filled-in basin: its surface is the sea-level fill over natural
        // terrain below sea level, coloured flat like the rest of the basin (no rim shading)
        let filled_basin =
            def.liquid.is_none() && h == sea && data.terrain.get_height(key.face, su, sv) < sea;
        // below the column's water level: a lake's or the sea's (Ice: always sea level)
        let water = data.terrain.water_level(key.face, su, sv);
        let color = if h < water || filled_basin {
            match def.liquid {
                // a glowing liquid (lava) is opaque; its glow isn't modelled here
                Some(liquid)
                    if matches!(liquid.behavior, crate::biome::LiquidBehavior::Glowing) =>
                {
                    liquid.shallow_color
                }
                // the voxel engine's water seen from above (shader.wgsl fs_water): the floor through
                // a body that darkens from the shallow to the deep colour with the depth
                Some(liquid) => lod_water_color(
                    surface,
                    &liquid,
                    (water - h) as f32,
                    def.atmosphere.sky_zenith,
                ),
                // liquid-less planet (e.g. Ice): the filled-in beach material
                // (PlanetData::exists's liquid-less solidity rule), not water — a filled
                // basin, or edits that dug below sea level
                None => def.palette.beach.color(),
            }
        } else if style == crate::lowpoly::TerrainStyle::LowPoly
            && h != data.effective_height(key.face, su, sv)
        {
            // an edit moved the column's top: that block's own colour, flat like the voxel facets
            surface.color()
        } else if style == crate::lowpoly::TerrainStyle::LowPoly {
            crate::material::lowpoly_material(
                &data.terrain,
                &def.palette,
                key.face,
                su,
                sv,
                data.smooth_height(key.face, su, sv),
                slope,
            )
            .color()
        } else {
            let below = data
                .block_type(BlockId {
                    face: key.face,
                    layer: h.saturating_sub(1),
                    u: su,
                    v: sv,
                })
                .unwrap_or(surface);
            lod_land_color(surface, below, slope)
        };
        (pos, normal, color)
    }

    // the geomorph targets of generate_lod_mesh's vertices (same order: the grid, then the skirts): the
    // parent node's piecewise-linear surface at each vertex. Even vertices take the parent's own
    // sample, odd ones the midpoint of the parent edge they lie on (for a quad's centre, the diagonal
    // tr-bl generate_lod_mesh splits it along), so at morph 1 the mesh lies exactly on its parent's
    // triangles. A root node (no parent, `logical_size` = the quadtree's) morphs to itself.
    pub fn generate_lod_morph(
        key: crate::common::LodKey,
        data: &PlanetData,
        logical_size: u32,
    ) -> Vec<LodMorph> {
        Self::generate_lod_morph_styled(key, data, logical_size, crate::lowpoly::style_for(data))
    }

    pub fn generate_lod_morph_styled(
        key: crate::common::LodKey,
        data: &PlanetData,
        logical_size: u32,
        style: crate::lowpoly::TerrainStyle,
    ) -> Vec<LodMorph> {
        let grid = Self::LOD_GRID;
        let row_len = grid + 1;
        let own: Vec<_> = (0..=grid)
            .flat_map(|y| (0..=grid).map(move |x| (x, y)))
            .map(|(x, y)| Self::lod_sample(key, data, x as i32, y as i32, style))
            .collect();
        let mut targets: Vec<LodMorph> = if key.size >= logical_size {
            own.iter()
                .map(|&(_, n, c)| LodMorph::new(0.0, n, c))
                .collect()
        } else {
            let psize = key.size * 2;
            let parent = crate::common::LodKey {
                face: key.face,
                x: key.x - key.x % psize,
                y: key.y - key.y % psize,
                size: psize,
            };
            // this node's grid in half-steps of the parent's: 0 or LOD_GRID offset into its quadrant
            let (ox, oy) = (
                (key.x - parent.x) * grid / key.size,
                (key.y - parent.y) * grid / key.size,
            );
            // the parent's samples over this quadrant (indices ox/2..=ox/2 + grid/2)
            let half = grid / 2 + 1;
            let psamples: Vec<_> = (0..half)
                .flat_map(|y| (0..half).map(move |x| (x, y)))
                .map(|(x, y)| {
                    Self::lod_sample(
                        parent,
                        data,
                        (ox / 2 + x) as i32,
                        (oy / 2 + y) as i32,
                        style,
                    )
                })
                .collect();
            let p = |hx: u32, hy: u32| psamples[((hy - oy) / 2 * half + (hx - ox) / 2) as usize];
            let mut out = Vec::with_capacity(own.len());
            for y in 0..=grid {
                for x in 0..=grid {
                    let (hx, hy) = (x + ox, y + oy);
                    let (a, b) = match (hx % 2, hy % 2) {
                        (0, 0) => (p(hx, hy), p(hx, hy)),
                        (1, 0) => (p(hx - 1, hy), p(hx + 1, hy)),
                        (0, _) => (p(hx, hy - 1), p(hx, hy + 1)),
                        _ => (p(hx + 1, hy - 1), p(hx - 1, hy + 1)),
                    };
                    let pos = (a.0 + b.0) * 0.5;
                    let normal = (a.1 + b.1).normalize();
                    let color = [0, 1, 2].map(|i| (a.2[i] + b.2[i]) * 0.5);
                    let own_pos = own[(y * row_len + x) as usize].0;
                    out.push(LodMorph::new(
                        pos.length() - own_pos.length(),
                        normal,
                        color,
                    ));
                }
            }
            out
        };
        // skirts: copies of the edge vertices, in generate_lod_mesh's order (top, bottom, left, right)
        let edges = [
            (0..=grid).map(|x| (x, 0)).collect::<Vec<_>>(),
            (0..=grid).map(|x| (x, grid)).collect(),
            (0..=grid).map(|y| (0, y)).collect(),
            (0..=grid).map(|y| (grid, y)).collect(),
        ];
        for edge in edges {
            for (x, y) in edge {
                targets.push(targets[(y * row_len + x) as usize]);
            }
        }
        targets
    }

    pub fn generate_lod_mesh(
        key: crate::common::LodKey,
        data: &PlanetData,
    ) -> (Vec<Vertex>, Vec<u32>) {
        Self::generate_lod_mesh_styled(key, data, crate::lowpoly::style_for(data))
    }

    pub fn generate_lod_mesh_styled(
        key: crate::common::LodKey,
        data: &PlanetData,
        style: crate::lowpoly::TerrainStyle,
    ) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();

        let grid_res = Self::LOD_GRID;
        let row_len = grid_res + 1;

        // 1. Generate Vertices
        for vy in 0..=grid_res {
            for ux in 0..=grid_res {
                let (pos, normal, color) = Self::lod_sample(key, data, ux as i32, vy as i32, style);
                verts.push(Vertex {
                    pos: pos.to_array(),
                    color,
                    normal: normal.to_array(),
                    water: 0.0,
                    edge: Vertex::NO_EDGE,
                });
            }
        }

        // generate indices
        for y in 0..grid_res {
            for x in 0..grid_res {
                let tl = y * row_len + x;
                let tr = tl + 1;
                let bl = (y + 1) * row_len + x;
                let br = bl + 1;

                inds.push(tl);
                inds.push(bl);
                inds.push(tr);
                inds.push(tr);
                inds.push(bl);
                inds.push(br);
            }
        }

        // generate Skirts (hides physical gaps)
        let radius = CoordSystem::get_layer_radius(data.resolution / 2, data.resolution);
        let chunk_phys_size = (key.size as f32 / data.resolution as f32) * radius;

        let skirt_depth = (chunk_phys_size * 0.15).clamp(4.0, 500.0);

        let mut add_skirt_edge = |coord_pairs: &[(u32, u32)], reverse: bool| {
            let base_idx = verts.len() as u32;
            for &(ux, vy) in coord_pairs {
                let src_idx = vy * row_len + ux;
                let src_v = verts[src_idx as usize];

                // bend skirt inwards slightly to avoid poking through other meshes
                let p = glam::Vec3::from_array(src_v.pos);
                let down = -p.normalize() * skirt_depth;

                verts.push(Vertex {
                    pos: (p + down).to_array(),
                    color: src_v.color,
                    normal: src_v.normal,
                    water: 0.0,
                    edge: Vertex::NO_EDGE,
                });
            }
            let len = coord_pairs.len() as u32;
            for i in 0..(len - 1) {
                let s1 = coord_pairs[i as usize].1 * row_len + coord_pairs[i as usize].0;
                let s2 =
                    coord_pairs[(i + 1) as usize].1 * row_len + coord_pairs[(i + 1) as usize].0;
                let k1 = base_idx + i;
                let k2 = base_idx + i + 1;

                // winding
                if reverse {
                    inds.push(s1);
                    inds.push(k2);
                    inds.push(k1);
                    inds.push(s1);
                    inds.push(s2);
                    inds.push(k2);
                } else {
                    inds.push(s1);
                    inds.push(k1);
                    inds.push(k2);
                    inds.push(s1);
                    inds.push(k2);
                    inds.push(s2);
                }
            }
        };

        // define active edges positive logic
        let top: Vec<(u32, u32)> = (0..=grid_res).map(|x| (x, 0)).collect();
        let bottom: Vec<(u32, u32)> = (0..=grid_res).map(|x| (x, grid_res)).collect();
        let left: Vec<(u32, u32)> = (0..=grid_res).map(|y| (0, y)).collect();
        let right: Vec<(u32, u32)> = (0..=grid_res).map(|y| (grid_res, y)).collect();

        add_skirt_edge(&top, false);
        add_skirt_edge(&bottom, true);
        add_skirt_edge(&left, true);
        add_skirt_edge(&right, false);

        (verts, inds)
    }

    fn add_voxel(
        id: BlockId,
        data: &PlanetData,
        verts: &mut Vec<Vertex>,
        inds: &mut Vec<u32>,
        idx: &mut u32,
    ) {
        Self::add_voxel_with(id, data, verts, inds, idx, &|b| data.exists(b), false);
    }

    pub(crate) fn add_voxel_with(
        id: BlockId,
        data: &PlanetData,
        verts: &mut Vec<Vertex>,
        inds: &mut Vec<u32>,
        idx: &mut u32,
        solid: &dyn Fn(BlockId) -> bool,
        // one colour per face (the low-poly mesher's flat shading takes each triangle's first vertex,
        // so the top face's per-corner AO would split it into two shades): the corners' mean
        flat: bool,
    ) {
        let res = data.resolution;

        // neighbor existence check
        let check = |d_face: u8, d_layer: i32, d_u: i32, d_v: i32| -> bool {
            let l = id.layer as i32 + d_layer;
            let u = id.u as i32 + d_u;
            let v = id.v as i32 + d_v;
            if l >= 0 && u >= 0 && u < res as i32 && v >= 0 && v < res as i32 {
                return solid(BlockId {
                    face: d_face,
                    layer: l as u32,
                    u: u as u32,
                    v: v as u32,
                });
            }
            l < 0 // Core is solid
        };

        // --- FACE CHECKS ---
        let has_top = check(id.face, 1, 0, 0);
        let has_btm = check(id.face, -1, 0, 0);
        let has_right = check(id.face, 0, 1, 0);
        let has_left = check(id.face, 0, -1, 0);
        let has_back = check(id.face, 0, 0, 1);
        let has_front = check(id.face, 0, 0, -1);

        if has_top && has_btm && has_left && has_right && has_front && has_back {
            return;
        }

        // --- SKY LIGHT ---
        // a face is darkened when the air cell it looks into has something solid within 8 layers above
        // it (tunnels, overhangs); open cliff faces stay lit even though their own column continues up
        let sky = |du: i32, dv: i32| -> f32 {
            if (1..=8).any(|i| check(id.face, i, du, dv)) {
                0.15
            } else {
                1.0
            }
        };
        let base_color = data.block_type(id).unwrap_or(BlockType::Dirt).color();

        // geometry Helpers
        let p = |u_off: u32, v_off: u32, l_off: u32| {
            CoordSystem::get_vertex_pos(id.face, id.u + u_off, id.v + v_off, id.layer + l_off, res)
        };
        let i_bl = p(0, 0, 0);
        let i_br = p(1, 0, 0);
        let i_tl = p(0, 1, 0);
        let i_tr = p(1, 1, 0);
        let o_bl = p(0, 0, 1);
        let o_br = p(1, 0, 1);
        let o_tl = p(0, 1, 1);
        let o_tr = p(1, 1, 1);

        let block_center = (i_bl + o_tr) * 0.5;
        // caustics: the water surface over the cell each face looks into (the neighbour column for side
        // faces, clamped to this face's grid), 0 when that column holds no water
        let water = |du: i32, dv: i32| {
            let nu = (id.u as i32 + du).clamp(0, res as i32 - 1) as u32;
            let nv = (id.v as i32 + dv).clamp(0, res as i32 - 1) as u32;
            data.water_surface_radius(id.face, nu, nv)
        };
        let apply =
            |ao: f32| -> [f32; 3] { [base_color[0] * ao, base_color[1] * ao, base_color[2] * ao] };
        let side = |du: i32, dv: i32| {
            let c = apply(0.8 * sky(du, dv));
            [c, c, c, c]
        };

        if !has_top {
            let n = |u, v| check(id.face, 1, u, v);
            let ao_bl = Self::calculate_ao(n(-1, 0), n(0, -1), n(-1, -1));
            let ao_br = Self::calculate_ao(n(1, 0), n(0, -1), n(1, -1));
            let ao_tr = Self::calculate_ao(n(1, 0), n(0, 1), n(1, 1));
            let ao_tl = Self::calculate_ao(n(-1, 0), n(0, 1), n(-1, 1));
            let top_sky = sky(0, 0);
            let apply = |ao: f32| apply(ao * top_sky);
            let (ao_bl, ao_br, ao_tr, ao_tl) = if flat {
                let mean = (ao_bl + ao_br + ao_tr + ao_tl) / 4.0;
                (mean, mean, mean, mean)
            } else {
                (ao_bl, ao_br, ao_tr, ao_tl)
            };
            // quad() splits along vertex 0-2; split along the darker diagonal so the AO gradient stays symmetric
            if ao_bl + ao_tr > ao_br + ao_tl {
                Self::quad(
                    verts,
                    inds,
                    idx,
                    [o_br, o_tr, o_tl, o_bl],
                    [apply(ao_br), apply(ao_tr), apply(ao_tl), apply(ao_bl)],
                    true,
                    block_center,
                    water(0, 0),
                    [true; 4],
                );
            } else {
                Self::quad(
                    verts,
                    inds,
                    idx,
                    [o_bl, o_br, o_tr, o_tl],
                    [apply(ao_bl), apply(ao_br), apply(ao_tr), apply(ao_tl)],
                    true,
                    block_center,
                    water(0, 0),
                    [true; 4],
                );
            }
        }

        if !has_btm {
            let c = apply(0.4);
            Self::quad(
                verts,
                inds,
                idx,
                [i_tl, i_tr, i_br, i_bl],
                [c, c, c, c],
                true,
                block_center,
                water(0, 0),
                [true; 4],
            );
        }

        if !has_front {
            Self::quad(
                verts,
                inds,
                idx,
                [i_bl, i_br, o_br, o_bl],
                side(0, -1),
                false,
                block_center,
                water(0, -1),
                [true; 4],
            );
        }
        if !has_back {
            Self::quad(
                verts,
                inds,
                idx,
                [o_tl, o_tr, i_tr, i_tl],
                side(0, 1),
                false,
                block_center,
                water(0, 1),
                [true; 4],
            );
        }
        if !has_left {
            Self::quad(
                verts,
                inds,
                idx,
                [i_tl, i_bl, o_bl, o_tl],
                side(-1, 0),
                false,
                block_center,
                water(-1, 0),
                [true; 4],
            );
        }
        if !has_right {
            Self::quad(
                verts,
                inds,
                idx,
                [i_br, i_tr, o_tr, o_br],
                side(1, 0),
                false,
                block_center,
                water(1, 0),
                [true; 4],
            );
        }
    }
    pub fn generate_cylinder(radius: f32, height: f32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        let color = [0.0, 0.5, 1.0];

        for i in 0..=segments {
            let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            let normal = Vec3::new(x, 0.0, z).normalize().to_array();

            verts.push(Vertex {
                pos: [x, 0.0, z],
                color,
                normal,
                water: 0.0,
                edge: Vertex::NO_EDGE,
            });

            verts.push(Vertex {
                pos: [x, height, z],
                color,
                normal,
                water: 0.0,
                edge: Vertex::NO_EDGE,
            });
        }

        for i in 0..segments {
            let bottom1 = i * 2;
            let top1 = bottom1 + 1;
            let bottom2 = bottom1 + 2;
            let top2 = bottom1 + 3;

            inds.push(bottom1);
            inds.push(top1);
            inds.push(bottom2);
            inds.push(bottom2);
            inds.push(top1);
            inds.push(top2);
        }

        let center_idx = verts.len() as u32;
        verts.push(Vertex {
            pos: [0.0, height, 0.0],
            color,
            normal: [0.0, 1.0, 0.0],
            water: 0.0,
            edge: Vertex::NO_EDGE,
        });
        for i in 0..=segments {
            let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            verts.push(Vertex {
                pos: [x, height, z],
                color,
                normal: [0.0, 1.0, 0.0],
                water: 0.0,
                edge: Vertex::NO_EDGE,
            });
        }
        for i in 0..segments {
            inds.push(center_idx);
            inds.push(center_idx + 1 + i);
            inds.push(center_idx + 1 + i + 1);
        }

        (verts, inds)
    }

    pub fn generate_crosshair() -> (Vec<Vertex>, Vec<u32>) {
        let s = 0.02; // size relative to screen (2%)
        let color = [1.0, 1.0, 1.0];
        let normal = [0.0, 0.0, 1.0];

        let verts = vec![
            Vertex {
                pos: [-s, 0.0, 0.0],
                color,
                normal,
                water: 0.0,
                edge: Vertex::NO_EDGE,
            },
            Vertex {
                pos: [s, 0.0, 0.0],
                color,
                normal,
                water: 0.0,
                edge: Vertex::NO_EDGE,
            },
            Vertex {
                pos: [0.0, -s, 0.0],
                color,
                normal,
                water: 0.0,
                edge: Vertex::NO_EDGE,
            },
            Vertex {
                pos: [0.0, s, 0.0],
                color,
                normal,
                water: 0.0,
                edge: Vertex::NO_EDGE,
            },
        ];
        let inds = vec![0, 1, 2, 3];
        (verts, inds)
    }

    // the normal is flipped to point away from block_center: the (u, v, layer) basis is left-handed on
    // some cube faces, so the winding alone doesn't tell which side is outside
    fn quad(
        verts: &mut Vec<Vertex>,
        inds: &mut Vec<u32>,
        idx: &mut u32,
        pos: [Vec3; 4],
        colors: [[f32; 3]; 4],
        force_radial: bool,
        block_center: Vec3,
        water: f32,       // Vertex::water for all four corners
        bevel: [bool; 4], // which edges pos[k] → pos[k + 1] are rounded off (bevel.rs)
    ) {
        let normal = if force_radial {
            let center = (pos[0] + pos[1] + pos[2] + pos[3]) * 0.25;
            center.normalize().to_array()
        } else {
            (pos[1] - pos[0])
                .cross(pos[2] - pos[0])
                .normalize()
                .to_array()
        };
        let quad_center = (pos[0] + pos[1] + pos[2] + pos[3]) * 0.25;
        let normal = if Vec3::from_array(normal).dot(quad_center - block_center) < 0.0 {
            (-Vec3::from_array(normal)).to_array()
        } else {
            normal
        };

        let edges = crate::bevel::quad_edges(pos, bevel);
        for i in 0..4 {
            verts.push(Vertex {
                pos: pos[i].to_array(),
                color: colors[i],
                normal,
                water,
                edge: crate::bevel::pack(edges[i]),
            });
        }

        inds.push(*idx);
        inds.push(*idx + 1);
        inds.push(*idx + 2);
        inds.push(*idx + 2);
        inds.push(*idx + 3);
        inds.push(*idx);
        *idx += 4;
    }
}

#[cfg(test)]
mod biome_tests {
    use super::*;

    #[test]
    fn fractional_layer_radius_matches_integer_layers() {
        for res in [49u32, 200, 337] {
            for layer in [0u32, res / 4, res / 2, res / 2 + 7, res] {
                let a = CoordSystem::get_layer_radius(layer, res);
                let b = CoordSystem::get_layer_radius_f(layer as f32, res);
                assert!((a - b).abs() < 1e-3, "res {res} layer {layer}: {a} vs {b}");
            }
        }
    }
    use crate::biome::PlanetType;
    use crate::common::PlanetData;

    // deterministically finds a chunk that actually has an underwater column, rather than
    // guessing a chunk key and hoping — otherwise this test could pass for the wrong reason
    // (an all-land chunk legitimately has no water mesh on *any* planet type)
    fn chunk_with_ocean(planet: &PlanetData) -> crate::common::ChunkKey {
        let sea = planet.terrain.sea_level();
        for face in 0..6u8 {
            for u in 0..planet.resolution {
                for v in 0..planet.resolution {
                    if planet.terrain.get_height(face, u, v) < sea {
                        return PlanetData::chunk_key(crate::common::BlockId {
                            face,
                            u,
                            v,
                            layer: 0,
                        });
                    }
                }
            }
        }
        panic!("test planet has no ocean at all");
    }

    #[test]
    fn earth_like_ocean_chunk_has_a_water_mesh() {
        let planet = PlanetData::new(32);
        let key = chunk_with_ocean(&planet);
        let (verts, _inds) = MeshGen::build_water(key, &planet);
        assert!(
            !verts.is_empty(),
            "sanity check: Earth-like should mesh water here"
        );
    }

    fn water_quads(planet: &PlanetData, face: u8, u: u32, v: u32) -> usize {
        let key = PlanetData::chunk_key(crate::common::BlockId {
            face,
            u,
            v,
            layer: 0,
        });
        MeshGen::build_water_styled(key, planet, crate::lowpoly::TerrainStyle::Cubes)
            .0
            .len()
            / 4
    }

    // digging a land column down below sea level: the hole gets a water surface (water table)
    #[test]
    fn a_hole_dug_below_sea_level_gets_a_water_surface() {
        let mut planet = PlanetData::new(32);
        let (face, u, v) = crate::common::tests::first_land_column(&planet, 2);
        let sea = planet.terrain.sea_level();
        let before = water_quads(&planet, face, u, v);
        for layer in (sea..=planet.terrain.get_height(face, u, v)).rev() {
            planet
                .remove_block(crate::common::BlockId { face, layer, u, v })
                .unwrap();
        }
        assert_eq!(water_quads(&planet, face, u, v), before + 1);
    }

    // a tunnel dug at sea level under land: water to swim in, but no surface pressed against its
    // ceiling (the two would flicker)
    #[test]
    fn a_tunnel_at_sea_level_gets_no_water_surface() {
        let mut planet = PlanetData::new(32);
        let (face, u, v) = crate::common::tests::first_land_column(&planet, 2);
        let sea = planet.terrain.sea_level();
        let before = water_quads(&planet, face, u, v);
        planet
            .remove_block(crate::common::BlockId {
                face,
                layer: sea,
                u,
                v,
            })
            .unwrap();
        assert_eq!(water_quads(&planet, face, u, v), before);
    }

    // a block placed into the ocean at sea level: no water surface drawn on top of it
    #[test]
    fn a_block_at_sea_level_has_no_water_surface_on_it() {
        let mut planet = PlanetData::new(32);
        let key = chunk_with_ocean(&planet);
        let sea = planet.terrain.sea_level();
        let (face, u, v) = (0..CHUNK_SIZE * CHUNK_SIZE)
            .map(|i| {
                (
                    key.face,
                    key.u_idx * CHUNK_SIZE + i / CHUNK_SIZE,
                    key.v_idx * CHUNK_SIZE + i % CHUNK_SIZE,
                )
            })
            .find(|&(f, u, v)| planet.terrain.get_height(f, u, v) < sea)
            .unwrap();
        let before = water_quads(&planet, face, u, v);
        planet
            .add_block(
                crate::common::BlockId {
                    face,
                    layer: sea,
                    u,
                    v,
                },
                crate::material::BlockType::Stone,
            )
            .unwrap();
        assert_eq!(water_quads(&planet, face, u, v), before - 1);
    }

    fn lod_key_for(face: u8, u: u32, v: u32) -> crate::common::LodKey {
        // the finest LOD node size (two chunks) containing the column: one vertex per column
        let size = CHUNK_SIZE * 2;
        crate::common::LodKey {
            face,
            x: u / size * size,
            y: v / size * size,
            size,
        }
    }

    // a placed tower changes the LOD mesh where it is
    #[test]
    fn lod_mesh_follows_edits() {
        let mut planet = PlanetData::new(64);
        let (face, u, v) = crate::common::tests::first_land_column(&planet, 1);
        let key = lod_key_for(face, u, v);
        let (before, _) =
            MeshGen::generate_lod_mesh_styled(key, &planet, crate::lowpoly::TerrainStyle::Cubes);
        let h = planet.terrain.get_height(face, u, v);
        let top = (h + 4).min(planet.build_ceiling());
        assert!(
            top >= h + 2,
            "sanity check: room for a two-block tower under the ceiling"
        );
        for layer in h + 1..=top {
            planet
                .add_block(
                    crate::common::BlockId { face, layer, u, v },
                    crate::material::BlockType::Stone,
                )
                .unwrap();
        }
        let (after, _) =
            MeshGen::generate_lod_mesh_styled(key, &planet, crate::lowpoly::TerrainStyle::Cubes);
        let raised = before.iter().zip(&after).any(|(a, b)| {
            glam::Vec3::from(b.pos).length() > glam::Vec3::from(a.pos).length() + 1.0
        });
        assert!(raised, "no LOD vertex rose with the tower");
    }

    // on a liquid-less planet the filled-in basins keep their unshaded beach colour from afar, as
    // before the LOD meshes sampled the edited surface, and dry land the natural top and step blocks
    // (every vertex, rims of basins included)
    #[test]
    fn unedited_ice_lod_colours_are_unchanged() {
        // the start planet's resolution: enough relief for steep shores around the basins
        let mut planet = PlanetData::new(337);
        planet.switch_planet_type(PlanetType::Ice);
        let def = planet.planet_type.def();
        let sea = planet.terrain.sea_level();
        let (res, size, row) = (
            planet.resolution,
            planet.resolution.next_power_of_two(),
            65u32,
        );
        let mut basin_vertices = 0;
        for face in 0..6u8 {
            let key = crate::common::LodKey {
                face,
                x: 0,
                y: 0,
                size,
            };
            let (verts, _) = MeshGen::generate_lod_mesh_styled(
                key,
                &planet,
                crate::lowpoly::TerrainStyle::Cubes,
            );
            // the 65 x 65 grid (skirt vertices follow it)
            for (k, vert) in verts.iter().take((row * row) as usize).enumerate() {
                let (ux, vy) = (k as u32 % row, k as u32 / row);
                let (su, sv) = ((ux * size / 64).min(res - 1), (vy * size / 64).min(res - 1));
                let h = planet.terrain.get_height(face, su, sv);
                let expected = if h < sea {
                    basin_vertices += 1;
                    def.palette.beach.color()
                } else {
                    let pos = glam::Vec3::from(vert.pos).normalize();
                    let slope = glam::Vec3::from(vert.normal).dot(pos).abs();
                    let floor = planet.mining_floor();
                    let natural = |layer| {
                        crate::material::natural_type(
                            &planet.terrain,
                            &def.palette,
                            floor,
                            face,
                            su,
                            sv,
                            layer,
                        )
                    };
                    lod_land_color(natural(h), natural(h - 1), slope)
                };
                assert_eq!(
                    vert.color, expected,
                    "face {face} vertex {k} (column {su}, {sv})"
                );
            }
        }
        assert!(
            basin_vertices > 0,
            "sanity check: the planet has filled-in basins"
        );
    }

    // natural terrain looks exactly as before: same positions and colours
    #[test]
    fn unedited_lod_mesh_is_unchanged() {
        let planet = PlanetData::new(64);
        let key = crate::common::LodKey {
            face: 1,
            x: 0,
            y: 0,
            size: 64,
        };
        let (verts, _) =
            MeshGen::generate_lod_mesh_styled(key, &planet, crate::lowpoly::TerrainStyle::Cubes);
        let sea = planet.terrain.sea_level();
        let def = planet.planet_type.def();
        // vertex 0 samples column (0, 0)
        let h = planet.terrain.get_height(1, 0, 0);
        let expected = CoordSystem::get_vertex_pos(1, 0, 0, h.max(sea), planet.resolution);
        assert!((glam::Vec3::from(verts[0].pos) - expected).length() < 1e-4);
        if h < sea {
            let floor = planet
                .block_type(BlockId {
                    face: 1,
                    layer: h,
                    u: 0,
                    v: 0,
                })
                .unwrap();
            let liquid = def.liquid.unwrap();
            let zenith = def.atmosphere.sky_zenith;
            assert_eq!(
                verts[0].color,
                lod_water_color(floor, &liquid, (sea - h) as f32, zenith)
            );
        }
    }

    // LOD water looks like the engine's from above: the floor shows through shallow water, deep
    // water takes the deep colour
    #[test]
    fn lod_water_darkens_with_depth() {
        let liquid = PlanetType::EarthLike.def().liquid.unwrap();
        let zenith = PlanetType::EarthLike.def().atmosphere.sky_zenith;
        let shallow = lod_water_color(BlockType::Sand, &liquid, 1.0, zenith);
        let deep = lod_water_color(BlockType::Sand, &liquid, 40.0, zenith);
        let brightness = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(brightness(shallow) > brightness(deep));
        let dist = |c: [f32; 3]| Vec3::from(c).distance(Vec3::from(liquid.deep_color));
        assert!(dist(deep) < dist(shallow)); // fs_water keeps 5 % of the floor (opacity <= 0.95)
    }

    // low-poly shores: the drawn terrain crosses the water level between column centres, so the
    // water plane reaches one column onto the land beside every water column (hidden where the
    // terrain is higher); without it dry shore columns drawn below the water level were unflooded
    // dips you could look into and under the water surface
    #[test]
    fn low_poly_water_reaches_one_column_onto_the_shore() {
        use crate::lowpoly::TerrainStyle;
        let planet = PlanetData::new(64);
        let res = planet.resolution;
        let n = res.div_ceil(CHUNK_SIZE);
        let quads = |style: TerrainStyle| {
            let mut cols = std::collections::HashSet::new();
            for face in 0..6u8 {
                for u_idx in 0..n {
                    for v_idx in 0..n {
                        let (verts, _) = MeshGen::build_water_styled(
                            ChunkKey { face, u_idx, v_idx },
                            &planet,
                            style,
                        );
                        for q in verts.chunks_exact(4) {
                            let c = q.iter().map(|vx| Vec3::from_array(vx.pos)).sum::<Vec3>() / 4.0;
                            let id =
                                CoordSystem::pos_to_id(c.normalize() * (res as f32 / 2.0), res)
                                    .unwrap();
                            cols.insert((id.face, id.u, id.v));
                        }
                    }
                }
            }
            cols
        };
        let (lowpoly, cubes) = (quads(TerrainStyle::LowPoly), quads(TerrainStyle::Cubes));
        let mut dips = 0;
        for face in 0..6u8 {
            for v in 0..res {
                for u in 0..res {
                    if planet.holds_water(face, u, v) {
                        continue;
                    }
                    let level = planet.terrain.water_level(face, u, v);
                    let shore = planet
                        .neighbour_columns(face, u, v)
                        .into_iter()
                        .any(|(f, cu, cv)| planet.holds_water(f, cu, cv));
                    // dry columns get no water in cube style
                    assert!(
                        !cubes.contains(&(face, u, v)),
                        "cube water on dry ({face}, {u}, {v})"
                    );
                    if shore {
                        assert!(
                            lowpoly.contains(&(face, u, v)),
                            "dry shore ({face}, {u}, {v}) not covered"
                        );
                    }
                    // the defect: a dry column drawn below its water level must be covered
                    if planet.smooth_height(face, u, v) < level as f32 {
                        dips += 1;
                        assert!(
                            lowpoly.contains(&(face, u, v)),
                            "unflooded dip at ({face}, {u}, {v})"
                        );
                    }
                }
            }
        }
        assert!(
            dips > 0,
            "test planet has no shore dips: the check is vacuous"
        );
    }

    #[test]
    fn ice_planets_generate_no_water_mesh() {
        let mut planet = PlanetData::new(32);
        let key = chunk_with_ocean(&planet); // same chunk that has water on Earth-like
        planet.switch_planet_type(PlanetType::Ice);
        let (verts, inds) = MeshGen::build_water(key, &planet);
        assert!(verts.is_empty() && inds.is_empty());
    }
    #[test]
    fn lake_water_surfaces_sit_at_the_lake_level() {
        let (planet, (face, u, v)) = crate::common::tests::lake_planet(PlanetType::EarthLike);
        let key = ChunkKey {
            face,
            u_idx: u / CHUNK_SIZE,
            v_idx: v / CHUNK_SIZE,
        };
        let (verts, _) = MeshGen::build_water(key, &planet);
        let level = planet.terrain.water_level(face, u, v);
        let lake_r = CoordSystem::get_layer_radius(level + 1, planet.resolution);
        assert!(
            verts
                .iter()
                .any(|vx| (Vec3::from_array(vx.pos).length() - lake_r).abs() < 1e-3),
            "no water quad at the lake's level"
        );
    }
    // from afar a lake is flat water at its own level, coloured by its depth there
    #[test]
    fn lod_meshes_flatten_lakes_to_their_level() {
        let (planet, (face, u, v)) = crate::common::tests::lake_planet(PlanetType::EarthLike);
        // size 64 with the 64-cell LOD grid: one vertex per column
        let key = crate::common::LodKey {
            face,
            x: u - u % 64,
            y: v - v % 64,
            size: 64,
        };
        let (verts, _) =
            MeshGen::generate_lod_mesh_styled(key, &planet, crate::lowpoly::TerrainStyle::Cubes);
        let vx = verts[((v - key.y) * 65 + (u - key.x)) as usize];
        let level = planet.terrain.water_level(face, u, v);
        let expected = CoordSystem::get_vertex_pos(face, u, v, level, planet.resolution).length();
        assert!((Vec3::from_array(vx.pos).length() - expected).abs() < 1e-3);
        let h = planet.surface(face, u, v);
        let floor = planet
            .block_type(BlockId {
                face,
                layer: h,
                u,
                v,
            })
            .unwrap();
        let liquid = PlanetType::EarthLike.def().liquid.unwrap();
        let zenith = PlanetType::EarthLike.def().atmosphere.sky_zenith;
        let expected = lod_water_color(floor, &liquid, (level - h) as f32, zenith);
        assert_eq!(vx.color, expected);
    }
    // faces under a lake carry its surface radius; a dry rim's wall facing the lake does too
    #[test]
    fn voxel_faces_carry_the_water_radius_of_the_cell_they_face() {
        let (planet, (face, u, v)) = crate::common::tests::lake_planet(PlanetType::EarthLike);
        let lake_r = planet.water_surface_radius(face, u, v);
        let key = ChunkKey {
            face,
            u_idx: u / CHUNK_SIZE,
            v_idx: v / CHUNK_SIZE,
        };
        let (verts, _) = MeshGen::build_chunk(key, &planet);
        assert!(
            verts.iter().any(|vx| (vx.water - lake_r).abs() < 1e-3),
            "no vertex under the lake"
        );
        assert!(verts
            .iter()
            .all(|vx| vx.water == 0.0 || vx.water >= Vec3::from_array(vx.pos).length() - 1.5));
    }

    // at morph factor 1 a LOD node shows its parent's surface: even vertices sit on the parent's own
    // vertices, odd ones (moved along their own up) on the parent's triangles
    fn check_lod_morph_targets(style: crate::lowpoly::TerrainStyle) {
        let planet = PlanetData::new(256);
        let parent = crate::common::LodKey {
            face: 2,
            x: 128,
            y: 0,
            size: 128,
        };
        let key = crate::common::LodKey {
            face: 2,
            x: 192,
            y: 64,
            size: 64,
        };
        let (pv, _) = MeshGen::generate_lod_mesh_styled(parent, &planet, style);
        let (verts, _) = MeshGen::generate_lod_mesh_styled(key, &planet, style);
        let morph = MeshGen::generate_lod_morph_styled(key, &planet, 256, style);
        assert_eq!(morph.len(), verts.len());
        let grid = MeshGen::LOD_GRID;
        let pos =
            |v: &[Vertex], x: u32, y: u32| Vec3::from_array(v[(y * (grid + 1) + x) as usize].pos);
        let step = pos(&pv, 1, 0).distance(pos(&pv, 0, 0));
        for y in 0..=grid {
            for x in 0..=grid {
                let own = pos(&verts, x, y);
                let m = morph[(y * (grid + 1) + x) as usize];
                let morphed = own + own.normalize() * m.height;
                let (hx, hy) = (x + grid, y + grid); // this node is the parent's bottom-right quadrant
                let p = |hx: u32, hy: u32| pos(&pv, hx / 2, hy / 2);
                let expected = match (hx % 2, hy % 2) {
                    (0, 0) => p(hx, hy),
                    (1, 0) => (p(hx - 1, hy) + p(hx + 1, hy)) * 0.5,
                    (0, _) => (p(hx, hy - 1) + p(hx, hy + 1)) * 0.5,
                    _ => (p(hx + 1, hy - 1) + p(hx - 1, hy + 1)) * 0.5,
                };
                assert!(
                    morphed.distance(expected) < 0.02 * step,
                    "({x}, {y}): {morphed} vs {expected}"
                );
                if hx % 2 == 0 && hy % 2 == 0 {
                    assert_eq!(m.height, 0.0);
                }
            }
        }
    }

    #[test]
    fn lod_morph_targets_lie_on_the_parent_mesh() {
        check_lod_morph_targets(crate::lowpoly::TerrainStyle::Cubes);
    }

    // geomorphing keeps working on the smooth low-poly heights
    #[test]
    fn lod_morph_targets_lie_on_the_parent_mesh_lowpoly() {
        check_lod_morph_targets(crate::lowpoly::TerrainStyle::LowPoly);
    }

    // a root node has no parent: it morphs to itself
    #[test]
    fn root_lod_nodes_morph_to_themselves() {
        let planet = PlanetData::new(64);
        let key = crate::common::LodKey {
            face: 0,
            x: 0,
            y: 0,
            size: 64,
        };
        assert!(MeshGen::generate_lod_morph_styled(
            key,
            &planet,
            64,
            crate::lowpoly::TerrainStyle::Cubes
        )
        .iter()
        .all(|m| m.height == 0.0));
    }
}

// the cube style's meshes, pinned while hex columns (hex.rs) were added beside them: a fingerprint of
// the chunk and water meshes of a few chunks (a face corner, an interior one, an edited one, one with
// water). build_chunk_cubes collects its blocks in a HashSet, so the fingerprint is order-independent:
// a wrapping sum of per-triangle hashes
#[cfg(test)]
mod cube_fingerprint {
    use super::*;
    use crate::common::{ChunkKey, PlanetData};

    // every cube block face is bevelled: each vertex lies on two of its face's edges
    #[test]
    fn cube_faces_carry_edge_distances() {
        let planet = PlanetData::new(32);
        let key = ChunkKey {
            face: 0,
            u_idx: 0,
            v_idx: 0,
        };
        let (verts, _) = MeshGen::build_chunk_cubes(key, &planet);
        assert!(!verts.is_empty());
        for v in &verts {
            let e = crate::bevel::unpack(v.edge);
            let on_edge = e.iter().filter(|d| d.abs() < 1e-3).count();
            let far = e.iter().filter(|&&d| d > 0.5 && d < 2.0).count();
            assert_eq!((on_edge, far), (2, 2), "vertex {e:?}");
        }
    }

    fn triangle_sum(verts: &[Vertex], inds: &[u32]) -> (usize, u64) {
        let mut sum = 0u64;
        for tri in inds.chunks_exact(3) {
            // FNV-1a over the triangle's vertex bits
            let mut h = 0xcbf29ce484222325u64;
            for &i in tri {
                let v = verts[i as usize];
                for f in v
                    .pos
                    .iter()
                    .chain(&v.color)
                    .chain(&v.normal)
                    .chain([&v.water])
                {
                    for b in f.to_bits().to_le_bytes() {
                        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                    }
                }
            }
            sum = sum.wrapping_add(h);
        }
        (inds.len() / 3, sum)
    }

    pub(crate) fn fingerprint(planet: &PlanetData) -> Vec<(usize, u64)> {
        let keys = [
            ChunkKey {
                face: 0,
                u_idx: 0,
                v_idx: 0,
            },
            ChunkKey {
                face: 2,
                u_idx: 1,
                v_idx: 1,
            },
            ChunkKey {
                face: 4,
                u_idx: 1,
                v_idx: 0,
            },
            ChunkKey {
                face: 5,
                u_idx: 0,
                v_idx: 1,
            },
        ];
        let mut out = Vec::new();
        for key in keys {
            let (v, i) = MeshGen::build_chunk_cubes(key, planet);
            out.push(triangle_sum(&v, &i));
            let (v, i) =
                MeshGen::build_water_styled(key, planet, crate::lowpoly::TerrainStyle::Cubes);
            out.push(triangle_sum(&v, &i));
        }
        out
    }

    pub(crate) fn edited_planet() -> PlanetData {
        let mut planet = PlanetData::new(64);
        // dig a pit and build a small tower in chunk (4, 1, 0), and dig at the face edge of (0, 0, 0)
        for (face, u, v) in [(4u8, 40u32, 10u32), (4, 41, 10), (0, 0, 5)] {
            let h = planet.surface(face, u, v);
            for layer in h.saturating_sub(3)..=h {
                let _ = planet.remove_block(BlockId { face, layer, u, v });
            }
        }
        let h = planet.surface(4, 45, 12);
        for layer in h + 1..h + 4 {
            let _ = planet.add_block(
                BlockId {
                    face: 4,
                    layer,
                    u: 45,
                    v: 12,
                },
                crate::material::BlockType::Stone,
            );
        }
        planet
    }

    #[test]
    fn cube_meshes_are_unchanged() {
        let got = fingerprint(&edited_planet());
        let want = vec![
            (3276, 10882691970822256782),
            (458, 2367537745188859897),
            (2982, 3011035848723356312),
            (364, 12695061272210051173),
            (3066, 62841614696602891),
            (742, 14632956052046115683),
            (2992, 2153993205474278395),
            (1158, 8045856442357342524),
        ];
        assert_eq!(got, want);
    }
}

// hex columns (hex.rs): the voxel meshes close up, inside faces, across cube-face seams and around edits
#[cfg(test)]
mod hex_mesh_tests {
    use super::*;
    use crate::common::{CellShape, ChunkKey, PlanetData};
    use std::collections::HashMap;

    // hex block faces are bevelled: no vertex is unbevelled or negative, and every triangle has at least
    // two vertices on one of its edges (fan corners, wall corners); res 33 has seam cells with pieces
    #[test]
    fn hex_mesh_bevel_distances() {
        for res in [32u32, 33] {
            let mut planet = PlanetData::new(res);
            planet.cells = CellShape::Hex;
            for key in all_keys(&planet) {
                let (verts, inds) = MeshGen::build_chunk_hex(key, &planet);
                let edges: Vec<_> = verts.iter().map(|v| crate::bevel::unpack(v.edge)).collect();
                for e in &edges {
                    let min = e.iter().cloned().fold(f32::MAX, f32::min);
                    assert!(min > -1e-4 && min < crate::bevel::NO_EDGE, "{e:?}");
                }
                for tri in inds.chunks_exact(3) {
                    let on = tri
                        .iter()
                        .filter(|&&i| edges[i as usize].iter().any(|d| d.abs() < 1e-3))
                        .count();
                    assert!(on >= 2, "triangle {tri:?}");
                }
            }
        }
    }

    fn all_keys(planet: &PlanetData) -> Vec<ChunkKey> {
        let n = planet.resolution.div_ceil(CHUNK_SIZE);
        (0..6u8)
            .flat_map(|face| {
                (0..n)
                    .flat_map(move |u_idx| (0..n).map(move |v_idx| ChunkKey { face, u_idx, v_idx }))
            })
            .collect()
    }

    // how many triangles use each (undirected) edge of the whole planet's voxel mesh
    fn edge_uses(planet: &PlanetData) -> HashMap<([i64; 3], [i64; 3]), u32> {
        let q = |p: [f32; 3]| p.map(|c| (c as f64 * 4096.0).round() as i64);
        let mut uses = HashMap::new();
        for key in all_keys(planet) {
            let (verts, inds) = MeshGen::build_chunk_hex(key, planet);
            for tri in inds.chunks_exact(3) {
                for k in 0..3 {
                    let (a, b) = (
                        q(verts[tri[k] as usize].pos),
                        q(verts[tri[(k + 1) % 3] as usize].pos),
                    );
                    if a != b {
                        *uses.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
                    }
                }
            }
        }
        uses
    }

    // closed: every edge is used by an even number of triangles — two, or four where a seam point has
    // two cells on each face and they alternate high and low around it (like cubes at a diagonal)
    fn assert_closed(planet: &PlanetData) {
        let uses = edge_uses(planet);
        assert!(!uses.is_empty());
        let open: Vec<_> = uses.iter().filter(|(_, &n)| n % 2 != 0).take(5).collect();
        assert!(
            open.is_empty(),
            "{} of {} edges open, e.g. {open:?}",
            uses.values().filter(|&&n| n % 2 != 0).count(),
            uses.len()
        );
    }

    #[test]
    fn hex_planet_mesh_is_closed() {
        for res in [32u32, 33] {
            let mut planet = PlanetData::new(res);
            planet.cells = CellShape::Hex;
            assert_closed(&planet);
        }
    }

    #[test]
    fn edited_hex_planet_mesh_is_closed() {
        let mut planet = super::cube_fingerprint::edited_planet();
        planet.cells = CellShape::Hex;
        // a pit and a tower on a cube-face seam too
        for (face, u, v) in [(2u8, 63u32, 30u32), (2, 0, 7)] {
            let h = planet.surface(face, u, v);
            let _ = planet.remove_block(BlockId {
                face,
                layer: h,
                u,
                v,
            });
            let _ = planet.add_block(
                BlockId {
                    face,
                    layer: h + 3,
                    u: (u + 1).min(63),
                    v,
                },
                BlockType::Stone,
            );
        }
        assert_closed(&planet);
    }

    // the hex style's water: one fan per wet column, centred over that column's own cell
    #[test]
    fn hex_water_fans_sit_over_their_cells() {
        let mut planet = PlanetData::new(32);
        planet.cells = CellShape::Hex;
        let mut fans = 0;
        for key in all_keys(&planet) {
            let (verts, inds) =
                MeshGen::build_water_styled(key, &planet, crate::lowpoly::TerrainStyle::Hex);
            // each fan starts with its centre and is followed by its corners
            let mut i = 0;
            while i < inds.len() {
                let centre = inds[i] as usize;
                let corners = inds[i..]
                    .chunks_exact(3)
                    .take_while(|t| t[0] as usize == centre)
                    .count();
                let p = Vec3::from_array(verts[centre].pos);
                let id = planet.cell_at(p * 0.999).unwrap();
                assert!(planet.holds_water(id.face, id.u, id.v));
                fans += 1;
                i += corners * 3;
            }
        }
        assert!(fans > 0);
    }
}
