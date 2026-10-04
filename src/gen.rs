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

    pub fn build_chunk(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
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

    // generates a simplified heightmap mesh for distant terrain
    // the water surface of a chunk: one quad at the top of layer sea level (flush with sea-level beaches)
    // over every column whose sea-level cell holds water (PlanetData::sea_cell_is_water: natural ocean
    // and holes dug below sea level alike) and whose cell above is open — under a ceiling (a tunnel dug at
    // sea level) the surface would lie against the solid face above and flicker; that water has none.
    pub fn build_water(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
        if data.planet_type.def().liquid.is_none() {
            return (Vec::new(), Vec::new());
        }
        let (mut verts, mut inds, mut idx) = (Vec::new(), Vec::new(), 0u32);
        let res = data.resolution;
        let sea = data.terrain.sea_level();
        let u_start = key.u_idx * CHUNK_SIZE;
        let v_start = key.v_idx * CHUNK_SIZE;
        for u in u_start..(u_start + CHUNK_SIZE).min(res) {
            for v in v_start..(v_start + CHUNK_SIZE).min(res) {
                let above = BlockId {
                    face: key.face,
                    layer: sea + 1,
                    u,
                    v,
                };
                if !data.sea_cell_is_water(key.face, u, v) || data.exists(above) {
                    continue;
                }
                let p =
                    |du, dv| CoordSystem::get_vertex_pos(key.face, u + du, v + dv, sea + 1, res);
                let corners = [p(0, 0), p(1, 0), p(1, 1), p(0, 1)];
                // liquid is Some here (checked above); color is unused by fs_water (which shades
                // purely from the biome uniform) but should still match the active planet type
                let color = data.planet_type.def().liquid.unwrap().shallow_color;
                for c in corners {
                    verts.push(Vertex {
                        pos: c.to_array(),
                        color,
                        normal: c.normalize().to_array(),
                    });
                }
                inds.extend_from_slice(&[idx, idx + 1, idx + 2, idx + 2, idx + 3, idx]);
                idx += 4;
            }
        }
        (verts, inds)
    }

    pub fn generate_lod_mesh(
        key: crate::common::LodKey,
        data: &PlanetData,
    ) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        let def = data.planet_type.def();

        let grid_res = 64;
        let row_len = grid_res + 1;

        // calculate global pos for any grid index (even outside this chunk)
        // this allows us to "peek" into neighbor chunks for perfect normals.
        let get_sample_pos = |gx: i32, gy: i32| -> glam::Vec3 {
            let step_u = (gx as i64 * key.size as i64) / grid_res as i64;
            let step_v = (gy as i64 * key.size as i64) / grid_res as i64;

            // calculate absolute U/V
            let abs_u = (key.x as i64 + step_u).clamp(0, data.resolution as i64) as u32;
            let abs_v = (key.y as i64 + step_v).clamp(0, data.resolution as i64) as u32;

            // the edited surface (natural height where unedited, LOD surfaces sit at layer h like the
            // land); oceans and liquid-less basins are flat at sea level from afar, like the water table
            let h = data
                .surface(key.face, abs_u, abs_v)
                .max(data.terrain.sea_level());
            CoordSystem::get_vertex_pos(key.face, abs_u, abs_v, h, data.resolution)
        };

        // 1. Generate Vertices
        for vy in 0..=grid_res {
            for ux in 0..=grid_res {
                let pos = get_sample_pos(ux as i32, vy as i32);

                // seamless normal fix
                // instead of clamping to grid edges, we look -1 and +1 in global grid Space
                // this ensures the normal at the chunk edge matches the neighbor's normal perfectly

                let p_right = get_sample_pos(ux as i32 + 1, vy as i32);
                let p_left = get_sample_pos(ux as i32 - 1, vy as i32);
                let p_down = get_sample_pos(ux as i32, vy as i32 + 1);
                let p_up = get_sample_pos(ux as i32, vy as i32 - 1);

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
                let offset_u = (ux * key.size) / grid_res;
                let offset_v = (vy * key.size) / grid_res;
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
                let shade = if slope < 0.85 { 0.75 } else { 1.0 }; // steep parts read like voxel sides
                let color = if h < data.terrain.sea_level() {
                    match def.liquid {
                        // distant water: the planet type's own shallow liquid color, not a
                        // hardcoded Earth-specific one
                        Some(liquid) => liquid.shallow_color,
                        // liquid-less planet (e.g. Ice): the LOD surface here is really the
                        // filled-in beach material (PlanetData::exists's liquid-less solidity rule),
                        // not water, so color it as such instead
                        None => def.palette.beach.color(),
                    }
                } else {
                    surface.color().map(|c| c * shade)
                };

                verts.push(Vertex {
                    pos: pos.to_array(),
                    color,
                    normal: normal.to_array(),
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
        let res = data.resolution;

        // neighbor existence check
        let check = |d_face: u8, d_layer: i32, d_u: i32, d_v: i32| -> bool {
            let l = id.layer as i32 + d_layer;
            let u = id.u as i32 + d_u;
            let v = id.v as i32 + d_v;
            if l >= 0 && u >= 0 && u < res as i32 && v >= 0 && v < res as i32 {
                return data.exists(BlockId {
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
            });

            verts.push(Vertex {
                pos: [x, height, z],
                color,
                normal,
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
        });
        for i in 0..=segments {
            let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            verts.push(Vertex {
                pos: [x, height, z],
                color,
                normal: [0.0, 1.0, 0.0],
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
            },
            Vertex {
                pos: [s, 0.0, 0.0],
                color,
                normal,
            },
            Vertex {
                pos: [0.0, -s, 0.0],
                color,
                normal,
            },
            Vertex {
                pos: [0.0, s, 0.0],
                color,
                normal,
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

        for i in 0..4 {
            verts.push(Vertex {
                pos: pos[i].to_array(),
                color: colors[i],
                normal,
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
        MeshGen::build_water(key, planet).0.len() / 4
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
        let (before, _) = MeshGen::generate_lod_mesh(key, &planet);
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
        let (after, _) = MeshGen::generate_lod_mesh(key, &planet);
        let raised = before.iter().zip(&after).any(|(a, b)| {
            glam::Vec3::from(b.pos).length() > glam::Vec3::from(a.pos).length() + 1.0
        });
        assert!(raised, "no LOD vertex rose with the tower");
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
        let (verts, _) = MeshGen::generate_lod_mesh(key, &planet);
        let sea = planet.terrain.sea_level();
        let def = planet.planet_type.def();
        // vertex 0 samples column (0, 0)
        let h = planet.terrain.get_height(1, 0, 0);
        let expected = CoordSystem::get_vertex_pos(1, 0, 0, h.max(sea), planet.resolution);
        assert!((glam::Vec3::from(verts[0].pos) - expected).length() < 1e-4);
        if h < sea {
            assert_eq!(verts[0].color, def.liquid.unwrap().shallow_color);
        }
    }

    #[test]
    fn ice_planets_generate_no_water_mesh() {
        let mut planet = PlanetData::new(32);
        let key = chunk_with_ocean(&planet); // same chunk that has water on Earth-like
        planet.switch_planet_type(PlanetType::Ice);
        let (verts, inds) = MeshGen::build_water(key, &planet);
        assert!(verts.is_empty() && inds.is_empty());
    }
}
