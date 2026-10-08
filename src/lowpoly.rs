// lowpoly.rs
// Low-poly terrain: marching cubes over a smoothed height field (PlanetData::smooth), flat colour per
// facet, normals from screen-space derivatives. Blocks stay the game's data; this only draws them.

use crate::common::{BlockId, ChunkKey, PlanetData, Vertex, CHUNK_SIZE};
use crate::gen::CoordSystem;
use crate::mc_tables::{CUBE_CORNER_OFFSETS, EDGE_VERTEX_PAIRS, TRI_TABLE};
use glam::Vec3;
use std::sync::atomic::{AtomicU8, Ordering};

// how terrain is drawn: low-poly (the game's look), the original cubes (development comparison) or hex
// columns (hex.rs; the planet's cells are hexagonal then, PlanetData::cells), console /terrain_style.
// Read once per mesh build at the entry points (MeshGen::build_chunk, build_water, generate_lod_mesh,
// generate_lod_morph, via style_for) and passed down, so tests can pick a style explicitly
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainStyle {
    Cubes,
    LowPoly,
    Hex,
}

static STYLE: AtomicU8 = AtomicU8::new(1);

pub fn style() -> TerrainStyle {
    match STYLE.load(Ordering::Relaxed) {
        0 => TerrainStyle::Cubes,
        2 => TerrainStyle::Hex,
        _ => TerrainStyle::LowPoly,
    }
}

pub fn set_style(s: TerrainStyle) {
    let v = match s {
        TerrainStyle::Cubes => 0,
        TerrainStyle::LowPoly => 1,
        TerrainStyle::Hex => 2,
    };
    STYLE.store(v, Ordering::Relaxed);
}

// the style `data` is meshed in: hex whenever its cells are hexagonal (the geometry is the planet's, so
// meshes and gameplay can't disagree), else the global look
pub fn style_for(data: &PlanetData) -> TerrainStyle {
    match data.cells {
        crate::common::CellShape::Hex => TerrainStyle::Hex,
        crate::common::CellShape::Square => match style() {
            TerrainStyle::Hex => TerrainStyle::Cubes,
            s => s,
        },
    }
}

// the cell shape a style plays with
pub fn cells_for(s: TerrainStyle) -> crate::common::CellShape {
    match s {
        TerrainStyle::Hex => crate::common::CellShape::Hex,
        _ => crate::common::CellShape::Square,
    }
}

// a cell is solid above this density
pub const ISO: f32 = 0.5;

// the density of a natural (unmined) cell at `layer` in a column of smooth height `smooth_height`:
// linear, falling by one per layer and unclamped, crossing ISO at layer coordinate smooth_height + 1
// over the column centre — so flat ground (an integer smooth height) lies exactly on the block tops, and
// between two columns the surface interpolates their heights however steep the slope (a 0..1 clamp
// saturated where neighbours differ by a layer or more and pinned those crossings to the midpoint:
// stair steps with near-vertical risers)
pub fn natural_density(smooth_height: f32, layer: u32) -> f32 {
    smooth_height - layer as f32 + 1.0
}

// where the surface crosses a column of smooth height `smooth_height`: the solid cell's layer and the
// fraction (0..=1) of the way to the centre of the empty cell above it
pub fn crossing(smooth_height: f32) -> (u32, f32) {
    let h = smooth_height.max(0.0);
    // the highest solid layer (density > ISO, i.e. layer < h + 1 - ISO); the densities of it and the
    // layer above differ by exactly 1
    let layer = ((h + 1.0 - ISO).ceil() - 1.0).max(0.0);
    (layer as u32, h - layer + 1.0 - ISO)
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

// the density of a mined cell: anything below ISO (only its sign is used, see Field::density)
const MINED_DENSITY: f32 = 0.0;

// a facet whose open side has solid cells within this many layers above is darkened (caves, tunnels,
// overhangs, under builds) by COVER_DARKEN, like the cubes' sky term
const COVER_LAYERS: u32 = 8;
const COVER_DARKEN: f32 = 0.2;

// a voxel chunk's low-poly terrain: marching cubes over the dual grid of block centres (cells of 2×2×2
// centres) on the density natural_density(smooth height) with mined cells empty, one flat-coloured facet
// per triangle (three unshared vertices each), plus the chunk's placed blocks as cubes. A chunk owns the
// cells whose smallest corner column (face, u, v) is one of its own; the other corners come from the
// neighbouring chunks' or faces' columns (neighbor_column)
pub fn build_chunk_lowpoly(key: ChunkKey, data: &PlanetData) -> (Vec<Vertex>, Vec<u32>) {
    let res = data.resolution;
    let u_start = key.u_idx * CHUNK_SIZE;
    let v_start = key.v_idx * CHUNK_SIZE;
    let u_end = (u_start + CHUNK_SIZE).min(res);
    let v_end = (v_start + CHUNK_SIZE).min(res);
    let field = Field::new(data);

    // tunnels below the natural surface need cells down there: the deepest mined cell of this chunk,
    // its same-face neighbours and, for chunks on a face edge, the other faces
    let n = res.div_ceil(CHUNK_SIZE);
    let on_face_edge = key.u_idx == 0 || key.v_idx == 0 || key.u_idx + 1 == n || key.v_idx + 1 == n;
    let near = |k: &ChunkKey| {
        (k.face == key.face && k.u_idx.abs_diff(key.u_idx) <= 1 && k.v_idx.abs_diff(key.v_idx) <= 1)
            || (on_face_edge && k.face != key.face)
    };
    let deepest_mined = data
        .edits
        .chunks
        .iter()
        .filter(|(k, _)| near(k))
        .flat_map(|(_, m)| m.mined.iter().map(|id| id.layer))
        .min()
        .unwrap_or(u32::MAX);

    let mut verts = Vec::new();
    for u in u_start..u_end {
        for v in v_start..v_end {
            // each of the four cells around the column, emitted by its smallest column, so a cell
            // across a cube-face edge is built once whichever side of the edge its owner lies on
            for (su, sv) in [(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)] {
                let column = |du: i32, dv: i32| data.neighbor_column(key.face, u, v, du, dv);
                let (Some(c1), Some(c3)) = (column(su, 0), column(0, sv)) else {
                    continue;
                };
                // the fourth column: across a cube-face edge neighbor_column's diagonal is unreliable at
                // face-corner columns, so step on from the column that stays on the face. At a three-face
                // cube corner there is no fourth column: the cell closes on c1 twice
                let own = (key.face, u, v);
                let c2 = match (c1.0 == own.0, c3.0 == own.0) {
                    (true, true) => column(su, sv),
                    (false, true) => data.neighbor_column(c3.0, c3.1, c3.2, su, 0),
                    (true, false) => data.neighbor_column(c1.0, c1.1, c1.2, 0, sv),
                    (false, false) => None,
                }
                .filter(|c| *c != c1 && *c != c3)
                .unwrap_or(c1);
                if [c1, c2, c3].iter().any(|c| *c < own) {
                    continue;
                }
                let cols = [own, c1, c2, c3];
                let heights = cols.map(|(f, cu, cv)| data.smooth_height(f, cu, cv));
                let lo = heights.iter().copied().fold(f32::MAX, f32::min).floor() as u32;
                let hi = heights.iter().copied().fold(f32::MIN, f32::max).ceil() as u32 + 2;
                for layer in lo.min(deepest_mined).saturating_sub(1)..hi {
                    cell(data, &field, &cols, layer, &mut verts);
                }
            }
        }
    }
    let mut inds: Vec<u32> = (0..verts.len() as u32).collect();

    // placed blocks: cubes, faces culled only against other placed blocks
    if let Some(mods) = data.edits.chunks.get(&key) {
        let mut idx = verts.len() as u32;
        for &id in mods.placed.keys() {
            crate::gen::MeshGen::add_voxel_with(
                id,
                data,
                &mut verts,
                &mut inds,
                &mut idx,
                &|b| field.placed(b),
                true,
            );
        }
    }
    (verts, inds)
}

// the drawn terrain: densities from the smooth heights, carved cells (see carved) empty
struct Field<'a> {
    data: &'a PlanetData,
}

impl<'a> Field<'a> {
    fn new(data: &'a PlanetData) -> Self {
        Self { data }
    }

    fn mods(&self, id: BlockId) -> Option<&crate::common::ChunkMods> {
        self.data.edits.chunks.get(&PlanetData::chunk_key(id))
    }

    fn placed(&self, id: BlockId) -> bool {
        self.mods(id).is_some_and(|m| m.placed.contains_key(&id))
    }

    // a cell carved out of the drawn terrain: mined, or natural air above a mined column top. Over a dip
    // the smooth surface lies above the column's top block (smooth_height > h + 0.5 draws the air cell
    // h + 1 solid), and air can't be mined, so without the second rule mining the top block would leave
    // a lid the player can never remove
    fn carved(&self, id: BlockId) -> bool {
        let Some(m) = self.mods(id) else {
            return false;
        };
        if m.mined.contains(&id) {
            return true;
        }
        let top = self.data.effective_height(id.face, id.u, id.v);
        id.layer > top && m.mined.contains(&BlockId { layer: top, ..id })
    }

    // the cell's density and whether it is carved: carved cells are empty (MINED_DENSITY) and cell()
    // puts the vertices on their edges half-way, at the block boundary, since the unclamped natural
    // density says nothing about where a wall should be
    fn sample(&self, id: BlockId) -> (f32, bool) {
        if self.carved(id) {
            (MINED_DENSITY, true)
        } else {
            (
                natural_density(self.data.smooth_height(id.face, id.u, id.v), id.layer),
                false,
            )
        }
    }

    fn density(&self, id: BlockId) -> f32 {
        self.sample(id).0
    }

    // solid as drawn: the smooth terrain or a placed block
    fn drawn_solid(&self, id: BlockId) -> bool {
        self.placed(id) || self.density(id) > ISO
    }
}

// one marching-cubes cell: corners are the block centres of `cols` (CUBE_CORNER_OFFSETS' (x, y) order:
// (0,0), (1,0), (1,1), (0,1)) at layers `layer` and `layer + 1`
fn cell(
    data: &PlanetData,
    field: &Field,
    cols: &[(u8, u32, u32); 4],
    layer: u32,
    verts: &mut Vec<Vertex>,
) {
    let res = data.resolution;
    let mut ids = [BlockId {
        face: 0,
        layer: 0,
        u: 0,
        v: 0,
    }; 8];
    let mut d = [0.0f32; 8];
    let mut mined = [false; 8]; // carved: mined, or air above a mined column top
    let mut case = 0usize;
    for (i, &(dx, dy, dz)) in CUBE_CORNER_OFFSETS.iter().enumerate() {
        let (face, u, v) = match (dx, dy) {
            (0, 0) => cols[0],
            (1, 0) => cols[1],
            (1, 1) => cols[2],
            _ => cols[3],
        };
        ids[i] = BlockId {
            face,
            u,
            v,
            layer: layer + dz,
        };
        (d[i], mined[i]) = field.sample(ids[i]);
        if d[i] > ISO {
            case |= 1 << i;
        }
    }
    if case == 0 || case == 255 {
        return;
    }
    let def = data.planet_type.def();
    let mut emitted: Vec<[[u32; 3]; 3]> = Vec::new();
    let centre = |id: BlockId| CoordSystem::get_block_center(id.face, id.u, id.v, id.layer, res);
    for tri in TRI_TABLE[case].chunks(3).take_while(|t| t[0] >= 0) {
        let mut p = [Vec3::ZERO; 3];
        let mut outward = Vec3::ZERO;
        let mut top_solid: Option<BlockId> = None;
        let mut open = ids[0];
        let mut wall = false; // touches a mined cell: a pit or tunnel wall
        for (k, &e) in tri.iter().enumerate() {
            let (a, b) = EDGE_VERTEX_PAIRS[e as usize];
            let (s, o) = if case >> a & 1 == 1 { (a, b) } else { (b, a) };
            let (ps, po) = (centre(ids[s]), centre(ids[o]));
            // into a mined cell: half-way, the block boundary (pit and tunnel walls)
            let t = if mined[o] {
                0.5
            } else {
                ((d[s] - ISO) / (d[s] - d[o]).max(1e-6)).clamp(0.0, 1.0)
            };
            p[k] = ps + (po - ps) * t;
            outward += po - ps;
            if top_solid.is_none_or(|ts| ids[s].layer > ts.layer) {
                top_solid = Some(ids[s]);
            }
            wall |= mined[o];
            open = ids[o];
        }
        // a cell at a cube corner repeats a column, so some of its triangles collapse or double up
        if p[0] == p[1] || p[1] == p[2] || p[0] == p[2] {
            continue;
        }
        let mut key = p.map(|q| q.to_array().map(f32::to_bits));
        key.sort();
        if emitted.contains(&key) {
            continue;
        }
        emitted.push(key);
        let mut n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        if n.dot(outward) < 0.0 {
            n = -n;
            p.swap(1, 2);
        }
        let centroid = (p[0] + p[1] + p[2]) / 3.0;
        let solid = top_solid.unwrap_or(ids[0]);
        let ty = if wall { data.block_type(solid) } else { None }.unwrap_or_else(|| {
            let col = CoordSystem::pos_to_id(centroid, res).unwrap_or(solid);
            let height = CoordSystem::layer_of_radius(centroid.length(), res) - 1.0;
            crate::material::lowpoly_material(
                &data.terrain,
                &def.palette,
                col.face,
                col.u,
                col.v,
                height,
                n.dot(centroid.normalize()),
            )
        });
        let covered = (1..=COVER_LAYERS).any(|i| {
            field.drawn_solid(BlockId {
                layer: open.layer + i,
                ..open
            })
        });
        let shade = if covered { COVER_DARKEN } else { 1.0 };
        let color = ty.color().map(|c| c * shade);
        let water = data.water_surface_radius(open.face, open.u, open.v);
        for q in p {
            verts.push(Vertex {
                pos: q.to_array(),
                color,
                normal: n.to_array(),
                water,
                edge: Vertex::NO_EDGE,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::Vertex;

    // terrain facets stay unbevelled (only placed cubes, drawn by add_voxel_with, get edges)
    #[test]
    fn lowpoly_facets_carry_no_edge() {
        let planet = crate::common::PlanetData::new(32);
        let key = crate::common::ChunkKey {
            face: 0,
            u_idx: 0,
            v_idx: 0,
        };
        let (verts, _) = build_chunk_lowpoly(key, &planet);
        assert!(!verts.is_empty());
        assert!(verts.iter().all(|v| v.edge == Vertex::NO_EDGE));
    }
    use crate::material::BlockType;

    // every chunk of the planet, meshed low-poly
    fn mesh_planet(planet: &PlanetData) -> Vec<(Vec<Vertex>, Vec<u32>)> {
        let n = planet.resolution.div_ceil(CHUNK_SIZE);
        let mut out = Vec::new();
        for face in 0..6u8 {
            for u_idx in 0..n {
                for v_idx in 0..n {
                    out.push(build_chunk_lowpoly(ChunkKey { face, u_idx, v_idx }, planet));
                }
            }
        }
        out
    }

    // triangles bucketed by the columns their vertices lie over, so a ray only tests its own and the
    // neighbouring columns' triangles
    pub(crate) struct MeshIndex {
        res: u32,
        buckets: std::collections::HashMap<(u8, u32, u32), Vec<[Vec3; 3]>>,
    }

    impl MeshIndex {
        pub(crate) fn new(meshes: &[(Vec<Vertex>, Vec<u32>)], res: u32) -> Self {
            let mut buckets: std::collections::HashMap<(u8, u32, u32), Vec<[Vec3; 3]>> =
                Default::default();
            for (verts, inds) in meshes {
                for t in inds.chunks_exact(3) {
                    let tri = [0, 1, 2].map(|k| Vec3::from_array(verts[t[k] as usize].pos));
                    let mut cols: Vec<(u8, u32, u32)> = tri
                        .iter()
                        .filter_map(|p| {
                            CoordSystem::pos_to_id(p.normalize() * (res as f32 / 2.0), res)
                        })
                        .map(|id| (id.face, id.u, id.v))
                        .collect();
                    cols.dedup();
                    for c in cols {
                        buckets.entry(c).or_default().push(tri);
                    }
                }
            }
            Self { res, buckets }
        }

        // the radius of the outermost surface along `dir`, casting inward from radius `from`
        // (Möller–Trumbore in f64 over the triangles of the ray's column and its eight neighbours: probes
        // between column centres run exactly along triangle edges, which f32 rounding at radius `from`
        // would miss at random)
        pub(crate) fn hit(&self, dir: Vec3, from: f32) -> Option<f32> {
            let id = CoordSystem::pos_to_id(dir * (self.res as f32 / 2.0), self.res)?;
            let dir = dir.as_dvec3();
            let (origin, ray) = (dir * from as f64, -dir);
            let mut best: Option<f32> = None;
            let mut seen = std::collections::HashSet::new();
            for dv in -1..=1 {
                for du in -1..=1 {
                    let Some(c) =
                        CoordSystem::neighbor_column(id.face, id.u, id.v, du, dv, self.res)
                    else {
                        continue;
                    };
                    if !seen.insert(c) {
                        continue;
                    }
                    for [a, b, c] in self.buckets.get(&c).into_iter().flatten() {
                        let (a, b, c) = (a.as_dvec3(), b.as_dvec3(), c.as_dvec3());
                        let (e1, e2) = (b - a, c - a);
                        let p = ray.cross(e2);
                        let det = e1.dot(p);
                        if det.abs() < 1e-9 {
                            continue;
                        }
                        let s = origin - a;
                        let bu = s.dot(p) / det;
                        let q = s.cross(e1);
                        let bv = ray.dot(q) / det;
                        if bu < -1e-5 || bv < -1e-5 || bu + bv > 1.0 + 1e-5 {
                            continue;
                        }
                        let t = e2.dot(q) / det;
                        if t > 0.0 {
                            let r = from - t as f32;
                            best = Some(best.map_or(r, |x: f32| x.max(r)));
                        }
                    }
                }
            }
            best
        }
    }

    // probe directions: points between neighbouring column centres (also across chunk borders and
    // cube-face edges) and column centres nudged off the vertical edges
    fn probes(planet: &PlanetData) -> Vec<Vec3> {
        let res = planet.resolution;
        let mut out = Vec::new();
        for face in 0..6u8 {
            for v in 0..res {
                for u in 0..res {
                    let c = CoordSystem::get_block_center(face, u, v, res / 2, res).normalize();
                    for (du, dv) in [(1, 0), (0, 1), (1, 1)] {
                        if let Some((f, nu, nv)) = planet.neighbor_column(face, u, v, du, dv) {
                            let n =
                                CoordSystem::get_block_center(f, nu, nv, res / 2, res).normalize();
                            out.push((c + n).normalize());
                            out.push((c * 0.7 + n * 0.3).normalize());
                        }
                    }
                }
            }
        }
        out
    }

    // no holes: every probe ray from outside hits the terrain, at chunk borders, cube-face edges and the
    // three-face corners alike
    #[test]
    fn low_poly_terrain_is_watertight() {
        for planet in [
            PlanetData::new(64), // 2×2 chunks per face: chunk borders and face edges
            PlanetData::new_for_type(32, crate::noise::HOME_SEED, crate::biome::PlanetType::Ice),
        ] {
            let index = MeshIndex::new(&mesh_planet(&planet), planet.resolution);
            let from = planet.resolution as f32 * 2.0;
            for dir in probes(&planet) {
                assert!(
                    index.hit(dir, from).is_some(),
                    "hole at {dir:?} (res {})",
                    planet.resolution
                );
            }
        }
    }

    // over a column centre the surface is exactly at surface_radius of the column's smooth height
    #[test]
    fn surface_sits_at_the_smooth_height() {
        let planet = PlanetData::new(32);
        let index = MeshIndex::new(&mesh_planet(&planet), 32);
        for (u, v) in [(5u32, 5u32), (10, 20), (16, 16), (25, 7)] {
            let dir = CoordSystem::get_block_center(0, u, v, 16, 32).normalize();
            let r = index.hit(dir, 64.0).unwrap();
            let expected = surface_radius(planet.smooth_height(0, u, v), 32);
            assert!((r - expected).abs() < 1e-3, "({u}, {v}): {r} vs {expected}");
        }
    }

    // a mined pit is carved into the smooth terrain and stays closed, also across a cube-face edge
    #[test]
    fn mined_pits_are_carved_and_closed() {
        for (u, v) in [(10u32, 12u32), (0, 12)] {
            let mut planet = PlanetData::new(64);
            let top = planet.surface(0, u, v);
            for layer in top - 2..=top {
                for (du, dv) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let _ = planet.remove_block(BlockId {
                        face: 0,
                        layer,
                        u: u + du,
                        v: v + dv,
                    });
                }
            }
            let index = MeshIndex::new(&mesh_planet(&planet), 64);
            let dir = CoordSystem::get_block_center(0, u, v, 32, 64).normalize();
            let r = index.hit(dir, 128.0).expect("hole through the pit");
            let before = surface_radius(PlanetData::new(64).smooth_height(0, u, v), 64);
            assert!(r < before - 0.5, "({u}, {v}) not carved: {r} vs {before}");
            for dir in probes(&planet) {
                assert!(
                    index.hit(dir, 128.0).is_some(),
                    "hole near the pit at {dir:?}"
                );
            }
        }
    }

    // a placed block is a full cube, even half-buried in the smooth surface (faces are culled only
    // against other placed blocks)
    #[test]
    fn placed_blocks_are_full_cubes() {
        let mut planet = PlanetData::new(32);
        let (u, v) = (12u32, 14u32);
        let id = BlockId {
            face: 0,
            layer: planet.surface(0, u, v) + 1,
            u,
            v,
        };
        planet.add_block(id, BlockType::Stone).unwrap();
        let (verts, _) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: u / CHUNK_SIZE,
                v_idx: v / CHUNK_SIZE,
            },
            &planet,
        );
        let has = |p: Vec3| {
            verts
                .iter()
                .any(|vx| (Vec3::from_array(vx.pos) - p).length() < 1e-4)
        };
        // all eight cube corners: top corners alone would mean the side and bottom faces were culled
        for (du, dv, dl) in (0..8).map(|i| (i & 1, i >> 1 & 1, i >> 2 & 1)) {
            let corner = CoordSystem::get_vertex_pos(0, u + du, v + dv, id.layer + dl, 32);
            assert!(
                has(corner),
                "corner ({du}, {dv}, {dl}) missing: faces culled against the terrain"
            );
        }
    }

    // a placed block's faces are one colour each, also the top face next to another placed block
    // (its per-corner AO would otherwise split it diagonally into two shades)
    #[test]
    fn placed_block_faces_are_one_colour() {
        let mut planet = PlanetData::new(32);
        let (u, v) = (12u32, 14u32);
        let layer = planet.surface(0, u, v) + 1;
        for (du, dl) in [(0, 0), (1, 0), (1, 1)] {
            planet
                .add_block(
                    BlockId {
                        face: 0,
                        layer: layer + dl,
                        u: u + du,
                        v,
                    },
                    BlockType::Stone,
                )
                .unwrap();
        }
        let (verts, inds) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: 0,
                v_idx: 0,
            },
            &planet,
        );
        // the placed cubes follow the terrain facets (indexed 0, 1, 2, ...) as quads (0, 1, 2, 2, 3, 0)
        let terrain = inds
            .iter()
            .enumerate()
            .position(|(i, &x)| x as usize != i)
            .unwrap()
            - 3; // the first quad's first triangle continues the sequence
        let mut quads = 0;
        for t in inds[terrain..].chunks_exact(6) {
            assert!(t[3] == t[2] && t[5] == t[0]);
            let c = verts[t[0] as usize].color;
            for k in 1..4 {
                assert_eq!(verts[t[0] as usize + k].color, c, "two-tone face");
            }
            quads += 1;
        }
        assert!(quads >= 10, "{quads} quads");
    }

    // facets are one flat colour each
    #[test]
    fn facets_are_flat_coloured() {
        let planet = PlanetData::new(32);
        let (verts, inds) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: 0,
                v_idx: 0,
            },
            &planet,
        );
        assert!(!inds.is_empty());
        for tri in inds.chunks_exact(3) {
            let c = verts[tri[0] as usize].color;
            assert_eq!(verts[tri[1] as usize].color, c);
            assert_eq!(verts[tri[2] as usize].color, c);
        }
    }

    // small planets: cells near the core don't underflow and mesh normally
    #[test]
    fn tiny_planets_mesh_without_panicking() {
        let planet = PlanetData::new(16);
        let meshes = mesh_planet(&planet);
        assert!(meshes.iter().all(|(v, _)| !v.is_empty()));
    }

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

    // the surface rises steadily with the smooth height, also across the switch to the next layer; over
    // a column centre it sits at layer coordinate smooth height + 1 (between the centres it interpolates
    // the radius linearly)
    #[test]
    fn surface_radius_is_continuous_and_rising() {
        for h in [10.0f32, 10.25, 10.5, 10.75] {
            let (l, t) = crossing(h);
            assert!(
                (l as f32 + 0.5 + t - (h + 1.0)).abs() < 1e-5,
                "{h}: ({l}, {t})"
            );
        }
        let mut last = surface_radius(10.0, 32);
        for i in 1..=200 {
            let r = surface_radius(10.0 + i as f32 * 0.01, 32);
            assert!(r > last, "not rising at {}", 10.0 + i as f32 * 0.01);
            assert!(r - last < 0.05, "jump at {}", 10.0 + i as f32 * 0.01);
            last = r;
        }
    }

    // density: linear, one less per layer up, unclamped; solid at and below the smooth height's layer,
    // empty above it (17.3: the surface over the column centre is at layer coordinate 18.3, below the
    // centre of layer 18)
    #[test]
    fn natural_density_brackets_the_smooth_height() {
        assert!((natural_density(17.3, 17) - 1.3).abs() < 1e-5);
        assert!((natural_density(17.3, 18) - 0.3).abs() < 1e-5);
        assert!((natural_density(17.3, 19) + 0.7).abs() < 1e-5);
        assert!((natural_density(17.3, 3) - 15.3).abs() < 1e-5);
        assert!(natural_density(17.3, 17) > ISO && natural_density(17.3, 18) <= ISO);
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

    // vertex positions quantised for matching shared corners
    fn quant(p: [f32; 3]) -> [i64; 3] {
        p.map(|c| (c as f64 * 1e4).round() as i64)
    }

    // terrain triangles (no placed blocks) as quantised vertex triples
    fn terrain_triangles(planet: &PlanetData) -> Vec<[[i64; 3]; 3]> {
        let mut out = Vec::new();
        for (verts, inds) in mesh_planet(planet) {
            for t in inds.chunks_exact(3) {
                out.push([0, 1, 2].map(|k| quant(verts[t[k] as usize].pos)));
            }
        }
        out
    }

    // each cell is built exactly once and the surface is closed: no two triangles share their vertices,
    // and every undirected edge is used by exactly two triangles
    fn assert_manifold(planet: &PlanetData) {
        let tris = terrain_triangles(planet);
        let mut seen = std::collections::HashSet::new();
        let mut edges: std::collections::HashMap<([i64; 3], [i64; 3]), u32> = Default::default();
        for t in &tris {
            let mut sorted = *t;
            sorted.sort();
            assert!(seen.insert(sorted), "duplicate triangle {t:?}");
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edges
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_default() += 1;
            }
        }
        let bad: Vec<_> = edges.iter().filter(|(_, c)| **c != 2).collect();
        let at = |p: [i64; 3]| {
            let v = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 1e4;
            CoordSystem::pos_to_id(v, planet.resolution)
        };
        assert!(
            bad.is_empty(),
            "{} of {} edges not shared by two triangles, e.g. {:?}",
            bad.len(),
            edges.len(),
            bad.iter()
                .take(5)
                .map(|(k, c)| (at(k.0), **c))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn low_poly_terrain_is_a_closed_manifold() {
        assert_manifold(&PlanetData::new(32));
        assert_manifold(&PlanetData::new(16));
        assert_manifold(&PlanetData::new_for_type(
            48,
            crate::noise::HOME_SEED,
            crate::biome::PlanetType::Ice,
        ));
    }

    #[test]
    fn mined_pits_keep_the_surface_a_closed_manifold() {
        let mut planet = PlanetData::new(32);
        let (u, v) = (10u32, 12u32);
        let top = planet.surface(0, u, v);
        for layer in top - 2..=top {
            for (du, dv) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let _ = planet.remove_block(BlockId {
                    face: 0,
                    layer,
                    u: u + du,
                    v: v + dv,
                });
            }
        }
        assert_manifold(&planet);
    }

    // mining the top block of a column in a dip opens it: the smooth surface there lies above the
    // block (smooth_height > h + 0.5 draws the air cell h + 1 solid, and air can't be mined), so the
    // cells above a mined column top are carved too; the pit stays a closed manifold
    #[test]
    fn mining_a_dip_opens_it() {
        let mut planet = PlanetData::new(64);
        let res = planet.resolution;
        let (u, v, h) = (2..res - 2)
            .flat_map(|v| (2..res - 2).map(move |u| (u, v)))
            .map(|(u, v)| (u, v, planet.effective_height(0, u, v)))
            .find(|&(u, v, h)| planet.smooth_height(0, u, v) > h as f32 + 0.6)
            .expect("no dip");
        let dir = CoordSystem::get_block_center(0, u, v, h, res).normalize();
        let before = MeshIndex::new(&mesh_planet(&planet), res)
            .hit(dir, res as f32 * 2.0)
            .unwrap();
        assert!(before > CoordSystem::get_layer_radius(h + 1, res) + 0.05);
        planet
            .remove_block(BlockId {
                face: 0,
                layer: h,
                u,
                v,
            })
            .unwrap();
        let r = MeshIndex::new(&mesh_planet(&planet), res)
            .hit(dir, res as f32 * 2.0)
            .unwrap();
        // the pit's floor is the top of block h - 1
        assert!(
            r < CoordSystem::get_layer_radius(h, res) + 0.05,
            "({u}, {v}) still covered: hit {r}, block top {}",
            CoordSystem::get_layer_radius(h + 1, res)
        );
        assert_manifold(&planet);
    }

    // a flat-ish, dry grass column at least 3 layers above the sea, with a 7×7 neighbourhood inside one chunk of face 0
    // within one layer of its height: (u, v, top layer)
    fn tunnel_site(planet: &PlanetData) -> (u32, u32, u32) {
        let def = planet.planet_type.def();
        for v in 2..planet.resolution - 4 {
            for u in 2..planet.resolution - 4 {
                // the 7×7 neighbourhood stays inside one chunk
                if u / CHUNK_SIZE != (u + 4) / CHUNK_SIZE
                    || v / CHUNK_SIZE != (v + 4) / CHUNK_SIZE
                    || u % CHUNK_SIZE < 2
                    || v % CHUNK_SIZE < 2
                {
                    continue;
                }
                let h = planet.terrain.get_height(0, u, v);
                let flat = (u - 2..=u + 4).all(|uu| {
                    (v - 2..=v + 4).all(|vv| planet.terrain.get_height(0, uu, vv).abs_diff(h) <= 1)
                });
                if flat
                    && h >= planet.terrain.sea_level() + 3
                    && crate::material::surface_type(&planet.terrain, &def.palette, 0, u, v)
                        == def.palette.ground
                {
                    return (u, v, planet.surface(0, u, v));
                }
            }
        }
        panic!("no flat grass site");
    }

    // 3×3 columns, three layers (top-4..top-2), closed above by two solid layers: a tunnel room
    fn tunnel_planet() -> (PlanetData, (u32, u32, u32)) {
        let mut planet = PlanetData::new(128);
        let (u, v, top) = tunnel_site(&planet);
        for layer in top - 4..=top - 2 {
            for du in 0..3 {
                for dv in 0..3 {
                    planet
                        .remove_block(BlockId {
                            face: 0,
                            layer,
                            u: u + du,
                            v: v + dv,
                        })
                        .unwrap();
                }
            }
        }
        (planet, (u, v, top))
    }

    // the room's facets: centroids between its floor and ceiling, over its columns
    fn room_facets(
        planet: &PlanetData,
        (u, v, top): (u32, u32, u32),
    ) -> (Vec<Vertex>, Vec<[usize; 3]>) {
        let res = planet.resolution;
        let (verts, inds) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: u / CHUNK_SIZE,
                v_idx: v / CHUNK_SIZE,
            },
            planet,
        );
        let mut out = Vec::new();
        for t in inds.chunks_exact(3) {
            let c = t
                .iter()
                .map(|&i| Vec3::from_array(verts[i as usize].pos))
                .sum::<Vec3>()
                / 3.0;
            let layer = CoordSystem::layer_of_radius(c.length(), res);
            let Some(col) = CoordSystem::pos_to_id(c, res) else {
                continue;
            };
            if col.face == 0
                && (u..u + 3).contains(&col.u)
                && (v..v + 3).contains(&col.v)
                && layer > top as f32 - 4.5
                && layer < top as f32 - 1.0
            {
                out.push([t[0] as usize, t[1] as usize, t[2] as usize]);
            }
        }
        assert!(!out.is_empty(), "no facets in the room");
        (verts, out)
    }

    // the room's walls take the colour of the solid block they are cut from (dirt/stone), not the
    // surface material, and are darkened by COVER_DARKEN (solid cells above)
    #[test]
    fn tunnel_walls_use_the_block_type_and_are_covered() {
        let (planet, site @ (u, v, top)) = tunnel_planet();
        let def = planet.planet_type.def();
        let (verts, facets) = room_facets(&planet, site);
        // the block types the room is cut from
        let mut solid: Vec<[f32; 3]> = Vec::new();
        for layer in top - 5..=top - 1 {
            for uu in u - 1..u + 4 {
                for vv in v - 1..v + 4 {
                    if let Some(t) = planet.block_type(BlockId {
                        face: 0,
                        layer,
                        u: uu,
                        v: vv,
                    }) {
                        solid.push(t.color().map(|c| c * COVER_DARKEN));
                    }
                }
            }
        }
        let grass = def.palette.ground.color().map(|c| c * COVER_DARKEN);
        for tri in &facets {
            let c = verts[tri[0]].color;
            assert_ne!(c, grass, "a wall took the surface material");
            assert!(
                solid.contains(&c),
                "facet colour {c:?} is not a darkened block type of the room"
            );
        }
    }

    // a facet with open air above is not darkened (the surface)
    #[test]
    fn open_surface_facets_are_not_darkened() {
        let planet = PlanetData::new(128);
        let (u, v, _) = tunnel_site(&planet);
        let (verts, inds) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: u / CHUNK_SIZE,
                v_idx: v / CHUNK_SIZE,
            },
            &planet,
        );
        let def = planet.planet_type.def();
        let ground = def.palette.ground.color();
        assert!(
            inds.iter().any(|&i| verts[i as usize].color == ground),
            "no full-brightness ground facet"
        );
    }

    // the room is mined below the smooth surface: deepest_mined extends the meshed layer range down to
    // its floor
    #[test]
    fn deep_tunnels_are_meshed_down_to_their_floor() {
        let (planet, (u, v, top)) = tunnel_planet();
        let res = planet.resolution;
        let (verts, _) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: u / CHUNK_SIZE,
                v_idx: v / CHUNK_SIZE,
            },
            &planet,
        );
        let lowest = (u..u + 3)
            .flat_map(|uu| (v..v + 3).map(move |vv| (uu, vv)))
            .map(|(uu, vv)| planet.smooth_height(0, uu, vv).floor())
            .fold(f32::MAX, f32::min);
        let natural_floor = CoordSystem::get_layer_radius_f(lowest - 2.0, res);
        assert!(top as f32 - 4.0 < lowest - 2.0);
        assert!(
            verts.iter().any(|vx| {
                let p = Vec3::from_array(vx.pos);
                CoordSystem::pos_to_id(p, res).is_some_and(|c| {
                    c.face == 0 && (u..u + 3).contains(&c.u) && (v..v + 3).contains(&c.v)
                }) && p.length() < natural_floor
            }),
            "no floor vertices below the natural range"
        );
    }

    // lake-floor facets carry the water surface radius of their lake, so the shader can tint them
    #[test]
    fn lake_floor_facets_carry_the_water_radius() {
        let (planet, (face, u, v)) =
            crate::common::tests::lake_planet(crate::biome::PlanetType::EarthLike);
        let res = planet.resolution;
        let key = ChunkKey {
            face,
            u_idx: u / CHUNK_SIZE,
            v_idx: v / CHUNK_SIZE,
        };
        let (verts, inds) = build_chunk_lowpoly(key, &planet);
        let expected = planet.water_surface_radius(face, u, v);
        assert!(expected > 0.0);
        let mut found = 0;
        for t in inds.chunks_exact(3) {
            let c = t
                .iter()
                .map(|&i| Vec3::from_array(verts[i as usize].pos))
                .sum::<Vec3>()
                / 3.0;
            if CoordSystem::pos_to_id(c, res)
                .is_some_and(|id| (id.face, id.u, id.v) == (face, u, v))
            {
                found += 1;
                for &i in t {
                    assert!(
                        (verts[i as usize].water - expected).abs() < 1e-3,
                        "{} vs {expected}",
                        verts[i as usize].water
                    );
                }
            }
        }
        assert!(found > 0, "no facets over the lake column");
    }

    // LOD vertices lie on the voxel low-poly surface (column corners: the mean smooth height of the four
    // columns there), so the LOD hand-over keeps the shape
    #[test]
    fn lod_vertices_lie_on_the_voxel_surface() {
        let planet = PlanetData::new(64);
        let index = MeshIndex::new(&mesh_planet(&planet), 64);
        let key = crate::common::LodKey {
            face: 0,
            x: 0,
            y: 0,
            size: 64,
        };
        let (lod, _) =
            crate::gen::MeshGen::generate_lod_mesh_styled(key, &planet, TerrainStyle::LowPoly);
        let (mut worst, mut sum, mut n) = (0.0f32, 0.0f32, 0.0f32);
        // grid vertices only (the skirt vertices follow them); interior, dry land
        for vy in 2..63u32 {
            for vx in 2..63u32 {
                let p = Vec3::from_array(lod[(vy * 65 + vx) as usize].pos);
                if planet.terrain.water_level(0, vx, vy) >= planet.terrain.get_height(0, vx, vy) {
                    continue;
                }
                let r = index.hit(p.normalize(), 128.0).unwrap();
                let err = (p.length() - r).abs();
                worst = worst.max(err);
                sum += err;
                n += 1.0;
            }
        }
        println!("lod vs voxel: n {n} worst {worst} mean {}", sum / n);
        assert!(n > 500.0);
        assert!(worst < 0.5, "worst {worst}");
        assert!(sum / n < 0.15, "mean {}", sum / n);
    }

    // in an edited chunk the LOD vertices still lie on the voxel low-poly surface: a tunnel leaves the
    // columns' tops unchanged, and a mined top lowers the column's corners by a layer, like the pit (the
    // cube path's block corners r(h) sat a layer low, with the voxel colours)
    #[test]
    fn lod_vertices_lie_on_the_voxel_surface_in_edited_chunks() {
        let mut planet = PlanetData::new(64);
        let res = planet.resolution;
        let (tu, tv) = (6u32, 7u32);
        let h = planet.effective_height(0, tu, tv);
        planet
            .remove_block(BlockId {
                face: 0,
                layer: h - 3,
                u: tu,
                v: tv,
            })
            .unwrap();
        let (pu, pv) = (20u32, 21u32);
        for (du, dv) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (u, v) = (pu + du, pv + dv);
            planet
                .remove_block(BlockId {
                    face: 0,
                    layer: planet.effective_height(0, u, v),
                    u,
                    v,
                })
                .unwrap();
        }
        assert!(planet.column_edited(0, 2, 2));
        let index = MeshIndex::new(&mesh_planet(&planet), res);
        let key = crate::common::LodKey {
            face: 0,
            x: 0,
            y: 0,
            size: 64,
        };
        let (lod, _) =
            crate::gen::MeshGen::generate_lod_mesh_styled(key, &planet, TerrainStyle::LowPoly);
        let (mut worst, mut sum, mut n) = (0.0f32, 0.0f32, 0.0f32);
        // grid vertices over the edited chunk (columns 0..32), dry land
        for vy in 2..31u32 {
            for vx in 2..31u32 {
                let p = Vec3::from_array(lod[(vy * 65 + vx) as usize].pos);
                if planet.terrain.water_level(0, vx, vy) >= planet.terrain.get_height(0, vx, vy) {
                    continue;
                }
                // the pit's middle corner: all four columns lowered
                let pit = (vx, vy) == (pu + 1, pv + 1);
                if !pit && vx.abs_diff(pu + 1) <= 1 && vy.abs_diff(pv + 1) <= 1 {
                    continue; // the pit's rim: half its columns lowered, the LOD point-samples one
                }
                let r = index.hit(p.normalize(), 128.0).unwrap();
                let err = (p.length() - r).abs();
                worst = worst.max(err);
                sum += err;
                n += 1.0;
            }
        }
        println!("edited lod vs voxel: n {n} worst {worst} mean {}", sum / n);
        assert!(n > 300.0);
        assert!(worst < 0.5, "worst {worst}");
        assert!(sum / n < 0.15, "mean {}", sum / n);
    }

    // the game's start planet (galaxy seed 1, planet #1, Earth-like, res 337): larger and steeper than
    // the small test planets above. Slow in a debug build (~25 s), hence ignored; run with
    // cargo test --release start_planet_is_a_closed_manifold -- --ignored
    #[test]
    #[ignore]
    fn start_planet_is_a_closed_manifold() {
        let planet = crate::galaxy::Galaxy::generate(1).planets[0].bake();
        assert_eq!(planet.resolution, 337);
        assert_manifold(&planet);
    }

    // how far the drawn surface over a column centre (layer coordinate smooth + 1) lies from the top of
    // its top block (h + 1), in layers, over every column: (max, 99th percentile, mean, share > 1 layer)
    fn smooth_mismatch(planet: &PlanetData) -> (f32, f32, f32, f32) {
        let res = planet.resolution;
        let mut d: Vec<f32> = (0..6u8)
            .flat_map(|f| (0..res).flat_map(move |v| (0..res).map(move |u| (f, u, v))))
            .map(|(f, u, v)| {
                (planet.smooth_height(f, u, v) - planet.effective_height(f, u, v) as f32).abs()
            })
            .collect();
        d.sort_by(f32::total_cmp);
        let p99 = d[(d.len() * 99) / 100];
        let mean = d.iter().sum::<f32>() / d.len() as f32;
        let over = d.iter().filter(|&&x| x > 1.0).count() as f32 / d.len() as f32;
        (*d.last().unwrap(), p99, mean, over)
    }

    // the drawn surface vs the blocks on a small planet: measured max 1.67, p99 0.78, mean 0.17 layers
    #[test]
    fn smooth_surface_stays_near_the_blocks() {
        let (max, p99, mean, over) = smooth_mismatch(&PlanetData::new(64));
        println!("res 64: max {max} p99 {p99} mean {mean} share>1 {over}");
        assert!(max < 2.0 && p99 < 1.0 && mean < 0.25, "{max} {p99} {mean}");
    }

    // the same on the game's start planet (res 337): measured max 2.78, p99 0.56, mean 0.16 layers —
    // up to ~3 layers on crests and valleys (cargo test --release start_planet_smooth -- --ignored)
    #[test]
    #[ignore]
    fn start_planet_smooth_surface_stays_near_the_blocks() {
        let planet = crate::galaxy::Galaxy::generate(1).planets[0].bake();
        let (max, p99, mean, over) = smooth_mismatch(&planet);
        println!("start planet: max {max} p99 {p99} mean {mean} share>1 {over}");
        assert!(max < 4.0 && p99 < 1.0 && mean < 0.3, "{max} {p99} {mean}");
    }

    // a planar ramp rising `slope` layers per column along u over face 0 (the rest of face 0 flattened
    // onto its ends): its low-poly facets, as (normal · up, centroid) for those over columns 11..21
    fn ramp_facets(slope: f32) -> Vec<(f32, Vec3)> {
        let mut planet = PlanetData::new(64);
        let res = planet.resolution;
        let mut smooth = (*planet.smooth).clone();
        for v in 0..res {
            for u in 0..res {
                smooth[(v * res + u) as usize] = 16.0 + slope * (u.clamp(8, 24) as f32 - 16.0);
            }
        }
        planet.smooth = std::sync::Arc::new(smooth);
        let (verts, inds) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: 0,
                v_idx: 0,
            },
            &planet,
        );
        let mut out = Vec::new();
        for t in inds.chunks_exact(3) {
            let c = t
                .iter()
                .map(|&i| Vec3::from_array(verts[i as usize].pos))
                .sum::<Vec3>()
                / 3.0;
            let col = CoordSystem::pos_to_id(c, res).unwrap();
            if col.face == 0 && (11..=21).contains(&col.u) && (11..=21).contains(&col.v) {
                let up = Vec3::from_array(verts[t[0] as usize].normal).dot(c.normalize());
                out.push((up, c));
            }
        }
        assert!(!out.is_empty());
        out
    }

    // a steep but even slope is one sloped plane of facets, not stair steps: with columns 1.5 and 2.5
    // layers apart (56° and 68° in layer units) every facet tilts alike (normal · up within 0.1; the
    // spread is the planet's curvature and the layers' changing thickness). The old 0..1-clamped density
    // pinned crossings between such columns to the midpoint: treads and risers, normal · up spread over
    // 0.37..0.67 at 1.5 and 0 (vertical risers)..0.67 at 2.5
    #[test]
    fn steep_even_slopes_are_one_sloped_plane() {
        for slope in [1.5f32, 2.5] {
            let f = ramp_facets(slope);
            let lo = f.iter().map(|x| x.0).fold(f32::MAX, f32::min);
            let hi = f.iter().map(|x| x.0).fold(f32::MIN, f32::max);
            assert!(
                hi - lo < 0.1,
                "slope {slope}: normal·up spread over {lo}..{hi}"
            );
        }
    }

    // over the steepest-sloped chunk of the game's start planet, near-vertical facets (normal · up < 0.3)
    // cover little area: 7.0 % with the linear density, 13.7 % with the old clamped one (and over the
    // whole planet 0.19 % vs 1.0 %)
    #[test]
    fn start_planet_slopes_have_few_near_vertical_facets() {
        let planet = crate::galaxy::Galaxy::generate(1).planets[0].bake();
        let res = planet.resolution;
        let n = res.div_ceil(CHUNK_SIZE);
        // the chunk with the most face-0 neighbour pairs one to two smooth layers apart (45–63°)
        let steep = |key: &ChunkKey| {
            let mut count = 0;
            for u in key.u_idx * CHUNK_SIZE..((key.u_idx + 1) * CHUNK_SIZE).min(res - 1) {
                for v in key.v_idx * CHUNK_SIZE..((key.v_idx + 1) * CHUNK_SIZE).min(res - 1) {
                    let h = planet.smooth_height(key.face, u, v);
                    for (du, dv) in [(1, 0), (0, 1)] {
                        let d = (h - planet.smooth_height(key.face, u + du, v + dv)).abs();
                        count += (1.0..2.0).contains(&d) as u32;
                    }
                }
            }
            count
        };
        let key = (0..n)
            .flat_map(|u_idx| {
                (0..n).map(move |v_idx| ChunkKey {
                    face: 0,
                    u_idx,
                    v_idx,
                })
            })
            .max_by_key(steep)
            .unwrap();
        let (verts, inds) = build_chunk_lowpoly(key, &planet);
        let (mut area, mut vertical) = (0.0f32, 0.0f32);
        for t in inds.chunks_exact(3) {
            let p = [0, 1, 2].map(|k| Vec3::from_array(verts[t[k] as usize].pos));
            let a = (p[1] - p[0]).cross(p[2] - p[0]).length() / 2.0;
            let up = Vec3::from_array(verts[t[0] as usize].normal)
                .dot(((p[0] + p[1] + p[2]) / 3.0).normalize());
            area += a;
            if up < 0.3 {
                vertical += a;
            }
        }
        assert!(
            vertical / area < 0.09,
            "{key:?}: {:.1} % of the area near-vertical",
            vertical / area * 100.0
        );
    }

    // a mined pit's walls sit at the block boundary between the solid and the mined cell: the wall
    // vertex on a horizontal edge from a solid into a mined block centre is half-way
    #[test]
    fn pit_walls_sit_at_the_block_boundary() {
        let mut planet = PlanetData::new(64);
        let (u, v) = (10u32, 12u32);
        let top = planet.surface(0, u, v);
        for layer in top - 2..=top {
            planet
                .remove_block(BlockId {
                    face: 0,
                    layer,
                    u,
                    v,
                })
                .unwrap();
        }
        let res = planet.resolution;
        let (verts, _) = build_chunk_lowpoly(
            ChunkKey {
                face: 0,
                u_idx: 0,
                v_idx: 0,
            },
            &planet,
        );
        let mut checked = 0;
        for layer in top - 2..top {
            // walls toward the four neighbours, at a mined layer well below the smooth surface
            let pit = CoordSystem::get_block_center(0, u, v, layer, res);
            for (nu, nv) in [(u + 1, v), (u - 1, v), (u, v + 1), (u, v - 1)] {
                if planet.smooth_height(0, nu, nv) < layer as f32 + 1.0 {
                    continue; // the neighbour isn't solid at this layer
                }
                let side = CoordSystem::get_block_center(0, nu, nv, layer, res);
                let boundary = (pit + side) / 2.0;
                let near = verts
                    .iter()
                    .map(|vx| (Vec3::from_array(vx.pos) - boundary).length());
                let d = near.fold(f32::MAX, f32::min);
                assert!(
                    d < 0.25,
                    "layer {layer} toward ({nu}, {nv}): nearest vertex {d} from the boundary"
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    // meshing cost of every chunk of a radius-128 planet in both styles (cargo test --release
    // lowpoly_bench -- --ignored --nocapture)
    #[test]
    #[ignore]
    fn lowpoly_bench() {
        let planet = PlanetData::new(256);
        let n = 256 / CHUNK_SIZE;
        let keys: Vec<ChunkKey> = (0..6u8)
            .flat_map(|face| {
                (0..n)
                    .flat_map(move |u_idx| (0..n).map(move |v_idx| ChunkKey { face, u_idx, v_idx }))
            })
            .collect();
        for style in [TerrainStyle::Cubes, TerrainStyle::LowPoly] {
            let start = std::time::Instant::now();
            let (mut tris, mut verts) = (0, 0);
            for &key in &keys {
                let (v, i) = match style {
                    TerrainStyle::Cubes => crate::gen::MeshGen::build_chunk_cubes(key, &planet),
                    TerrainStyle::LowPoly => build_chunk_lowpoly(key, &planet),
                    TerrainStyle::Hex => crate::gen::MeshGen::build_chunk_hex(key, &planet),
                };
                tris += i.len() / 3;
                verts += v.len();
            }
            let t = start.elapsed();
            println!(
                "{style:?}: {:.2} ms/chunk, {tris} triangles, {:.1} MB vertices",
                t.as_secs_f64() * 1000.0 / keys.len() as f64,
                (verts * std::mem::size_of::<Vertex>()) as f64 / 1e6
            );
        }
    }
}
