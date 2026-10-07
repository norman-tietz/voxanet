//common.rs

use crate::biome::PlanetType;
use crate::material::{self, BlockType};
use crate::noise::PlanetTerrain;
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// --- CONSTANTS ---
pub const CHUNK_SIZE: u32 = 32;

// the smooth height map (low-poly terrain, lowpoly.rs) averages each column over this many columns
// around it in every direction: 1 → 3×3
pub const SMOOTH_RADIUS: i32 = 1;

// edit limits (PlanetData::build_ceiling, mining_floor, add_block/remove_block)
const BUILD_MARGIN: u32 = 24; // layers above the highest natural peak that can be built on
const BEDROCK_DEPTH: u32 = 32; // layers below the lowest natural column that can be mined
pub const MAX_EDITS: usize = 1_000_000; // placed + mined entries per planet (~60 MB)
                                        // cloud shell radius in planet radii; must match CLOUD_ALT in atmosphere.wgsl (checked by a test)
pub const CLOUD_ALT: f32 = 1.32;

// why an edit was refused
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditRefused {
    BuildLimit,  // above build_ceiling
    MiningFloor, // below mining_floor
    EditLimit,   // MAX_EDITS reached
}

// --- DATA TYPES ---

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct BlockId {
    pub face: u8,
    pub layer: u32,
    pub u: u32,
    pub v: u32,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct ChunkKey {
    pub face: u8,
    pub u_idx: u32,
    pub v_idx: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
    pub water: f32, // water surface radius over the cell this face looks into, 0 = dry (caustics)
}

impl Vertex {
    // shader.wgsl VertexIn
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32];
}

// geomorphing (LOD meshes only, a second vertex buffer beside Vertex): what the parent LOD level shows at
// this vertex, which vs_lod blends toward by the mesh's morph factor (LocalUniform params.z), so a node
// appears with exactly its parent's shape and sharpens into its own as the camera closes in
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LodMorph {
    pub height: f32, // parent surface radius minus this vertex's radius (moved along its own up)
    pub normal: u32, // the parent's normal, Snorm8x4
    pub color: u32,  // the parent's colour, Unorm8x4
}

impl LodMorph {
    // shader.wgsl LodMorphIn
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![4 => Float32, 5 => Snorm8x4, 6 => Unorm8x4];

    pub fn new(height: f32, normal: glam::Vec3, color: [f32; 3]) -> Self {
        let snorm = |x: f32| ((x.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8) as u32;
        let unorm = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
        Self {
            height,
            normal: snorm(normal.x) | snorm(normal.y) << 8 | snorm(normal.z) << 16,
            color: unorm(color[0]) | unorm(color[1]) << 8 | unorm(color[2]) << 16,
        }
    }
}

pub struct ChunkMesh {
    pub v_buf: wgpu::Buffer,
    pub i_buf: wgpu::Buffer,
    pub num_inds: u32,
    pub num_verts: usize,
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub center: glam::Vec3,
    pub radius: f32,
    pub blas: Option<wgpu::Blas>, // for hardware ray-traced shadows (hw_rt.rs), when supported
    pub water: Option<WaterMesh>, // voxel chunks with ocean columns
    pub morph: Option<wgpu::Buffer>, // LOD meshes: LodMorph per vertex (drawn with the vs_lod pipeline)
    pub morph_factor: f32, // LOD meshes: 1 = the parent's shape, 0 = its own (Renderer::update_view)
    pub params: [f32; 4],  // LocalUniform params last written to uniform_buf
}

// translucent water surface of a voxel chunk, drawn by fs_water after the deferred lighting
pub struct WaterMesh {
    pub v_buf: wgpu::Buffer,
    pub i_buf: wgpu::Buffer,
    pub num_inds: u32,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct LodKey {
    pub face: u8,
    pub x: u32,
    pub y: u32,
    pub size: u32,
}

#[derive(Clone)]
pub struct ChunkMods {
    pub mined: HashSet<BlockId>,
    pub placed: HashMap<BlockId, BlockType>,
}

// a planet's player edits: everything a revisit (and later a save file) needs to restore them. The map
// is shared copy-on-write, so cloning a PlanetData for a mesh worker doesn't copy the edits
#[derive(Clone, Default)]
pub struct PlanetEdits {
    pub chunks: Arc<HashMap<ChunkKey, ChunkMods>>,
    count: usize,    // placed + mined entries, for MAX_EDITS
    resolution: u32, // the terrain these edits were made on
    noise_seed: u32,
}

impl PlanetEdits {
    fn for_terrain(resolution: u32, noise_seed: u32) -> Self {
        Self {
            resolution,
            noise_seed,
            ..Self::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[cfg(test)]
    pub fn set_count_for_test(&mut self, count: usize) {
        self.count = count;
    }
}

impl ChunkMods {
    pub fn new() -> Self {
        Self {
            mined: HashSet::new(),
            placed: HashMap::new(),
        }
    }
}

#[derive(Clone)]
pub struct PlanetData {
    pub edits: PlanetEdits,
    edit_generation: u64, // bumped by every successful edit (an outdated near impostor), not by restoring
    pub resolution: u32,
    pub terrain: crate::noise::PlanetTerrain,
    // per column the mean effective height of its (2·SMOOTH_RADIUS + 1)² neighbourhood, in layers: the
    // surface the low-poly terrain is drawn on (blocks stay the truth; lowpoly.rs)
    pub smooth: Arc<Vec<f32>>,
    pub planet_type: PlanetType,
    // the noise seed this planet was baked with (GalaxyPlanet::noise_seed, HOME_SEED for test planets);
    // edits record it so they're only restored onto the same terrain
    pub seed: u32,
}

impl PlanetData {
    pub fn new(resolution: u32) -> Self {
        Self::new_seeded(resolution, crate::noise::HOME_SEED)
    }

    pub fn new_seeded(resolution: u32, seed: u32) -> Self {
        println!("Generating Terrain Noise Map for res {}...", resolution);
        let terrain = PlanetTerrain::new(resolution, seed); // calculate once
        println!("Terrain Generation Complete.");
        Self::from_terrain(resolution, seed, terrain, PlanetType::EarthLike)
    }

    // a planet baked for its type: liquid planets get lakes (decided here, at bake time; switching the
    // type later keeps the terrain)
    pub fn new_for_type(resolution: u32, seed: u32, planet_type: PlanetType) -> Self {
        println!("Generating Terrain Noise Map for res {}...", resolution);
        let terrain = if planet_type.def().liquid.is_some() {
            PlanetTerrain::with_lakes(resolution, seed)
        } else {
            PlanetTerrain::new(resolution, seed)
        };
        println!("Terrain Generation Complete.");
        Self::from_terrain(resolution, seed, terrain, planet_type)
    }

    fn from_terrain(
        resolution: u32,
        seed: u32,
        terrain: PlanetTerrain,
        planet_type: PlanetType,
    ) -> Self {
        let smooth = Self::bake_smooth(&terrain, resolution, planet_type);
        Self {
            edits: PlanetEdits::for_terrain(resolution, seed),
            edit_generation: 0,
            resolution,
            terrain,
            smooth,
            planet_type,
            seed,
        }
    }

    // switches the active planet type in place: clears edits (they reference the old palette)
    // but keeps the same terrain shape — phase 1 reuses one noise map for every planet type
    #[cfg(test)]
    pub fn switch_planet_type(&mut self, to: PlanetType) {
        self.planet_type = to;
        self.edits = PlanetEdits::for_terrain(self.resolution, self.seed);
        self.smooth = Self::bake_smooth(&self.terrain, self.resolution, to);
    }

    // the highest layer blocks can be placed in: BUILD_MARGIN above the highest natural peak, but not
    // into the cloud shell (on small planets the clouds are only a few layers above the peaks)
    pub fn build_ceiling(&self) -> u32 {
        let (_, peak) = self.terrain.height_range();
        let cloud_radius = CLOUD_ALT * self.resolution as f32 / 2.0;
        let mut top = peak + BUILD_MARGIN;
        while top > peak
            && crate::gen::CoordSystem::get_layer_radius(top + 1, self.resolution) > cloud_radius
        {
            top -= 1;
        }
        top
    }

    // the lowest layer that can be mined: BEDROCK_DEPTH below the lowest natural column, never into the
    // core; everything below is bedrock. Deeper, the exponential layers shrink blocks too far to move in.
    pub fn mining_floor(&self) -> u32 {
        let (lowest, _) = self.terrain.height_range();
        lowest
            .saturating_sub(BEDROCK_DEPTH)
            .max(material::CORE_LAYERS)
    }

    // stored edit entries (placed + mined)
    #[cfg(test)]
    pub fn edit_count(&self) -> usize {
        self.edits.count
    }

    // hands the edits over (for keeping them while another planet is installed); the planet is left unedited
    pub fn take_edits(&mut self) -> PlanetEdits {
        std::mem::replace(
            &mut self.edits,
            PlanetEdits::for_terrain(self.resolution, self.seed),
        )
    }

    // puts edits taken from an earlier bake of this planet back; refused (false, nothing restored) when
    // they were made on another terrain. Not an edit: the generation stays, the near impostor built
    // after restoring is current
    pub fn restore_edits(&mut self, edits: PlanetEdits) -> bool {
        if edits.resolution != self.resolution || edits.noise_seed != self.seed {
            return false;
        }
        self.edits = edits;
        true
    }

    pub fn edit_generation(&self) -> u64 {
        self.edit_generation
    }

    fn get_chunk_key(id: BlockId) -> ChunkKey {
        ChunkKey {
            face: id.face,
            u_idx: id.u / CHUNK_SIZE,
            v_idx: id.v / CHUNK_SIZE,
        }
    }

    pub fn add_block(&mut self, id: BlockId, ty: BlockType) -> Result<(), EditRefused> {
        if id.layer > self.build_ceiling() {
            return Err(EditRefused::BuildLimit);
        }
        let natural = self.natural_type(id);
        let key = Self::get_chunk_key(id);
        let (was_mined, was_placed) = self.edits.chunks.get(&key).map_or((false, false), |m| {
            (m.mined.contains(&id), m.placed.contains_key(&id))
        });
        // putting back what was mined there just restores the terrain
        let restores = natural == Some(ty);
        if !restores && !was_mined && !was_placed && self.edits.count >= MAX_EDITS {
            return Err(EditRefused::EditLimit);
        }

        let mods = Arc::make_mut(&mut self.edits.chunks)
            .entry(key)
            .or_insert_with(ChunkMods::new);
        let before = mods.mined.len() + mods.placed.len();
        mods.mined.remove(&id);
        if restores {
            mods.placed.remove(&id);
        } else {
            mods.placed.insert(id, ty);
        }
        self.edits.count = self.edits.count + mods.mined.len() + mods.placed.len() - before;
        self.edit_generation += 1;
        Ok(())
    }

    pub fn remove_block(&mut self, id: BlockId) -> Result<(), EditRefused> {
        if id.layer < self.mining_floor() {
            return Err(EditRefused::MiningFloor);
        }
        let terrain_below = id.layer <= self.terrain.get_height(id.face, id.u, id.v);
        let key = Self::get_chunk_key(id);
        let (was_mined, was_placed) = self.edits.chunks.get(&key).map_or((false, false), |m| {
            (m.mined.contains(&id), m.placed.contains_key(&id))
        });
        if terrain_below && !was_mined && !was_placed && self.edits.count >= MAX_EDITS {
            return Err(EditRefused::EditLimit);
        }

        let mods = Arc::make_mut(&mut self.edits.chunks)
            .entry(key)
            .or_insert_with(ChunkMods::new);
        let before = mods.mined.len() + mods.placed.len();
        mods.placed.remove(&id);
        if terrain_below {
            mods.mined.insert(id);
        }
        self.edits.count = self.edits.count + mods.mined.len() + mods.placed.len() - before;
        self.edit_generation += 1;
        Ok(())
    }

    pub fn exists(&self, id: BlockId) -> bool {
        let key = Self::get_chunk_key(id);
        if let Some(mods) = self.edits.chunks.get(&key) {
            if mods.placed.contains_key(&id) {
                return true;
            }
            if mods.mined.contains(&id) {
                return false;
            }
        }

        // instead of a flat floor, we check the pre-calculated noise map
        id.layer <= self.effective_height(id.face, id.u, id.v)
    }

    // the height `exists()`/mesh generation should treat as solid: on a liquid-less planet (no
    // ocean mesh to fill the gap visually), terrain is solid all the way up to sea level instead
    // of just its natural height
    pub fn effective_height(&self, face: u8, u: u32, v: u32) -> u32 {
        let height = self.terrain.get_height(face, u, v);
        if self.planet_type.def().liquid.is_none() {
            height.max(self.terrain.sea_level())
        } else {
            height
        }
    }

    // the column's top solid layer after edits (u, v clamped to the face like the height map): the natural
    // effective height where its chunk is unedited, else searched down from the build ceiling. Distant
    // views (LOD meshes, the near impostor) sample it, so large builds and pits show from afar
    pub fn surface(&self, face: u8, u: u32, v: u32) -> u32 {
        let (u, v) = (u.min(self.resolution - 1), v.min(self.resolution - 1));
        let edited = self.column_edited(face, u, v);
        if !edited {
            return self.effective_height(face, u, v);
        }
        let floor = self.mining_floor();
        let mut layer = self.build_ceiling();
        while layer >= floor && !self.exists(BlockId { face, layer, u, v }) {
            layer -= 1;
        }
        layer // below the floor everything is bedrock
    }

    // picks a spawn direction: `preferred` unless the active liquid is damaging and `preferred`'s
    // column is underwater (an unescapable death loop, since floating alone still ticks damage), in
    // which case it searches a dense, evenly-spread set of directions over the whole sphere for a dry
    // one
    pub fn safe_spawn_direction(&self, preferred: glam::Vec3) -> glam::Vec3 {
        let damaging = self.planet_type.def().liquid.is_some_and(|l| l.damaging);
        if !damaging {
            return preferred;
        }
        // whether the column under `dir` holds the liquid (a lake or the sea)
        let probe_wet = |dir: glam::Vec3| {
            crate::gen::CoordSystem::pos_to_id(
                dir * (self.resolution as f32 / 2.0),
                self.resolution,
            )
            .map(|id| self.holds_water(id.face, id.u, id.v))
        };
        if probe_wet(preferred) != Some(true) {
            return preferred;
        }
        // Fibonacci-sphere sampling: far denser than the 6 cube-axis points this used to check
        // (which could all land in ocean on a small or heavily-watered planet), still cheap since
        // each sample is just a height-map lookup
        const SAMPLES: u32 = 256;
        (0..SAMPLES)
            .map(|i| {
                let phi = (1.0 + 5.0_f32.sqrt()) / 2.0; // golden ratio
                let t = (i as f32 + 0.5) / SAMPLES as f32;
                let incl = (1.0 - 2.0 * t).acos();
                let azim = 2.0 * std::f32::consts::PI * (i as f32) / phi;
                glam::Vec3::new(incl.sin() * azim.cos(), incl.sin() * azim.sin(), incl.cos())
            })
            .find(|&dir| probe_wet(dir) == Some(false))
            .unwrap_or(preferred)
    }

    // the column next to (face, u, v) in direction (du, dv) as (face, u, v); across a cube-face edge that
    // is a column of the neighbouring face, found by continuing the line from the inner neighbour outward
    pub fn neighbor_column(
        &self,
        face: u8,
        u: u32,
        v: u32,
        du: i32,
        dv: i32,
    ) -> Option<(u8, u32, u32)> {
        crate::gen::CoordSystem::neighbor_column(face, u, v, du, dv, self.resolution)
    }

    pub fn neighbor_height(&self, face: u8, u: u32, v: u32, du: i32, dv: i32) -> u32 {
        self.neighbor_column(face, u, v, du, dv)
            .map_or(0, |(f, nu, nv)| self.effective_height(f, nu, nv))
    }

    fn bake_smooth(
        terrain: &PlanetTerrain,
        resolution: u32,
        planet_type: PlanetType,
    ) -> Arc<Vec<f32>> {
        use rayon::prelude::*;
        let res = resolution;
        let (fill, sea) = (planet_type.def().liquid.is_none(), terrain.sea_level());
        // effective_height's rule: liquid-less planets are solid up to sea level
        let eff = |f: u8, u: u32, v: u32| {
            let h = terrain.get_height(f, u, v);
            (if fill { h.max(sea) } else { h }) as f32
        };
        let mut out = vec![0.0f32; (6 * res * res) as usize];
        out.par_chunks_mut(res as usize)
            .enumerate()
            .for_each(|(row, out)| {
                let face = (row as u32 / res) as u8;
                let v = row as u32 % res;
                for u in 0..res {
                    let (mut sum, mut n) = (0.0, 0.0);
                    for dv in -SMOOTH_RADIUS..=SMOOTH_RADIUS {
                        for du in -SMOOTH_RADIUS..=SMOOTH_RADIUS {
                            if let Some((f, cu, cv)) =
                                crate::gen::CoordSystem::neighbor_column(face, u, v, du, dv, res)
                            {
                                sum += eff(f, cu, cv);
                                n += 1.0;
                            }
                        }
                    }
                    out[u as usize] = sum / n;
                }
            });
        Arc::new(out)
    }

    // the column's smooth height in layers (u, v clamped to the face like the height map)
    pub fn smooth_height(&self, face: u8, u: u32, v: u32) -> f32 {
        let res = self.resolution;
        let (u, v) = (u.min(res - 1), v.min(res - 1));
        self.smooth[(face as u32 * res * res + v * res + u) as usize]
    }

    // the smooth height at grid corner (u, v) (0..=res): the mean of the up to four columns of this face
    // touching it
    pub fn smooth_corner(&self, face: u8, u: u32, v: u32) -> f32 {
        let res = self.resolution;
        let (mut sum, mut n) = (0.0, 0.0);
        for (cu, cv) in [
            (u.wrapping_sub(1), v.wrapping_sub(1)),
            (u, v.wrapping_sub(1)),
            (u.wrapping_sub(1), v),
            (u, v),
        ] {
            if cu < res && cv < res {
                sum += self.smooth_height(face, cu, cv);
                n += 1.0;
            }
        }
        sum / n
    }

    // whether the column's chunk has any edits (placed or mined blocks)
    pub fn column_edited(&self, face: u8, u: u32, v: u32) -> bool {
        self.edits
            .chunks
            .get(&Self::get_chunk_key(BlockId {
                face,
                layer: 0,
                u,
                v,
            }))
            .is_some_and(|m| !m.placed.is_empty() || !m.mined.is_empty())
    }

    pub fn chunk_key(id: BlockId) -> ChunkKey {
        Self::get_chunk_key(id)
    }

    // the water rule: the planet has a liquid and the column's water-level cell is empty — natural lake
    // or ocean, a pit dug into a lake bed, a hole dug below sea level elsewhere (no flow between them)
    pub fn holds_water(&self, face: u8, u: u32, v: u32) -> bool {
        let layer = self.terrain.water_level(face, u, v);
        self.planet_type.def().liquid.is_some() && !self.exists(BlockId { face, layer, u, v })
    }

    // the radius of this column's water surface, 0 when it holds none
    pub fn water_surface_radius(&self, face: u8, u: u32, v: u32) -> f32 {
        if !self.holds_water(face, u, v) {
            return 0.0;
        }
        crate::gen::CoordSystem::get_layer_radius(
            self.terrain.water_level(face, u, v) + 1,
            self.resolution,
        )
    }

    // how far `pos` lies below its column's water surface (negative above it), or None outside water
    // (holds_water)
    pub fn water_depth(&self, pos: glam::Vec3) -> Option<f32> {
        let id = crate::gen::CoordSystem::pos_to_id(pos, self.resolution)?;
        if !self.holds_water(id.face, id.u, id.v) {
            return None;
        }
        Some(self.water_surface_radius(id.face, id.u, id.v) - pos.length())
    }

    // the type of an existing block, None for air
    pub fn block_type(&self, id: BlockId) -> Option<BlockType> {
        if let Some(mods) = self.edits.chunks.get(&Self::get_chunk_key(id)) {
            if let Some(&ty) = mods.placed.get(&id) {
                return Some(ty);
            }
            if mods.mined.contains(&id) {
                return None;
            }
        }
        self.natural_type(id)
    }

    // the terrain's own block at this position, ignoring edits
    fn natural_type(&self, id: BlockId) -> Option<BlockType> {
        let height = self.terrain.get_height(id.face, id.u, id.v);
        let def = self.planet_type.def();
        let effective_height = if def.liquid.is_none() {
            height.max(self.terrain.sea_level())
        } else {
            height
        };
        if id.layer > effective_height {
            return None;
        }
        if id.layer > height {
            // filled-in liquid-less "ocean": solid, but it's the beach material, not real terrain depth
            return Some(def.palette.beach);
        }
        Some(material::natural_type(
            &self.terrain,
            &def.palette,
            self.mining_floor(),
            id.face,
            id.u,
            id.v,
            id.layer,
        ))
    }

    // the BlockType at `pos`'s column, at the layer pos itself sits in (for "what is the player
    // standing on" checks — entity.rs's slippery-ice friction)
    pub fn ground_block(&self, pos: glam::Vec3) -> Option<BlockType> {
        let id = crate::gen::CoordSystem::pos_to_id(pos, self.resolution)?;
        self.block_type(id)
    }
}

// --- FRUSTUM CULLING HELPER ---

pub struct Frustum {
    planes: [glam::Vec4; 6],
}

impl Frustum {
    pub fn from_matrix(m: glam::Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);

        let mut planes = [
            r3 + r0, // Left
            r3 - r0, // Right
            r3 + r1, // Bottom
            r3 - r1, // Top
            r3 + r2, // Near
            r3 - r2, // Far
        ];

        // normalize planes
        for plane in &mut planes {
            let len = glam::Vec3::new(plane.x, plane.y, plane.z).length();
            *plane /= len;
        }

        Self { planes }
    }

    // returns true if a sphere is partly or fully inside the frustum
    pub fn intersects_sphere(&self, center: glam::Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;

            if dist < -radius {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::biome::PlanetType;

    // res 32: small enough to generate fast in a test, large enough that continent noise
    // reliably produces both land and ocean columns
    const TEST_RES: u32 = 32;
    // a baked planet with a lake at least 2 layers above the sea, and one of its columns at least 2
    // layers under the lake's water
    pub(crate) fn lake_planet(planet_type: PlanetType) -> (PlanetData, (u8, u32, u32)) {
        for seed in 1..80 {
            let planet = PlanetData::new_for_type(128, seed, planet_type);
            for face in 0..6u8 {
                for v in 0..128 {
                    for u in 0..128 {
                        let w = planet.terrain.water_level(face, u, v);
                        if w >= planet.terrain.sea_level() + 2
                            && planet.terrain.get_height(face, u, v) + 2 <= w
                        {
                            return (planet, (face, u, v));
                        }
                    }
                }
            }
        }
        panic!("no seed gave a lake at resolution 128");
    }

    #[test]
    fn baking_a_liquid_planet_carves_lakes() {
        let (planet, _) = lake_planet(PlanetType::EarthLike);
        assert!(planet.terrain.lake_count() >= 1);
    }

    #[test]
    fn ice_bake_has_no_lakes() {
        for seed in 1..10 {
            assert_eq!(
                PlanetData::new_for_type(128, seed, PlanetType::Ice)
                    .terrain
                    .lake_count(),
                0
            );
        }
    }

    #[test]
    fn new_uses_the_home_seed() {
        assert_eq!(PlanetData::new(16).seed, crate::noise::HOME_SEED);
    }

    // a land column at least `above` layers higher than sea level, for digging below it
    pub(crate) fn first_land_column(planet: &PlanetData, above: u32) -> (u8, u32, u32) {
        let sea = planet.terrain.sea_level();
        for face in 0..6u8 {
            for u in 0..planet.resolution {
                for v in 0..planet.resolution {
                    if planet.terrain.get_height(face, u, v) >= sea + above {
                        return (face, u, v);
                    }
                }
            }
        }
        panic!("test planet has no land column {above} above sea level");
    }

    fn center(planet: &PlanetData, face: u8, u: u32, v: u32, layer: u32) -> glam::Vec3 {
        crate::gen::CoordSystem::get_block_center(face, u, v, layer, planet.resolution)
    }

    // digging a land column down below sea level leaves a hole that holds water (water table):
    // any empty cell at sea level is water, whatever the column's natural height
    #[test]
    fn a_hole_dug_below_sea_level_holds_water() {
        let mut planet = PlanetData::new(TEST_RES);
        let (face, u, v) = first_land_column(&planet, 2);
        let sea = planet.terrain.sea_level();
        let pos = center(&planet, face, u, v, sea);
        assert!(planet.water_depth(pos).is_none(), "sanity check: dry land");
        for layer in (sea - 1..=planet.terrain.get_height(face, u, v)).rev() {
            planet.remove_block(BlockId { face, layer, u, v }).unwrap();
        }
        assert!(planet.water_depth(pos).is_some_and(|d| d > 0.0));
    }

    // a block placed into the ocean at sea level: no water there to swim in any more
    #[test]
    fn a_block_placed_at_sea_level_displaces_the_water() {
        let mut planet = PlanetData::new(TEST_RES);
        let (face, u, v) = first_underwater_column(&planet);
        let sea = planet.terrain.sea_level();
        let pos = center(&planet, face, u, v, sea + 1);
        assert!(planet.water_depth(pos).is_some(), "sanity check: ocean");
        planet
            .add_block(
                BlockId {
                    face,
                    layer: sea,
                    u,
                    v,
                },
                crate::material::BlockType::Stone,
            )
            .unwrap();
        assert!(planet.water_depth(pos).is_none());
    }

    fn id(face: u8, u: u32, v: u32, layer: u32) -> BlockId {
        BlockId { face, layer, u, v }
    }

    // the build ceiling: blocks can be placed up to it, not above, and a refusal stores nothing
    #[test]
    fn placing_stops_at_the_build_ceiling() {
        let mut planet = PlanetData::new(TEST_RES);
        let top = planet.build_ceiling();
        assert_eq!(planet.add_block(id(0, 3, 3, top), BlockType::Stone), Ok(()));
        let before = planet.edit_count();
        assert_eq!(
            planet.add_block(id(0, 3, 3, top + 1), BlockType::Stone),
            Err(EditRefused::BuildLimit)
        );
        assert_eq!(planet.edit_count(), before);
        assert!(!planet.exists(id(0, 3, 3, top + 1)));
    }

    // the ceiling sits above the highest peak and below the cloud shell, on small and large planets
    #[test]
    fn the_build_ceiling_is_above_the_peaks_and_below_the_clouds() {
        for res in [80u32, 160, 337] {
            let planet = PlanetData::new(res);
            let (_, peak) = planet.terrain.height_range();
            let top = planet.build_ceiling();
            let cloud_radius = CLOUD_ALT * res as f32 / 2.0;
            assert!(top > peak, "res {res}: ceiling {top} not above peak {peak}");
            assert!(
                crate::gen::CoordSystem::get_layer_radius(top + 1, res) <= cloud_radius,
                "res {res}: ceiling {top} reaches into the clouds"
            );
        }
    }

    // CLOUD_ALT has to agree with the shader's cloud shell
    #[test]
    fn cloud_alt_matches_the_shader() {
        let wgsl = include_str!("atmosphere.wgsl");
        let line = wgsl
            .lines()
            .find(|l| l.trim_start().starts_with("const CLOUD_ALT "))
            .unwrap();
        let value: f32 = line
            .split('=')
            .nth(1)
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(value, CLOUD_ALT);
    }

    // nothing below the mining floor can be mined, and it is bedrock
    #[test]
    fn mining_stops_at_the_bedrock_floor() {
        let mut planet = PlanetData::new(TEST_RES);
        let floor = planet.mining_floor();
        let (lowest, _) = planet.terrain.height_range();
        assert_eq!(
            floor,
            lowest
                .saturating_sub(BEDROCK_DEPTH)
                .max(material::CORE_LAYERS)
        );
        assert_eq!(planet.remove_block(id(0, 3, 3, floor)), Ok(()));
        assert_eq!(
            planet.remove_block(id(0, 3, 3, floor - 1)),
            Err(EditRefused::MiningFloor)
        );
        assert!(planet.exists(id(0, 3, 3, floor - 1)));
        assert_eq!(
            planet.block_type(id(0, 3, 3, floor - 1)),
            Some(BlockType::Bedrock)
        );
        assert_ne!(
            planet.block_type(id(0, 3, 4, floor)),
            Some(BlockType::Bedrock)
        );
    }

    // a copy handed to a mesh worker shares the edits until the original is edited again
    #[test]
    fn copies_share_edits_until_an_edit() {
        let mut planet = PlanetData::new(TEST_RES);
        planet
            .remove_block(id(0, 3, 3, planet.terrain.get_height(0, 3, 3)))
            .unwrap();
        let copy = planet.clone();
        assert!(Arc::ptr_eq(&planet.edits.chunks, &copy.edits.chunks));
        planet
            .remove_block(id(0, 4, 4, planet.terrain.get_height(0, 4, 4)))
            .unwrap();
        assert!(!Arc::ptr_eq(&planet.edits.chunks, &copy.edits.chunks));
        assert_eq!(copy.edit_count(), 1, "the copy keeps its own snapshot");
        assert_eq!(planet.edit_count(), 2);
    }

    // at the cap, new entries are refused, but undoing edits still works
    #[test]
    fn the_edit_cap_refuses_new_entries_but_allows_undo() {
        let mut planet = PlanetData::new(TEST_RES);
        let h = planet.terrain.get_height(0, 3, 3);
        planet.remove_block(id(0, 3, 3, h)).unwrap();
        planet.edits.set_count_for_test(MAX_EDITS); // pretend the planet is full
        let h2 = planet.terrain.get_height(0, 5, 5);
        assert_eq!(
            planet.remove_block(id(0, 5, 5, h2)),
            Err(EditRefused::EditLimit)
        );
        assert_eq!(
            planet.add_block(id(0, 5, 5, h2 + 1), BlockType::Stone),
            Err(EditRefused::EditLimit)
        );
        // putting the mined block back restores the terrain: one entry fewer
        let restore = planet.natural_type(id(0, 3, 3, h)).unwrap();
        assert_eq!(planet.add_block(id(0, 3, 3, h), restore), Ok(()));
        assert_eq!(planet.edit_count(), MAX_EDITS - 1);
    }

    // edits taken from a planet and restored into a fresh bake of the same terrain come back intact
    #[test]
    fn edits_survive_take_and_restore() {
        let mut planet = PlanetData::new(TEST_RES);
        let h = planet.terrain.get_height(0, 3, 3);
        planet.remove_block(id(0, 3, 3, h)).unwrap();
        let h5 = planet.terrain.get_height(0, 5, 5);
        planet
            .add_block(id(0, 5, 5, h5 + 1), BlockType::Stone)
            .unwrap();
        let edits = planet.take_edits();
        assert!(planet.edits.is_empty(), "taking leaves the planet unedited");
        let mut again = PlanetData::new(TEST_RES);
        assert!(again.restore_edits(edits));
        assert!(!again.exists(id(0, 3, 3, h)));
        assert!(again.exists(id(0, 5, 5, h5 + 1)));
    }

    // edits never land on a different terrain
    #[test]
    fn edits_are_not_restored_into_another_terrain() {
        let mut planet = PlanetData::new(TEST_RES);
        planet
            .remove_block(id(0, 3, 3, planet.terrain.get_height(0, 3, 3)))
            .unwrap();
        let edits = planet.take_edits();
        let mut other_res = PlanetData::new(TEST_RES + 8);
        assert!(!other_res.restore_edits(edits.clone()));
        assert!(other_res.edits.is_empty());
        let mut other_seed = PlanetData::new_seeded(TEST_RES, crate::noise::HOME_SEED + 1);
        assert!(!other_seed.restore_edits(edits));
        assert!(other_seed.edits.is_empty());
    }

    // the edit count travels with the edits, so MAX_EDITS still holds after a revisit
    #[test]
    fn restore_keeps_the_edit_count() {
        let mut planet = PlanetData::new(TEST_RES);
        planet
            .remove_block(id(0, 3, 3, planet.terrain.get_height(0, 3, 3)))
            .unwrap();
        planet
            .remove_block(id(0, 4, 4, planet.terrain.get_height(0, 4, 4)))
            .unwrap();
        let edits = planet.take_edits();
        let mut again = PlanetData::new(TEST_RES);
        again.restore_edits(edits);
        assert_eq!(again.edit_count(), 2);
    }

    // successful edits bump the generation (the near impostor is outdated), refused ones don't
    #[test]
    fn edit_generation_counts_successful_edits_only() {
        let mut planet = PlanetData::new(TEST_RES);
        let g0 = planet.edit_generation();
        planet
            .remove_block(id(0, 3, 3, planet.terrain.get_height(0, 3, 3)))
            .unwrap();
        assert_eq!(planet.edit_generation(), g0 + 1);
        let top = planet.build_ceiling();
        assert!(planet
            .add_block(id(0, 3, 3, top + 1), BlockType::Stone)
            .is_err());
        assert_eq!(planet.edit_generation(), g0 + 1);
    }

    // restoring isn't an edit: the near impostor built after it is current
    #[test]
    fn restore_does_not_bump_the_edit_generation() {
        let mut planet = PlanetData::new(TEST_RES);
        planet
            .remove_block(id(0, 3, 3, planet.terrain.get_height(0, 3, 3)))
            .unwrap();
        let edits = planet.take_edits();
        let mut again = PlanetData::new(TEST_RES);
        let g = again.edit_generation();
        again.restore_edits(edits);
        assert_eq!(again.edit_generation(), g);
    }

    // the surface follows edits: a placed tower raises it, a mined pit lowers it; unedited columns
    // keep their natural (effective) height
    #[test]
    fn surface_follows_placed_and_mined_blocks() {
        let mut planet = PlanetData::new(TEST_RES);
        let (h3, h5) = (
            planet.terrain.get_height(0, 3, 3),
            planet.terrain.get_height(0, 5, 5),
        );
        assert_eq!(planet.surface(0, 3, 3), planet.effective_height(0, 3, 3));
        let top = (h3 + 3).min(planet.build_ceiling());
        for layer in h3 + 1..=top {
            planet
                .add_block(id(0, 3, 3, layer), BlockType::Stone)
                .unwrap();
        }
        assert_eq!(planet.surface(0, 3, 3), top);
        planet.remove_block(id(0, 5, 5, h5)).unwrap();
        planet.remove_block(id(0, 5, 5, h5 - 1)).unwrap();
        assert_eq!(planet.surface(0, 5, 5), h5 - 2);
        // a column of an edited chunk that wasn't itself edited: natural height
        assert_eq!(planet.surface(0, 7, 7), planet.effective_height(0, 7, 7));
    }

    // columns past the face edge clamp like the height map does
    #[test]
    fn surface_clamps_to_the_face() {
        let planet = PlanetData::new(TEST_RES);
        assert_eq!(
            planet.surface(0, TEST_RES, 0),
            planet.surface(0, TEST_RES - 1, 0)
        );
    }

    fn first_underwater_column(planet: &PlanetData) -> (u8, u32, u32) {
        let sea = planet.terrain.sea_level();
        for face in 0..6u8 {
            for u in 0..planet.resolution {
                for v in 0..planet.resolution {
                    if planet.terrain.get_height(face, u, v) < sea {
                        return (face, u, v);
                    }
                }
            }
        }
        panic!(
            "test planet at res {} has no underwater column",
            planet.resolution
        );
    }

    #[test]
    fn earth_like_ocean_column_is_not_solid_at_sea_level() {
        let planet = PlanetData::new(TEST_RES);
        let (face, u, v) = first_underwater_column(&planet);
        let sea = planet.terrain.sea_level();
        assert!(!planet.exists(BlockId {
            face,
            u,
            v,
            layer: sea
        }));
    }

    #[test]
    fn ice_planet_ocean_column_is_solid_ice_at_sea_level() {
        let mut planet = PlanetData::new(TEST_RES);
        planet.switch_planet_type(PlanetType::Ice);
        let (face, u, v) = first_underwater_column(&planet);
        let sea = planet.terrain.sea_level();
        let id = BlockId {
            face,
            u,
            v,
            layer: sea,
        };
        assert!(planet.exists(id));
        assert_eq!(planet.block_type(id), Some(crate::material::BlockType::Ice));
    }

    #[test]
    fn water_depth_is_none_on_ice_planets() {
        let mut planet = PlanetData::new(TEST_RES);
        let (face, u, v) = first_underwater_column(&planet);
        let sea = planet.terrain.sea_level();
        let pos = crate::gen::CoordSystem::get_vertex_pos(face, u, v, sea, planet.resolution);
        assert!(
            planet.water_depth(pos).is_some(),
            "sanity check: Earth-like should report a depth here"
        );

        planet.switch_planet_type(PlanetType::Ice);
        assert!(planet.water_depth(pos).is_none());
    }

    #[test]
    fn switch_planet_type_clears_edits_but_keeps_terrain_shape() {
        let mut planet = PlanetData::new(TEST_RES);
        let id = BlockId {
            face: 0,
            u: 5,
            v: 5,
            layer: planet.terrain.get_height(0, 5, 5) + 1,
        };
        planet
            .add_block(id, crate::material::BlockType::Stone)
            .unwrap();
        assert!(planet.edits.chunks.values().any(|m| !m.placed.is_empty()));

        let height_before = planet.terrain.get_height(0, 5, 5);
        planet.switch_planet_type(PlanetType::Volcanic);
        assert!(planet
            .edits
            .chunks
            .values()
            .all(|m| m.placed.is_empty() && m.mined.is_empty()));
        assert_eq!(planet.terrain.get_height(0, 5, 5), height_before);
        assert_eq!(planet.planet_type, PlanetType::Volcanic);
    }

    #[test]
    fn ground_block_finds_the_surface_material_under_a_position() {
        let planet = PlanetData::new(TEST_RES);
        // somewhere well above sea level, away from the ocean scan above
        let face = 0;
        let (u, v) = (planet.resolution / 2, planet.resolution / 2);
        let h = planet.terrain.get_height(face, u, v);
        let pos = crate::gen::CoordSystem::get_vertex_pos(face, u, v, h, planet.resolution);
        assert!(planet.ground_block(pos).is_some());
    }

    // regression test for the friction probe fix (entity.rs's walk branch): the feet rest just
    // above the ground, so `ground_block` must be probed slightly below the feet position, not at
    // it, to reliably see the solid block instead of the air above it
    #[test]
    fn ground_block_just_above_an_ice_ocean_column_resolves_to_ice_when_probed_below() {
        let mut planet = PlanetData::new(TEST_RES);
        planet.switch_planet_type(PlanetType::Ice);
        let (face, u, v) = first_underwater_column(&planet);
        let sea = planet.terrain.sea_level();

        // a world position just above the effective (sea-level-filled) ice surface, the way a
        // standing player's feet would rest
        let pos = crate::gen::CoordSystem::get_vertex_pos(face, u, v, sea + 1, planet.resolution);
        let up = pos.normalize();

        assert_eq!(
            planet.ground_block(pos - up * 0.1),
            Some(crate::material::BlockType::Ice)
        );
    }

    // regression test for the `effective_height` refactor (Critical 4): a liquid-less planet's
    // effective height must be clamped up to sea level over an underwater column, while a planet
    // with a liquid keeps reporting the raw (lower) natural height for the very same column
    #[test]
    fn effective_height_clamps_to_sea_level_only_on_liquid_less_planets() {
        let mut planet = PlanetData::new(TEST_RES);
        let (face, u, v) = first_underwater_column(&planet);
        let raw_height = planet.terrain.get_height(face, u, v);
        let sea = planet.terrain.sea_level();
        assert!(raw_height < sea, "sanity check: column must be underwater");

        // Earth-like (has liquid): effective height is the raw, un-clamped height
        assert_eq!(planet.effective_height(face, u, v), raw_height);

        // Ice (no liquid): effective height is clamped up to sea level
        planet.switch_planet_type(PlanetType::Ice);
        assert_eq!(planet.effective_height(face, u, v), sea);
    }
    #[test]
    fn lake_columns_hold_water_at_the_lake_level() {
        let (mut planet, (face, u, v)) = lake_planet(PlanetType::EarthLike);
        let level = planet.terrain.water_level(face, u, v);
        assert!(planet.holds_water(face, u, v));
        // water_depth is measured from the lake's surface, not the sea's
        let floor = planet.terrain.get_height(face, u, v);
        let pos =
            crate::gen::CoordSystem::get_block_center(face, u, v, floor + 1, planet.resolution);
        let depth = planet.water_depth(pos).unwrap();
        let expected =
            crate::gen::CoordSystem::get_layer_radius(level + 1, planet.resolution) - pos.length();
        assert!(
            (depth - expected).abs() < 1e-3 && depth > 1.0,
            "{depth} vs {expected}"
        );
        assert!(planet.water_surface_radius(face, u, v) > 0.0);
        // mining the bed keeps it flooded to the lake level
        planet
            .remove_block(BlockId {
                face,
                layer: floor,
                u,
                v,
            })
            .unwrap();
        assert!(planet.holds_water(face, u, v));
        // a block on the water cell displaces the water
        planet
            .add_block(
                BlockId {
                    face,
                    layer: level,
                    u,
                    v,
                },
                BlockType::Stone,
            )
            .unwrap();
        assert!(!planet.holds_water(face, u, v));
        assert_eq!(planet.water_surface_radius(face, u, v), 0.0);
    }

    #[test]
    fn a_hole_beside_a_lake_fills_only_to_sea_level() {
        let (mut planet, (face, u, v)) = lake_planet(PlanetType::EarthLike);
        let level = planet.terrain.water_level(face, u, v);
        // walk away from the lake until the column is no lake column
        let mut du = u;
        while planet.terrain.water_level(face, du, v) == level && du + 1 < planet.resolution {
            du += 1;
        }
        assert_eq!(
            planet.terrain.water_level(face, du, v),
            planet.terrain.sea_level()
        );
        let (h, sea) = (
            planet.terrain.get_height(face, du, v),
            planet.terrain.sea_level(),
        );
        assert!(
            level >= sea + 2,
            "lake level {level} leaves no room between it and the sea"
        );
        // down to just above the sea: below the lake's level
        for layer in (sea + 1..=h).rev() {
            let _ = planet.remove_block(BlockId {
                face,
                layer,
                u: du,
                v,
            });
        }
        assert!(!planet.exists(BlockId {
            face,
            layer: level - 1,
            u: du,
            v
        }));
        // dug below the lake's level but above the sea: dry (no flow from the lake)
        assert!(!planet.holds_water(face, du, v));
    }
    // Vertex mirrors shader.wgsl's VertexIn: pos, color, normal (vec3 each), water (f32) at offset 36
    #[test]
    fn vertex_layout_carries_the_water_radius() {
        assert_eq!(std::mem::size_of::<Vertex>(), 40);
        assert_eq!(std::mem::offset_of!(Vertex, water), 36);
        assert_eq!(Vertex::ATTRIBUTES[3].offset, 36);
        assert_eq!(Vertex::ATTRIBUTES[3].shader_location, 3);
    }

    // the smooth map is the mean effective height of the 3×3 columns around each column
    #[test]
    fn smooth_height_is_the_mean_of_the_3x3_neighbourhood() {
        let planet = PlanetData::new(32);
        let (face, u, v) = (0u8, 10u32, 12u32);
        let mut sum = 0.0;
        for dv in -1..=1 {
            for du in -1..=1 {
                let (f, cu, cv) = planet.neighbor_column(face, u, v, du, dv).unwrap();
                sum += planet.effective_height(f, cu, cv) as f32;
            }
        }
        assert!((planet.smooth_height(face, u, v) - sum / 9.0).abs() < 1e-5);
    }

    // at a cube-face edge the neighbourhood continues on the next face
    #[test]
    fn smooth_height_crosses_face_edges() {
        let planet = PlanetData::new(32);
        let (face, u, v) = (0u8, 0u32, 12u32);
        let mut sum = 0.0;
        let mut other_face = false;
        for dv in -1..=1 {
            for du in -1..=1 {
                let (f, cu, cv) = planet.neighbor_column(face, u, v, du, dv).unwrap();
                other_face |= f != face;
                sum += planet.effective_height(f, cu, cv) as f32;
            }
        }
        assert!(
            other_face,
            "the edge column has no neighbour on another face"
        );
        assert!((planet.smooth_height(face, u, v) - sum / 9.0).abs() < 1e-5);
    }

    // liquid-less planets smooth their sea-level fill: nothing dips below sea level
    #[test]
    fn liquid_less_smooth_heights_follow_the_basin_fill() {
        let planet = PlanetData::new_for_type(32, crate::noise::HOME_SEED, PlanetType::Ice);
        let sea = planet.terrain.sea_level() as f32;
        for face in 0..6u8 {
            for v in 0..32 {
                for u in 0..32 {
                    assert!(planet.smooth_height(face, u, v) >= sea - 1e-5);
                }
            }
        }
    }

    // baking is deterministic
    #[test]
    fn smooth_map_is_deterministic() {
        let a = PlanetData::new(32);
        let b = PlanetData::new(32);
        assert_eq!(a.smooth, b.smooth);
    }

    // a grid corner's smooth height is the mean of the up to four columns touching it
    #[test]
    fn smooth_corner_is_the_mean_of_its_columns() {
        let planet = PlanetData::new(32);
        let (face, u, v) = (0u8, 10u32, 12u32);
        let mean = (planet.smooth_height(face, 9, 11)
            + planet.smooth_height(face, 10, 11)
            + planet.smooth_height(face, 9, 12)
            + planet.smooth_height(face, 10, 12))
            / 4.0;
        assert!((planet.smooth_corner(face, u, v) - mean).abs() < 1e-5);
        // at the face's far edge only the columns on this face count
        let edge = (planet.smooth_height(face, 31, 11) + planet.smooth_height(face, 31, 12)) / 2.0;
        assert!((planet.smooth_corner(face, 32, 12) - edge).abs() < 1e-5);
    }

    // layer_of_radius inverts get_layer_radius_f
    #[test]
    fn layer_of_radius_inverts_the_layer_radius() {
        for layer in [3.0f32, 16.0, 17.25, 40.5] {
            let r = crate::gen::CoordSystem::get_layer_radius_f(layer, 32);
            assert!((crate::gen::CoordSystem::layer_of_radius(r, 32) - layer).abs() < 1e-3);
        }
    }
}
