//common.rs

use crate::biome::PlanetType;
use crate::material::{self, BlockType};
use crate::noise::PlanetTerrain;
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// --- CONSTANTS ---
pub const CHUNK_SIZE: u32 = 32;

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
    // the player's edits, shared (copy-on-write): every mesh worker gets a clone of the planet, and
    // cloning the Arc instead of every entry keeps that instant however much was edited
    pub chunks: Arc<HashMap<ChunkKey, ChunkMods>>,
    edits: usize, // entries in chunks (placed + mined), for MAX_EDITS
    pub resolution: u32,
    pub terrain: crate::noise::PlanetTerrain,
    pub planet_type: PlanetType,
    // the noise seed this planet was baked with (GalaxyPlanet::noise_seed, HOME_SEED for test planets);
    // nothing reads it at runtime since resizing is gone, kept as a record (and checked by tests)
    #[allow(dead_code)]
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

        Self {
            chunks: Arc::default(),
            edits: 0,
            resolution,
            terrain, // <--- Store it
            planet_type: PlanetType::EarthLike,
            seed,
        }
    }

    // switches the active planet type in place: clears edits (they reference the old palette)
    // but keeps the same terrain shape — phase 1 reuses one noise map for every planet type
    pub fn switch_planet_type(&mut self, to: PlanetType) {
        self.planet_type = to;
        self.chunks = Arc::default();
        self.edits = 0;
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
        self.edits
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
        let (was_mined, was_placed) = self.chunks.get(&key).map_or((false, false), |m| {
            (m.mined.contains(&id), m.placed.contains_key(&id))
        });
        // putting back what was mined there just restores the terrain
        let restores = natural == Some(ty);
        if !restores && !was_mined && !was_placed && self.edits >= MAX_EDITS {
            return Err(EditRefused::EditLimit);
        }

        let mods = Arc::make_mut(&mut self.chunks)
            .entry(key)
            .or_insert_with(ChunkMods::new);
        let before = mods.mined.len() + mods.placed.len();
        mods.mined.remove(&id);
        if restores {
            mods.placed.remove(&id);
        } else {
            mods.placed.insert(id, ty);
        }
        self.edits = self.edits + mods.mined.len() + mods.placed.len() - before;
        Ok(())
    }

    pub fn remove_block(&mut self, id: BlockId) -> Result<(), EditRefused> {
        if id.layer < self.mining_floor() {
            return Err(EditRefused::MiningFloor);
        }
        let terrain_below = id.layer <= self.terrain.get_height(id.face, id.u, id.v);
        let key = Self::get_chunk_key(id);
        let (was_mined, was_placed) = self.chunks.get(&key).map_or((false, false), |m| {
            (m.mined.contains(&id), m.placed.contains_key(&id))
        });
        if terrain_below && !was_mined && !was_placed && self.edits >= MAX_EDITS {
            return Err(EditRefused::EditLimit);
        }

        let mods = Arc::make_mut(&mut self.chunks)
            .entry(key)
            .or_insert_with(ChunkMods::new);
        let before = mods.mined.len() + mods.placed.len();
        mods.placed.remove(&id);
        if terrain_below {
            mods.mined.insert(id);
        }
        self.edits = self.edits + mods.mined.len() + mods.placed.len() - before;
        Ok(())
    }

    pub fn exists(&self, id: BlockId) -> bool {
        let key = Self::get_chunk_key(id);
        if let Some(mods) = self.chunks.get(&key) {
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

    // picks a spawn direction: `preferred` unless the active liquid is damaging and `preferred`'s
    // column is underwater (an unescapable death loop, since floating alone still ticks damage), in
    // which case it searches a dense, evenly-spread set of directions over the whole sphere for one
    // at or above sea level
    pub fn safe_spawn_direction(&self, preferred: glam::Vec3) -> glam::Vec3 {
        let damaging = self.planet_type.def().liquid.is_some_and(|l| l.damaging);
        if !damaging {
            return preferred;
        }
        let sea_level = self.terrain.sea_level();
        let probe_height = |dir: glam::Vec3| {
            crate::gen::CoordSystem::pos_to_id(
                dir * (self.resolution as f32 / 2.0),
                self.resolution,
            )
            .map(|id| self.terrain.get_height(id.face, id.u, id.v))
        };
        if !probe_height(preferred).is_some_and(|h| h < sea_level) {
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
            .find(|&dir| probe_height(dir).is_some_and(|h| h >= sea_level))
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
        let res = self.resolution;
        let (nu, nv) = (u as i32 + du, v as i32 + dv);
        if nu >= 0 && nv >= 0 && nu < res as i32 && nv < res as i32 {
            return Some((face, nu as u32, nv as u32));
        }
        let mid = res / 2;
        let here = crate::gen::CoordSystem::get_block_center(face, u, v, mid, res);
        let inner = crate::gen::CoordSystem::get_block_center(
            face,
            (u as i32 - du) as u32,
            (v as i32 - dv) as u32,
            mid,
            res,
        );
        crate::gen::CoordSystem::pos_to_id(here * 2.0 - inner, res).map(|id| (id.face, id.u, id.v))
    }

    pub fn neighbor_height(&self, face: u8, u: u32, v: u32, du: i32, dv: i32) -> u32 {
        self.neighbor_column(face, u, v, du, dv)
            .map_or(0, |(f, nu, nv)| self.effective_height(f, nu, nv))
    }

    pub fn chunk_key(id: BlockId) -> ChunkKey {
        Self::get_chunk_key(id)
    }

    // whether the column's sea-level cell holds the planet's liquid. Water table: any empty cell at sea
    // level does, natural ocean or dug out, so holes dug below sea level fill (not only those that
    // connect to the sea). Shared by the rendered water (MeshGen::build_water) and swimming.
    pub fn sea_cell_is_water(&self, face: u8, u: u32, v: u32) -> bool {
        let layer = self.terrain.sea_level();
        self.planet_type.def().liquid.is_some() && !self.exists(BlockId { face, layer, u, v })
    }

    // how far `pos` lies below the sea surface (negative above it), or None outside water: the column's
    // sea-level cell must hold water (sea_cell_is_water)
    pub fn water_depth(&self, pos: glam::Vec3) -> Option<f32> {
        let id = crate::gen::CoordSystem::pos_to_id(pos, self.resolution)?;
        if !self.sea_cell_is_water(id.face, id.u, id.v) {
            return None;
        }
        let sea = self.terrain.sea_level();
        Some(crate::gen::CoordSystem::get_layer_radius(sea + 1, self.resolution) - pos.length())
    }

    // the type of an existing block, None for air
    pub fn block_type(&self, id: BlockId) -> Option<BlockType> {
        if let Some(mods) = self.chunks.get(&Self::get_chunk_key(id)) {
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
        assert!(Arc::ptr_eq(&planet.chunks, &copy.chunks));
        planet
            .remove_block(id(0, 4, 4, planet.terrain.get_height(0, 4, 4)))
            .unwrap();
        assert!(!Arc::ptr_eq(&planet.chunks, &copy.chunks));
        assert_eq!(copy.edit_count(), 1, "the copy keeps its own snapshot");
        assert_eq!(planet.edit_count(), 2);
    }

    // at the cap, new entries are refused, but undoing edits still works
    #[test]
    fn the_edit_cap_refuses_new_entries_but_allows_undo() {
        let mut planet = PlanetData::new(TEST_RES);
        let h = planet.terrain.get_height(0, 3, 3);
        planet.remove_block(id(0, 3, 3, h)).unwrap();
        planet.edits = MAX_EDITS; // pretend the planet is full
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
        assert_eq!(planet.edits, MAX_EDITS - 1);
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
        assert!(planet.chunks.values().any(|m| !m.placed.is_empty()));

        let height_before = planet.terrain.get_height(0, 5, 5);
        planet.switch_planet_type(PlanetType::Volcanic);
        assert!(planet
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
}
