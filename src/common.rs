//common.rs

use std::collections::{HashMap, HashSet};
use bytemuck::{Pod, Zeroable};
use crate::noise::PlanetTerrain;
use crate::material::{self, BlockType};

// --- CONSTANTS ---
pub const CHUNK_SIZE: u32 = 32;

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
        Self { mined: HashSet::new(), placed: HashMap::new() }
    }
}

#[derive(Clone)] 
pub struct PlanetData {
    pub chunks: HashMap<ChunkKey, ChunkMods>, 
    pub resolution: u32,
    pub has_core: bool,
    pub terrain: crate::noise::PlanetTerrain,
}

impl PlanetData {
    pub fn new(resolution: u32) -> Self {
        println!("Generating Terrain Noise Map for res {}...", resolution);
        let terrain = PlanetTerrain::new(resolution); // calculate once
        println!("Terrain Generation Complete.");
        
        Self {
            chunks: HashMap::new(),
            resolution,
            has_core: true,
            terrain, // <--- Store it
        }
    }

pub fn resize(&mut self, increase: bool) {
        if increase {
            // multiply by 1.2
            // i use .max(self.resolution + 1) to ensure it always grows by at least 1 block
            let new_res = (self.resolution as f32 * 1.2) as u32;
            self.resolution = new_res.max(self.resolution + 1).min(16384); 
        } else {
            // divide by 1.2
            let new_res = (self.resolution as f32 / 1.2) as u32;
            self.resolution = new_res.max(8);
        }
        

        self.chunks.clear();
        
        // regenerate noise map for new resolution
        println!("Regenerating Terrain for new res {}...", self.resolution);
        self.terrain = PlanetTerrain::new(self.resolution); 
    }

    fn get_chunk_key(id: BlockId) -> ChunkKey {
        ChunkKey {
            face: id.face,
            u_idx: id.u / CHUNK_SIZE,
            v_idx: id.v / CHUNK_SIZE,
        }
    }

    pub fn add_block(&mut self, id: BlockId, ty: BlockType) {
        let natural = self.natural_type(id);
        let key = Self::get_chunk_key(id);
        let mods = self.chunks.entry(key).or_insert_with(ChunkMods::new);

        mods.mined.remove(&id);
        // putting back what was mined there just restores the terrain
        if natural == Some(ty) {
            mods.placed.remove(&id);
        } else {
            mods.placed.insert(id, ty);
        }
    }

pub fn remove_block(&mut self, id: BlockId) {
        // protect the bottom 4 layers as the unbreakable core
        if self.has_core && id.layer < 6 {
            return; 
        }
        
        let terrain_below = id.layer <= self.terrain.get_height(id.face, id.u, id.v);
        let key = Self::get_chunk_key(id);
        let mods = self.chunks.entry(key).or_insert_with(ChunkMods::new);

        mods.placed.remove(&id);
        if terrain_below {
            mods.mined.insert(id);
        }
    }
    
    pub fn exists(&self, id: BlockId) -> bool {
        let key = Self::get_chunk_key(id);
        if let Some(mods) = self.chunks.get(&key) {
            if mods.placed.contains_key(&id) { return true; }
            if mods.mined.contains(&id) { return false; }
        }
        

        // instead of a flat floor, we check the pre-calculated noise map
        let height = self.terrain.get_height(id.face, id.u, id.v);
        id.layer <= height
    }

    // height of the column next to (face, u, v) in direction (du, dv); across a cube-face edge that is
    // a column of the neighbouring face, found by continuing the line from the inner neighbour outward
    pub fn neighbor_height(&self, face: u8, u: u32, v: u32, du: i32, dv: i32) -> u32 {
        let res = self.resolution;
        let (nu, nv) = (u as i32 + du, v as i32 + dv);
        if nu >= 0 && nv >= 0 && nu < res as i32 && nv < res as i32 {
            return self.terrain.get_height(face, nu as u32, nv as u32);
        }
        let mid = res / 2;
        let here = crate::gen::CoordSystem::get_block_center(face, u, v, mid, res);
        let inner = crate::gen::CoordSystem::get_block_center(face, (u as i32 - du) as u32, (v as i32 - dv) as u32, mid, res);
        match crate::gen::CoordSystem::pos_to_id(here * 2.0 - inner, res) {
            Some(id) => self.terrain.get_height(id.face, id.u, id.v),
            None => 0,
        }
    }

    // the type of an existing block, None for air
    pub fn block_type(&self, id: BlockId) -> Option<BlockType> {
        if let Some(mods) = self.chunks.get(&Self::get_chunk_key(id)) {
            if let Some(&ty) = mods.placed.get(&id) { return Some(ty); }
            if mods.mined.contains(&id) { return None; }
        }
        self.natural_type(id)
    }

    // the terrain's own block at this position, ignoring edits
    fn natural_type(&self, id: BlockId) -> Option<BlockType> {
        if id.layer > self.terrain.get_height(id.face, id.u, id.v) { return None; }
        Some(material::natural_type(&self.terrain, self.has_core, id.face, id.u, id.v, id.layer))
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