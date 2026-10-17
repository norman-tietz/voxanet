// Plants as natural blocks: a pure function of (planet seed, planet type, column). Anchors sit on a jittered
// grid of ANCHOR_CELL-column cells, are accepted by a forest mask and ground rules, and expand into hex-ring
// templates (rings over `column_neighbors`, so they follow the planet's cell shape).

use crate::meshing::gen::CoordSystem;
use crate::planet::biome::PlanetType;
use crate::planet::data::{BlockId, ChunkKey, PlanetData, CHUNK_SIZE};
use crate::planet::material::BlockType;
use crate::planet::noise::NoiseGenerator;
use crate::weather::smoothstep;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

pub const ANCHOR_CELL: u32 = 6;
pub const MAX_PLANT_HEIGHT: u32 = 12;
pub const MAX_PLANT_REACH: u32 = 3;

// canopy radius in rings
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Template {
    Blob(u32),
    Cone(u32),
    Column,
    Arms,
    Crown(u32),
    Mushroom(u32),
    Spire,
    Prisms,
    Boulders,
}

#[derive(Clone, Copy, Debug)]
pub struct Species {
    pub name: &'static str,
    pub trunk: BlockType,
    pub crown: BlockType,
    pub tip: Option<BlockType>,
    pub ground: &'static [BlockType],
    pub height: (u32, u32),
    pub template: Template,
    pub density: f32,
    // allowed column height as a share of the peak above sea level
    pub band: (f32, f32),
    pub beach_ok: bool,
}

#[allow(clippy::too_many_arguments)]
const fn sp(
    name: &'static str,
    trunk: BlockType,
    crown: BlockType,
    tip: Option<BlockType>,
    ground: &'static [BlockType],
    height: (u32, u32),
    template: Template,
    density: f32,
    band: (f32, f32),
    beach_ok: bool,
) -> Species {
    Species {
        name,
        trunk,
        crown,
        tip,
        ground,
        height,
        template,
        density,
        band,
        beach_ok,
    }
}

use BlockType as B;
const EARTH_LIKE: &[Species] = &[
    sp(
        "Oak",
        B::Wood,
        B::Leaves,
        None,
        &[B::Grass, B::Dirt],
        (4, 6),
        Template::Blob(2),
        1.0,
        (0.0, 0.25),
        false,
    ),
    sp(
        "Pine",
        B::Wood,
        B::Pine,
        None,
        &[B::Grass, B::Dirt, B::Stone],
        (6, 9),
        Template::Cone(2),
        0.8,
        (0.25, 0.55),
        false,
    ),
];

const OCEAN: &[Species] = &[sp(
    "Palm",
    B::Wood,
    B::Palm,
    None,
    &[B::Sand, B::Dirt, B::Grass, B::Stone],
    (5, 8),
    Template::Crown(2),
    1.0,
    (0.0, 1.0),
    true,
)];
const DESERT: &[Species] = &[
    sp(
        "Cactus",
        B::Cactus,
        B::Cactus,
        None,
        &[B::Sand],
        (2, 5),
        Template::Arms,
        0.6,
        (0.0, 0.7),
        false,
    ),
    sp(
        "Dry shrub",
        B::DeadWood,
        B::DeadWood,
        None,
        &[B::Sand],
        (1, 2),
        Template::Column,
        0.4,
        (0.0, 1.0),
        false,
    ),
];
const ICE: &[Species] = &[
    sp(
        "Frozen pine",
        B::Wood,
        B::Snow,
        None,
        &[B::Stone, B::Snow, B::Ice],
        (5, 8),
        Template::Cone(2),
        0.4,
        (0.0, 1.0),
        false,
    ),
    sp(
        "Ice cluster",
        B::Ice,
        B::Ice,
        None,
        &[B::Stone, B::Snow, B::Ice],
        (2, 4),
        Template::Prisms,
        0.25,
        (0.0, 1.0),
        false,
    ),
];
const VOLCANIC: &[Species] = &[
    sp(
        "Obsidian spire",
        B::Obsidian,
        B::Obsidian,
        Some(B::Ember),
        &[B::Basalt, B::Ash, B::Obsidian],
        (3, 8),
        Template::Spire,
        0.5,
        (0.0, 1.0),
        false,
    ),
    sp(
        "Charred stump",
        B::DeadWood,
        B::DeadWood,
        None,
        &[B::Basalt, B::Ash, B::Obsidian],
        (1, 3),
        Template::Column,
        0.5,
        (0.0, 1.0),
        false,
    ),
];
const TOXIC: &[Species] = &[sp(
    "Giant mushroom",
    B::Stem,
    B::Cap,
    None,
    &[B::Dirt, B::Ash, B::Obsidian, B::Stone],
    (3, 6),
    Template::Mushroom(2),
    0.45,
    (0.0, 1.0),
    true,
)];
const SHATTERED: &[Species] = &[sp(
    "Crystal cluster",
    B::Crystal,
    B::Crystal,
    None,
    &[B::Basalt, B::Ember],
    (3, 7),
    Template::Prisms,
    0.6,
    (0.0, 1.0),
    false,
)];
const CRATERED: &[Species] = &[sp(
    "Boulder pile",
    B::Stone,
    B::Basalt,
    None,
    &[B::MoonDust, B::Basalt, B::Stone],
    (2, 4),
    Template::Boulders,
    0.7,
    (0.0, 1.0),
    true,
)];

pub fn species_for(t: PlanetType) -> &'static [Species] {
    match t {
        PlanetType::EarthLike => EARTH_LIKE,
        PlanetType::Ocean => OCEAN,
        PlanetType::Desert => DESERT,
        PlanetType::Ice => ICE,
        PlanetType::Volcanic => VOLCANIC,
        PlanetType::Toxic => TOXIC,
        PlanetType::Shattered => SHATTERED,
        PlanetType::Cratered => CRATERED,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FloraConfig {
    pub planet_type: PlanetType,
    pub seed: u32,
    pub density: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plant {
    pub face: u8,
    pub u: u32,
    pub v: u32,
    pub species: usize,
    pub height: u32,
    pub hash: u32,
}

use crate::weather::lightning::{pcg, unit};

fn hash(seed: u32, face: u8, a: u32, b: u32, salt: u32) -> u32 {
    let mut h = pcg(seed ^ 0x5eed_f10a);
    h = pcg(h ^ face as u32);
    h = pcg(h ^ a);
    h = pcg(h ^ b);
    pcg(h ^ salt)
}

// the forest mask: 0 in a clearing, 1 in a grove (seeded low-frequency noise through a smoothstep).
// fbm(3 octaves) of this noise spans about -0.5..0.5; the thresholds put ~10-25 % of the sphere on each side
pub fn forest_mask(cfg: &FloraConfig, dir: Vec3) -> f32 {
    thread_local! {
        // the noise tables are built once per thread and seed (this runs per anchor and per LOD vertex)
        static GEN: std::cell::RefCell<Option<(u32, NoiseGenerator)>> = const { std::cell::RefCell::new(None) };
    }
    GEN.with(|g| {
        let mut g = g.borrow_mut();
        if g.as_ref().is_none_or(|(seed, _)| *seed != cfg.seed) {
            *g = Some((cfg.seed, NoiseGenerator::new(cfg.seed ^ 0xF04E57)));
        }
        let n = g
            .as_ref()
            .unwrap()
            .1
            .fbm(dir * 3.0 + Vec3::new(5.1, -2.7, 8.3), 3);
        smoothstep(MASK_LO, MASK_HI, n)
    })
}
const MASK_LO: f32 = -0.1;
const MASK_HI: f32 = 0.1;

// the species that fit column (face, u, v) by ground block, height band and beach rule (everything
// `anchor_at` asks before the hashes and the mask), with the column's ground height
fn fitting_species(data: &PlanetData, cfg: &FloraConfig, face: u8, u: u32, v: u32) -> Vec<usize> {
    let all = species_for(cfg.planet_type);
    let h = data.effective_height(face, u, v);
    let sea = data.terrain.sea_level();
    if all.is_empty()
        || h <= sea
        || data.terrain.water_level(face, u, v) >= h
        || data.terrain.in_lake_region(face, u, v)
    {
        return Vec::new();
    }
    let peak = data.terrain.height_range().1;
    let share = (h - sea) as f32 / (peak.saturating_sub(sea)).max(1) as f32;
    let Some(ground) = data.natural_type(BlockId {
        face,
        layer: h,
        u,
        v,
    }) else {
        return Vec::new();
    };
    (0..all.len())
        .filter(|&i| {
            let sp = &all[i];
            sp.ground.contains(&ground)
                && share >= sp.band.0
                && share <= sp.band.1
                && (sp.beach_ok || h >= sea + 2)
        })
        .collect()
}

// the LOD tint of a column: the share (0..=1) of the colour that leans toward a species' crown (the first
// species that fits the column's ground and height band): the forest mask times its density. None without
// flora, over water, or where no species could grow
pub fn forest_crown(data: &PlanetData, face: u8, u: u32, v: u32) -> Option<(f32, BlockType)> {
    let f = data.flora.as_ref()?;
    let fits = fitting_species(data, &f.cfg, face, u, v);
    let sp = &species_for(f.cfg.planet_type)[*fits.first()?];
    let h = data.effective_height(face, u, v);
    let dir = CoordSystem::get_block_center(face, u, v, h, data.resolution).normalize();
    let t = (sp.density * f.cfg.density * forest_mask(&f.cfg, dir)).clamp(0.0, 1.0);
    Some((t, sp.crown))
}

#[cfg(test)]
pub fn forest_tint(data: &PlanetData, face: u8, u: u32, v: u32) -> f32 {
    forest_crown(data, face, u, v).map_or(0.0, |(t, _)| t)
}

// the anchor planted in column (face, u, v), if any
pub fn anchor_at(data: &PlanetData, cfg: &FloraConfig, face: u8, u: u32, v: u32) -> Option<Plant> {
    let all = species_for(cfg.planet_type);
    if all.is_empty() {
        return None;
    }
    let (cu, cv) = (u / ANCHOR_CELL, v / ANCHOR_CELL);
    let ju = hash(cfg.seed, face, cu, cv, 1) % ANCHOR_CELL;
    let jv = hash(cfg.seed, face, cu, cv, 2) % ANCHOR_CELL;
    if (u % ANCHOR_CELL, v % ANCHOR_CELL) != (ju, jv) {
        return None;
    }
    let res = data.resolution;
    let h = data.effective_height(face, u, v);
    let fits = fitting_species(data, cfg, face, u, v);
    if fits.is_empty() {
        return None;
    }
    let species = fits[(hash(cfg.seed, face, cu, cv, 4) as usize) % fits.len()];
    let sp = &all[species];
    let dir = CoordSystem::get_block_center(face, u, v, h, res).normalize();
    let accept = sp.density * cfg.density * forest_mask(cfg, dir);
    if unit(hash(cfg.seed, face, cu, cv, 3)) >= accept {
        return None;
    }
    for (nf, nu, nv) in data.column_neighbors(face, u, v) {
        if (data.effective_height(nf, nu, nv) as i64 - h as i64).abs() > 1 {
            return None;
        }
    }
    let span = sp.height.1 - sp.height.0 + 1;
    Some(Plant {
        face,
        u,
        v,
        species,
        height: sp.height.0 + hash(cfg.seed, face, cu, cv, 5) % span,
        hash: hash(cfg.seed, face, cu, cv, 6),
    })
}

// breadth-first rings over the column neighbours: ring 0 is the column itself
pub fn rings(data: &PlanetData, face: u8, u: u32, v: u32, max: u32) -> Vec<Vec<(u8, u32, u32)>> {
    let mut seen: HashSet<(u8, u32, u32)> = HashSet::new();
    seen.insert((face, u, v));
    let mut out = vec![vec![(face, u, v)]];
    for _ in 0..max {
        let mut next = Vec::new();
        for &(f, cu, cv) in out.last().unwrap() {
            for c in data.column_neighbors(f, cu, cv) {
                if seen.insert(c) {
                    next.push(c);
                }
            }
        }
        out.push(next);
    }
    out
}

// which columns of a ring a template cell fills
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Skip {
    // all of them
    No,
    // all but about one in five (an uneven outline)
    Ragged,
    // only the columns whose hash (0..6) is below the number: nested counts give columns of different heights
    Part(u32),
}

// (ring, layer offset above the anchor's ground, type, which columns); `hash` is the plant's own
fn template_cells(sp: &Species, h: u32, hash: u32) -> Vec<(u32, u32, BlockType, Skip)> {
    use Skip::{No, Part, Ragged};
    let mut c = Vec::new();
    match sp.template {
        Template::Blob(r) => {
            for o in 1..h {
                c.push((0, o, sp.trunk, No));
            }
            for ring in 0..=r {
                for o in [h - 1, h] {
                    if ring > 0 || o == h {
                        c.push((ring, o, sp.crown, if ring == r { Ragged } else { No }));
                    }
                }
            }
            for ring in 0..r {
                c.push((
                    ring,
                    h + 1,
                    sp.crown,
                    if ring + 1 == r { Ragged } else { No },
                ));
            }
            c.push((0, h + 2, sp.crown, No));
        }
        Template::Cone(r) => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
            let start = (h / 3).max(1);
            let top = h + 1;
            for o in start..=top {
                let t = (o - start) as f32 / (top - start).max(1) as f32;
                let radius = (r as f32 * (1.0 - t)).round() as u32;
                for ring in 1..=radius {
                    c.push((ring, o, sp.crown, No));
                }
            }
            c.push((0, top, sp.tip.unwrap_or(sp.crown), No));
        }
        Template::Crown(r) => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
            for ring in 1..=r {
                c.push((ring, h + 1, sp.crown, if ring == 1 { Ragged } else { No }));
            }
            // drooping fronds
            if r >= 2 {
                c.push((r, h, sp.crown, Part(3)));
            }
            c.push((0, h + 1, sp.crown, No));
        }
        Template::Column => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
        }
        Template::Arms => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
            if h >= 3 {
                // one or two arms: a ring-1 column out, then up
                let o = h / 2;
                c.push((1, o, sp.trunk, Part(1 + hash % 2)));
                c.push((1, o + 1, sp.trunk, Part(1 + hash % 2)));
                c.push((1, o + 2, sp.trunk, Part(1 + hash % 2)));
            }
        }
        Template::Spire => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
            for o in 1..=h / 2 {
                c.push((1, o, sp.trunk, No));
            }
            if let Some(tip) = sp.tip {
                if hash.is_multiple_of(4) {
                    c.push((0, h, tip, No));
                }
            }
        }
        Template::Mushroom(r) => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
            for ring in 0..=r {
                c.push((ring, h + 1, sp.crown, No));
            }
            c.push((r, h, sp.crown, No));
        }
        Template::Prisms => {
            for o in 1..=h {
                c.push((0, o, sp.trunk, No));
            }
            for (cols, lower) in [(3, 3), (2, 2), (1, 1)] {
                for o in 1..=h.saturating_sub(lower) {
                    c.push((1, o, sp.trunk, Part(cols)));
                }
            }
        }
        Template::Boulders => {
            // mostly one stone, now and then the dark one
            let ty = if hash.is_multiple_of(3) {
                sp.crown
            } else {
                sp.trunk
            };
            for o in 1..=(h / 2 + 1).min(h) {
                c.push((0, o, ty, No));
            }
            c.push((1, 1, ty, Part(3)));
            c.push((1, 2, ty, Part(1)));
            if h >= 3 {
                c.push((2, 1, ty, Part(1)));
            }
        }
    }
    c
}

// the plant's blocks above the ground of their own columns
pub fn plant_blocks(
    data: &PlanetData,
    plant: &Plant,
    cfg: &FloraConfig,
) -> Vec<(BlockId, BlockType)> {
    let sp = &species_for(cfg.planet_type)[plant.species];
    let base = data.effective_height(plant.face, plant.u, plant.v);
    let cells = template_cells(sp, plant.height, plant.hash);
    let reach = cells.iter().map(|c| c.0).max().unwrap_or(0);
    let rs = rings(data, plant.face, plant.u, plant.v, reach);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (ring, offset, ty, skip) in cells {
        let Some(columns) = rs.get(ring as usize) else {
            continue;
        };
        let layer = base + offset;
        for &(face, u, v) in columns {
            // the same window `PlanetData::decor_block` accepts, so the cache and `exists` agree
            let ground = data.effective_height(face, u, v);
            if layer <= ground || layer > ground + MAX_PLANT_HEIGHT {
                continue;
            }
            let skipped = match skip {
                Skip::No => false,
                Skip::Ragged => hash(plant.hash, face, u, v, layer).is_multiple_of(5),
                Skip::Part(n) => hash(plant.hash, face, u, v, 77) % 6 >= n,
            };
            if skipped {
                continue;
            }
            let id = BlockId { face, layer, u, v };
            if seen.insert(id) {
                out.push((id, ty));
            }
        }
    }
    out
}

// the largest ring any template of the type's species reaches (at most MAX_PLANT_REACH)
fn max_reach(t: PlanetType) -> u32 {
    // computed once per type (this runs for every chunk)
    static REACH: std::sync::OnceLock<Vec<(PlanetType, u32)>> = std::sync::OnceLock::new();
    REACH
        .get_or_init(|| {
            PlanetType::ALL
                .iter()
                .map(|&t| (t, compute_reach(t)))
                .collect()
        })
        .iter()
        .find(|(k, _)| *k == t)
        .map_or(0, |(_, r)| *r)
}

fn compute_reach(t: PlanetType) -> u32 {
    species_for(t)
        .iter()
        .flat_map(|sp| (sp.height.0..=sp.height.1).map(move |h| (sp, h)))
        .flat_map(|(sp, h)| template_cells(sp, h, 0).into_iter().map(|c| c.0))
        .max()
        .unwrap_or(0)
        .min(MAX_PLANT_REACH)
}

// the plant blocks of one chunk, and per column of the chunk its highest plant layer
pub struct ChunkDecor {
    pub blocks: HashMap<BlockId, BlockType>,
    pub tops: HashMap<(u32, u32), u32>,
    // per column of the chunk (u % CHUNK_SIZE * CHUNK_SIZE + v % CHUNK_SIZE) the height of its highest
    // plant block above the column's ground, 0 = no plant: most `exists` queries end on this one read
    col_top: Vec<u8>,
}

impl ChunkDecor {
    fn empty() -> Self {
        Self {
            blocks: HashMap::new(),
            tops: HashMap::new(),
            col_top: vec![0; (CHUNK_SIZE * CHUNK_SIZE) as usize],
        }
    }

    fn slot(u: u32, v: u32) -> usize {
        ((u % CHUNK_SIZE) * CHUNK_SIZE + v % CHUNK_SIZE) as usize
    }

    // the block at `id` for a column whose ground is at `ground` (id must be in this chunk)
    pub fn block_above(&self, id: BlockId, ground: u32) -> Option<BlockType> {
        let top = self.col_top[Self::slot(id.u, id.v)] as u32;
        if top == 0 || id.layer > ground + top {
            return None;
        }
        self.blocks.get(&id).copied()
    }
}

// Pure: every anchor whose plant can reach the chunk is generated whole and its blocks inside the chunk kept.
// Such an anchor lies in the chunk or within MAX_PLANT_REACH rings of it; the outside ones are found by
// expanding the chunk's outermost columns through `column_neighbors`, which crosses cube-face seams.
pub fn decor_chunk(data: &PlanetData, cfg: &FloraConfig, key: ChunkKey) -> ChunkDecor {
    let mut out = ChunkDecor::empty();
    if species_for(cfg.planet_type).is_empty() {
        return out;
    }
    let res = data.resolution;
    let reach = max_reach(cfg.planet_type);
    let (u0, v0) = (key.u_idx * CHUNK_SIZE, key.v_idx * CHUNK_SIZE);
    let (u1, v1) = ((u0 + CHUNK_SIZE).min(res), (v0 + CHUNK_SIZE).min(res));
    let mut candidates: Vec<(u8, u32, u32)> = Vec::new();
    if u0 >= reach && v0 >= reach && u1 + reach <= res && v1 + reach <= res {
        // away from the face edges `reach` rings stay within `reach` columns in u and v: only the anchor
        // cells over that square are looked at, each holding at most one candidate
        let (a0, a1) = ((u0 - reach) / ANCHOR_CELL, (u1 + reach - 1) / ANCHOR_CELL);
        let (b0, b1) = ((v0 - reach) / ANCHOR_CELL, (v1 + reach - 1) / ANCHOR_CELL);
        for cu in a0..=a1 {
            for cv in b0..=b1 {
                let u = cu * ANCHOR_CELL + hash(cfg.seed, key.face, cu, cv, 1) % ANCHOR_CELL;
                let v = cv * ANCHOR_CELL + hash(cfg.seed, key.face, cu, cv, 2) % ANCHOR_CELL;
                if (u0 - reach..u1 + reach).contains(&u) && (v0 - reach..v1 + reach).contains(&v) {
                    candidates.push((key.face, u, v));
                }
            }
        }
    } else {
        let mut seen: HashSet<(u8, u32, u32)> =
            HashSet::with_capacity(((u1 - u0) * (v1 - v0)) as usize + 512);
        let mut frontier = Vec::new();
        for u in u0..u1 {
            for v in v0..v1 {
                seen.insert((key.face, u, v));
                candidates.push((key.face, u, v));
                if u == u0 || u + 1 == u1 || v == v0 || v + 1 == v1 {
                    frontier.push((key.face, u, v));
                }
            }
        }
        for _ in 0..reach {
            let mut next = Vec::new();
            for &(f, u, v) in &frontier {
                for c in data.column_neighbors(f, u, v) {
                    if seen.insert(c) {
                        next.push(c);
                        candidates.push(c);
                    }
                }
            }
            frontier = next;
        }
    }
    // overlapping plants resolve by anchor order (face, u, v): the later anchor's block wins, in any chunk
    candidates.sort_unstable();
    for (f, u, v) in candidates {
        let Some(plant) = anchor_at(data, cfg, f, u, v) else {
            continue;
        };
        for (id, t) in plant_blocks(data, &plant, cfg) {
            if PlanetData::chunk_key(id) == key {
                let top = out.tops.entry((id.u, id.v)).or_insert(id.layer);
                *top = (*top).max(id.layer);
                out.blocks.insert(id, t);
            }
        }
    }
    for (&(u, v), &top) in &out.tops {
        let ground = data.effective_height(key.face, u, v);
        out.col_top[ChunkDecor::slot(u, v)] = (top - ground).min(u8::MAX as u32) as u8;
    }
    out
}

// hard cap of cached chunks (a full flight over forests would otherwise grow without bound)
pub const MAX_CACHED_CHUNKS: usize = 2048;

// The natural plants of a planet: its config and the per-chunk cache, shared (behind an Arc) by the
// planet's clones on the mesh worker threads. Generation runs outside the lock.
pub struct Flora {
    pub cfg: FloraConfig,
    cache: RwLock<HashMap<ChunkKey, Arc<ChunkDecor>>>,
}

impl Flora {
    pub fn new(cfg: FloraConfig) -> Self {
        Self {
            cfg,
            cache: RwLock::new(HashMap::new()),
        }
    }

    // the chunk's decor: a read-locked hit, else generated with no lock held and inserted under a short
    // write lock (a racing thread's equal result may be stored first; whichever is stored is returned)
    pub fn chunk(&self, data: &PlanetData, key: ChunkKey) -> Arc<ChunkDecor> {
        if let Some(hit) = self.cache.read().unwrap().get(&key) {
            return hit.clone();
        }
        let made = Arc::new(decor_chunk(data, &self.cfg, key));
        let mut map = self.cache.write().unwrap();
        if map.len() >= MAX_CACHED_CHUNKS && !map.contains_key(&key) {
            // drop one entry, not the whole cache (a flight over forests would regenerate everything)
            if let Some(&old) = map.keys().next() {
                map.remove(&old);
            }
        }
        map.entry(key).or_insert(made).clone()
    }

    // the plant block at `id` over a column with the given ground, answered under the read lock (no Arc
    // clone); a miss generates the chunk first
    pub fn block_above(&self, data: &PlanetData, id: BlockId, ground: u32) -> Option<BlockType> {
        let key = PlanetData::chunk_key(id);
        if let Some(hit) = self.cache.read().unwrap().get(&key) {
            return hit.block_above(id, ground);
        }
        self.chunk(data, key).block_above(id, ground)
    }

    pub fn evict(&self, key: ChunkKey) {
        self.cache.write().unwrap().remove(&key);
    }

    // drops the cached chunks `keep` rejects
    pub fn retain(&self, keep: impl Fn(&ChunkKey) -> bool) {
        self.cache.write().unwrap().retain(|k, _| keep(k));
    }

    #[cfg(test)]
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::data::CellShape;

    fn planet() -> (PlanetData, FloraConfig) {
        let mut p = PlanetData::new_for_type(128, 7, PlanetType::EarthLike);
        p.cells = CellShape::Hex; // the game's cell shape: six neighbours
        let cfg = FloraConfig {
            planet_type: PlanetType::EarthLike,
            seed: 7,
            density: 1.0,
        };
        (p, cfg)
    }

    // all anchors of one face of a planet
    fn anchors(p: &PlanetData, cfg: &FloraConfig, face: u8) -> Vec<Plant> {
        (0..p.resolution)
            .flat_map(|u| (0..p.resolution).map(move |v| (u, v)))
            .filter_map(|(u, v)| anchor_at(p, cfg, face, u, v))
            .collect()
    }

    #[test]
    fn anchors_are_deterministic_sparse_and_clustered() {
        let (p, cfg) = planet();
        let a = anchors(&p, &cfg, 0);
        assert_eq!(a, anchors(&p, &cfg, 0));
        assert!(!a.is_empty(), "an Earth-like face has plants");
        let columns = (p.resolution * p.resolution) as f32;
        let share = a.len() as f32 / columns;
        eprintln!("tree share {share}");
        assert!(
            share > 0.002 && share < 0.06,
            "tree share of columns {share}"
        );
        // at most one anchor per anchor cell
        let mut cells = HashSet::new();
        for pl in &a {
            assert!(
                cells.insert((pl.u / ANCHOR_CELL, pl.v / ANCHOR_CELL)),
                "two anchors in one cell"
            );
        }
    }

    #[test]
    fn anchors_respect_the_ground_rules() {
        let (p, cfg) = planet();
        let sea = p.terrain.sea_level();
        let peak = p.terrain.height_range().1;
        for face in 0..6u8 {
            for pl in anchors(&p, &cfg, face) {
                let sp = &species_for(cfg.planet_type)[pl.species];
                let h = p.effective_height(face, pl.u, pl.v);
                assert!(h > sea, "dry land: {h} vs sea {sea}");
                assert!(
                    p.terrain.water_level(face, pl.u, pl.v) < h,
                    "above its own water"
                );
                let ground = p
                    .block_type(BlockId {
                        face,
                        layer: h,
                        u: pl.u,
                        v: pl.v,
                    })
                    .unwrap();
                assert!(sp.ground.contains(&ground), "{:?} on {:?}", sp.name, ground);
                for (nf, nu, nv) in p.column_neighbors(face, pl.u, pl.v) {
                    assert!(
                        (p.effective_height(nf, nu, nv) as i64 - h as i64).abs() <= 1,
                        "slope"
                    );
                }
                let share = (h - sea) as f32 / (peak - sea).max(1) as f32;
                assert!(
                    share >= sp.band.0 && share <= sp.band.1,
                    "{} at {share}",
                    sp.name
                );
                assert!((sp.height.0..=sp.height.1).contains(&pl.height));
                // independent of the formulas above: the rules as the player would see them
                assert_ne!(ground, BlockType::Sand, "{} on a beach", sp.name);
                assert!(
                    !p.terrain.in_lake_region(face, pl.u, pl.v),
                    "{} in a lake region",
                    sp.name
                );
                let above = BlockId {
                    face,
                    layer: h + 1,
                    u: pl.u,
                    v: pl.v,
                };
                assert!(!p.exists(above), "the cell over the ground is open");
                if !sp.beach_ok {
                    for ring in rings(&p, face, pl.u, pl.v, 1) {
                        for (nf, nu, nv) in ring {
                            assert!(
                                p.effective_height(nf, nu, nv) > p.terrain.water_level(nf, nu, nv),
                                "{} next to open water",
                                sp.name
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn no_plants_grow_in_a_lake_region() {
        let seed = (1..60u32)
            .find(|&s| {
                PlanetData::new_for_type(256, s, PlanetType::EarthLike)
                    .terrain
                    .lake_count()
                    > 0
            })
            .expect("a seed with lakes");
        let p = PlanetData::new_for_type(256, seed, PlanetType::EarthLike);
        let cfg = FloraConfig {
            planet_type: PlanetType::EarthLike,
            seed,
            density: 3.0,
        };
        let sea = p.terrain.sea_level();
        let mut shore_land = 0;
        for face in 0..6u8 {
            for u in 0..256 {
                for v in 0..256 {
                    if p.terrain.in_lake_region(face, u, v) {
                        shore_land += (p.effective_height(face, u, v) > sea) as usize;
                        assert!(anchor_at(&p, &cfg, face, u, v).is_none());
                    }
                }
            }
        }
        assert!(
            shore_land > 0,
            "the lakes have dry columns in their regions"
        );
    }

    #[test]
    fn the_forest_mask_makes_groves_and_clearings() {
        let (_, cfg) = planet();
        let dirs: Vec<Vec3> = (0..4000)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / 4000.0;
                let r = (1.0 - y * y).sqrt();
                let a = i as f32 * 2.399_963;
                Vec3::new(r * a.cos(), y, r * a.sin())
            })
            .collect();
        let m: Vec<f32> = dirs.iter().map(|&d| forest_mask(&cfg, d)).collect();
        assert!(m.iter().all(|v| (0.0..=1.0).contains(v)));
        let high = m.iter().filter(|&&v| v > 0.6).count() as f32 / m.len() as f32;
        let low = m.iter().filter(|&&v| v < 0.2).count() as f32 / m.len() as f32;
        eprintln!("groves {high}, clearings {low}");
        assert!(high > 0.1 && low > 0.1, "groves {high}, clearings {low}");
    }

    #[test]
    fn rings_are_disjoint_discs_around_the_column() {
        let (p, _) = planet();
        let rs = rings(&p, 0, 40, 40, MAX_PLANT_REACH);
        assert_eq!(rs.len() as u32, MAX_PLANT_REACH + 1);
        assert_eq!(rs[0], vec![(0, 40, 40)]);
        assert_eq!(
            rs[1].len(),
            6,
            "a hex column has six neighbours away from the seams"
        );
        let mut seen = HashSet::new();
        for ring in &rs {
            for c in ring {
                assert!(seen.insert(*c), "{c:?} twice");
            }
        }
        // crossing a cube-face seam still produces full rings
        let edge = rings(&p, 0, 0, 40, MAX_PLANT_REACH);
        assert!(edge[2].len() >= 8);
    }

    #[test]
    fn an_oak_is_a_trunk_with_a_crown_above_the_ground() {
        let (p, cfg) = planet();
        let oak = anchors(&p, &cfg, 0)
            .into_iter()
            .find(|pl| species_for(cfg.planet_type)[pl.species].name == "Oak")
            .expect("an oak exists");
        let blocks = plant_blocks(&p, &oak, &cfg);
        let h = p.effective_height(oak.face, oak.u, oak.v);
        assert!(
            blocks
                .iter()
                .all(|(id, _)| id.layer > p.effective_height(id.face, id.u, id.v)),
            "only above the ground"
        );
        assert!(blocks
            .iter()
            .all(|(id, _)| id.layer <= h + MAX_PLANT_HEIGHT));
        let trunk: Vec<_> = blocks
            .iter()
            .filter(|(id, t)| id.u == oak.u && id.v == oak.v && *t == BlockType::Wood)
            .collect();
        assert!(trunk.len() as u32 >= oak.height - 1);
        assert!(
            blocks
                .iter()
                .filter(|(_, t)| *t == BlockType::Leaves)
                .count()
                >= 8
        );
        // no block is generated twice
        let mut ids = HashSet::new();
        assert!(blocks.iter().all(|(id, _)| ids.insert(*id)));
    }

    #[test]
    fn chunks_agree_on_every_tree_across_borders_and_seams() {
        for (res, cells) in [
            (64u32, CellShape::Hex),
            (100, CellShape::Hex),
            (100, CellShape::Square),
            (64, CellShape::Square),
        ] {
            let mut p = PlanetData::new_for_type(res, 11, PlanetType::EarthLike);
            p.cells = cells;
            let cfg = FloraConfig {
                planet_type: PlanetType::EarthLike,
                seed: 11,
                density: 3.0,
            };
            let mut whole = HashMap::new();
            let mut plants = 0;
            for face in 0..6u8 {
                for u in 0..res {
                    for v in 0..res {
                        if let Some(pl) = anchor_at(&p, &cfg, face, u, v) {
                            plants += 1;
                            for (id, t) in plant_blocks(&p, &pl, &cfg) {
                                whole.insert(id, t);
                            }
                        }
                    }
                }
            }
            assert!(plants > 20, "res {res}: only {plants} plants");
            let chunks = res.div_ceil(CHUNK_SIZE);
            let mut union = HashMap::new();
            for face in 0..6u8 {
                for cu in 0..chunks {
                    for cv in 0..chunks {
                        let key = ChunkKey {
                            face,
                            u_idx: cu,
                            v_idx: cv,
                        };
                        for (id, t) in decor_chunk(&p, &cfg, key).blocks {
                            assert_eq!(
                                PlanetData::chunk_key(id),
                                key,
                                "a block in the wrong chunk"
                            );
                            assert!(
                                union.insert(id, t).is_none(),
                                "{id:?} generated by two chunks"
                            );
                        }
                    }
                }
            }
            assert_eq!(whole.len(), union.len(), "res {res}");
            assert!(
                whole.iter().all(|(id, t)| union.get(id) == Some(t)),
                "res {res}"
            );
        }
    }

    #[test]
    fn decor_chunk_is_deterministic_and_has_tops() {
        let p = PlanetData::new_for_type(96, 5, PlanetType::EarthLike);
        let cfg = FloraConfig {
            planet_type: PlanetType::EarthLike,
            seed: 5,
            density: 2.0,
        };
        let key = ChunkKey {
            face: 0,
            u_idx: 1,
            v_idx: 1,
        };
        let a = decor_chunk(&p, &cfg, key);
        let b = decor_chunk(&p, &cfg, key);
        assert_eq!(a.blocks, b.blocks);
        for (&(u, v), &top) in &a.tops {
            let best = a
                .blocks
                .keys()
                .filter(|id| id.u == u && id.v == v)
                .map(|id| id.layer)
                .max();
            assert_eq!(best, Some(top));
        }
    }

    #[test]
    fn generating_a_chunk_is_fast() {
        // a loose bound that catches an accidental O(n^2): 16 chunks in well under 320 ms in release
        let p = PlanetData::new_for_type(128, 5, PlanetType::EarthLike);
        let cfg = FloraConfig {
            planet_type: PlanetType::EarthLike,
            seed: 5,
            density: 2.0,
        };
        let t = std::time::Instant::now();
        for k in 0..16u32 {
            let _ = decor_chunk(
                &p,
                &cfg,
                ChunkKey {
                    face: (k % 6) as u8,
                    u_idx: (k / 6) % 4,
                    v_idx: k % 4,
                },
            );
        }
        eprintln!("16 chunks: {:?}", t.elapsed());
        assert!(t.elapsed().as_millis() < 320, "{:?}", t.elapsed());
    }

    #[test]
    fn every_template_stays_within_the_plant_reach_and_height() {
        for t in PlanetType::ALL {
            for sp in species_for(t) {
                for h in sp.height.0..=sp.height.1 {
                    for (ring, offset, _, _) in template_cells(sp, h, 0)
                        .into_iter()
                        .chain(template_cells(sp, h, 1))
                    {
                        assert!(ring <= MAX_PLANT_REACH, "{} ring {ring}", sp.name);
                        assert!(offset <= MAX_PLANT_HEIGHT, "{} offset {offset}", sp.name);
                    }
                }
            }
        }
    }

    #[test]
    fn every_planet_type_has_plants_and_they_generate() {
        for t in PlanetType::ALL {
            assert!(!species_for(t).is_empty(), "{t:?} has no species");
            let p = PlanetData::new_for_type(128, 9, t);
            let cfg = FloraConfig {
                planet_type: t,
                seed: 9,
                density: 4.0,
            };
            let (mut total_blocks, mut anchors) = (0usize, 0usize);
            let mut kinds = HashSet::new();
            for face in 0..6u8 {
                for u in 0..128 {
                    for v in 0..128 {
                        if let Some(pl) = anchor_at(&p, &cfg, face, u, v) {
                            anchors += 1;
                            kinds.insert(pl.species);
                            let b = plant_blocks(&p, &pl, &cfg);
                            assert!(
                                !b.is_empty(),
                                "{t:?}/{} produced no blocks",
                                species_for(t)[pl.species].name
                            );
                            assert!(b
                                .iter()
                                .all(|(id, _)| id.layer > p.effective_height(id.face, id.u, id.v)));
                            assert!(b.iter().all(|(id, _)| id.layer
                                <= p.effective_height(face, pl.u, pl.v) + MAX_PLANT_HEIGHT));
                            total_blocks += b.len();
                        }
                    }
                }
            }
            eprintln!("{t:?}: {anchors} anchors, {total_blocks} blocks, species {kinds:?}");
            assert!(total_blocks > 50, "{t:?}: {total_blocks} blocks");
            if species_for(t).len() > 1 && !matches!(t, PlanetType::Ice | PlanetType::Volcanic) {
                assert!(kinds.len() >= 2, "{t:?}: {kinds:?}");
            }
        }
    }

    #[test]
    fn species_use_their_own_blocks() {
        use BlockType::*;
        let uses = |t: PlanetType, want: BlockType| {
            let p = PlanetData::new_for_type(128, 9, t);
            let cfg = FloraConfig {
                planet_type: t,
                seed: 9,
                density: 4.0,
            };
            (0..6u8).any(|f| {
                (0..128u32).any(|u| {
                    (0..128u32).any(|v| {
                        anchor_at(&p, &cfg, f, u, v).is_some_and(|pl| {
                            plant_blocks(&p, &pl, &cfg).iter().any(|(_, b)| *b == want)
                        })
                    })
                })
            })
        };
        assert!(uses(PlanetType::Ocean, Palm));
        assert!(uses(PlanetType::Desert, Cactus));
        assert!(uses(PlanetType::Toxic, Cap) && uses(PlanetType::Toxic, Stem));
        assert!(uses(PlanetType::Shattered, Crystal));
        assert!(uses(PlanetType::Volcanic, Obsidian));
        assert!(uses(PlanetType::Ice, Snow) || uses(PlanetType::Ice, Ice));
        assert!(uses(PlanetType::Cratered, Stone) || uses(PlanetType::Cratered, Basalt));
    }

    #[test]
    fn column_neighbours_are_symmetric_on_every_planet_size() {
        for (res, cells) in [
            (33u32, CellShape::Hex),
            (64, CellShape::Hex),
            (100, CellShape::Hex),
            (64, CellShape::Square),
            (100, CellShape::Square),
        ] {
            let mut p = PlanetData::new(res);
            p.cells = cells;
            let mut links = HashSet::new();
            for face in 0..6u8 {
                for u in 0..res {
                    for v in 0..res {
                        let n = p.column_neighbors(face, u, v);
                        assert!((4..=7).contains(&n.len()), "res {res} {n:?}");
                        for c in n {
                            links.insert(((face, u, v), c));
                        }
                    }
                }
            }
            let asymmetric = links
                .iter()
                .filter(|&&(a, b)| !links.contains(&(b, a)))
                .count();
            assert_eq!(asymmetric, 0, "res {res} {cells:?}");
        }
    }

    // cargo test --release decor_chunk_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn decor_chunk_bench() {
        for t in [PlanetType::EarthLike, PlanetType::Ice, PlanetType::Toxic] {
            let p = PlanetData::new_for_type(500, 7, t);
            let cfg = FloraConfig {
                planet_type: t,
                seed: 7,
                density: 1.0,
            };
            let n = (500 / 32) as u32;
            let start = std::time::Instant::now();
            let mut count = 0;
            for i in 0..100u32 {
                let key = ChunkKey {
                    face: (i % 6) as u8,
                    u_idx: (i * 7) % n,
                    v_idx: (i * 11) % n,
                };
                count += decor_chunk(&p, &cfg, key).blocks.len();
            }
            eprintln!(
                "{t:?}: decor_chunk x100 {:?}, {count} blocks",
                start.elapsed()
            );
        }
    }
}
