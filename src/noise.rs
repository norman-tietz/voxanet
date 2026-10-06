use crate::gen::CoordSystem;
use glam::Vec3;
use std::sync::Arc;

// --- TERRAIN SHAPE ---
// Heights are layers relative to sea level (layer res/2). Continents and mountain ranges are sized
// relative to the planet; hills keep roughly the same size in blocks at any resolution. The overall
// relief grows with the square root of the radius, capped at 20% of the radius: layers are about one
// unit thick at any size, so on small planets a fixed number of layers would be huge spikes and pits.

pub(crate) const LAKE_WARP: f32 = 0.35; // region outline wanders by ±35% of the radius
pub(crate) const LAKE_CAP: f32 = 1.0 + LAKE_WARP; // no region reaches beyond radius × this
const LAKE_CANDIDATES: u32 = 400; // most are rejected at the first height sample (ocean, low land)
const LAKE_MIN_CENTER: f32 = 2.0; // layers above sea level at the centre
const LAKE_MIN_EDGE: f32 = 2.0; // layers above sea level all along the edge: never joins the ocean
pub(crate) const LAKE_MIN_WATER_SHARE: f32 = 0.3;

// one carved lake: the terrain is lowered inside a noise-warped region around `center`, and water fills
// what ends up below `level` — so the shoreline follows the terrain's own contours
pub(crate) struct Lake {
    pub center: Vec3, // unit direction
    pub radius: f32,  // influence radius, radians on the unit sphere
    pub level: u32,   // absolute layer whose top is the water surface
    depression: f32,  // layers the centre is lowered by
    e1: Vec3,         // tangent frame at the centre, for the outline's azimuth
    e2: Vec3,
    warp_offset: Vec3,
}

impl Lake {
    // outline scale for the azimuth of `dir` around the centre: 1 ± LAKE_WARP
    fn warp(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let phi = dir.dot(self.e2).atan2(dir.dot(self.e1));
        let n = g.fbm(
            Vec3::new(phi.cos(), phi.sin(), 0.0) * 1.5 + self.warp_offset,
            2,
        );
        1.0 + LAKE_WARP * (n * 2.0).clamp(-1.0, 1.0)
    }

    // normalised distance from the centre: < 1 inside the region, 1 on its edge
    pub(crate) fn distance(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let a = dir.dot(self.center).clamp(-1.0, 1.0).acos();
        a / (self.radius * self.warp(g, dir))
    }

    // the direction `t` (0 = centre, 1 = edge) of the way out along azimuth `phi`
    fn along(&self, g: &NoiseGenerator, phi: f32, t: f32) -> Vec3 {
        let tangent = self.e1 * phi.cos() + self.e2 * phi.sin();
        let angle = t * self.radius * self.warp(g, tangent);
        self.center * angle.cos() + tangent * angle.sin()
    }

    pub(crate) fn edge_dir(&self, g: &NoiseGenerator, phi: f32) -> Vec3 {
        self.along(g, phi, 1.0)
    }

    fn falloff(d: f32) -> f32 {
        1.0 - smoothstep(0.35, 1.0, d)
    }
}

pub(crate) struct TerrainSample {
    pub height: f32,         // layers relative to sea level, lakes carved
    pub lake: Option<usize>, // the lake whose region contains the direction
}

pub(crate) struct TerrainShape {
    relief: f32,    // highest mountains above sea level, in layers
    hill_freq: f32, // noise frequencies on the unit sphere (cycles per radian)
    ridge_freq: f32,
    range_freq: f32,
    resolution: u32,
    lakes: Vec<Lake>,
}

impl TerrainShape {
    pub(crate) fn new(resolution: u32, g: &NoiseGenerator, with_lakes: bool) -> Self {
        let radius = resolution as f32 / 2.0;
        let mut shape = Self {
            relief: (3.0 * radius.sqrt()).min(0.2 * radius),
            hill_freq: (radius / 35.0).max(2.0),
            ridge_freq: (radius / 120.0).max(2.5),
            range_freq: (radius / 400.0).max(1.5),
            resolution,
            lakes: Vec::new(),
        };
        if with_lakes {
            shape.lakes = shape.place_lakes(g);
        }
        shape
    }

    pub(crate) fn lakes(&self) -> &[Lake] {
        &self.lakes
    }

    pub(crate) fn height(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        self.sample(g, dir).height
    }

    pub(crate) fn sample(&self, g: &NoiseGenerator, dir: Vec3) -> TerrainSample {
        let mut height = self.natural_height(g, dir);
        let mut lake = None;
        for (k, l) in self.lakes.iter().enumerate() {
            if dir.dot(l.center) < (l.radius * LAKE_CAP).cos() {
                continue; // outside its bounding cap: cheap reject
            }
            let d = l.distance(g, dir);
            if d < 1.0 {
                height -= l.depression * Lake::falloff(d);
                lake = Some(k);
            }
        }
        TerrainSample { height, lake }
    }

    // share of a lake's region under water, area-weighted on a 12 × 24 polar grid
    pub(crate) fn water_share(&self, g: &NoiseGenerator, lake: &Lake) -> f32 {
        let sea = (self.resolution / 2) as f32;
        let (mut wet, mut total) = (0.0, 0.0);
        for i in 0..12 {
            let t = (i as f32 + 0.5) / 12.0;
            for j in 0..24 {
                let phi = std::f32::consts::TAU * j as f32 / 24.0;
                let dir = lake.along(g, phi, t);
                let h = self.natural_height(g, dir) - lake.depression * Lake::falloff(t);
                total += t;
                if ((sea + h).round() as u32) < lake.level {
                    wet += t;
                }
            }
        }
        wet / total
    }

    #[cfg(test)]
    // how far out (0..1 of the region) water reaches along azimuth `phi`: the outermost wet sample
    pub(crate) fn shore_reach(&self, g: &NoiseGenerator, lake: &Lake, phi: f32) -> f32 {
        let sea = (self.resolution / 2) as f32;
        (0..64)
            .map(|i| (i as f32 + 0.5) / 64.0)
            .filter(|&t| {
                let h = self.natural_height(g, lake.along(g, phi, t))
                    - lake.depression * Lake::falloff(t);
                ((sea + h).round() as u32) < lake.level
            })
            .fold(0.0, f32::max)
    }

    fn place_lakes(&self, g: &NoiseGenerator) -> Vec<Lake> {
        let r_blocks = self.resolution as f32 / 2.0;
        let sea = (self.resolution / 2) as f32;
        let target = ((3.0 + r_blocks / 50.0) as usize).min(8);
        let depth = (0.25 * self.relief).clamp(3.0, 10.0);
        let mut lakes: Vec<Lake> = Vec::new();
        for k in 0..LAKE_CANDIDATES {
            if lakes.len() >= target {
                break;
            }
            let unit = |salt: u32| lake_hash(g.seed(), k, salt);
            let center = {
                let (z, a) = (unit(1) * 2.0 - 1.0, unit(2) * std::f32::consts::TAU);
                let s = (1.0 - z * z).max(0.0).sqrt();
                Vec3::new(s * a.cos(), s * a.sin(), z)
            };
            let radius = (0.2 * r_blocks).clamp(6.0, 50.0) * (0.7 + 0.3 * unit(3)) / r_blocks;
            let e1 = center.any_orthonormal_vector();
            let mut lake = Lake {
                center,
                radius,
                level: 0,
                depression: 0.0,
                e1,
                e2: center.cross(e1),
                warp_offset: Vec3::new(unit(4), unit(5), unit(6)) * 100.0,
            };
            let center_height = self.natural_height(g, center);
            if center_height < LAKE_MIN_CENTER {
                continue;
            }
            let overlaps = lakes.iter().any(|o| {
                center.dot(o.center).clamp(-1.0, 1.0).acos()
                    < (o.radius + radius) * LAKE_CAP + o.radius.max(radius)
            });
            if overlaps {
                continue;
            }
            // the edge, about one sample per block along the outline
            let samples =
                ((std::f32::consts::TAU * radius * LAKE_CAP * r_blocks).ceil() as usize).max(64);
            let edge_min = (0..samples)
                .map(|i| {
                    let phi = std::f32::consts::TAU * i as f32 / samples as f32;
                    self.natural_height(g, lake.edge_dir(g, phi))
                })
                .fold(f32::MAX, f32::min);
            if edge_min < LAKE_MIN_EDGE {
                continue;
            }
            lake.level = (sea + edge_min).round() as u32 - 1;
            lake.depression = (center_height - (lake.level as f32 - sea)) + depth;
            if self.water_share(g, &lake) < LAKE_MIN_WATER_SHARE {
                continue;
            }
            lakes.push(lake);
        }
        lakes
    }

    // height above (or below) sea level for a direction on the unit sphere, before lakes are carved
    pub(crate) fn natural_height(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        // continents: low-frequency, positive = land
        let c = g.fbm(dir * 1.3 + Vec3::new(17.1, 3.7, 9.2), 4) + 0.08;
        let land = smoothstep(-0.02, 0.12, c);
        // short continental shelves: coasts are a narrow band, not wide flats around sea level
        let base = if c < 0.0 {
            -0.4 * self.relief * smoothstep(0.0, 0.12, -c) // ocean floor
        } else {
            0.12 * self.relief * smoothstep(0.0, 0.12, c) // lowlands rise inland
        };

        let hills = g.fbm(dir * self.hill_freq + Vec3::new(-5.3, 11.9, 2.4), 4)
            * 0.15
            * self.relief
            * (0.4 + 0.6 * land);

        // mountain ranges: ridged noise, only where the range mask is high and on land
        let range = smoothstep(
            -0.05,
            0.25,
            g.fbm(dir * self.range_freq + Vec3::new(8.8, -2.6, 31.5), 3),
        ) * land;
        let mountains = if range > 0.0 {
            g.ridged(dir * self.ridge_freq + Vec3::new(1.9, 23.3, -7.7), 5)
                .powf(1.5)
                * self.relief
                * range
        } else {
            0.0
        };

        base + hills + mountains
    }
}

// a seeded value in 0..1 for lake candidate `k` (splitmix64 over seed, k, salt)
fn lake_hash(seed: u32, k: u32, salt: u32) -> f32 {
    let mut x =
        ((seed as u64) << 32 | k as u64) ^ (salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- PLANET TERRAIN DATA ---

// the home planet's noise seed (the value every planet used before galaxy planets had their own)
pub const HOME_SEED: u32 = 42;

#[derive(Clone)]
pub struct PlanetTerrain {
    // Flattened height map
    heights: Arc<Vec<u16>>,
    // per column: which water body's level is its water level (0 = the sea, k + 1 = lake k), separate
    // from the heights so solidity checks (heights only, the hot path) don't fetch it
    water_body: Arc<Vec<u8>>,
    water_levels: Arc<Vec<u32>>, // layer whose top is the water surface, per water body
    resolution: u32,
    height_range: (u32, u32), // lowest and highest column
    sea_level: u32,
}

impl PlanetTerrain {
    pub fn new(resolution: u32, seed: u32) -> Self {
        Self::generate(resolution, seed, false)
    }

    // a liquid planet's terrain: lakes carved, each with its own water level
    pub fn with_lakes(resolution: u32, seed: u32) -> Self {
        Self::generate(resolution, seed, true)
    }

    fn generate(resolution: u32, seed: u32, lakes: bool) -> Self {
        use rayon::prelude::*;
        let generator = NoiseGenerator::new(seed);
        let shape = TerrainShape::new(resolution, &generator, lakes);
        let sea_level = resolution / 2;
        // index 0: the sea; k + 1: lake k
        let water_levels: Vec<u32> = std::iter::once(sea_level)
            .chain(shape.lakes().iter().map(|l| l.level))
            .collect();
        let n = (6 * resolution * resolution) as usize;
        let (mut heights, mut water_body) = (vec![0u16; n], vec![0u8; n]);

        // rows are independent, so generate them in parallel
        heights
            .par_chunks_mut(resolution as usize)
            .zip(water_body.par_chunks_mut(resolution as usize))
            .enumerate()
            .for_each(|(row, (out, body))| {
                let face = (row as u32 / resolution) as u8;
                let v = row as u32 % resolution;
                for u in 0..resolution as usize {
                    let dir = CoordSystem::get_direction(face, u as u32, v, resolution);
                    let s = shape.sample(&generator, dir);
                    let h = (sea_level as f32 + s.height).round().max(1.0) as u16;
                    out[u] = h;
                    // a lake column: inside a region and below that lake's level (islands stay dry)
                    body[u] = match s.lake {
                        Some(k) if (h as u32) < water_levels[k + 1] => (k + 1) as u8,
                        _ => 0,
                    };
                }
            });

        let height_range = (
            heights.iter().copied().min().unwrap_or(0) as u32,
            heights.iter().copied().max().unwrap_or(0) as u32,
        );

        // Wrap in Arc for cheap cloning
        Self {
            heights: Arc::new(heights),
            water_body: Arc::new(water_body),
            water_levels: Arc::new(water_levels),
            resolution,
            height_range,
            sea_level,
        }
    }

    // the layer whose top is this column's water surface: its lake's level, else sea level (the water
    // table dug holes fill to). Whether the column holds water is PlanetData::holds_water's question
    pub fn water_level(&self, face: u8, u: u32, v: u32) -> u32 {
        let (u, v) = (u.min(self.resolution - 1), v.min(self.resolution - 1));
        self.water_levels[self.water_body[Self::get_index(face, u, v, self.resolution)] as usize]
    }

    #[cfg(test)]
    pub fn lake_count(&self) -> usize {
        self.water_levels.len() - 1
    }

    // the layer of the water surface: columns at or below it are sea floor
    pub fn sea_level(&self) -> u32 {
        self.sea_level
    }

    pub fn height_range(&self) -> (u32, u32) {
        self.height_range
    }

    #[inline(always)]
    fn get_index(face: u8, u: u32, v: u32, res: u32) -> usize {
        let face_offset = (face as usize) * (res as usize) * (res as usize);
        let row_offset = (v as usize) * (res as usize);
        face_offset + row_offset + (u as usize)
    }

    pub fn get_height(&self, face: u8, u: u32, v: u32) -> u32 {
        let u_safe = u.min(self.resolution - 1);
        let v_safe = v.min(self.resolution - 1);

        let idx = Self::get_index(face, u_safe, v_safe, self.resolution);
        self.heights[idx] as u32
    }
}

// --- NOISE GENERATOR ---

pub(crate) struct NoiseGenerator {
    perm: [u8; 512],
    seed: u32,
}

impl NoiseGenerator {
    pub(crate) fn new(seed: u32) -> Self {
        let mut p = [0u8; 512];
        let mut permutation: Vec<u8> = (0..=255).collect();
        let mut state = seed;
        for i in (1..256).rev() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (state as usize) % (i + 1);
            permutation.swap(i, j);
        }

        for i in 0..256 {
            p[i] = permutation[i];
            p[i + 256] = permutation[i];
        }
        Self { perm: p, seed }
    }

    pub(crate) fn seed(&self) -> u32 {
        self.seed
    }

    // fractal Brownian motion, roughly -1..1
    fn fbm(&self, p: Vec3, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm, mut freq) = (0.0, 1.0, 0.0, 1.0);
        for _ in 0..octaves {
            sum += self.perlin(p * freq) * amp;
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    // ridged multifractal, 0..1: sharp crests where the noise crosses zero
    fn ridged(&self, p: Vec3, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm, mut freq, mut weight) = (0.0, 1.0, 0.0, 1.0, 1.0);
        for _ in 0..octaves {
            let r = 1.0 - self.perlin(p * freq).abs();
            let r = r * r * weight;
            weight = (r * 2.0).clamp(0.0, 1.0); // detail mostly on the crests
            sum += r * amp;
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    // --- PERLIN MATH ---

    fn perlin(&self, pos: Vec3) -> f32 {
        let x = pos.x.floor();
        let y = pos.y.floor();
        let z = pos.z.floor();

        let xi = x as i32 & 255;
        let yi = y as i32 & 255;
        let zi = z as i32 & 255;

        let x = pos.x - x;
        let y = pos.y - y;
        let z = pos.z - z;

        let u = fade(x);
        let v = fade(y);
        let w = fade(z);

        let a = self.perm[xi as usize] as usize + yi as usize;
        let aa = self.perm[a] as usize + zi as usize;
        let ab = self.perm[a + 1] as usize + zi as usize;
        let b = self.perm[xi as usize + 1] as usize + yi as usize;
        let ba = self.perm[b] as usize + zi as usize;
        let bb = self.perm[b + 1] as usize + zi as usize;

        lerp(
            w,
            lerp(
                v,
                lerp(
                    u,
                    grad(self.perm[aa], x, y, z),
                    grad(self.perm[ba], x - 1.0, y, z),
                ),
                lerp(
                    u,
                    grad(self.perm[ab], x, y - 1.0, z),
                    grad(self.perm[bb], x - 1.0, y - 1.0, z),
                ),
            ),
            lerp(
                v,
                lerp(
                    u,
                    grad(self.perm[aa + 1], x, y, z - 1.0),
                    grad(self.perm[ba + 1], x - 1.0, y, z - 1.0),
                ),
                lerp(
                    u,
                    grad(self.perm[ab + 1], x, y - 1.0, z - 1.0),
                    grad(self.perm[bb + 1], x - 1.0, y - 1.0, z - 1.0),
                ),
            ),
        )
    }
}

// ---MATH-HELPERS---

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
fn lerp(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}
fn grad(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else {
        if h == 12 || h == 14 {
            x
        } else {
            z
        }
    };
    (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
}

#[cfg(test)]
mod tests {
    use super::*;

    // FNV-1a over every column height: a cheap fingerprint of a whole terrain
    fn fingerprint(t: &PlanetTerrain) -> u64 {
        t.heights.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, &h| {
            (a ^ h as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    // the value was measured on the code before seeds existed (hardcoded NoiseGenerator::new(42)):
    // the home planet must come out bit-identical
    #[test]
    fn home_seed_reproduces_the_original_terrain() {
        assert_eq!(
            fingerprint(&PlanetTerrain::new(64, HOME_SEED)),
            0x8940_2259_76b1_98a6
        );
    }

    #[test]
    fn different_seeds_give_different_terrain() {
        assert_ne!(
            fingerprint(&PlanetTerrain::new(64, 1)),
            fingerprint(&PlanetTerrain::new(64, 2))
        );
    }

    fn lake_shape(res: u32, seed: u32) -> (TerrainShape, NoiseGenerator) {
        let g = NoiseGenerator::new(seed);
        (TerrainShape::new(res, &g, true), g)
    }

    // the first seeds that give `res` at least `min` lakes
    fn seeds_with_lakes(res: u32, min: usize) -> Vec<u32> {
        (1..60)
            .filter(|&s| lake_shape(res, s).0.lakes().len() >= min)
            .collect()
    }

    #[test]
    fn lakes_are_deterministic_and_seeded() {
        let a: Vec<_> = lake_shape(256, 7)
            .0
            .lakes()
            .iter()
            .map(|l| (l.center, l.level))
            .collect();
        let b: Vec<_> = lake_shape(256, 7)
            .0
            .lakes()
            .iter()
            .map(|l| (l.center, l.level))
            .collect();
        let c: Vec<_> = lake_shape(256, 8)
            .0
            .lakes()
            .iter()
            .map(|l| (l.center, l.level))
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn lake_count_follows_the_planet_size() {
        let g = NoiseGenerator::new(3);
        assert!(TerrainShape::new(256, &g, false).lakes().is_empty());
        for seed in 1..20 {
            let n = lake_shape(256, seed).0.lakes().len();
            assert!(
                n <= 5,
                "seed {seed}: {n} lakes (target min(3 + 128/50, 8) = 5)"
            );
        }
        assert!(
            !seeds_with_lakes(256, 3).is_empty(),
            "no seed reaches 3 lakes"
        );
    }

    #[test]
    fn lake_edges_stay_above_sea_and_lakes_dont_overlap() {
        for seed in seeds_with_lakes(256, 1).into_iter().take(8) {
            let (shape, g) = lake_shape(256, seed);
            let sea = 128;
            for (k, lake) in shape.lakes().iter().enumerate() {
                assert!(
                    lake.level >= sea + 1,
                    "seed {seed} lake {k}: level {}",
                    lake.level
                );
                for o in &shape.lakes()[k + 1..] {
                    let gap = lake.center.dot(o.center).clamp(-1.0, 1.0).acos();
                    assert!(gap >= (lake.radius + o.radius) * LAKE_CAP + lake.radius.max(o.radius));
                }
                // every edge point: natural height at least one layer above the water
                for i in 0..64 {
                    let dir = lake.edge_dir(&g, std::f32::consts::TAU * i as f32 / 64.0);
                    let h = (sea as f32 + shape.natural_height(&g, dir)).round() as u32;
                    assert!(
                        h >= lake.level + 1,
                        "seed {seed} lake {k}: edge {h} vs level {}",
                        lake.level
                    );
                }
            }
        }
    }

    #[test]
    fn lake_regions_are_at_least_30_percent_water() {
        for seed in seeds_with_lakes(256, 1).into_iter().take(8) {
            let (shape, g) = lake_shape(256, seed);
            for lake in shape.lakes() {
                assert!(shape.water_share(&g, lake) >= LAKE_MIN_WATER_SHARE);
            }
        }
    }

    // natural shapes, not bowls: the shoreline's distance from the centre varies by more than ±15%
    #[test]
    fn lake_shorelines_are_not_round() {
        let mut found = false;
        for seed in seeds_with_lakes(256, 1).into_iter().take(8) {
            let (shape, g) = lake_shape(256, seed);
            for lake in shape.lakes() {
                let reach: Vec<f32> = (0..48)
                    .map(|i| shape.shore_reach(&g, lake, std::f32::consts::TAU * i as f32 / 48.0))
                    .collect();
                let mean = reach.iter().sum::<f32>() / reach.len() as f32;
                let (lo, hi) = reach
                    .iter()
                    .fold((f32::MAX, 0.0f32), |(a, b), &r| (a.min(r), b.max(r)));
                if mean > 0.0 && (hi / mean > 1.15 || lo / mean < 0.85) {
                    found = true;
                }
            }
        }
        assert!(found, "every lake came out round");
    }

    #[test]
    fn carving_only_lowers_terrain_inside_a_region() {
        let (shape, g) = lake_shape(256, seeds_with_lakes(256, 1)[0]);
        let lake = &shape.lakes()[0];
        let inside = lake.center;
        assert!(shape.height(&g, inside) < shape.natural_height(&g, inside));
        assert_eq!(shape.sample(&g, inside).lake, Some(0));
        let far = -lake.center;
        assert_eq!(shape.height(&g, far), shape.natural_height(&g, far));
    }
    #[test]
    fn water_level_is_sea_level_except_in_lakes() {
        let seed = seeds_with_lakes(128, 1)[0];
        let t = PlanetTerrain::with_lakes(128, seed);
        assert!(t.lake_count() >= 1);
        let (mut lake_cols, mut sea_cols) = (0, 0);
        for face in 0..6u8 {
            for v in 0..128 {
                for u in 0..128 {
                    let (h, w) = (t.get_height(face, u, v), t.water_level(face, u, v));
                    if w == t.sea_level() {
                        sea_cols += 1;
                    } else {
                        assert!(
                            w > t.sea_level() && h < w,
                            "lake column {face}/{u}/{v}: h {h} w {w}"
                        );
                        lake_cols += 1;
                    }
                }
            }
        }
        assert!(lake_cols > 0 && sea_cols > lake_cols);
        let plain = PlanetTerrain::new(128, seed);
        assert_eq!(plain.lake_count(), 0);
        assert_eq!(plain.water_level(0, 5, 5), plain.sea_level());
    }

    // no wall of water: a lake column never borders a dry column lower than the lake's level
    #[test]
    fn lakes_have_no_water_walls() {
        for res in [80u32, 256] {
            for seed in seeds_with_lakes(res, 1).into_iter().take(4) {
                let t = PlanetTerrain::with_lakes(res, seed);
                for face in 0..6u8 {
                    for v in 1..res - 1 {
                        for u in 1..res - 1 {
                            let w = t.water_level(face, u, v);
                            if w == t.sea_level() {
                                continue;
                            }
                            for (du, dv) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                                let (nu, nv) = ((u as i32 + du) as u32, (v as i32 + dv) as u32);
                                let dry = t.water_level(face, nu, nv) != w;
                                assert!(
                                    !dry || t.get_height(face, nu, nv) >= w,
                                    "res {res} seed {seed}: wall at {face}/{u}/{v} -> {nu}/{nv}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
