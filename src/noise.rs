use glam::Vec3;
use crate::gen::CoordSystem;
use std::sync::Arc; 

// --- TERRAIN SHAPE ---
// Heights are layers relative to sea level (layer res/2). Continents and mountain ranges are sized
// relative to the planet; hills keep roughly the same size in blocks at any resolution. The overall
// relief grows with the square root of the radius, capped at 20% of the radius: layers are about one
// unit thick at any size, so on small planets a fixed number of layers would be huge spikes and pits.

struct TerrainShape {
    relief: f32,      // highest mountains above sea level, in layers
    hill_freq: f32,   // noise frequencies on the unit sphere (cycles per radian)
    ridge_freq: f32,
    range_freq: f32,
}

impl TerrainShape {
    fn new(resolution: u32) -> Self {
        let radius = resolution as f32 / 2.0;
        Self {
            relief: (3.0 * radius.sqrt()).min(0.2 * radius),
            hill_freq: (radius / 35.0).max(2.0),
            ridge_freq: (radius / 120.0).max(2.5),
            range_freq: (radius / 400.0).max(1.5),
        }
    }

    // height above (or below) sea level for a direction on the unit sphere
    fn height(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        // continents: low-frequency, positive = land
        let c = g.fbm(dir * 1.3 + Vec3::new(17.1, 3.7, 9.2), 4) + 0.08;
        let land = smoothstep(-0.02, 0.12, c);
        // short continental shelves: coasts are a narrow band, not wide flats around sea level
        let base = if c < 0.0 {
            -0.4 * self.relief * smoothstep(0.0, 0.12, -c) // ocean floor
        } else {
            0.12 * self.relief * smoothstep(0.0, 0.12, c) // lowlands rise inland
        };

        let hills = g.fbm(dir * self.hill_freq + Vec3::new(-5.3, 11.9, 2.4), 4) * 0.15 * self.relief * (0.4 + 0.6 * land);

        // mountain ranges: ridged noise, only where the range mask is high and on land
        let range = smoothstep(-0.05, 0.25, g.fbm(dir * self.range_freq + Vec3::new(8.8, -2.6, 31.5), 3)) * land;
        let mountains = if range > 0.0 {
            g.ridged(dir * self.ridge_freq + Vec3::new(1.9, 23.3, -7.7), 5).powf(1.5) * self.relief * range
        } else {
            0.0
        };

        base + hills + mountains
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- PLANET TERRAIN DATA ---

pub struct PlanetTerrain {
    // Flattened height map
    heights: Arc<Vec<u16>>, 
    resolution: u32,
    height_range: (u32, u32), // lowest and highest column
    sea_level: u32,
}

impl PlanetTerrain {
    pub fn new(resolution: u32) -> Self {
        use rayon::prelude::*;
        let generator = NoiseGenerator::new(42); // Seed 42
        let shape = TerrainShape::new(resolution);
        let sea_level = resolution / 2;
        let mut heights = vec![0u16; (6 * resolution * resolution) as usize];

        // rows are independent, so generate them in parallel
        heights.par_chunks_mut(resolution as usize).enumerate().for_each(|(row, out)| {
            let face = (row as u32 / resolution) as u8;
            let v = row as u32 % resolution;
            for (u, h) in out.iter_mut().enumerate() {
                let dir = CoordSystem::get_direction(face, u as u32, v, resolution);
                *h = (sea_level as f32 + shape.height(&generator, dir)).round().max(1.0) as u16;
            }
        });

        let height_range = (
            heights.iter().copied().min().unwrap_or(0) as u32,
            heights.iter().copied().max().unwrap_or(0) as u32,
        );

        // Wrap in Arc for cheap cloning
        Self { heights: Arc::new(heights), resolution, height_range, sea_level } 
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

impl Clone for PlanetTerrain {
    fn clone(&self) -> Self {
        Self {
            heights: self.heights.clone(),
            resolution: self.resolution,
            height_range: self.height_range,
            sea_level: self.sea_level,
        }
    }
}


// --- NOISE GENERATOR ---

struct NoiseGenerator {
    perm: [u8; 512],
}

impl NoiseGenerator {
    fn new(seed: u32) -> Self {
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
        Self { perm: p }
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

        lerp(w, lerp(v, lerp(u, grad(self.perm[aa], x, y, z),
                                grad(self.perm[ba], x - 1.0, y, z)),
                        lerp(u, grad(self.perm[ab], x, y - 1.0, z),
                                grad(self.perm[bb], x - 1.0, y - 1.0, z))),
                lerp(v, lerp(u, grad(self.perm[aa + 1], x, y, z - 1.0),
                                grad(self.perm[ba + 1], x - 1.0, y, z - 1.0)),
                        lerp(u, grad(self.perm[ab + 1], x, y - 1.0, z - 1.0),
                                grad(self.perm[bb + 1], x - 1.0, y - 1.0, z - 1.0))))
    }
}

// ---MATH-HELPERS---

fn fade(t: f32) -> f32 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
fn lerp(t: f32, a: f32, b: f32) -> f32 { a + t * (b - a) }
fn grad(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 { y } else { if h == 12 || h == 14 { x } else { z } };
    (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
}