use crate::meshing::gen::CoordSystem;
use crate::planet::biome::PlanetType;
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

// per-planet terrain variation (PlanetTraits::terrain): relief scales R before its 20 %-of-radius cap,
// land_shift adds to the continent mask's land_offset
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainTweaks {
    pub relief: f32,
    pub land_shift: f32,
}

impl TerrainTweaks {
    pub const NEUTRAL: Self = Self {
        relief: 1.0,
        land_shift: 0.0,
    };
}

pub(crate) struct TerrainShape {
    relief: f32,    // highest mountains above sea level, in layers
    hill_freq: f32, // noise frequencies on the unit sphere (cycles per radian)
    ridge_freq: f32,
    range_freq: f32,
    resolution: u32,
    planet_type: PlanetType,
    land_shift: f32,
    lakes: Vec<Lake>,
}

// the relief R (highest mountains above sea level, in layers) of a planet of this radius, `scale` being
// its PlanetTraits::relief; also the impostors' storm fog base (impostor::storm_base), which has no shape
pub(crate) fn relief_for(radius: f32, scale: f32) -> f32 {
    (3.0 * radius.sqrt() * scale).min(0.2 * radius)
}

impl TerrainShape {
    #[cfg(test)]
    pub(crate) fn new(
        resolution: u32,
        g: &NoiseGenerator,
        planet_type: PlanetType,
        with_lakes: bool,
    ) -> Self {
        Self::tweaked(
            resolution,
            g,
            planet_type,
            with_lakes,
            TerrainTweaks::NEUTRAL,
        )
    }

    pub(crate) fn tweaked(
        resolution: u32,
        g: &NoiseGenerator,
        planet_type: PlanetType,
        with_lakes: bool,
        tweaks: TerrainTweaks,
    ) -> Self {
        let radius = resolution as f32 / 2.0;
        let mut shape = Self {
            relief: relief_for(radius, tweaks.relief),
            hill_freq: (radius / 35.0).max(2.0),
            ridge_freq: (radius / 120.0).max(2.5),
            range_freq: (radius / 400.0).max(1.5),
            resolution,
            planet_type,
            land_shift: tweaks.land_shift,
            lakes: Vec::new(),
        };
        if with_lakes {
            shape.lakes = shape.place_lakes(g);
        }
        shape
    }

    // highest mountains above sea level, in layers (the impostors' colour bands use it)
    pub(crate) fn relief(&self) -> f32 {
        self.relief
    }

    pub(crate) fn lakes(&self) -> &[Lake] {
        &self.lakes
    }

    #[cfg(test)]
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

    // the continent mask (low-frequency, positive = land), including the type's land offset
    pub(crate) fn continent(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        g.fbm(dir * 1.3 + Vec3::new(17.1, 3.7, 9.2), 4)
            + land_offset(self.planet_type)
            + self.land_shift
    }

    // the three crater layers of Cratered: cell size in world units at sea level, depth of the bowl and
    // height of the rim as shares of the relief, how sharp the rim is, the salt, and the least bowl depth and
    // rim height in whole layers (on small planets the relief is a few layers and the shares would vanish)
    fn layer_params(layer: CraterLayer) -> LayerParams {
        let (blocks, bowl, rim, rim_width, salt, min_bowl, min_rim) = match layer {
            CraterLayer::Large => (
                150.0,
                0.16,
                0.025,
                0.10,
                Vec3::new(2.5, -7.1, 4.4),
                3.0,
                1.3,
            ), // old, worn, shallow
            CraterLayer::Medium => (55.0, 0.12, 0.030, 0.07, Vec3::new(-6.3, 1.9, 8.2), 2.5, 1.3),
            CraterLayer::Small => (20.0, 0.10, 0.030, 0.05, Vec3::new(9.1, 5.5, -3.7), 1.5, 1.0), // fresh, deep, sharp
        };
        LayerParams {
            blocks,
            bowl,
            rim,
            rim_width,
            salt,
            min_bowl,
            min_rim,
        }
    }

    // the crater cells shrink with the planet below radius 250 (never under 8 world units), so a small
    // planet keeps a sensible number of craters of a sensible relative size; unchanged from radius 250 up
    fn crater_cell(&self, blocks: f32) -> f32 {
        (blocks * (self.resolution as f32 / 2.0 / 250.0).min(1.0)).max(8.0)
    }

    // one crater layer's height contribution in layers: a smooth bowl inside the crater, a raised rim at its
    // edge, nothing between craters; fades to 0 at the cell boundary (no cliffs)
    pub(crate) fn crater_layer(&self, g: &NoiseGenerator, dir: Vec3, layer: CraterLayer) -> f32 {
        let p = Self::layer_params(layer);
        let (f1, f2, h) = self.cells(g, dir, self.crater_cell(p.blocks), p.salt);
        // half of the cells hold a crater, radius 0.25..0.45 of the cell by the cell's hash
        if h < 0.45 {
            return 0.0;
        }
        let rc = 0.25 + 0.2 * (h - 0.45) / 0.55;
        let bowl = smoothstep(rc, rc * 0.4, f1) * (p.bowl * self.relief).max(p.min_bowl);
        let rim = (-((f1 - rc) / p.rim_width).powi(2)).exp() * (p.rim * self.relief).max(p.min_rim);
        (rim - bowl) * smoothstep(0.0, EDGE_FADE, f2 - f1)
    }

    // 1 on the maria (the low part of the continent mask, about 30 % of the surface), 0 on the highlands
    pub(crate) fn mare_mask(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        1.0 - smoothstep(MARE_LO, MARE_HI, self.continent(g, dir))
    }

    // lunar highlands with craters at three sizes and a few large smooth maria; heights never below the
    // fill (liquid-less planets are filled flat up to sea level, so a deeper floor would be a plateau)
    fn cratered(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let mare = self.mare_mask(g, dir);
        // the highlands are high enough to hold the large bowls
        let base = (0.30 * r).max(Self::layer_params(CraterLayer::Large).min_bowl + 2.0);
        let highland = base + g.fbm(dir * self.hill_freq + Vec3::new(1.3, 8.1, -2.2), 4) * 0.10 * r;
        let large = self.crater_layer(g, dir, CraterLayer::Large);
        let medium = self.crater_layer(g, dir, CraterLayer::Medium);
        let small = self.crater_layer(g, dir, CraterLayer::Small);
        // maria are flooded: only the small fresh craters survive, at a quarter of their strength
        let craters = (large + medium) * (1.0 - mare) + small * (1.0 - 0.75 * mare);
        let surface = highland * (1.0 - mare) + craters;
        // the fill: a smooth floor at MIN_FLOOR (never below), maria sit exactly on it
        let floor = MIN_FLOOR;
        let h = floor + softplus_above(surface - floor, 0.5);
        self.soft_cap(h)
    }

    // height above (or below) sea level for a direction on the unit sphere, before lakes are carved
    pub(crate) fn natural_height(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        match self.planet_type {
            PlanetType::EarthLike => self.earth_like(g, dir),
            PlanetType::Volcanic => self.volcanic(g, dir),
            PlanetType::Ice => self.ice(g, dir),
            PlanetType::Desert => self.desert(g, dir),
            PlanetType::Toxic => self.toxic(g, dir),
            PlanetType::Ocean => self.ocean(g, dir),
            PlanetType::Shattered => self.shattered(g, dir),
            PlanetType::Cratered => self.cratered(g, dir),
        }
    }

    // the ocean floor / lowland base Earth-like uses, for continent mask `c`
    fn base(&self, c: f32) -> f32 {
        if c < 0.0 {
            -0.4 * self.relief * smoothstep(0.0, 0.12, -c)
        } else {
            0.12 * self.relief * smoothstep(0.0, 0.12, c)
        }
    }

    // a smooth ceiling at 1.2R: heights below ~0.8R unchanged, never above 1.2R
    fn soft_cap(&self, h: f32) -> f32 {
        let cap = 1.2 * self.relief;
        let knee = 0.8 * self.relief;
        if h <= knee {
            h
        } else {
            knee + (cap - knee) * ((h - knee) / (cap - knee)).tanh()
        }
    }

    // cellular noise at a cell spacing of `blocks` world units at sea level (about columns near a face centre; constant on every planet size)
    fn cells(&self, g: &NoiseGenerator, dir: Vec3, blocks: f32, salt: Vec3) -> (f32, f32, f32) {
        g.cellular(dir * (self.resolution as f32 / 2.0 / blocks) + salt)
    }

    // jagged ridges on all land, calderas with lava-filled bowls, narrow fissures
    fn volcanic(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let c = self.continent(g, dir);
        let land = smoothstep(-0.02, 0.12, c);
        if land == 0.0 {
            return self.soft_cap(self.base(c)); // every other term is scaled by `land`: skip the noise
        }
        let ridges = g
            .ridged(dir * self.ridge_freq * 1.5 + Vec3::new(3.1, -8.2, 5.5), 5)
            .powi(2)
            * 0.8
            * r
            * land;
        let (f1, f2, h) = self.cells(g, dir, 60.0, Vec3::new(-4.4, 2.2, 9.9));
        let edge = smoothstep(0.0, EDGE_FADE, f2 - f1); // features reach 0 at a cell boundary: no cliffs
        let caldera = if h < 0.5 {
            let rim = (-((f1 - 0.45) / 0.08).powi(2)).exp() * 0.5 * r;
            let bowl = smoothstep(0.4, 0.2, f1) * 1.0 * r;
            (rim - bowl) * land * edge
        } else {
            0.0
        };
        let fissure = self.fissure(g, dir) * 0.4 * r * land;
        self.soft_cap(self.base(c) + ridges + caldera - fissure)
    }

    // the Volcanic fissure term in 0..1: a band around the zero crossings of one noise octave (period
    // FISSURE_PERIOD world units), whose edge is set by the distance to the crossing in world units
    // (|noise| / |gradient|, by finite differences), so it is at least FISSURE_WIDTH units across
    // wherever the gradient is, and on every planet size (see the width test)
    pub(crate) fn fissure(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let p = dir * (self.resolution as f32 / 2.0 / FISSURE_PERIOD) + Vec3::new(12.0, 1.0, -3.0);
        let n = g.perlin(p);
        const E: f32 = 0.02;
        let grad = Vec3::new(
            g.perlin(p + Vec3::X * E) - n,
            g.perlin(p + Vec3::Y * E) - n,
            g.perlin(p + Vec3::Z * E) - n,
        ) / E;
        let dist = n.abs() / grad.length().max(1e-3) * FISSURE_PERIOD; // world units
        smoothstep(FISSURE_WIDTH, 0.0, dist)
    }

    // smooth billow land, frozen seas (the fill), rare wide ice spikes
    fn ice(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let c = self.continent(g, dir);
        let land = smoothstep(-0.02, 0.12, c);
        let hills =
            g.billow(dir * self.hill_freq * 0.6 + Vec3::new(7.7, 0.3, -2.9), 2) * 0.15 * r * land;
        // a spike per lucky cell (cells of 40 world units): a flat top ~2.4 columns across, sloping out to a
        // 10-column radius, so neighbouring columns differ by <= ~0.2R (the width test) and the low-poly
        // smoothing keeps it
        let (f1, f2, h) = self.cells(g, dir, 40.0, Vec3::new(1.5, 6.6, -7.1));
        let spike = if h < 0.15 {
            smoothstep(0.25, 0.03, f1) * 1.0 * r * smoothstep(0.0, EDGE_FADE, f2 - f1)
        } else {
            0.0
        };
        self.soft_cap(self.base(c) * 0.5 + hills + spike)
    }

    // terraced mesas, canyons ending just above the flats, low dunes
    fn desert(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let c = self.continent(g, dir);
        let land = smoothstep(-0.02, 0.12, c);
        let mesa =
            g.ridged(dir * self.range_freq * 2.0 + Vec3::new(-6.0, 4.0, 2.0), 4) * 0.7 * r * land;
        let stepped = terrace(self.base(c) + mesa, (r / 8.0).max(2.0), 0.15);
        let dunes =
            g.billow(dir * self.hill_freq * 1.5 + Vec3::new(2.0, -9.0, 4.4), 3) * 0.05 * r * land;
        let h = stepped + dunes;
        let canyon = smoothstep(
            0.88,
            0.96,
            g.ridged(dir * self.hill_freq * 0.5 + Vec3::new(9.1, 9.2, 9.3), 3),
        ) * land;
        // canyons cut toward sea level + 1, never below (the fill would flatten them)
        let h = if h > 1.0 { h - canyon * (h - 1.0) } else { h };
        self.soft_cap(h)
    }

    // flat land pocked with craters whose floors lie below sea level
    fn toxic(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let c = self.continent(g, dir);
        let land = smoothstep(-0.02, 0.12, c);
        let flat = if c < 0.0 {
            self.base(c)
        } else {
            0.05 * r * smoothstep(0.0, 0.12, c)
        } + g.fbm(dir * self.hill_freq + Vec3::new(0.7, -0.7, 3.3), 3) * 0.03 * r * land;
        let (f1, f2, h) = self.cells(g, dir, 30.0, Vec3::new(-2.2, -5.5, 8.8));
        let rc = 0.15 + 0.2 * h; // crater radius in cell units, by the cell's hash
        let bowl = smoothstep(rc, rc * 0.4, f1) * 0.4 * r;
        let rim = (-((f1 - rc) / 0.05).powi(2)).exp() * 0.1 * r;
        self.soft_cap(flat + (rim - bowl) * land * smoothstep(0.0, EDGE_FADE, f2 - f1))
    }

    // deep sea, steep rocky islands where the mask breaks through
    fn ocean(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let c = self.continent(g, dir);
        if c < 0.0 {
            return -0.6 * r * smoothstep(0.0, 0.15, -c);
        }
        let rise = smoothstep(0.0, 0.03, c);
        let hills = g.fbm(dir * self.hill_freq + Vec3::new(4.0, 4.0, -1.0), 4) * 0.3 * r;
        self.soft_cap(rise * (0.5 * r + hills))
    }

    // tectonic plates at their own heights, deep cracks between them, shard ridges on top
    fn shattered(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let r = self.relief;
        let c = self.continent(g, dir);
        let land = smoothstep(-0.02, 0.12, c);
        if land == 0.0 {
            return self.base(c);
        }
        let (f1, f2, h) = self.cells(g, dir, 50.0, Vec3::new(5.0, -1.0, 7.0));
        let tilt = g.fbm(
            dir * self.range_freq * 4.0 + Vec3::new(h * 10.0, 0.0, 0.0),
            2,
        ) * 0.1
            * r;
        let plate = (-0.3 + 0.9 * h) * r + tilt;
        let shards = g.ridged(dir * self.hill_freq + Vec3::new(-8.0, 3.0, 1.0), 3) * 0.2 * r;
        let top = self.base(c) + (plate + shards) * land;
        // the crack: where the two nearest cells are about equally near (>= 2 columns wide at 50-unit cells)
        let crack = smoothstep(0.08, 0.0, f2 - f1) * land;
        self.soft_cap(top + crack * (-0.3 * r - top))
    }

    pub(crate) fn earth_like(&self, g: &NoiseGenerator, dir: Vec3) -> f32 {
        let c = self.continent(g, dir);
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

// a height quantised into steps of `step` with the step edges rounded over the last `softness` of each
// step (0..1): flat treads, steep risers — mesas and strata
pub(crate) fn terrace(h: f32, step: f32, softness: f32) -> f32 {
    let k = h / step;
    let f = k.floor();
    (f + smoothstep(1.0 - softness, 1.0, k - f)) * step
}

// the continent mask's offset per planet type: its land share (Earth-like's is today's 0.08)
pub(crate) fn land_offset(t: PlanetType) -> f32 {
    match t {
        PlanetType::Desert => 0.5,
        PlanetType::Ocean => -0.25,
        _ => 0.08,
    }
}

// per-cell features fade to 0 over this much f2 - f1 (cell units) so they have no cliffs at cell boundaries
const EDGE_FADE: f32 = 0.1;
const MARE_LO: f32 = -0.09; // continent-mask values below this are full mare ...
const MARE_HI: f32 = 0.04; // ... above this, highlands (tune so the mare share is 0.22..0.38)
const MIN_FLOOR: f32 = 1.0; // layers above sea level: liquid-less planets are filled flat to sea level

struct LayerParams {
    blocks: f32,
    bowl: f32,
    rim: f32,
    rim_width: f32,
    salt: Vec3,
    min_bowl: f32,
    min_rim: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CraterLayer {
    Large,
    Medium,
    Small,
}

// a smooth max(x, 0): x for x >> 0, 0 for x << 0 (`k` layers of blend), so a floor has no crease
fn softplus_above(x: f32, k: f32) -> f32 {
    if x > 8.0 * k {
        x
    } else {
        k * (1.0 + (x / k).exp()).ln()
    }
}

// Volcanic fissures: one noise octave per this many world units
const FISSURE_PERIOD: f32 = 40.0;
// the band reaches out to this distance from the crossing (half depth at half of it), so >= 2 units wide
const FISSURE_WIDTH: f32 = 6.0;

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- PLANET TERRAIN DATA ---

// the home planet's noise seed (the value every planet used before planets had their own)
pub const HOME_SEED: u32 = 42;

#[derive(Clone)]
pub struct PlanetTerrain {
    // Flattened height map
    heights: Arc<Vec<u16>>,
    // per column: which water body's level is its water level (0 = the sea, k + 1 = lake k's region), separate
    // from the heights so solidity checks (heights only, the hot path) don't fetch it
    water_body: Arc<Vec<u8>>,
    water_levels: Arc<Vec<u32>>, // layer whose top is the water surface, per water body
    resolution: u32,
    height_range: (u32, u32), // lowest and highest column
    sea_level: u32,
}

impl PlanetTerrain {
    pub fn new(resolution: u32, seed: u32) -> Self {
        Self::for_type(resolution, seed, PlanetType::EarthLike, false)
    }

    // a liquid planet's terrain: lakes carved, each with its own water level
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_lakes(resolution: u32, seed: u32) -> Self {
        Self::for_type(resolution, seed, PlanetType::EarthLike, true)
    }

    // the terrain of a planet type (lakes only where the caller says so)
    pub fn for_type(resolution: u32, seed: u32, planet_type: PlanetType, lakes: bool) -> Self {
        Self::for_planet(resolution, seed, planet_type, lakes, TerrainTweaks::NEUTRAL)
    }

    // the same with a planet's own relief and land share
    pub fn for_planet(
        resolution: u32,
        seed: u32,
        planet_type: PlanetType,
        lakes: bool,
        tweaks: TerrainTweaks,
    ) -> Self {
        use rayon::prelude::*;
        let generator = NoiseGenerator::new(seed);
        let shape = TerrainShape::tweaked(resolution, &generator, planet_type, lakes, tweaks);
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
                    // every column inside a lake's region takes that lake's level: below it the column
                    // holds water; above it (shores, islands) its beach band and any hole dug into it
                    // follow the lake, not the sea
                    body[u] = s.lake.map_or(0, |k| (k + 1) as u8);
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

    // billow noise, 0..1: soft rounded mounds (1 − |perlin| per octave)
    pub(crate) fn billow(&self, p: Vec3, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm, mut freq) = (0.0, 1.0, 0.0, 1.0);
        for _ in 0..octaves {
            sum += (1.0 - self.perlin(p * freq).abs()) * amp;
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        (sum / norm).clamp(0.0, 1.0)
    }

    // Voronoi (cellular) noise over the 3D lattice: one feature point per integer cell, jittered by a
    // hash of the cell and the seed. Returns the distances to the nearest (f1) and second-nearest (f2)
    // feature point and the nearest cell's own hash in 0..1 (per-cell choices: which cells get a
    // caldera, how high a plate sits)
    pub(crate) fn cellular(&self, p: Vec3) -> (f32, f32, f32) {
        let base = p.floor();
        let (mut f1, mut f2, mut h1) = (f32::MAX, f32::MAX, 0.0);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let cell = base + Vec3::new(dx as f32, dy as f32, dz as f32);
                    let [jx, jy, jz, h] = self.cell_hash(cell);
                    // feature points stay 0.1 inside their cell, so a point two cells away is never nearer than
                    // the 3x3x3 neighbourhood's second-nearest: the search is exact enough
                    let d = (cell + Vec3::splat(0.1) + Vec3::new(jx, jy, jz) * 0.8 - p).length();
                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                        h1 = h;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
        }
        (f1, f2, h1)
    }

    // four values in 0..1 for an integer cell (splitmix64 over the seed and the cell coordinates)
    fn cell_hash(&self, c: Vec3) -> [f32; 4] {
        let (x, y, z) = (
            c.x as i32 as u32 as u64,
            c.y as i32 as u32 as u64,
            c.z as i32 as u32 as u64,
        );
        let mut s = (self.seed as u64)
            ^ x.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ y.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ z.wrapping_mul(0x1656_67B1_9E37_79F9);
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut v = s;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            v ^= v >> 31;
            (v >> 40) as f32 / (1u64 << 24) as f32
        };
        [next(), next(), next(), next()]
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

    #[test]
    fn cellular_distances_are_ordered_and_continuous() {
        let g = NoiseGenerator::new(HOME_SEED);
        let mut prev: Option<(f32, f32)> = None;
        for k in 0..2000 {
            let p = Vec3::new(k as f32 * 0.01, 0.37, -1.3);
            let (f1, f2, h) = g.cellular(p);
            assert!(f1 >= 0.0 && f1 <= f2, "{p}: {f1} {f2}");
            assert!((0.0..1.0).contains(&h));
            if let Some((a, _)) = prev {
                assert!((f1 - a).abs() <= 0.011, "f1 continuous at {p}");
            }
            prev = Some((f1, f2));
        }
    }

    #[test]
    fn billow_and_terrace_behave() {
        let g = NoiseGenerator::new(7);
        for k in 0..500 {
            let b = g.billow(Vec3::new(k as f32 * 0.13, 1.1, 2.7), 3);
            assert!((0.0..=1.0).contains(&b), "{b}");
        }
        // terrace: monotonic, flat mid-step, steps of `step`
        let mut last = f32::MIN;
        for k in 0..400 {
            let h = k as f32 * 0.05;
            let t = terrace(h, 2.0, 0.15);
            assert!(t >= last - 1e-5);
            last = t;
        }
        assert_eq!(terrace(1.0, 2.0, 0.15), 0.0);
        assert_eq!(terrace(3.0, 2.0, 0.15), 2.0);
        assert!((terrace(3.99, 2.0, 0.15) - 4.0).abs() < 0.1);
    }

    // Review Focus 3
    #[test]
    fn earth_like_profile_is_todays_formula() {
        let g = NoiseGenerator::new(HOME_SEED);
        let shape = TerrainShape::new(128, &g, PlanetType::EarthLike, false);
        assert_eq!(land_offset(PlanetType::EarthLike), 0.08);
        for k in 0..64 {
            let dir = Vec3::new((k as f32 * 0.7).sin(), (k as f32 * 0.3).cos(), 0.4).normalize();
            assert_eq!(shape.natural_height(&g, dir), shape.earth_like(&g, dir));
        }
    }

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
        (TerrainShape::new(res, &g, PlanetType::EarthLike, true), g)
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
        assert!(TerrainShape::new(256, &g, PlanetType::EarthLike, false)
            .lakes()
            .is_empty());
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
                        // a lake region's column: underwater (h < w) or its shore/islands
                        assert!(w > t.sea_level(), "lake column {face}/{u}/{v}: h {h} w {w}");
                        if h < w {
                            lake_cols += 1;
                        }
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
                            if w == t.sea_level() || t.get_height(face, u, v) >= w {
                                continue; // only wet lake columns
                            }
                            for (du, dv) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                                let (nu, nv) = ((u as i32 + du) as u32, (v as i32 + dv) as u32);
                                let dry = t.water_level(face, nu, nv) != w
                                    || t.get_height(face, nu, nv) >= w;
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

    fn fib_dirs(n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                let r = (1.0 - y * y).sqrt();
                let a = i as f32 * 2.399_963;
                Vec3::new(r * a.cos(), y, r * a.sin())
            })
            .collect()
    }

    fn shape_of(t: PlanetType, res: u32, seed: u32) -> (TerrainShape, NoiseGenerator) {
        let g = NoiseGenerator::new(seed);
        (TerrainShape::new(res, &g, t, false), g)
    }

    // share of sampled directions whose natural height satisfies `f`
    fn share(
        t: PlanetType,
        res: u32,
        f: impl Fn(&TerrainShape, &NoiseGenerator, Vec3, f32) -> bool,
    ) -> f32 {
        let (s, g) = shape_of(t, res, 11);
        let dirs = fib_dirs(20000);
        dirs.iter()
            .filter(|&&d| f(&s, &g, d, s.natural_height(&g, d)))
            .count() as f32
            / dirs.len() as f32
    }

    #[test]
    fn ocean_is_mostly_water() {
        let land = share(PlanetType::Ocean, 256, |_, _, _, h| h > 0.0);
        println!("ocean land {land}");
        assert!((0.02..0.10).contains(&land), "{land}");
    }

    #[test]
    fn desert_is_land_with_canyons_above_the_flats() {
        let land = share(PlanetType::Desert, 256, |_, _, _, h| h > 0.0);
        let thin = share(PlanetType::Desert, 256, |_, _, _, h| h > 0.0 && h < 0.99);
        println!("desert land {land} thin {thin}");
        assert!(land >= 0.85);
        // canyon floors stop at sea level + 1: almost nothing lies just above sea level
        assert!(thin < 0.01);
    }

    #[test]
    fn ice_spikes_are_rare_and_wide() {
        let res = 256;
        let (s, g) = shape_of(PlanetType::Ice, res, 11);
        let r = s.relief;
        let high = share(PlanetType::Ice, res, |_, _, _, h| h > 0.6 * r);
        println!("ice spikes {high}");
        assert!(high > 0.0 && high < 0.03, "{high}");
        // every spike column has high neighbours one column away along two tangent axes
        let step = 1.0 / (res as f32 / 2.0);
        for d in fib_dirs(20000) {
            if s.natural_height(&g, d) > 0.6 * r {
                let a = d.cross(Vec3::Y).normalize_or_zero();
                let a = if a == Vec3::ZERO { Vec3::X } else { a };
                let b = d.cross(a);
                for o in [a, -a, b, -b] {
                    let n = (d + o * step).normalize();
                    assert!(s.natural_height(&g, n) > 0.4 * r, "narrow spike at {d}");
                }
            }
        }
    }

    // calderas, craters and cracks: land interior below sea level (Earth has almost none)
    #[test]
    fn hollows_reach_below_sea_level() {
        let inland_low = |t| share(t, 256, |s, g, d, h| s.continent(g, d) > 0.15 && h < 0.0);
        assert!(inland_low(PlanetType::EarthLike) < 0.005);
        for t in [
            PlanetType::Volcanic,
            PlanetType::Toxic,
            PlanetType::Shattered,
        ] {
            println!("{t:?} hollows {}", inland_low(t));
            assert!(inland_low(t) > 0.01, "{t:?}: {}", inland_low(t));
        }
    }

    #[test]
    fn toxic_land_is_flat_and_shattered_is_not() {
        let (s, _) = shape_of(PlanetType::Toxic, 256, 11);
        let r = s.relief;
        let dry = share(PlanetType::Toxic, 256, |s, g, d, h| {
            s.continent(g, d) > 0.15 && h >= 0.0
        });
        let flat = share(PlanetType::Toxic, 256, |s, g, d, h| {
            s.continent(g, d) > 0.15 && h >= 0.0 && h <= 0.25 * r
        });
        println!("toxic flat {flat} of dry {dry}");
        assert!(flat >= 0.8 * dry, "{flat} of {dry}");
        // Shattered: plates at different heights (spread of inland heights)
        let (s, g) = shape_of(PlanetType::Shattered, 256, 11);
        let hs: Vec<f32> = fib_dirs(20000)
            .into_iter()
            .filter(|&d| s.continent(&g, d) > 0.15)
            .map(|d| s.natural_height(&g, d))
            .collect();
        let mean = hs.iter().sum::<f32>() / hs.len() as f32;
        let sd = (hs.iter().map(|h| (h - mean).powi(2)).sum::<f32>() / hs.len() as f32).sqrt();
        println!("shattered sd {sd} relief {}", s.relief);
        assert!(sd > 0.15 * s.relief, "{sd}");
    }

    // `n` points `step` columns apart along the great circle number `k` (a column is 1 world unit)
    fn circle(k: usize, res: u32, step: f32, arc_columns: f32) -> Vec<Vec3> {
        let a = fib_dirs(64)[k];
        let b = a.cross(Vec3::new(0.3, 0.9, 0.31)).normalize();
        let n = (arc_columns / step) as usize;
        let r = res as f32 / 2.0;
        (0..n)
            .map(|i| {
                let t = i as f32 * step / r;
                (a * t.cos() + b * t.sin()).normalize()
            })
            .collect()
    }

    // every contiguous run of "fissure deeper than half" is at least 2 columns long
    #[test]
    fn volcanic_fissures_are_at_least_two_columns_wide() {
        for res in [64, 500] {
            let (s, g) = shape_of(PlanetType::Volcanic, res, 11);
            let (mut runs, mut shortest) = (0, f32::MAX);
            for k in 0..40 {
                let pts = circle(k, res, 0.25, 2.0 * std::f32::consts::PI * res as f32 / 2.0);
                let deep: Vec<bool> = pts.iter().map(|&d| s.fissure(&g, d) > 0.5).collect();
                // skip a run touching either end of the line
                let mut i = deep.iter().position(|&x| !x).unwrap_or(deep.len());
                while i < deep.len() {
                    if deep[i] {
                        let start = i;
                        while i < deep.len() && deep[i] {
                            i += 1;
                        }
                        if i < deep.len() {
                            runs += 1;
                            shortest = shortest.min((i - start) as f32 * 0.25);
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            println!("res {res}: {runs} fissure runs, shortest {shortest} columns");
            assert!(runs > 20, "res {res}: only {runs} fissures sampled");
            assert!(
                shortest >= 2.0,
                "res {res}: a fissure {shortest} columns wide"
            );
        }
    }

    // per-cell features fade out at cell boundaries: no cliffs between neighbouring samples
    #[test]
    fn cell_features_have_no_cliffs() {
        let res = 500;
        for t in [PlanetType::Volcanic, PlanetType::Toxic, PlanetType::Ice] {
            let (s, g) = shape_of(t, res, 3);
            let mut worst = 0.0_f32;
            for k in 0..10 {
                let pts = circle(k, res, 0.01, 300.0);
                let mut prev = s.natural_height(&g, pts[0]);
                for &d in &pts[1..] {
                    let h = s.natural_height(&g, d);
                    worst = worst.max((h - prev).abs());
                    prev = h;
                }
            }
            println!("{t:?} max jump over 0.01 columns: {worst} layers");
            assert!(worst < 0.5, "{t:?}: {worst}");
        }
    }

    // Review Focus 5 and 1
    #[test]
    fn profiles_stay_within_the_relief_cap() {
        for res in [64, 500] {
            for t in [
                PlanetType::Volcanic,
                PlanetType::Ice,
                PlanetType::Desert,
                PlanetType::Toxic,
                PlanetType::Ocean,
                PlanetType::Shattered,
            ] {
                let (s, g) = shape_of(t, res, 3);
                let max = fib_dirs(20000)
                    .into_iter()
                    .map(|d| s.natural_height(&g, d))
                    .fold(f32::MIN, f32::max);
                assert!(
                    max <= 1.2 * s.relief + 1e-3,
                    "{t:?} res {res}: {max} > 1.2 × {}",
                    s.relief
                );
            }
        }
    }

    #[test]
    fn profiles_differ_and_are_deterministic() {
        let d = Vec3::new(0.3, 0.2, 0.93).normalize();
        let heights: Vec<f32> = PlanetType::ALL
            .iter()
            .map(|&t| {
                let (s, g) = shape_of(t, 128, 9);
                s.natural_height(&g, d)
            })
            .collect();
        let (s, g) = shape_of(PlanetType::Desert, 128, 9);
        assert_eq!(
            s.natural_height(&g, d),
            heights[PlanetType::ALL
                .iter()
                .position(|&t| t == PlanetType::Desert)
                .unwrap()]
        );
        let distinct = heights
            .iter()
            .enumerate()
            .all(|(i, a)| heights.iter().skip(i + 1).all(|b| a != b));
        assert!(distinct, "{heights:?}");
    }

    #[test]
    #[ignore] // cargo test --release terrain_bake_bench -- --ignored --nocapture
    fn terrain_bake_bench() {
        for t in PlanetType::ALL {
            let start = std::time::Instant::now();
            let _ = PlanetTerrain::for_type(500, 3, t, t.def().liquid.is_some());
            println!("{t:?}: {:?}", start.elapsed());
        }
    }

    #[test]
    fn cratered_never_dips_below_the_fill() {
        for (res, seed) in [(64, 3), (80, 5), (256, 11), (500, 19)] {
            let (s, g) = shape_of(PlanetType::Cratered, res, seed);
            for d in fib_dirs(20000) {
                let h = s.natural_height(&g, d);
                assert!(h.is_finite() && h >= MIN_FLOOR - 1e-4, "res {res}: {h}");
            }
        }
    }

    #[test]
    fn cratered_has_maria_on_about_a_third_of_the_surface() {
        let (s, g) = shape_of(PlanetType::Cratered, 256, 11);
        let dirs = fib_dirs(20000);
        let mare =
            dirs.iter().filter(|&&d| s.mare_mask(&g, d) > 0.5).count() as f32 / dirs.len() as f32;
        println!("mare share {mare}");
        assert!((0.22..=0.38).contains(&mare), "{mare}");
    }

    #[test]
    fn mare_floors_are_flat_and_highlands_are_rough() {
        let (s, g) = shape_of(PlanetType::Cratered, 256, 11);
        let r = s.relief;
        let dirs = fib_dirs(20000);
        let mare: Vec<f32> = dirs
            .iter()
            .filter(|&&d| s.mare_mask(&g, d) > 0.9)
            .map(|&d| s.natural_height(&g, d))
            .collect();
        let high: Vec<f32> = dirs
            .iter()
            .filter(|&&d| s.mare_mask(&g, d) < 0.1)
            .map(|&d| s.natural_height(&g, d))
            .collect();
        assert!(mare.len() > 500 && high.len() > 500);
        let flat = mare.iter().filter(|&&h| h <= MIN_FLOOR + 0.1 * r).count();
        assert!(
            flat as f32 >= 0.7 * mare.len() as f32,
            "{flat} of {}",
            mare.len()
        );
        let sd = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|h| (h - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        assert!(
            sd(&high) > 2.0 * sd(&mare),
            "{} vs {}",
            sd(&high),
            sd(&mare)
        );
    }

    // each layer digs bowls and raises rims, and the crater sizes grow from Small to Large: along a
    // great circle the sign of the layer changes less often the larger its cells
    #[test]
    fn the_three_crater_layers_have_three_sizes() {
        let (s, g) = shape_of(PlanetType::Cratered, 500, 11);
        let axes = fib_dirs(24);
        let circles: Vec<(Vec3, Vec3)> = (0..10)
            .map(|i| {
                let a = axes[i * 2 + 1];
                (
                    a,
                    a.cross(Vec3::new(0.3, 0.9, 0.31 + 0.1 * i as f32))
                        .normalize(),
                )
            })
            .collect();
        let changes = |layer: CraterLayer| {
            let mut total = 0;
            for &(a, b) in &circles {
                let mut prev = 0.0_f32;
                for i in 0..6000 {
                    let t = std::f32::consts::TAU * i as f32 / 6000.0;
                    let v = s.crater_layer(&g, (a * t.cos() + b * t.sin()).normalize(), layer);
                    if v * prev < 0.0 {
                        total += 1;
                    }
                    if v != 0.0 {
                        prev = v;
                    }
                }
            }
            total as f32 / circles.len() as f32
        };
        let (large, medium, small) = (
            changes(CraterLayer::Large),
            changes(CraterLayer::Medium),
            changes(CraterLayer::Small),
        );
        println!("average sign changes large {large} medium {medium} small {small}");
        assert!(large < medium && medium < small);
    }

    // the start system's Cratered planet is tiny (res 111): its craters must still show in whole layers
    const START_RES: u32 = 111;
    const START_SEED: u32 = 1558184414;

    #[test]
    fn small_planet_craters_are_visible_in_whole_layers() {
        let (s, g) = shape_of(PlanetType::Cratered, START_RES, START_SEED);
        let dirs = fib_dirs(40000);
        for layer in [CraterLayer::Large, CraterLayer::Medium] {
            let vals: Vec<f32> = dirs.iter().map(|&d| s.crater_layer(&g, d, layer)).collect();
            let lo = vals.iter().cloned().fold(0.0_f32, f32::min);
            let hi = vals.iter().cloned().fold(0.0_f32, f32::max);
            println!("{layer:?}: bowl {lo} rim {hi}");
            assert!(lo <= -2.0, "{layer:?} bowl {lo}");
            assert!(hi >= 1.0, "{layer:?} rim {hi}");
        }
    }

    #[test]
    fn small_planet_has_terraces_of_heights_and_a_flat_share() {
        let (s, g) = shape_of(PlanetType::Cratered, START_RES, START_SEED);
        let dirs = fib_dirs(40000);
        let mut levels = std::collections::HashSet::new();
        let mut floor = 0;
        for &d in &dirs {
            let h = s.natural_height(&g, d);
            if h <= 1.5 {
                floor += 1;
            }
            if s.mare_mask(&g, d) < 0.1 {
                levels.insert(h.round() as i32);
            }
        }
        let share = floor as f32 / dirs.len() as f32;
        println!("highland levels {} floor share {share}", levels.len());
        assert!(levels.len() >= 6, "{levels:?}");
        assert!((0.15..=0.6).contains(&share), "{share}");
    }

    // no cliffs at cell boundaries: no step between neighbouring columns beyond what Toxic (also cellular
    // craters) has
    #[test]
    fn cratered_has_no_cliffs() {
        let max_step = |t: PlanetType| {
            let (s, g) = shape_of(t, 500, 11);
            let mut worst = 0.0_f32;
            for k in 0..16 {
                let pts = circle(k, 500, 1.0, 4000.0);
                for w in pts.windows(2) {
                    worst =
                        worst.max((s.natural_height(&g, w[0]) - s.natural_height(&g, w[1])).abs());
                }
            }
            worst
        };
        let (c, t) = (max_step(PlanetType::Cratered), max_step(PlanetType::Toxic));
        println!("max step cratered {c} toxic {t}");
        assert!(c <= 1.5 * t.max(1.0), "{c} vs {t}");
    }

    #[test]
    fn cratered_is_deterministic_and_seed_dependent() {
        let (a, ga) = shape_of(PlanetType::Cratered, 128, 1);
        let (b, gb) = shape_of(PlanetType::Cratered, 128, 1);
        let (c, gc) = shape_of(PlanetType::Cratered, 128, 2);
        let d = Vec3::new(0.3, 0.5, 0.8).normalize();
        assert_eq!(a.natural_height(&ga, d), b.natural_height(&gb, d));
        let differs = fib_dirs(200)
            .iter()
            .any(|&d| a.natural_height(&ga, d) != c.natural_height(&gc, d));
        assert!(differs);
    }

    // bake time at res 500 against Toxic's (run on purpose: cargo test --release cratered_bake -- --ignored --nocapture)
    #[test]
    #[ignore]
    fn cratered_bake_time_is_at_most_twice_toxics() {
        let time = |t: PlanetType| {
            let start = std::time::Instant::now();
            let _ =
                crate::planet::noise::PlanetTerrain::for_type(500, 11, t, t.def().liquid.is_some());
            start.elapsed().as_secs_f32()
        };
        let (c, t) = (time(PlanetType::Cratered), time(PlanetType::Toxic));
        println!("bake cratered {c:.3}s toxic {t:.3}s");
        assert!(c <= 2.0 * t, "{c} vs {t}");
    }
}
