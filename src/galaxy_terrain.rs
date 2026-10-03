// galaxy_terrain.rs
// Sub-project #2 (galaxy terrain impostors): generates a terrain-displaced icosphere mesh for one
// GalaxyPlanet, sampling the same TerrainShape/NoiseGenerator the real single-planet engine uses —
// but directly, with no baked PlanetTerrain heightmap. Baking a heightmap for every planet in the
// galaxy would defeat sub-project #1's "many planets, bounded memory" design (see that feature's
// design discussion). Colors are a simplified, slope-free height-banded rule — material::natural_type
// needs neighbor-slope data that's meaningless on this sparse, widely-spaced mesh.

use crate::common::Vertex;
use crate::galaxy::GalaxyPlanet;
use crate::noise::{NoiseGenerator, TerrainShape};
use glam::Vec3;

// fractions of this planet's own peak height (shape_relief) above sea level; mirrors
// material.rs's ROCK_LINE/SNOW_LINE spirit but as new, impostor-specific constants (material.rs's
// are private and tuned for integer voxel layers, not a continuous float height)
const ROCK_LINE: f32 = 0.35;
const SNOW_LINE: f32 = 0.70;

// unit-sphere vertex positions, displaced outward/inward by this planet's own TerrainShape, plus
// per-vertex biome color and a real face-normal (not a noise-sampled one, so shading matches
// whatever faceting exists at this subdivision level rather than implying smoothness the mesh
// doesn't have)
pub fn generate_planet_mesh(planet: &GalaxyPlanet, subdivision: u32) -> (Vec<Vertex>, Vec<u32>) {
    let (unit_verts, indices) = crate::icosphere::generate(subdivision);
    let shape = TerrainShape::new((planet.radius * 2.0) as u32);
    let generator = NoiseGenerator::new(planet.noise_seed);

    // TerrainShape::height already returns an offset from sea level in units that are ~1 world
    // unit thick near the surface (same as the real engine's exponential layers, which are
    // approximately linear near res/2) — so no extra scale factor is needed, the planet's own
    // radius plus this value directly gives a world-space displaced position
    let heights: Vec<f32> = unit_verts
        .iter()
        .map(|&dir| shape.height(&generator, dir))
        .collect();
    let def = planet.planet_type.def();
    let positions: Vec<Vec3> = unit_verts
        .iter()
        .zip(heights.iter())
        .map(|(&dir, &h)| {
            // mirrors PlanetData::effective_height (src/common.rs): on a liquid-less planet there's
            // no water mesh to fill the gap visually, so the surface is solid up to sea level — the
            // *color* logic below still uses the raw height to pick palette.beach for these columns,
            // same as the real engine's natural_type() does for filled-in liquid-less "ocean"
            let display_h = if def.liquid.is_none() { h.max(0.0) } else { h };
            dir * (planet.radius + display_h)
        })
        .collect();

    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let face_normal = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += face_normal;
        normals[b] += face_normal;
        normals[c] += face_normal;
    }

    let relief = shape_relief(planet.radius);

    let verts = (0..positions.len())
        .map(|i| {
            let height = heights[i];
            let color = if height < 0.0 {
                match def.liquid {
                    Some(l) => l.shallow_color,
                    None => def.palette.beach.color(),
                }
            } else {
                let frac = (height / relief).clamp(0.0, 1.0);
                if frac > SNOW_LINE {
                    def.palette.peak.color()
                } else if frac > ROCK_LINE {
                    def.palette.rock.color()
                } else {
                    def.palette.ground.color()
                }
            };
            Vertex {
                pos: positions[i].to_array(),
                color,
                normal: normals[i].normalize_or_zero().to_array(),
            }
        })
        .collect();

    (verts, indices)
}

// same formula TerrainShape::new uses internally (its own `relief` field is private) — kept here
// as the single place both this function and any future caller derive "this planet's peak height"
// from, without needing to widen TerrainShape's field visibility too
fn shape_relief(radius: f32) -> f32 {
    (3.0 * radius.sqrt()).min(0.2 * radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biome::PlanetType;

    fn test_planet(noise_seed: u32, planet_type: PlanetType) -> GalaxyPlanet {
        GalaxyPlanet {
            orbit_radius: 8_000.0,
            orbit_speed: 0.01,
            orbit_phase: 0.0,
            radius: 100.0,
            planet_type,
            noise_seed,
        }
    }

    #[test]
    fn vertex_and_index_counts_match_icosphere() {
        for subdivision in [0, 1, 2] {
            let (icosphere_verts, icosphere_indices) = crate::icosphere::generate(subdivision);
            let (verts, indices) =
                generate_planet_mesh(&test_planet(1, PlanetType::EarthLike), subdivision);
            assert_eq!(verts.len(), icosphere_verts.len());
            assert_eq!(indices.len(), icosphere_indices.len());
        }
    }

    #[test]
    fn colors_are_from_the_active_palette_or_liquid() {
        let planet = test_planet(1, PlanetType::EarthLike);
        let def = planet.planet_type.def();
        let allowed = [
            def.liquid.unwrap().shallow_color,
            def.palette.beach.color(),
            def.palette.ground.color(),
            def.palette.rock.color(),
            def.palette.peak.color(),
        ];
        let (verts, _) = generate_planet_mesh(&planet, 2);
        for v in &verts {
            assert!(allowed.contains(&v.color), "unexpected color {:?}", v.color);
        }
    }

    #[test]
    fn normals_are_unit_length() {
        let (verts, _) = generate_planet_mesh(&test_planet(1, PlanetType::EarthLike), 2);
        for v in &verts {
            let len = Vec3::from_array(v.normal).length();
            assert!(
                (len - 1.0).abs() < 1e-3,
                "normal {:?} has length {len}",
                v.normal
            );
        }
    }

    #[test]
    fn normals_point_roughly_outward() {
        let (verts, _) = generate_planet_mesh(&test_planet(1, PlanetType::EarthLike), 2);
        for v in &verts {
            let pos = Vec3::from_array(v.pos).normalize();
            let normal = Vec3::from_array(v.normal);
            assert!(
                normal.dot(pos) > 0.0,
                "normal {:?} faces inward at pos {:?}",
                v.normal,
                v.pos
            );
        }
    }

    #[test]
    fn different_noise_seeds_produce_different_terrain() {
        let a = test_planet(1, PlanetType::EarthLike);
        let b = test_planet(2, PlanetType::EarthLike);
        let (verts_a, _) = generate_planet_mesh(&a, 2);
        let (verts_b, _) = generate_planet_mesh(&b, 2);
        let differs = verts_a.iter().zip(verts_b.iter()).any(|(va, vb)| {
            let ra = Vec3::from_array(va.pos).length();
            let rb = Vec3::from_array(vb.pos).length();
            (ra - rb).abs() > 1e-3
        });
        assert!(differs, "two different noise seeds produced identical terrain — the per-planet seeding isn't taking effect");
    }
}
