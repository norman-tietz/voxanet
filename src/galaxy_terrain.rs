// galaxy_terrain.rs
// Sub-project #2 (galaxy terrain impostors): generates a terrain-displaced icosphere mesh for one
// GalaxyPlanet, sampling the same TerrainShape/NoiseGenerator the real single-planet engine uses —
// but directly, with no baked PlanetTerrain heightmap. Baking a heightmap for every planet in the
// galaxy would defeat sub-project #1's "many planets, bounded memory" design (see that feature's
// design discussion). Colors are a simplified, slope-free height-banded rule — material::natural_type
// needs neighbor-slope data that's meaningless on this sparse, widely-spaced mesh.

use crate::common::Vertex;
use crate::galaxy::GalaxyPlanet;
use crate::gen::CoordSystem;
use crate::noise::{NoiseGenerator, TerrainShape};
use glam::Vec3;

// fractions of this planet's own peak height (shape_relief) above sea level — the same fractions
// material.rs's natural_type uses for rock (45%) and snow (62%)
const ROCK_LINE: f32 = 0.45;
const SNOW_LINE: f32 = 0.62;

// unit-sphere vertex positions, displaced outward/inward by this planet's own TerrainShape, plus
// per-vertex biome color and a real face-normal (not a noise-sampled one, so shading matches
// whatever faceting exists at this subdivision level rather than implying smoothness the mesh
// doesn't have)
pub fn generate_planet_mesh(planet: &GalaxyPlanet, subdivision: u32) -> (Vec<Vertex>, Vec<u32>) {
    let (unit_verts, indices) = crate::icosphere::generate(subdivision);
    // the same resolution GalaxyPlanet::bake uses, so impostor and voxel world agree to the unit
    let res = planet.voxel_resolution();
    let sea_level = (res / 2) as f32;
    let def = planet.planet_type.def();
    let generator = NoiseGenerator::new(planet.noise_seed);
    // lakes like the baked planet (GalaxyPlanet::bake: liquid planet types)
    let shape = TerrainShape::new(res, &generator, def.liquid.is_some());

    // TerrainShape returns offsets in layers from sea level, like the real engine's; per vertex the
    // carved height and, inside a lake below its level, that level
    let samples: Vec<(f32, Option<f32>)> = unit_verts
        .iter()
        .map(|&dir| {
            let s = shape.sample(&generator, dir);
            let lake = s
                .lake
                .map(|k| shape.lakes()[k].level as f32 - sea_level)
                .filter(|&l| s.height < l);
            (s.height, lake)
        })
        .collect();
    let positions: Vec<Vec3> = unit_verts
        .iter()
        .zip(samples.iter())
        .map(|(&dir, &(h, lake))| {
            // like the engine's distant LOD meshes (MeshGen::generate_lod_mesh): lakes flat at their
            // level, oceans and liquid-less basins at sea level, layers spaced exponentially
            // (get_layer_radius). The colour below still uses the raw height, so underwater
            // columns get the liquid (or, liquid-less, the beach) colour.
            dir * CoordSystem::get_layer_radius_f(sea_level + lake.unwrap_or(h.max(0.0)), res)
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

    let relief = shape_relief(res as f32 / 2.0);

    let verts = (0..positions.len())
        .map(|i| {
            let (height, lake) = samples[i];
            let color = if lake.is_some() {
                // only liquid planets have lakes
                def.liquid
                    .map_or(def.palette.beach.color(), |l| l.shallow_color)
            } else if height < 0.0 {
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
                water: 0.0,
            }
        })
        .collect();

    (verts, indices)
}

// the near impostor: once a planet is baked, the voxel engine's own distant-terrain meshes for the
// six whole cube faces (MeshGen::generate_lod_mesh on the quadtree's root nodes) — exactly the shape
// and colours the engine shows from orbit, so the handover to the voxel engine doesn't pop. Built on
// the bake thread; the LOD skirts hang below the surface but are hidden inside the closed planet.
pub fn near_impostor_mesh(data: &crate::common::PlanetData) -> (Vec<Vertex>, Vec<u32>) {
    let size = data.resolution.next_power_of_two();
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for face in 0..6u8 {
        let key = crate::common::LodKey {
            face,
            x: 0,
            y: 0,
            size,
        };
        let (v, i) = crate::gen::MeshGen::generate_lod_mesh(key, data);
        let base = verts.len() as u32;
        verts.extend(v);
        indices.extend(i.into_iter().map(|k| k + base));
    }
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
    use crate::gen::CoordSystem;

    // the near impostor is exactly the engine's own whole-face distant-terrain meshes, so the later
    // handover to the voxel engine can't change the planet's shape or colours
    #[test]
    fn near_impostor_is_the_engines_whole_face_lod_meshes() {
        let mut p = test_planet(3, PlanetType::EarthLike);
        p.radius = 40.0; // smallest size: keeps the bake quick in a debug test build
        let data = p.bake();
        let (verts, indices) = near_impostor_mesh(&data);
        let size = data.resolution.next_power_of_two();
        let mut expected = Vec::new();
        for face in 0..6u8 {
            let key = crate::common::LodKey {
                face,
                x: 0,
                y: 0,
                size,
            };
            expected.extend(crate::gen::MeshGen::generate_lod_mesh(key, &data).0);
        }
        assert_eq!(verts.len(), expected.len());
        assert!(verts
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.pos == b.pos && a.color == b.color));
        assert!(
            indices.iter().all(|&k| (k as usize) < verts.len()),
            "index out of range"
        );
        assert_eq!(indices.len() % 3, 0);
    }

    // the impostor must have the shape the voxel engine shows from orbit: exponential layers
    // (CoordSystem::get_layer_radius), radius voxel_resolution()/2, oceans flattened to sea level on
    // every planet type (generate_lod_mesh flattens them all)
    #[test]
    fn vertices_sit_on_the_engines_layers_with_flat_oceans() {
        for planet_type in PlanetType::ALL {
            let planet = test_planet(1, planet_type);
            let res = planet.voxel_resolution();
            let generator = NoiseGenerator::new(planet.noise_seed);
            let shape = TerrainShape::new(res, &generator, planet_type.def().liquid.is_some());
            let (unit_verts, _) = crate::icosphere::generate(2);
            let (verts, _) = generate_planet_mesh(&planet, 2);
            let sea = (res / 2) as f32;
            for (dir, v) in unit_verts.iter().zip(&verts) {
                // lakes flat at their level, oceans at sea level
                let s = shape.sample(&generator, *dir);
                let lake_level = s.lake.map(|k| shape.lakes()[k].level as f32 - sea);
                let h = match lake_level {
                    Some(l) if s.height < l => l,
                    _ => s.height.max(0.0),
                };
                let expected = CoordSystem::get_layer_radius_f(sea + h, res);
                let actual = Vec3::from_array(v.pos).length();
                assert!(
                    (actual - expected).abs() < 1e-2,
                    "{planet_type:?}: {actual} vs {expected}"
                );
                assert!(actual >= CoordSystem::get_layer_radius_f(sea, res) - 1e-3);
            }
        }
    }

    fn test_planet(noise_seed: u32, planet_type: PlanetType) -> GalaxyPlanet {
        GalaxyPlanet {
            orbit_radius: 8_000.0,
            orbit_speed: 0.01,
            orbit_phase: 0.0,
            orbit_tilt: 0.0,
            orbit_node: 0.0,
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
    fn lake_test_planet(planet_type: PlanetType) -> GalaxyPlanet {
        (1..80)
            .map(|seed| test_planet(seed, planet_type))
            .find(|p| {
                let g = NoiseGenerator::new(p.noise_seed);
                !TerrainShape::new(p.voxel_resolution(), &g, true)
                    .lakes()
                    .is_empty()
            })
            .expect("no seed with a lake")
    }

    #[test]
    fn impostor_lakes_are_flat_liquid_at_their_level() {
        let planet = lake_test_planet(PlanetType::EarthLike);
        let res = planet.voxel_resolution();
        let g = NoiseGenerator::new(planet.noise_seed);
        let shape = TerrainShape::new(res, &g, true);
        let sea = (res / 2) as f32;
        let (unit_verts, _) = crate::icosphere::generate(5);
        let (verts, _) = generate_planet_mesh(&planet, 5);
        let mut lake_verts = 0;
        for (dir, v) in unit_verts.iter().zip(&verts) {
            let s = shape.sample(&g, *dir);
            if let Some(k) = s.lake {
                let level = shape.lakes()[k].level as f32;
                if sea + s.height < level {
                    lake_verts += 1;
                    let expected = CoordSystem::get_layer_radius_f(level, res);
                    assert!((Vec3::from_array(v.pos).length() - expected).abs() < 1e-2);
                    assert_eq!(
                        v.color,
                        PlanetType::EarthLike.def().liquid.unwrap().shallow_color
                    );
                }
            }
        }
        assert!(lake_verts > 0, "no impostor vertex fell into a lake");
    }
}
