// icosphere.rs
// A standalone icosphere generator for galaxy mode's placeholder planet/star meshes
// (src/galaxy_render.rs). Not voxel-related; deliberately separate from gen.rs/MeshGen, which only
// ever builds terrain geometry.

use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_subdivisions_is_the_bare_icosahedron() {
        let (verts, indices) = generate(0);
        assert_eq!(verts.len(), 12);
        assert_eq!(indices.len(), 60); // 20 faces * 3
    }

    #[test]
    fn subdivision_quadruples_face_count_each_level() {
        let (_, indices1) = generate(1);
        let (_, indices2) = generate(2);
        assert_eq!(indices1.len(), 80 * 3);
        assert_eq!(indices2.len(), 320 * 3);
    }

    #[test]
    fn all_vertices_are_unit_length() {
        for level in [0, 1, 2] {
            let (verts, _) = generate(level);
            for v in &verts {
                assert!(
                    (v.length() - 1.0).abs() < 1e-4,
                    "level {level}: {v:?} not unit length"
                );
            }
        }
    }

    #[test]
    fn all_indices_reference_valid_vertices() {
        let (verts, indices) = generate(2);
        for &i in &indices {
            assert!((i as usize) < verts.len());
        }
    }
}

fn base_icosahedron() -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let verts: Vec<Vec3> = [
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ]
    .iter()
    .map(|v| Vec3::from(*v).normalize())
    .collect();

    let faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    (verts, faces)
}

fn get_or_add_midpoint(
    a: u32,
    b: u32,
    verts: &mut Vec<Vec3>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
) -> u32 {
    let key = (a.min(b), a.max(b));
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let mid = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
    let idx = verts.len() as u32;
    verts.push(mid);
    cache.insert(key, idx);
    idx
}

// unit-sphere vertex positions (also the normals, being unit length from the origin) and triangle
// indices. `subdivisions` 0 = the bare 12-vertex icosahedron; each level ~4x's the triangle count.
pub fn generate(subdivisions: u32) -> (Vec<Vec3>, Vec<u32>) {
    let (mut verts, mut faces) = base_icosahedron();

    for _ in 0..subdivisions {
        let mut cache = std::collections::HashMap::new();
        let mut next_faces = Vec::with_capacity(faces.len() * 4);
        for f in &faces {
            let (a, b, c) = (f[0], f[1], f[2]);
            let ab = get_or_add_midpoint(a, b, &mut verts, &mut cache);
            let bc = get_or_add_midpoint(b, c, &mut verts, &mut cache);
            let ca = get_or_add_midpoint(c, a, &mut verts, &mut cache);
            next_faces.push([a, ab, ca]);
            next_faces.push([b, bc, ab]);
            next_faces.push([c, ca, bc]);
            next_faces.push([ab, bc, ca]);
        }
        faces = next_faces;
    }

    let indices: Vec<u32> = faces.into_iter().flatten().collect();
    (verts, indices)
}
