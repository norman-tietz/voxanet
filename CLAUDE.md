# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

voxanet is a single-binary Rust voxel engine that renders an explorable spherical planet with wgpu (0.19) / winit (0.29). There is no library crate, no test suite, and no CI config.

## Commands

```sh
cargo run --release   # debug builds are very slow for terrain/mesh generation
cargo build
cargo check
cargo clippy
```

There are no tests (`cargo test` builds but runs nothing).

## Architecture

### Data model: heightmap + sparse edit diffs
The planet is not stored as a voxel array. `PlanetData` (`src/common.rs`) holds:
- `terrain: PlanetTerrain` (`src/noise.rs`): a precomputed per-column height map indexed by `(face, u, v)`, generated once per resolution (in parallel with rayon). `TerrainShape` builds it from continent noise (ocean vs land around a sea level at layer `res/2`), hills, and ridged mountain noise inside low-frequency mountain zones; overall relief scales with `3·sqrt(res/2)`.
- `chunks: HashMap<ChunkKey, ChunkMods>`: only the player's edits per 32×32 column chunk (`CHUNK_SIZE`): `placed: HashMap<BlockId, BlockType>` and `mined: HashSet<BlockId>`.

`PlanetData::exists(id)` is the single source of truth for solidity: placed → true, mined → false, otherwise `layer <= height`. Meshing, physics, and raycasting all go through it. Layers below 6 are an unbreakable core (`remove_block`).

Block types live in `src/material.rs`. Natural blocks have no stored type: `material::natural_type` derives it from the column's height within the planet's height range (`PlanetTerrain::height_range`), the slope to neighbouring columns and smooth per-column jitter (sand on beaches and the sea floor, stone on steep steps and from 45% of the peak height above sea level, snow from 62%, dirt then stone below the surface). `PlanetData::block_type(id)` combines that with edits; voxel and LOD meshes both use it, so they agree.

### Coordinates (`src/gen.rs`, `CoordSystem`)
A `BlockId` is `{face: 0..6, layer, u, v}` on a cube-sphere. `resolution` is the voxel count per face edge. Cube→sphere uses the Nowell mapping (`cube_to_sphere`), and `cubize_point` is its inverse. Layers are radially **exponential**, not linear: `get_layer_radius(layer, res) = (res/2) * exp(K*(layer/(res/2) - 1))` with `K = 0.85`, so voxels keep a roughly cubic shape at every depth. Convert between world positions and blocks only through `CoordSystem` (`pos_to_id`, `get_local_coords`, `get_block_center`, `get_vertex_pos`). Never hand-roll the math.

### Renderer owns streaming and LOD (`src/renderer.rs`)
`Renderer` is the largest module. Besides the wgpu pipelines (fill/wire/line/shadow/UI plus glyphon text), it also owns world streaming:
- `update_view` walks a quadtree per face (`process_quadtree`, logical size `res.next_power_of_two()`). Nodes split on distance against `node_radius * lod_factor`, and `lod_factor` grows as node size shrinks. Leaf nodes at `CHUNK_SIZE` near the player become full voxel chunks (`ChunkKey` → `chunks`). Coarser nodes become heightfield LOD meshes (`LodKey` → `lod_chunks`).
- Mesh generation runs on `std::thread::spawn` workers that get a **cloned `PlanetData`** and send `(key, verts, indices)` back over `mpsc` channels (`mesh_tx/rx`, `lod_tx/rx`). Per-frame budgets cap spawns and GPU uploads. `pending_*` sets dedupe in-flight work. (`rayon` is a dependency but is currently unused.)
- An LOD mesh is kept alive until all overlapping voxel chunks have loaded, which prevents holes. Removed meshes go to `LodAnimator` (`src/lod_animation.rs`) to fade out, and new ones fade in.
- After a block edit, `refresh_neighbors` must be called so the affected chunks get remeshed. After `planet.resize`, call `force_reload_all`.

`MeshGen` (`src/gen.rs`) builds voxel chunk meshes (with AO and skirts), LOD meshes, and debug geometry (player cylinder, guide sphere, crosshair, collision boxes). `Vertex` is `bytemuck::Pod` and uploads directly. All shading (cascaded/texel-snapped shadows, 5×5 PCF, exp² fog, ACES, dithered transparency) is in `src/shader.wgsl`, loaded with `include_str!`. CPU-side uniform structs (`GlobalUniform`, `LocalUniform`) must match its layout.

### Gameplay loop (`src/main.rs`)
The winit closure runs per event: `controller.update_player` (physics), raycast to cursor, `renderer.update_cursor`, `renderer.update_view`, then input dispatch. `Physics` (`src/physics.rs`) is a custom kinematic solver. Up is `normalize(position)`, orientation uses `Quat::from_rotation_arc`, and collision samples `exists()` with a 5% edge "shave" margin. `Console` (`src/cmd.rs`) captures keyboard input while open.

Known quirks (don't "fix" these silently, mention them): `update_player` and the raycast run twice per event when the console is closed, and `src/lighting.rs` is not declared as a module, so it is dead code that isn't compiled.

## Runtime controls (useful for manual verification)
- WASD / Space / Left Ctrl (sprint); mouse look. LMB mines, RMB places; `1`–`5` choose the placed block type.
- `K` toggles first/third person. `F` toggles fly (first person only).
- `]` / `[` grows/shrinks planet resolution by ×1.2 (min 8, max 16384) and regenerates terrain.
- `` ` `` opens the console: `help`, `/debug_mode set true`, `/move_speed get|set <v>`, `/jump_force get|set <v>`.
- When debug mode is on: `P` wireframe, `O` collision boxes, `'` freezes frustum culling.
