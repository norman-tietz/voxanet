# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

voxanet is a single-binary Rust voxel engine that renders an explorable spherical planet with wgpu (30) / winit (0.30), glyphon for text, glam for math. There is no library crate, no test suite, and no CI config.

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
- `terrain: PlanetTerrain` (`src/noise.rs`): a precomputed per-column height map indexed by `(face, u, v)`, generated once per resolution (in parallel with rayon). `TerrainShape` builds it from continent noise (ocean vs land around a sea level at layer `res/2`), hills, and ridged mountain noise inside low-frequency mountain zones; overall relief scales with `3·sqrt(res/2)`, capped at 20% of the radius (layers stay ~1 unit thick, so small planets need proportionally less relief).
- `chunks: HashMap<ChunkKey, ChunkMods>`: only the player's edits per 32×32 column chunk (`CHUNK_SIZE`): `placed: HashMap<BlockId, BlockType>` and `mined: HashSet<BlockId>`.

`PlanetData::exists(id)` is the single source of truth for solidity: placed → true, mined → false, otherwise `layer <= height`. Meshing, physics, and raycasting all go through it. Layers below 6 are an unbreakable core (`remove_block`).

Block types live in `src/material.rs`. Natural blocks have no stored type: `material::natural_type` derives it from the column's height within the planet's height range (`PlanetTerrain::height_range`), the slope to neighbouring columns and smooth per-column jitter (sand on beaches and the sea floor, stone on steep steps and from 45% of the peak height above sea level, snow from 62%, dirt then stone below the surface). `PlanetData::block_type(id)` combines that with edits; voxel and LOD meshes both use it, so they agree.

### Sun shadows: ray marching, no shadow map
The sun is a fixed directional light (`sun_dir` in `Renderer::render`). Shadows are ray-marched per pixel, with no shadow map:
- `src/rt_shadow.rs` builds a window of solid/air bits per cube face (whole faces up to `WINDOW_SIZE` = 256 columns per face, otherwise windows on the faces turned toward the player) plus 8×8-column max-height tiles. It's rebuilt in `Renderer::update_rt_window` when the player moves far, changes face, or a block is edited (`rt_dirty`).
- `rt_shadow()` in `shader.wgsl` walks those cells toward the sun (3D DDA in block space, re-linearised every few units for the planet's curvature, bisection across cube-face edges, skipping segments above the max-height tiles).
- `src/rt_blur.rs` + `src/blur.wgsl`: an RT pass draws the scene with `fs_rt` into an offscreen texture (shadow, camera distance), a separable depth-aware blur softens the edges (`PENUMBRA_WIDTH`), and `fs_main` reads the result through bind group 2. Ray marching costs per pixel, so these targets are capped at `MAX_RT_PIXELS` (1.5 MP) and `shadow_factor()` upsamples them depth-aware (4 taps weighted by camera-distance match; exact single tap at full resolution). `GlobalUniform.screen` carries the screen size for that. Every pipeline using the scene layout must bind group 2.
- `RtParams`/`FaceWindow` in `rt_shadow.rs` must match the WGSL structs. The UI bind group uses `rt.enabled = 0`, which means "lit, don't read the shadow texture".

### Coordinates (`src/gen.rs`, `CoordSystem`)
A `BlockId` is `{face: 0..6, layer, u, v}` on a cube-sphere. `resolution` is the voxel count per face edge. Cube→sphere uses the Nowell mapping (`cube_to_sphere`), and `cubize_point` is its inverse. Layers are radially **exponential**, not linear: `get_layer_radius(layer, res) = (res/2) * exp(K*(layer/(res/2) - 1))` with `K = 0.85`, so voxels keep a roughly cubic shape at every depth. Convert between world positions and blocks only through `CoordSystem` (`pos_to_id`, `get_local_coords`, `get_block_center`, `get_vertex_pos`). Never hand-roll the math.

### Renderer owns streaming and LOD (`src/renderer.rs`)
`Renderer` is the largest module. Besides the wgpu pipelines (fill/wire/line/UI, the ray-marched shadow pass and blur, plus glyphon text), it also owns world streaming:
- `update_view` walks a quadtree per face (`process_quadtree`, logical size `res.next_power_of_two()`). Nodes split on distance against `node_radius * lod_factor`, and `lod_factor` grows as node size shrinks. Leaf nodes at `CHUNK_SIZE` near the player become full voxel chunks (`ChunkKey` → `chunks`). Coarser nodes become heightfield LOD meshes (`LodKey` → `lod_chunks`).
- Mesh generation runs on `std::thread::spawn` workers that get a **cloned `PlanetData`** and send `(key, verts, indices)` back over `mpsc` channels (`mesh_tx/rx`, `lod_tx/rx`). Per-frame budgets cap spawns and GPU uploads. `pending_*` sets dedupe in-flight work.
- An LOD mesh is kept alive until all overlapping voxel chunks have loaded, which prevents holes. Removed meshes go to `LodAnimator` (`src/lod_animation.rs`) to fade out, and new ones fade in.
- After a block edit, `refresh_neighbors` must be called so the affected chunks get remeshed. After `planet.resize`, call `force_reload_all`.

`MeshGen` (`src/gen.rs`) builds voxel chunk meshes (with AO and skirts), LOD meshes, and debug geometry (player cylinder, crosshair, collision boxes). `Vertex` is `bytemuck::Pod` and uploads directly. All shading (ray-marched sun shadows, exp² fog, ACES, dithered transparency) is in `src/shader.wgsl`, loaded with `include_str!`. CPU-side uniform structs (`GlobalUniform`, `LocalUniform`) must match its layout.

### Gameplay loop (`src/main.rs`)
`App` implements winit's `ApplicationHandler`; the window and `Game` (renderer, controller, player, planet, console) are created in `resumed`. `Game::tick` runs before every window/device event and on `about_to_wait`: `controller.update_player` (physics), raycast to cursor, `renderer.update_cursor`, `renderer.update_view`; then the event is dispatched. On macOS the surface is tagged Display P3 and `fs_main` converts its output from sRGB to P3 primaries (flag in `sun_dir.w`), otherwise colours show oversaturated. `Physics` (`src/physics.rs`) is a custom kinematic solver. Up is `normalize(position)`, orientation uses `Quat::from_rotation_arc`, and collision samples `exists()` with a 5% edge "shave" margin. `Console` (`src/cmd.rs`) captures keyboard input while open.

Known quirk (don't "fix" it silently, mention it): `src/lighting.rs` is not declared as a module, so it is dead code that isn't compiled. `update_player` runs exactly once per tick; movement and turning speeds are per second (Q/E turn at `TURN_SPEED` in `controller.rs`).

## Runtime controls (useful for manual verification)
- WASD / Space / Left Ctrl (sprint); mouse look (first person); `Q`/`E` turn left/right in both views. LMB mines, RMB places; `1`–`5` choose the placed block type.
- `K` toggles first/third person. `F` toggles fly (first person only).
- `]` / `[` grows/shrinks planet resolution by ×1.2 (min 8, max 16384) and regenerates terrain.
- `` ` `` opens the console: `help`, `/debug_mode set true`, `/move_speed get|set <v>`, `/jump_force get|set <v>`.
- When debug mode is on: `P` wireframe, `O` collision boxes, `'` freezes frustum culling. The overlay shows GPU time per part (`src/gpu_timer.rs`, timestamp queries). Apple GPUs overlap passes, so each part is measured from the previous part's end to its own end; use this overlay rather than FPS when profiling.
