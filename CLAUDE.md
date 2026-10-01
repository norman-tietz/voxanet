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

### Frame structure: deferred shading (`src/deferred.rs`)
`Renderer::render` draws the scene geometry once per frame:
1. **Geometry pass** (`fs_geom`, full screen resolution): vertex colour (`Rgba8Unorm`), normal (`Rgb10a2Unorm`, `n * 0.5 + 0.5`), camera distance (`R32Float`, 0 = sky) and depth. Chunks, LOD meshes, fading chunks and the third-person player are drawn here; fading chunks dither (discard) as before.
2. **Shadows** (below): `cs_gbuf_down` derives the shadow-resolution G-buffer from it, then one compute invocation per shadow texel, then the blur.
3. **Lighting pass** (`vs_full` + `fs_light`): one full-screen triangle; `shade()` in `shader.wgsl` (lighting, fog, ACES, P3 output) runs once per pixel. Sky pixels are discarded and keep the pass's clear colour.
4. **Forward overlays** in the same pass, depth-tested against the G-buffer depth: collision lines, cursor box (`fs_main`, which also calls `shade()`), crosshair, console; then text.

World positions are reconstructed from the camera distance and a camera ray basis (`GlobalUniform.ray_dirs`, built in f64 by `Renderer::ray_dirs` in homogeneous form so the per-pixel interpolation is exact). Don't use an f32 inverse view-projection for this: with near 0.1 / far 20000 it was off by ~0.5 units at 500 units, which made hardware rays self-shadow distant surfaces.

### Sun shadows: hardware ray tracing or ray marching, no shadow map
The sun is a fixed directional light (`sun_dir` in `Renderer::render`). There is no shadow map. Per frame (`src/rt_blur.rs`):
1. `Deferred::downsample` (`cs_gbuf_down`) fills `RtBlur::g_pos` / `g_nrm` (world position + camera distance, normal) at the shadow resolution, capped at `MAX_RT_PIXELS` (1.5 MP).
2. A compute pass computes the sharp shadow term once per texel into `RtBlur::target`, using one of two paths.
3. `blur.wgsl` blurs it (separable, depth-aware, `PENUMBRA_WIDTH`), and `shade()` upsamples it through bind group 2 (`shadow_at()`: 4 taps weighted by camera-distance match; `GlobalUniform.screen` carries the screen size). Every pipeline using the scene layout must bind group 2.

The two paths:
- **Hardware ray tracing** (`src/hw_rt.rs`, `src/rt_hw.wgsl`), the default when the device has `EXPERIMENTAL_RAY_QUERY` (opted in with `ExperimentalFeatures::enabled()`); console `/hw_shadows set true|false` switches (`Renderer::set_hw_shadows`). Every `ChunkMesh` gets a `blas` (vertex/index buffers need `BLAS_INPUT`); `HwRt::update` rebuilds the TLAS over all loaded chunks each frame; `HwRt::trace` casts one ray per texel. Don't put ray-query code into `shader.wgsl`: in the big scene shader it made the whole pass ~3× slower even unexecuted.
- **Ray marching** (fallback, `cs_march` in `shader.wgsl`, `RtBlur::march`): `src/rt_shadow.rs` builds a window of solid/air bits per cube face (whole faces up to `WINDOW_SIZE` = 256 columns per face, otherwise windows on the faces turned toward the player) plus 8×8-column max-height tiles, rebuilt in `Renderer::update_rt_window` when the player moves far, changes face, or a block is edited (`rt_dirty`). `rt_shadow()` walks those cells toward the sun: 3D DDA in block space on short segments (re-linearised for the planet's curvature), bisection across cube-face edges, and adaptive steps that double (up to `RT_MAX_STRIDE`) while a climbing ray passes above the max-height tiles.
- `RtParams`/`FaceWindow` in `rt_shadow.rs` must match the WGSL structs. The UI bind group uses `rt.enabled = 0` ("lit, don't read the shadow texture"); in hardware mode `update_rt_window` still writes `enabled = 1`.
- The debug overlay's GPU timer (`src/gpu_timer.rs`) shows geometry, rays (downsample + shadow compute), blur, lighting and text.

### Coordinates (`src/gen.rs`, `CoordSystem`)
A `BlockId` is `{face: 0..6, layer, u, v}` on a cube-sphere. `resolution` is the voxel count per face edge. Cube→sphere uses the Nowell mapping (`cube_to_sphere`), and `cubize_point` is its inverse. Layers are radially **exponential**, not linear: `get_layer_radius(layer, res) = (res/2) * exp(K*(layer/(res/2) - 1))` with `K = 0.85`, so voxels keep a roughly cubic shape at every depth. Convert between world positions and blocks only through `CoordSystem` (`pos_to_id`, `get_local_coords`, `get_block_center`, `get_vertex_pos`). Never hand-roll the math.

### Renderer owns streaming and LOD (`src/renderer.rs`)
`Renderer` is the largest module. Besides the frame passes (deferred geometry/lighting, shadows, overlays, glyphon text), it also owns world streaming:
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
