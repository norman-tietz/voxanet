# README screenshot (`resources/planet.png`)

How the showcase screenshot at the top of the README was arranged, and how to shoot it again after
engine changes.

## Re-shoot

```sh
docs/showcase/take-showcase.sh                 # overwrites resources/planet.png
docs/showcase/take-showcase.sh /tmp/try.png    # or write somewhere else to compare first
```

The script needs a clean `src/main.rs`, `src/renderer.rs` and `src/screenshot.rs`. It works like this:
1. Applies `showcase.patch` (throwaway capture hooks).
2. Builds the release binary and runs the game.
3. Captures at a fixed game time.
4. Removes the hooks again, even on failure.
5. Stores the shot as a compressed RGB PNG (needs Python with Pillow).

A window opens while the script runs (about 1–2 minutes); don't touch it.

## The arrangement

- **Galaxy:** `--seed 25` (Yellow star). The start planet is Earth-like #1. Volcanic #2 passes close to
  it at about 2.1° apparent radius around game time 1856 s. With the default seed 1 the volcanic
  planet only gets that close after about 54 minutes, and in early tries it was a tiny dot.
- **Time:** the shot is taken at game time `SHOT_T` = 1856.5 s. The game clock starts
  `SHOT_T - 60` s in (`VOXANET_CLOCK`), so the bake and streaming have a minute to settle. Orbits, the
  planet's spin, the sun direction and the clouds all follow that one clock, so the same time gives
  the same picture.
- **Renderer:** the planet engine (voxel world, fly mode) draws the planet, not a galaxy impostor.
  The camera sits 2.6 planet radii from the planet's centre, below the 3.5-radii liftoff handover.
- **Camera pose** (`VOXANET_SHOWCASE="2.6 86 25"`, solved every tick):
  - Find the view direction that puts the sun 86° and volcanic #2 25° from the view axis. If the
    current orbit positions can't satisfy that, the volcano angle is moved to the nearest one they
    can.
  - Place the camera on the opposite side, so the planet sits exactly in the screen centre.
  - Roll the camera so the sun lies toward the top-left corner. The star's corona is wider than
    the 60° corner of the 80° field of view, so only its edge shows.
  - Phase angle ≈ 98°, so the terminator crosses the disc: the day side faces the sun (upper left),
    the night side is on the right. The night side keeps about 12% sky light (`NIGHT_AMBIENT`), so
    it reads as dim rather than black.
- **Output:** 3840×2160. The hooks force a 4K render surface (the window itself can't be bigger
  than the display). They also skip the 1280-px downscale screenshots normally get, and hide the
  HUD text and the crosshair.

## Adjusting it

The knobs are at the top of `take-showcase.sh`:
- `POSE`: distance in radii (keep it below 3.5), sun angle, volcano angle.
- `SHOT_T`: the moment of the shot.
- `SEED`: the galaxy.

The hook prints `DBG showcase t=… ok=… volc_deg=… ang=…` every 2 s:
- `ok=false`: no pose was found at that moment.
- `volc_deg`: the volcano angle actually used.
- `ang`: volcano #2's apparent radius.

To pick a new seed or time, scan for moments when a volcanic planet appears large and lies 62–108°
from the sun, as seen from the Earth-like planet. That scan is how seed 25 and t ≈ 1856 s were found.

If the engine changes enough that `showcase.patch` no longer applies, recreate the hooks by hand and
regenerate the patch with `git diff > docs/showcase/showcase.patch`. The hooks are:
- the clock offset, after the bake in `Game::new`;
- the pose solver, after `controller.update_player` in planet mode;
- the 4K surface, in `Renderer::new` and `Renderer::resize`;
- no crosshair and no HUD text;
- no screenshot downscale.

All are gated on the `VOXANET_SHOWCASE` / `VOXANET_CLOCK` environment variables and marked
`DEBUG-HACK`.
