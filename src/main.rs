// engine main.rs

mod app;
mod meshing;
mod planet;
mod player;
mod render;
mod space;
mod weather;

use crate::app::cmd::Console;
use crate::app::system_diagnostics::SystemDiagnostics;
use crate::planet::data::PlanetData;
use crate::player::controller::Controller;
use crate::player::entity::Player;
use crate::render::renderer::Renderer;
use crate::space::solar_system::FlightFrame;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

// --biome <name> picks the starting planet type (default earth-like); matched case-insensitively
// against each PlanetType's def().name with spaces/hyphens stripped, so "Earth-like", "earthlike" and
// "EARTHLIKE" all match. Unrecognized values fall back to earth-like with a warning.
fn parse_biome_arg() -> crate::planet::biome::PlanetType {
    let Some(requested) = std::env::args().skip_while(|a| a != "--biome").nth(1) else {
        return crate::planet::biome::PlanetType::EarthLike;
    };
    crate::planet::biome::PlanetType::from_name(&requested).unwrap_or_else(|| {
        let names: Vec<_> = crate::planet::biome::PlanetType::ALL
            .iter()
            .map(|t| t.def().name)
            .collect();
        eprintln!(
            "Unknown --biome '{requested}', valid options: {}. Using Earth-like.",
            names.join(", ")
        );
        crate::planet::biome::PlanetType::EarthLike
    })
}

// --windowed: a normal window instead of the default borderless fullscreen (reproducible sizes for
// measurements and screenshots)
fn windowed(args: &[String]) -> bool {
    args.iter().skip(1).any(|a| a == "--windowed")
}

// --seed <n>: the solar system seed (default 1): sets the star type and the planets
fn parse_seed(args: &[String]) -> u64 {
    args.iter()
        .skip_while(|a| *a != "--seed")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

// the world-space distance from the planet center to just above the ground along `dir`, using the
// liquid-less-planet-aware effective height so this never spawns the player inside filled-in ocean
fn spawn_radius(planet: &PlanetData, dir: glam::Vec3, margin: f32) -> f32 {
    let res = planet.resolution;
    if let Some(id) = planet.cell_at(dir * (res as f32 / 2.0)) {
        crate::meshing::gen::CoordSystem::get_layer_radius(
            planet.effective_height(id.face, id.u, id.v),
            res,
        ) + margin
    } else {
        (res as f32 / 2.0) + 20.0
    }
}

fn main() {
    SystemDiagnostics::print_startup_info();
    let initial_biome = parse_biome_arg();
    let seed = parse_seed(&std::env::args().collect::<Vec<_>>());
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        game: None,
        initial_biome,
        seed,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    // created in `resumed`, once a window can be opened
    game: Option<Game>,
    initial_biome: crate::planet::biome::PlanetType, // from --biome, see parse_biome_arg()
    seed: u64, // the solar system seed, from --seed (parse_seed)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum GameMode {
    Planet,
    Space,
}

// Space mode has no Q/E key-yaw, so mouse-look must be active even if the player entered while in
// third person (Controller::raw_input gates it on first_person && !mouse_released). With the map open
// the cursor stays free for it and the captured state is what closing the map restores.
fn enter_space_controls(controller: &mut Controller, map: &mut Option<crate::space::map::MapView>) {
    controller.first_person = true;
    match map {
        Some(m) => m.restore_mouse_released = false,
        None => controller.mouse_released = false,
    }
}

// a planet baked on a background thread: its voxel world plus the near impostor built from it
struct BakedPlanet {
    system: usize, // the solar system it belongs to (Universe::current when the bake started)
    index: usize,
    data: PlanetData,
    near_verts: Vec<crate::planet::data::Vertex>,
    near_indices: Vec<u32>,
    impostor_generation: u64, // data.edit_generation() when the near impostor was built
}

// bakes a planet's voxel world and its near impostor: synchronously at start-up and for
// /system home, on a background thread on capture (start_bake)
fn bake_planet(
    system: &crate::space::solar_system::SolarSystem,
    system_index: usize,
    index: usize,
    edits: Option<crate::planet::data::PlanetEdits>,
) -> BakedPlanet {
    let data = system.planets[index].bake_with_edits(edits);
    let (near_verts, near_indices) = crate::space::impostor::near_impostor_mesh(&data);
    BakedPlanet {
        system: system_index,
        index,
        impostor_generation: data.edit_generation(),
        data,
        near_verts,
        near_indices,
    }
}

fn start_bake(
    system: &crate::space::solar_system::SolarSystem,
    system_index: usize,
    index: usize,
    edits: Option<crate::planet::data::PlanetEdits>,
) -> std::sync::mpsc::Receiver<BakedPlanet> {
    let (tx, rx) = std::sync::mpsc::channel();
    let target = system.planets[index];
    std::thread::spawn(move || {
        let data = target.bake_with_edits(edits);
        let (near_verts, near_indices) = crate::space::impostor::near_impostor_mesh(&data);
        let _ = tx.send(BakedPlanet {
            system: system_index,
            index,
            impostor_generation: data.edit_generation(),
            data,
            near_verts,
            near_indices,
        });
    });
    rx
}

// before planet `incoming` is installed, keeps the installed planet's edits for its next visit (nothing
// to keep at start-up, when re-installing the same planet, or when it's unedited)
fn stash_edits<K: std::hash::Hash + Eq + Copy>(
    store: &mut std::collections::HashMap<K, crate::planet::data::PlanetEdits>,
    installed: Option<K>,
    incoming: K,
    planet: &mut PlanetData,
) {
    let Some(installed) = installed.filter(|&i| i != incoming) else {
        return;
    };
    let edits = planet.take_edits();
    if !edits.is_empty() {
        store.insert(installed, edits);
    }
}

// whether the installed planet was edited since its near impostor was built
fn near_impostor_outdated(built_at: u64, planet: &PlanetData) -> bool {
    planet.edit_generation() != built_at
}

// a near impostor rebuilt in the background: (solar system, planet index, edit generation it shows, mesh)
type ImpostorRebuild = (
    usize,
    usize,
    u64,
    Vec<crate::planet::data::Vertex>,
    Vec<u32>,
);

// whether the voxel world `loaded` (system, planet) is planet `i` of the current solar system: a planet
// index alone names a different planet in every system
fn loaded_here(loaded: (usize, usize), current: usize, i: usize) -> bool {
    loaded == (current, i)
}

// makes a baked planet the voxel engine's world: palette, full mesh reload streaming from
// `stream_from` (planet frame), near impostor for the space renderer
fn install_world(
    renderer: &mut Renderer,
    controller: &mut Controller,
    planet: &mut PlanetData,
    baked: BakedPlanet,
    stream_from: glam::Vec3,
) {
    *planet = baked.data;
    // the terrain style's cells (/style hex) carry over to every planet installed
    planet.cells = crate::meshing::lowpoly::cells_for(crate::meshing::lowpoly::style());
    controller.selected_block =
        crate::planet::material::placeable(&planet.planet_type.def().palette)[0];
    renderer.force_reload_all(planet, stream_from);
    // the new planet's storms now, not at the first planet frame (a rebake in the handover frame popped)
    renderer.apply_storminess(planet.traits.storminess);
    renderer.set_near_impostor(baked.index, &baked.near_verts, &baked.near_indices);
    renderer.log_memory(planet);
}

// the block the player targets (selection box, mining, placing, picking): none while flying
fn cursor_target(
    flying: bool,
    hit: Option<crate::planet::data::BlockId>,
) -> Option<crate::planet::data::BlockId> {
    hit.filter(|_| !flying)
}

// the HUD status line for a refused edit; None: the block would overlap the player
fn edit_refused_message(why: Option<crate::planet::data::EditRefused>) -> &'static str {
    use crate::planet::data::EditRefused;
    match why {
        Some(EditRefused::BuildLimit) => "Build limit reached.",
        Some(EditRefused::MiningFloor) => "Bedrock: can't dig deeper.",
        Some(EditRefused::EditLimit) => "Too many edits on this planet.",
        None => "No room: you're in the way.",
    }
}

// where a player stands at local noon on planet `index` (whose voxel world is `planet`):
// the surface point facing the star, moved to dry land if that is lava
fn noon_spawn(
    planet: &PlanetData,
    system: &crate::space::solar_system::SolarSystem,
    index: usize,
    t: f64,
) -> glam::Vec3 {
    let noon = system.planets[index].sun_dir_in_planet_frame(system.star.position(), t);
    let dir = planet.safe_spawn_direction(noon);
    dir * spawn_radius(planet, dir, 10.0)
}

// a finished bake is installed unless the player has since gone home, landed (both leave system
// mode) or been captured by another planet; having merely left orbit again keeps it (spec §1:
// re-capturing the same planet doesn't re-bake). Capturing another planet also replaces bake_rx,
// so an older bake can't arrive after that.
fn bake_still_wanted(
    mode: GameMode,
    frame: FlightFrame,
    index: usize,
    bake_system: usize,
    current_system: usize,
) -> bool {
    bake_system == current_system
        && mode == GameMode::Space
        && match frame {
            FlightFrame::Free => true,
            FlightFrame::Captured(i) => i == index,
        }
}

struct Game {
    renderer: Renderer,
    controller: Controller,
    player: Player,
    planet: PlanetData,
    console: Console,
    last_time: Instant,
    current_cursor_locked: bool,
    mode: GameMode,
    system: crate::space::solar_system::SolarSystem,
    space_flight: crate::space::solar_system::SpaceFlight,
    flight_frame: FlightFrame, // space, or captured by a planet (solar_system.rs)
    bake_rx: Option<std::sync::mpsc::Receiver<BakedPlanet>>, // the running background bake, if any
    baking: Option<usize>,     // which planet it is for
    voxel_ready_announced: bool, // "voxel world ready" printed for the loaded planet
    clock: GameClock,          // shared game clock: system orbits and planet day/night
    bench: Option<Bench>,      // a --bench run in progress (perf.rs)
    pub quit: bool,            // the bench run is done: about_to_wait exits
    loaded_world: usize,       // the planet the voxel engine holds (always one)
    loaded_system: usize, // ... and the solar system it belongs to (a jump leaves it installed)
    universe: crate::space::universe::Universe, // the galaxy and the systems visited (Game.system: the current one)
    jump: Option<Jump>, // a jump to another solar system in progress (fade, switch, fade)
    start_planet: usize, // where the game starts and /system home returns to
    // each planet's edits while another planet is installed (stash_edits), handed to its next bake
    edits_by_planet: std::collections::HashMap<(usize, usize), crate::planet::data::PlanetEdits>, // (system, planet)
    impostor_generation: u64, // the installed planet's edit_generation when its near impostor was built
    impostor_rx: Option<std::sync::mpsc::Receiver<ImpostorRebuild>>, // a near impostor rebuild in flight
    map: Option<crate::space::map::MapView>, // the solar-system map, while it's open (M)
    galaxy_view: Option<crate::space::map::MapView>, // the galaxy view, while it's open (G)
    travel: Option<Travel>, // an automatic flight to a planet's orbit (space/travel.rs), while it runs
    trip_mouse: TripMouse,  // mouse motion during a trip, toward cancelling it
    visited: std::collections::HashSet<(usize, usize)>, // (system, planet) whose voxel world has been installed
    landed_at: Option<Instant>, // the last landing handover, for its cross-fade (landing::handover_overlay_opacity)
}

// a jump to another solar system as a light-speed flight (spec docs/superpowers/specs/2026-10-12-warp-jump-design.md,
// timeline in space/warp.rs): from the surface the travel climb first, then in space: align toward the
// target, accelerate into streaks, switch unseen at light speed, brake into the new system. /system home
// from another system ends with the start planet installed under a short black fade.
const HOME_FADE_SECS: f32 = 0.4;

enum JumpStage {
    Climb, // planet mode: climbing to the liftoff handover; in space: about to start the flight
    Fly {
        started: Instant,
        duration: f32,
        prev_frac: f32,
        initial_fwd: glam::Vec3, // where the view looked when the flight started (the align turn)
        initial_velocity: glam::DVec3, // the ship's velocity then, eased out over the align phase
    },
    HomeFadeIn(Instant), // /system home: the start planet installed, fading in from black
}

struct Jump {
    target: usize,
    then_home: bool, // /system home from another system: home runs at the end of the flight
    dir: glam::Vec3, // galaxy direction from the old system to the target (galaxy axes = system axes)
    switched: bool,
    stage: JumpStage,
}

impl Jump {
    fn new(target: usize, then_home: bool, dir: glam::Vec3) -> Self {
        Self {
            target,
            then_home,
            dir,
            switched: false,
            stage: JumpStage::Climb,
        }
    }

    // `/system home` from another system while this jump runs: one still on its way to the start system
    // goes home on arrival (true); any other is left alone (false: "Jump in progress.")
    fn queue_home(&mut self) -> bool {
        let goes_home = !self.switched && self.target == 0;
        if goes_home {
            self.then_home = true;
        }
        goes_home
    }

    // the climb or the flight: the jump owns space flight and blocks the player's input
    fn flies(&self) -> bool {
        !matches!(self.stage, JumpStage::HomeFadeIn(_))
    }
}

// the galaxy direction from the current system to `target` (the jump's travel direction)
fn jump_direction(universe: &crate::space::universe::Universe, target: usize) -> glam::Vec3 {
    let from = universe.galaxy.systems[universe.current].position;
    let to = universe.galaxy.systems[target].position;
    (to - from).normalize_or(glam::DVec3::NEG_Z).as_vec3()
}

// the black fade after /system home's bake at the end of a jump, `elapsed` seconds in
fn home_fade(elapsed: f32) -> f32 {
    (1.0 - elapsed / HOME_FADE_SECS).clamp(0.0, 1.0)
}

// the jump owns the controls (no movement, Esc, M, G) while it climbs or flies
fn jump_blocks_input(jump: &Option<Jump>) -> bool {
    jump.as_ref().is_some_and(|j| j.flies())
}

// the console commands that move the player or open an overlay, which a jump has to refuse
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JumpCommand {
    Home, // /system home
    Goto, // /system goto
    Map,
    Galaxy,
    Travel,
    Jump,
}

// whether `command` must wait for the running jump ("Jump in progress."): while it climbs or flies all of
// them, except /system home from another system, which Jump::queue_home decides (`home_elsewhere`:
// universe.current != 0); during home's fade-in the overlays work again (the controls are back), but
// nothing may move the player off the start planet the fade is revealing
fn refused_during_jump(jump: Option<&Jump>, command: JumpCommand, home_elsewhere: bool) -> bool {
    let Some(j) = jump else {
        return false;
    };
    match command {
        JumpCommand::Home => !(home_elsewhere && j.flies()),
        JumpCommand::Goto | JumpCommand::Travel | JumpCommand::Jump => true,
        JumpCommand::Map | JumpCommand::Galaxy => j.flies(),
    }
}

// a console command refused while a jump runs: said in the console (where it was typed; Space mode's HUD
// has no status line) and in planet mode's status line
fn say_jump_in_progress(renderer: &mut Renderer, console: &mut Console) {
    renderer.show_status("Jump in progress.");
    console.log("Jump in progress.", [1.0, 0.5, 0.0]);
}

// the window events a jump keeps from the controls while it climbs or flies: key presses and mouse
// clicks; releases pass, so a key held into the jump doesn't stay down after it
fn jump_swallows(event: &WindowEvent) -> bool {
    match event {
        WindowEvent::KeyboardInput { event, .. } => event.state == ElementState::Pressed,
        WindowEvent::MouseInput { state, .. } => *state == ElementState::Pressed,
        _ => false,
    }
}

// an automatic flight to a planet's orbit (space/travel.rs): climbing from the surface (planet mode), or
// about to plan the cruise (the first space frame), then the cruise itself
enum Travel {
    Climb { target: usize },
    Cruise(crate::space::travel::Trip),
}

// which space-mode arm of Game::tick runs: while the autopilot flies it owns the flight (no capture or
// release by the planets it passes, no heat push, no manual flight); only the manual arm calls
// next_flight_frame
#[derive(Clone, Copy, PartialEq, Debug)]
enum SpaceArm {
    Autopilot,
    Manual,
}

fn space_arm(travel: &Option<Travel>) -> SpaceArm {
    match travel {
        Some(_) => SpaceArm::Autopilot,
        None => SpaceArm::Manual,
    }
}

// the mouse turns the view only in first person with the cursor captured, the console and map closed
fn mouse_turns_view(
    first_person: bool,
    released: bool,
    console_open: bool,
    map_open: bool,
) -> bool {
    first_person && !released && !console_open && !map_open
}

const TRIP_MOUSE_GRACE: f32 = 0.4; // seconds after a trip starts or the mouse comes back before it counts
const TRIP_MOUSE_CANCEL: f32 = 40.0; // pixels of turning, summed, that take over a running trip

// the mouse motion that would turn the view during a trip, summed since it last started counting
struct TripMouse {
    since: Instant, // counting starts TRIP_MOUSE_GRACE after this
    sum: f32,
}

impl TripMouse {
    fn new(now: Instant) -> Self {
        Self {
            since: now,
            sum: 0.0,
        }
    }

    // adds a motion; true once the sum exceeds the cancel threshold
    fn add(&mut self, delta: (f64, f64), now: Instant) -> bool {
        if now.duration_since(self.since).as_secs_f32() < TRIP_MOUSE_GRACE {
            return false;
        }
        self.sum += delta.0.hypot(delta.1) as f32;
        self.sum > TRIP_MOUSE_CANCEL
    }
}

// 1..9 from the number row or the keypad
fn digit_key(code: KeyCode) -> Option<usize> {
    use KeyCode::*;
    [
        Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9,
    ]
    .iter()
    .position(|&c| c == code)
    .or_else(|| {
        [
            Numpad1, Numpad2, Numpad3, Numpad4, Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
        ]
        .iter()
        .position(|&c| c == code)
    })
    .map(|i| i + 1)
}

// opens the solar-system map: the keyboard and mouse go to it (no movement, the cursor freed)
fn open_map(controller: &mut Controller) -> crate::space::map::MapView {
    let view = crate::space::map::MapView::new(
        controller.mouse_released,
        crate::space::map::MapCamera::default_view(),
    );
    controller.release_keys();
    controller.mouse_released = true;
    view
}

// opens the galaxy view (space/galaxy.rs): like the map, with the galaxy camera
fn open_galaxy(controller: &mut Controller) -> crate::space::map::MapView {
    let view = crate::space::map::MapView::new(
        controller.mouse_released,
        crate::space::map::MapCamera::galaxy_view(),
    );
    controller.release_keys();
    controller.mouse_released = true;
    view
}

// closes the map or the galaxy view (both are MapViews), restoring the mouse
fn close_map(map: &mut Option<crate::space::map::MapView>, controller: &mut Controller) {
    if let Some(m) = map.take() {
        controller.mouse_released = m.restore_mouse_released;
    }
}

// the shared game clock: seconds since the start planet was baked, or frozen at 0 for a --bench run
struct GameClock {
    start: Instant,
    frozen: bool,
}

impl GameClock {
    fn new(frozen: bool) -> Self {
        Self {
            start: Instant::now(),
            frozen,
        }
    }

    fn now(&self) -> f64 {
        if self.frozen {
            0.0
        } else {
            self.start.elapsed().as_secs_f64()
        }
    }
}

// a --bench run: wait until the world is streamed in (+2 s), run the --exec commands, wait 1 s,
// record (perf.rs), quit
struct Bench {
    args: crate::app::perf::BenchArgs,
    stage: BenchStage,
}

enum BenchStage {
    Loading(Option<Instant>), // since when the view has been covered
    Settling(Instant),        // commands run at
    Recording,
}

impl Game {
    // the player takes over a running trip: from the cruise into free flight at boost speed at most, from
    // the climb into hovering fly mode
    fn cancel_travel(&mut self) {
        if let Some(Travel::Cruise(trip)) = self.travel.take() {
            let t = self.clock.now();
            self.space_flight.velocity = crate::space::travel::cancel_velocity(trip.velocity(t));
            self.space_flight.roll_rate = 0.0;
            self.space_flight.turn_rate = 0.0;
        }
        self.renderer.travel_hud = None;
        self.renderer.show_status("Travel cancelled.");
    }

    fn new(
        window: Arc<Window>,
        initial_biome: crate::planet::biome::PlanetType,
        seed: u64,
    ) -> Self {
        let mut renderer = pollster::block_on(Renderer::new(window));
        let mut controller = Controller::new();
        let mut player = Player::new();
        // --seed picks the galaxy; its start system (entry 0) has the galaxy seed, so it is the solar
        // system this seed always gave
        let universe =
            crate::space::universe::Universe::new(crate::space::galaxy::Galaxy::generate(seed));
        let system =
            crate::space::solar_system::SolarSystem::generate(universe.galaxy.systems[0].seed);
        println!(
            "Galaxy seed {seed}: start system {}, {} star",
            universe.galaxy.systems[0].name,
            system.star.star_type.def().name
        );
        let start_planet = system.start_planet_index(initial_biome);

        // every planet lives in the solar system: start on one of them, standing at local noon
        let mut planet = PlanetData::new(8); // placeholder, replaced right below
        let baked = bake_planet(&system, 0, start_planet, None);
        let impostor_generation = baked.impostor_generation;
        // the clock starts once the bake is done, so t = 0 really is now and the planet hasn't
        // turned on past noon while baking
        // a --bench run freezes it at t = 0 (noon, no day cycle, clouds and weather standing still)
        let bench = crate::app::perf::parse_bench_args(&std::env::args().collect::<Vec<_>>());
        let clock = GameClock::new(bench.is_some());
        let spawn = noon_spawn(&baked.data, &system, start_planet, 0.0);
        install_world(&mut renderer, &mut controller, &mut planet, baked, spawn);
        player.spawn(spawn);
        player.handover_altitude = Some(crate::player::landing::handover_altitude(
            system.planets[start_planet].radius,
        ));
        player.spin_rate = system.planets[start_planet].spin_rate() as f32;

        // fixed comparison poses (screenshots of terrain styles, docs/showcase-like captures): lift the
        // spawn into fly mode VOXANET_POSE_ALT units up, pitched VOXANET_POSE_PITCH radians
        if let Some(alt) = std::env::var("VOXANET_POSE_ALT")
            .ok()
            .and_then(|a| a.parse::<f32>().ok())
        {
            player.position += player.position.normalize() * alt;
            controller.fly_mode = true;
            player.cam_pitch = std::env::var("VOXANET_POSE_PITCH")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(-0.4);
        }

        let mut console = Console::new();
        console.log("Welcome to voxanet.", [0.0, 1.0, 0.0]);
        console.log(
            "Press the key left of 1 (` or ^) to open the console.",
            [1.0, 1.0, 1.0],
        );
        console.log(
            &format!(
                "Starting on #{} {}.",
                start_planet + 1,
                planet.planet_type.def().name
            ),
            [1.0, 1.0, 1.0],
        );

        Self {
            renderer,
            controller,
            player,
            planet,
            console,
            last_time: Instant::now(),
            current_cursor_locked: false,
            mode: GameMode::Planet,
            system,
            space_flight: crate::space::solar_system::SpaceFlight::new(glam::DVec3::new(
                0.0, 0.0, 120_000.0,
            )),
            flight_frame: FlightFrame::Free,
            bake_rx: None,
            baking: None,
            voxel_ready_announced: false,
            map: None,
            galaxy_view: None,
            travel: None,
            trip_mouse: TripMouse::new(Instant::now()),
            visited: std::collections::HashSet::new(),
            landed_at: None,
            clock,
            bench: bench.map(|args| Bench {
                args,
                stage: BenchStage::Loading(None),
            }),
            quit: false,
            loaded_world: start_planet,
            loaded_system: 0,
            universe,
            jump: None,
            start_planet,
            edits_by_planet: std::collections::HashMap::new(),
            impostor_generation,
            impostor_rx: None,
        } // mismatched on purpose, so tick()'s first diff locks the cursor
    }

    // runs before every event is handled
    fn tick(&mut self) {
        let Self {
            renderer,
            controller,
            player,
            planet,
            console,
            last_time,
            current_cursor_locked,
            mode,
            system,
            space_flight,
            flight_frame,
            bake_rx,
            baking,
            voxel_ready_announced,
            landed_at,
            map,
            galaxy_view,
            visited,
            travel,
            trip_mouse,
            clock,
            loaded_world,
            loaded_system,
            universe,
            jump,
            start_planet,
            edits_by_planet,
            impostor_generation,
            impostor_rx,
            bench,
            quit,
        } = self;

        let now = Instant::now();
        let dt = (now - *last_time).as_secs_f32();
        *last_time = now;

        // cursor locking logic: locked only in first person, and only while the mouse hasn't been
        // explicitly released (Escape) without leaving first person
        let want_locked = controller.first_person
            && !controller.mouse_released
            && map.is_none()
            && galaxy_view.is_none();
        if want_locked != *current_cursor_locked {
            *current_cursor_locked = want_locked;
            if *current_cursor_locked {
                let _ = renderer.window.set_cursor_grab(CursorGrabMode::Locked);
                renderer.window.set_cursor_visible(false);
            } else {
                let _ = renderer.window.set_cursor_grab(CursorGrabMode::None);
                renderer.window.set_cursor_visible(true);
            }
        }

        let mut land_on: Option<usize> = None; // landing handover this tick (planet index)
        let mut lift_off_from: Option<usize> = None; // liftoff handover this tick

        // a jump's climb or flight turns the view itself: the mouse turns nothing
        if jump_blocks_input(jump) {
            controller.mouse_delta = (0.0, 0.0);
        }
        // during a trip the mouse turns nothing: its motion only counts toward cancelling (device_event),
        // and the count restarts (with its grace period) whenever the mouse can't turn the view
        if travel.is_some() {
            controller.mouse_delta = (0.0, 0.0);
            if !mouse_turns_view(
                controller.first_person,
                controller.mouse_released,
                console.is_open,
                map.is_some() || galaxy_view.is_some(),
            ) {
                *trip_mouse = TripMouse::new(now);
            }
        }

        // physics & player Update (once per tick: movement and turning speeds are per second)
        match mode {
            GameMode::Planet => {
                let climbing = matches!(travel, Some(Travel::Climb { .. }))
                    || matches!(
                        jump,
                        Some(Jump {
                            stage: JumpStage::Climb,
                            ..
                        })
                    );
                if climbing {
                    // the climb moves the player itself: update_player mustn't integrate it again
                    player.velocity = glam::Vec3::ZERO;
                }
                controller.update_player(player, planet, dt);
                // the trip's climb: straight up, faster with altitude, until the liftoff handover below;
                // the velocity is set for the handover and the cruise's start direction
                if climbing {
                    let up = player.position.normalize_or_zero();
                    let altitude = player.position.length() - planet.resolution as f32 / 2.0;
                    let speed = crate::space::travel::climb_speed(altitude);
                    player.position += up * speed * dt;
                    player.velocity = up * speed;
                }

                // raycast & cursor Update
                let width = renderer.config.width as f32;
                let height = renderer.config.height as f32;
                let ray_result = controller.raycast(player, planet, width, height, false);
                controller.cursor_id =
                    cursor_target(controller.fly_mode, ray_result.map(|(id, _)| id));

                renderer.update_cursor(planet, controller.cursor_id);
                renderer.update_view(player.position, planet);

                // high enough above the planet: hand back to captured space flight
                if crate::player::landing::must_lift_off(
                    system,
                    *loaded_world,
                    player.position.as_dvec3(),
                    clock.now(),
                ) {
                    lift_off_from = Some(*loaded_world);
                }
            }
            GameMode::Space if jump_blocks_input(jump) => {
                // a jump's flight (space/warp.rs): no manual flight, no capture or heat push; the view turns
                // toward the target, the streaks speed up and down, the switch comes at light speed
                let j = jump.as_mut().expect("a jump flies");
                if let JumpStage::Climb = j.stage {
                    // the first space frame: fly from here, in space axes (the galaxy direction holds there)
                    let t = clock.now();
                    if let FlightFrame::Captured(_) = *flight_frame {
                        space_flight.change_frame(*flight_frame, FlightFrame::Free, system, t);
                        *flight_frame = FlightFrame::Free;
                    }
                    space_flight.roll_rate = 0.0;
                    space_flight.turn_rate = 0.0;
                    let ly = universe.galaxy.distance_ly(universe.current, j.target);
                    let duration = crate::space::warp::jump_duration(ly);
                    println!(
                        "Jumping to {} ({ly:.1} ly, {duration:.1} s)",
                        universe.galaxy.systems[j.target].name
                    );
                    j.stage = JumpStage::Fly {
                        started: now,
                        duration,
                        prev_frac: 0.0,
                        initial_fwd: space_flight.rotation * glam::Vec3::NEG_Z,
                        initial_velocity: space_flight.velocity,
                    };
                }
                if let JumpStage::Fly {
                    started,
                    duration,
                    prev_frac,
                    initial_fwd,
                    initial_velocity,
                } = &mut j.stage
                {
                    use crate::space::warp;
                    let frac = started.elapsed().as_secs_f32() / *duration;
                    // the align turn: from where the view looked to the travel direction over the align
                    // phase, incremental so the roll stays
                    let k = {
                        let x = (frac / warp::ALIGN).clamp(0.0, 1.0);
                        x * x * (3.0 - 2.0 * x)
                    };
                    let desired = glam::Quat::IDENTITY
                        .slerp(glam::Quat::from_rotation_arc(*initial_fwd, j.dir), k)
                        * *initial_fwd;
                    let cur = space_flight.rotation * glam::Vec3::NEG_Z;
                    space_flight.rotation = (glam::Quat::from_rotation_arc(cur, desired)
                        * space_flight.rotation)
                        .normalize();
                    // the ship coasts to a stop over the align phase instead of halting in one frame
                    // (a liftoff climb hands over its speed)
                    space_flight.velocity = *initial_velocity * (1.0 - k) as f64;
                    space_flight.position += space_flight.velocity * dt as f64;
                    if warp::crosses_switch(*prev_frac, frac) {
                        j.switched = true; // the switch block below parks and takes the systems
                    }
                    *prev_frac = frac;
                    renderer.warp = Some((warp::speed(frac), warp::scene_fade(frac), j.dir));
                    let name = &universe.galaxy.systems[j.target].name;
                    renderer.travel_hud = Some(format!("Jumping to {name}…"));
                }
            }
            GameMode::Space if space_arm(travel) == SpaceArm::Autopilot => {
                let t = clock.now();
                // the first space frame of a trip: plan the cruise from here, in space (Free frame), and
                // bake the target's voxel world meanwhile
                if let Some(Travel::Climb { target }) = *travel {
                    if let FlightFrame::Captured(_) = *flight_frame {
                        space_flight.change_frame(*flight_frame, FlightFrame::Free, system, t);
                        *flight_frame = FlightFrame::Free;
                    }
                    let fwd = (space_flight.rotation * glam::Vec3::NEG_Z).as_dvec3();
                    let dir = space_flight.velocity.try_normalize().unwrap_or(fwd);
                    let trip = crate::space::travel::Trip::plan(
                        system,
                        target,
                        space_flight.position,
                        dir,
                        fwd,
                        t,
                    );
                    *travel = Some(Travel::Cruise(trip));
                    let here =
                        loaded_here((*loaded_system, *loaded_world), universe.current, target);
                    if !here && *baking != Some(target) {
                        *bake_rx = Some(start_bake(
                            system,
                            universe.current,
                            target,
                            edits_by_planet.get(&(universe.current, target)).cloned(),
                        ));
                        *baking = Some(target);
                    }
                }
                if let Some(Travel::Cruise(trip)) = travel {
                    let n = trip.target;
                    let name = system.planets[n].planet_type.def().name;
                    if trip.done(t) {
                        // in the target's orbit, at rest relative to it (the trip ends co-moving),
                        // looking at its centre with the roll it flew with: the view turned into the
                        // planet's frame
                        let view = trip.view_rotation(system, t, space_flight.rotation);
                        *flight_frame = FlightFrame::Captured(n);
                        space_flight.position = trip.arrival_local;
                        space_flight.velocity = glam::DVec3::ZERO;
                        space_flight.rotation = system.planets[n].rotation_to_planet_frame(view, t);
                        space_flight.roll_rate = 0.0;
                        space_flight.turn_rate = 0.0;
                        renderer.travel_hud = None;
                        renderer.show_status(&format!("Arrived at #{} {name}.", n + 1));
                        *travel = None;
                    } else {
                        space_flight.position = trip.position(t);
                        space_flight.velocity = trip.velocity(t);
                        space_flight.rotation =
                            trip.view_rotation(system, t, space_flight.rotation);
                        renderer.travel_hud = Some(format!(
                            "-> #{} {name}  {:.0} s",
                            n + 1,
                            trip.remaining(t).ceil()
                        ));
                    }
                }
            }
            GameMode::Space => {
                debug_assert_eq!(space_arm(travel), SpaceArm::Manual);
                let (_, jump, down, sprint, mouse_delta) = controller.raw_input();
                let (input, keys) = controller.space_flight_input();
                // turning and climbing follow space's +Y in free flight, the local radial up
                // while captured (the planet's spin axis would pitch the view at its equator)
                let up = match *flight_frame {
                    FlightFrame::Free => None,
                    FlightFrame::Captured(_) => space_flight.position.as_vec3().try_normalize(),
                };
                space_flight.update(dt, input, jump, down, mouse_delta, sprint, keys, up);
                // captured flight turns with the planet only close to it (solar_system::co_rotation)
                if let FlightFrame::Captured(i) = *flight_frame {
                    let p = &system.planets[i];
                    let distance = (space_flight.position.length() / p.radius as f64) as f32;
                    let q = crate::space::solar_system::counter_spin(
                        p.spin_rate() as f32,
                        distance,
                        dt,
                    )
                    .as_dquat();
                    space_flight.position = q * space_flight.position;
                    space_flight.velocity = q * space_flight.velocity;
                    space_flight.rotation = (q.as_quat() * space_flight.rotation).normalize();
                }
                // the star's heat zone pushes free flight back out and can't be passed (solar_system.rs);
                // planets orbit outside it, so captured flight never meets it
                if *flight_frame == FlightFrame::Free {
                    let star = system.star.position();
                    let r = system.star.radius;
                    space_flight.velocity = crate::space::solar_system::star_heat_push(
                        space_flight.velocity,
                        space_flight.position - star,
                        r,
                        dt,
                    );
                    let kept =
                        crate::space::solar_system::keep_off_star(space_flight.position - star, r)
                            + star;
                    if kept != space_flight.position {
                        // at the limit: no further inward speed
                        let out = (kept - star).normalize();
                        let inward = space_flight.velocity.dot(out).min(0.0);
                        space_flight.velocity -= out * inward;
                        space_flight.position = kept;
                    }
                }
                let t = clock.now();
                // capture by / release from a planet's frame (solar_system.rs, CAPTURE_RADII)
                let next = crate::space::solar_system::next_flight_frame(
                    system,
                    *flight_frame,
                    space_flight.position,
                    t,
                );
                if next != *flight_frame {
                    space_flight.change_frame(*flight_frame, next, system, t);
                    match next {
                        FlightFrame::Captured(i) => println!("Entering the orbit of #{}", i + 1),
                        FlightFrame::Free => println!("Leaving orbit"),
                    }
                    *flight_frame = next;
                }
                if let FlightFrame::Captured(i) = *flight_frame {
                    // the impostor is all there is until the voxel world is ready: don't sink into it
                    space_flight.position = crate::player::landing::keep_above_planet(
                        space_flight.position,
                        system.planets[i].radius as f64,
                    );
                    // capture bakes the planet's voxel world in the background, unless it's loaded
                    // (the loaded world counts only if it belongs to this solar system)
                    let here = loaded_here((*loaded_system, *loaded_world), universe.current, i);
                    if !here && *baking != Some(i) {
                        // the stored edits stay in the store until this bake is installed, so a
                        // dropped bake loses nothing
                        *bake_rx = Some(start_bake(
                            system,
                            universe.current,
                            i,
                            edits_by_planet.get(&(universe.current, i)).cloned(),
                        ));
                        *baking = Some(i);
                    }
                    // once loaded, the voxel engine streams from the flight's planet-frame position,
                    // so its meshes are ready by the time the landing handover needs them
                    if here {
                        renderer.update_view(space_flight.position.as_vec3(), planet);
                        if !*voxel_ready_announced && renderer.view_covered() {
                            println!("Voxel world for #{} ready", i + 1);
                            *voxel_ready_announced = true;
                        }
                        if crate::player::landing::may_land_on(
                            system,
                            i,
                            space_flight.position,
                            t,
                            renderer.view_covered(),
                        ) {
                            land_on = Some(i);
                        }
                    }
                }
            }
        }

        if let Some(i) = land_on {
            // space flight -> voxel engine: same eye, same view direction and roll, in fly mode
            let (position, rotation, cam_pitch, cam_roll) =
                crate::player::landing::player_pose_from_flight(
                    space_flight.position.as_vec3(),
                    space_flight.rotation,
                );
            player.position = position;
            player.rotation = rotation;
            player.cam_pitch = cam_pitch;
            player.cam_roll = cam_roll;
            // both sides live in the planet frame: the flight carries on at its speed
            player.velocity = space_flight.velocity.as_vec3();
            player.roll_rate = space_flight.roll_rate; // a roll in progress carries on
            player.landing = false;
            player.taking_off = None;
            player.handover_altitude = Some(crate::player::landing::handover_altitude(
                system.planets[i].radius,
            ));
            player.spin_rate = system.planets[i].spin_rate() as f32;
            // dying here respawns on dry land below, not back in orbit
            let below = planet.safe_spawn_direction(position.normalize());
            player.spawn_point = below * spawn_radius(planet, below, 10.0);
            controller.fly_mode = true;
            controller.first_person = true;
            *mode = GameMode::Planet;
            *landed_at = Some(now);
            println!("Landing handover onto #{}", i + 1);
        }
        if let Some(i) = lift_off_from {
            // voxel engine -> captured space flight: same eye, same view direction and roll, same velocity
            *space_flight = crate::player::landing::liftoff_flight(
                player.position,
                player.rotation,
                player.cam_pitch,
                player.cam_roll,
                player.velocity,
            );
            space_flight.roll_rate = player.roll_rate; // a roll in progress carries on
            *flight_frame = FlightFrame::Captured(i);
            player.landing = false;
            *mode = GameMode::Space;
            enter_space_controls(controller, map);
            println!("Liftoff into the orbit of #{}", i + 1);
        }

        // a background bake finished: install it unless the player went home, landed or got
        // captured by another planet meanwhile
        let finished = bake_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(baked) = finished {
            *bake_rx = None;
            *baking = None;
            if bake_still_wanted(
                *mode,
                *flight_frame,
                baked.index,
                baked.system,
                universe.current,
            ) {
                // the engine streams in the planet frame; free flight is in space
                let local = match *flight_frame {
                    FlightFrame::Captured(_) => space_flight.position,
                    FlightFrame::Free => system.planets[baked.index]
                        .to_planet_frame(space_flight.position, clock.now()),
                };
                let index = baked.index;
                let generation = baked.impostor_generation;
                // the installed planet keeps its edits for its next visit; the new one has its own now
                let key = (baked.system, index);
                stash_edits(
                    edits_by_planet,
                    Some((*loaded_system, *loaded_world)),
                    key,
                    planet,
                );
                edits_by_planet.remove(&key);
                install_world(renderer, controller, planet, baked, local.as_vec3());
                *loaded_world = index;
                *loaded_system = key.0;
                *impostor_generation = generation;
                *voxel_ready_announced = false;
                println!("Baked #{} (res {})", index + 1, planet.resolution);
            }
        }

        // the solar system shows the installed planet's near impostor: rebuild it in the background when the
        // planet was edited since it was built (after liftoff, or leaving by /system goto)
        if *mode == GameMode::Space
            && *loaded_system == universe.current
            && impostor_rx.is_none()
            && near_impostor_outdated(*impostor_generation, planet)
        {
            let (tx, rx) = std::sync::mpsc::channel();
            let (system_index, index, generation, data) = (
                *loaded_system,
                *loaded_world,
                planet.edit_generation(),
                planet.clone(),
            );
            std::thread::spawn(move || {
                let (v, i) = crate::space::impostor::near_impostor_mesh(&data);
                let _ = tx.send((system_index, index, generation, v, i));
            });
            *impostor_rx = Some(rx);
        }
        let rebuilt = impostor_rx.as_ref().map(|rx| rx.try_recv());
        if let Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) = rebuilt {
            *impostor_rx = None; // the rebuild thread died: allow a new one
        }
        if let Some(Ok((built_system, index, generation, v, i))) = rebuilt {
            *impostor_rx = None;
            // dropped if another planet (the same index in another system too) was installed meanwhile,
            // or the player is in another system
            if built_system == universe.current
                && loaded_here((*loaded_system, *loaded_world), built_system, index)
            {
                renderer.set_near_impostor(index, &v, &i);
                *impostor_generation = generation;
            }
        }

        // UPDATE ANIMATION
        console.update_animation(dt);

        if let Some(path) = console.screenshot_request.take() {
            renderer.request_screenshot(path);
        }
        if let Some(request) = console.view_request.take() {
            if let Some(fp) = request {
                controller.first_person = fp;
                controller.cam_dist = if fp { 40.0 } else { 100.0 };
            }
            let view = if controller.first_person {
                "first person"
            } else {
                "third person"
            };
            console.log(&format!("View: {view}"), [0.0, 1.0, 0.0]);
        }
        // the switch, decided by the flight in the space-mode arm (JumpStage::Fly): park the current system,
        // take the target (generated on its first visit), arrive at its edge; the voxel world stays installed
        // with its edits until a planet here is baked. Unseen: at light speed, the scene faded out.
        if let Some(target) = jump
            .as_ref()
            .filter(|j| j.switched && j.flies())
            .map(|j| j.target)
        {
            if universe.current != target {
                let from = universe.galaxy.systems[universe.current].position;
                let to = universe.galaxy.systems[target].position;
                let placeholder = crate::space::solar_system::SolarSystem::generate(0);
                let old = std::mem::replace(system, placeholder);
                *system = match universe.switch(target, old) {
                    Ok(next) => next,
                    Err((back, _)) => back,
                };
                renderer.clear_system_caches();
                if *loaded_system == universe.current {
                    // back where the voxel world is installed: rebuild its near impostor
                    *impostor_generation = u64::MAX;
                }
                *bake_rx = None;
                *baking = None;
                // the arrival point; the view keeps its aligned rotation (already looking along the
                // travel direction, at the new star), so the streaks don't turn at the switch
                let (pos, _) = crate::space::universe::Universe::arrival_pose(
                    system,
                    (from - to).normalize_or(glam::DVec3::Z),
                );
                space_flight.position = pos;
                space_flight.velocity = glam::DVec3::ZERO;
                *flight_frame = FlightFrame::Free;
                let entry = &universe.galaxy.systems[universe.current];
                console.log(
                    &format!(
                        "Arrived in {} ({} star).",
                        entry.name,
                        system.star.star_type.def().name
                    ),
                    [1.0, 1.0, 1.0],
                );
            }
        }
        // the end of a flight: the controls back, /system home's step, or done
        let mut home_after_jump = false; // the system request below is the jump's own /system home
        if let Some(j) = jump.as_mut() {
            match &mut j.stage {
                JumpStage::Fly {
                    started, duration, ..
                } if crate::space::warp::phase(started.elapsed().as_secs_f32() / *duration)
                    == crate::space::warp::Phase::Done =>
                {
                    renderer.warp = None;
                    renderer.travel_hud = None;
                    controller.fly_mode = false;
                    // "mode Space, map closed" (overlays were blocked during the flight anyway)
                    close_map(map, controller);
                    close_map(galaxy_view, controller);
                    enter_space_controls(controller, map);
                    if j.then_home {
                        renderer.set_fade(1.0);
                        console.system_request = Some(crate::app::cmd::SystemRequest::Home);
                        home_after_jump = true;
                        j.stage = JumpStage::HomeFadeIn(Instant::now());
                    } else {
                        *jump = None;
                    }
                }
                JumpStage::HomeFadeIn(start) => {
                    let e = start.elapsed().as_secs_f32();
                    renderer.set_fade(home_fade(e));
                    if e >= HOME_FADE_SECS {
                        renderer.set_fade(0.0);
                        *jump = None;
                    }
                }
                _ => {}
            }
        }
        if let Some(request) = console.system_request.take() {
            use crate::app::cmd::SystemRequest;
            let say = |console: &mut Console, text: &str| {
                console.log(text, [1.0, 1.0, 1.0]);
                println!("{text}");
            };
            let command = match request {
                SystemRequest::Home => Some(JumpCommand::Home),
                SystemRequest::Goto { .. } => Some(JumpCommand::Goto),
                SystemRequest::Add { .. } => None,
            };
            let refused = !home_after_jump
                && command
                    .is_some_and(|c| refused_during_jump(jump.as_ref(), c, universe.current != 0));
            match request {
                _ if refused => say_jump_in_progress(renderer, console),
                SystemRequest::Home if universe.current != 0 => {
                    // home is in the start system: jump there first, then home (Jump::then_home)
                    match jump {
                        Some(j) => {
                            if !j.queue_home() {
                                say_jump_in_progress(renderer, console);
                            }
                        }
                        None => {
                            let dir = jump_direction(universe, 0);
                            close_map(map, controller);
                            close_map(galaxy_view, controller);
                            *travel = None;
                            renderer.travel_hud = None;
                            if *mode == GameMode::Planet {
                                controller.fly_mode = true;
                                player.landing = false;
                                player.taking_off = None;
                            }
                            *jump = Some(Jump::new(0, true, dir));
                            let name = &universe.galaxy.systems[0].name;
                            renderer.show_status(&format!("Jumping to {name}…"));
                        }
                    }
                }
                SystemRequest::Home => {
                    // a running bake was for wherever the player was flying: drop it, and the trip too
                    *bake_rx = None;
                    *baking = None;
                    *travel = None;
                    renderer.travel_hud = None;
                    let i = *start_planet;
                    if !loaded_here((*loaded_system, *loaded_world), 0, i) {
                        let baked = bake_planet(system, 0, i, edits_by_planet.remove(&(0, i)));
                        let generation = baked.impostor_generation;
                        // the time after the bake: the planet keeps turning while it bakes
                        let t = clock.now();
                        let spawn = noon_spawn(&baked.data, system, i, t);
                        stash_edits(
                            edits_by_planet,
                            Some((*loaded_system, *loaded_world)),
                            (0, i),
                            planet,
                        );
                        install_world(renderer, controller, planet, baked, spawn);
                        *impostor_generation = generation;
                        *loaded_world = i;
                        *loaded_system = 0;
                        // the bake ran under the black frame at the end of the jump home: fade in from now
                        if let Some(Jump {
                            stage: JumpStage::HomeFadeIn(start),
                            ..
                        }) = jump.as_mut()
                        {
                            *start = Instant::now();
                            renderer.set_fade(1.0);
                        }
                    }
                    let t = clock.now();
                    player.spawn(noon_spawn(planet, system, i, t));
                    player.handover_altitude = Some(crate::player::landing::handover_altitude(
                        system.planets[i].radius,
                    ));
                    player.spin_rate = system.planets[i].spin_rate() as f32;
                    controller.fly_mode = false;
                    *flight_frame = FlightFrame::Free;
                    *mode = GameMode::Planet;
                    say(
                        console,
                        &format!("Back on #{} {}.", i + 1, planet.planet_type.def().name),
                    );
                }
                SystemRequest::Add {
                    planet_type,
                    radius,
                } => match system.add_planet(planet_type, radius) {
                    Ok(i) => say(
                        console,
                        &format!(
                            "Added #{n} {} (radius {radius}). /system goto {n} 3 to visit.",
                            planet_type.def().name,
                            n = i + 1
                        ),
                    ),
                    Err(message) => console.log(message, [1.0, 0.5, 0.0]),
                },
                SystemRequest::Goto { planet: n, radii } => match system.planets.get(n - 1) {
                    None => console.log(
                        &format!(
                            "No planet #{n}: this solar system has 1..={}",
                            system.planets.len()
                        ),
                        [1.0, 0.5, 0.0],
                    ),
                    Some(target) => {
                        let t = clock.now();
                        // over the day side, looking at the planet's centre (roll levels out by itself)
                        let dir = target.sun_dir_in_planet_frame(system.star.position(), t);
                        *space_flight = crate::space::solar_system::SpaceFlight::new(
                            (dir * radii * target.radius).as_dvec3(),
                        );
                        space_flight.rotation =
                            glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, -dir);
                        *flight_frame = FlightFrame::Captured(n - 1);
                        *mode = GameMode::Space;
                        *travel = None;
                        renderer.travel_hud = None;
                        enter_space_controls(controller, map);
                        say(
                            console,
                            &format!(
                                "Orbiting #{n} {} at {radii} radii.",
                                target.planet_type.def().name
                            ),
                        );
                    }
                },
            }
        }
        // a jump to another solar system (/jump, or a pick in the galaxy view): after the system requests, so
        // "/system goto …; /jump …" in one tick (--bench --exec) goes to the orbit first and jumps from there
        if let Some(n) = console.jump_request.take() {
            if n >= universe.galaxy.systems.len() {
                console.log(
                    &format!(
                        "No system #{}: this galaxy has 1..={}",
                        n + 1,
                        universe.galaxy.systems.len()
                    ),
                    [1.0, 0.5, 0.0],
                );
            } else if n == universe.current {
                renderer.show_status("You are here.");
            } else if refused_during_jump(jump.as_ref(), JumpCommand::Jump, false) {
                say_jump_in_progress(renderer, console);
            } else {
                close_map(map, controller);
                close_map(galaxy_view, controller);
                *travel = None;
                renderer.travel_hud = None;
                if *mode == GameMode::Planet {
                    controller.fly_mode = true;
                    player.landing = false;
                    player.taking_off = None;
                }
                let dir = jump_direction(universe, n);
                *jump = Some(Jump::new(n, false, dir));
                let name = &universe.galaxy.systems[n].name;
                renderer.show_status(&format!("Jumping to {name}…"));
            }
        }
        // dev convenience: polled instead of a console command so screenshots can be scripted without
        // needing window focus or keyboard injection
        const SCREENSHOT_TRIGGER: &str = "/tmp/voxanet_screenshot.trigger";
        if let Ok(path) = std::fs::read_to_string(SCREENSHOT_TRIGGER) {
            let _ = std::fs::remove_file(SCREENSHOT_TRIGGER);
            renderer.request_screenshot(path.trim().to_string());
        }
        // --bench: wait for the world, run its commands, record, then quit (perf.rs)
        if let Some(b) = bench {
            match &mut b.stage {
                BenchStage::Loading(since) => {
                    if renderer.view_covered() {
                        let since = *since.get_or_insert_with(Instant::now);
                        if since.elapsed().as_secs_f32() >= 2.0 {
                            for cmd in &b.args.exec {
                                console.exec(cmd, player);
                            }
                            b.stage = BenchStage::Settling(Instant::now());
                        }
                    } else {
                        *since = None;
                    }
                }
                BenchStage::Settling(at) => {
                    if at.elapsed().as_secs_f32() >= 1.0 {
                        renderer.start_perf(b.args.seconds, &b.args.label);
                        b.stage = BenchStage::Recording;
                    }
                }
                BenchStage::Recording => {
                    if !renderer.perf_running() {
                        *quit = true;
                    }
                }
            }
        }
        const COMMAND_TRIGGER: &str = "/tmp/voxanet_cmd.trigger";
        if let Ok(cmd) = std::fs::read_to_string(COMMAND_TRIGGER) {
            let _ = std::fs::remove_file(COMMAND_TRIGGER);
            console.exec(cmd.trim(), player);
        }

        if let Some(cmd) = console.ao_request.take() {
            let msg = renderer.ao.apply(cmd);
            console.log(&msg, [0.0, 1.0, 0.0]);
            if !renderer.apply_ao_state(planet, player.position) && renderer.ao.enabled {
                console.log(
                    "Ray-traced AO needs hardware shadows; using vertex AO.",
                    [1.0, 0.5, 0.0],
                );
            }
        }
        if let Some(cmd) = console.plastic_request.take() {
            let msg = renderer.plastic.apply(cmd);
            console.log(&msg, [0.0, 1.0, 0.0]);
        }
        visited.insert((*loaded_system, *loaded_world));
        if let Some(request) = console.map_request.take() {
            match (request, map.is_some()) {
                _ if refused_during_jump(jump.as_ref(), JumpCommand::Map, false) => {
                    say_jump_in_progress(renderer, console)
                }
                (None, true) => close_map(map, controller),
                (selected, _) => {
                    // one overlay at a time: the galaxy view closes, as /galaxy closes the map
                    close_map(galaxy_view, controller);
                    let m = map.get_or_insert_with(|| open_map(controller));
                    if let Some(i) = selected.filter(|&i| i < system.planets.len()) {
                        m.selected = Some(i);
                    }
                }
            }
        }
        if let Some(request) = console.galaxy_request.take() {
            match (request, galaxy_view.is_some()) {
                _ if refused_during_jump(jump.as_ref(), JumpCommand::Galaxy, false) => {
                    say_jump_in_progress(renderer, console)
                }
                (None, true) => close_map(galaxy_view, controller),
                (selected, _) => {
                    close_map(map, controller);
                    let g = galaxy_view.get_or_insert_with(|| open_galaxy(controller));
                    if let Some(i) = selected.filter(|&i| i < universe.galaxy.systems.len()) {
                        g.selected = Some(i);
                    }
                }
            }
        }
        // a system picked in the galaxy view: jump there
        if let Some(n) = galaxy_view.as_mut().and_then(|g| g.travel_request.take()) {
            console.jump_request = Some(n);
        }
        // a trip picked on the map or with /travel
        let request = console
            .travel_request
            .take()
            .or_else(|| map.as_mut().and_then(|m| m.travel_request.take()));
        if let Some(n) = request {
            if n >= system.planets.len() {
                console.log(
                    &format!(
                        "No planet #{}: this solar system has 1..={}",
                        n + 1,
                        system.planets.len()
                    ),
                    [1.0, 0.5, 0.0],
                );
            } else if refused_during_jump(jump.as_ref(), JumpCommand::Travel, false) {
                say_jump_in_progress(renderer, console);
            } else if *mode == GameMode::Planet && *loaded_world == n {
                renderer.show_status("You are here.");
            } else {
                close_map(map, controller);
                if *mode == GameMode::Planet {
                    controller.fly_mode = true;
                    player.landing = false;
                    player.taking_off = None;
                }
                *travel = Some(Travel::Climb { target: n });
                *trip_mouse = TripMouse::new(now);
                let name = system.planets[n].planet_type.def().name;
                renderer.show_status(&format!("Travelling to #{} {name}...", n + 1));
            }
        }
        if let Some((secs, label)) = console.perf_request.take() {
            renderer.start_perf(secs, &label);
            console.log(
                &format!("Perf: recording {secs} s as '{label}'..."),
                [0.0, 1.0, 0.0],
            );
        }
        if let Some(msg) = renderer.take_perf_message() {
            console.log(&msg, [0.0, 1.0, 0.0]); // also printed to the terminal
        }
        if let Some(cmd) = console.weather_request.take() {
            use crate::weather::WeatherCommand;
            // in space flight the player's position and the loaded planet's type are stale: only the
            // global switch makes sense there
            if *mode == GameMode::Space
                && !matches!(
                    cmd,
                    WeatherCommand::On
                        | WeatherCommand::Off
                        | WeatherCommand::Particles(_)
                        | WeatherCommand::Fog(_)
                        | WeatherCommand::Lightning(_)
                )
            {
                console.log(
                    "Land on a planet to see or force its weather.",
                    [1.0, 0.5, 0.0],
                );
            } else {
                let kind = crate::weather::WeatherKind::for_planet(planet.planet_type);
                let t = (clock.now() % 3600.0) as f32;
                let (msg, changed) = renderer.weather.apply(cmd, kind, player.position, t);
                if changed {
                    renderer.weather_changed();
                }
                console.log(&msg, [0.0, 1.0, 0.0]);
            }
        }
        if let Some(cmd) = console.light_request.take() {
            let msg = renderer.light.apply(cmd);
            console.log(&msg, [0.0, 1.0, 0.0]);
        }
        if let Some(cmd) = console.shadow_request.take() {
            let msg = renderer.shadow.apply(cmd);
            console.log(&msg, [0.0, 1.0, 0.0]);
        }
        if let Some(cmd) = console.film_request.take() {
            let msg = renderer.film.apply(cmd);
            console.log(&msg, [0.0, 1.0, 0.0]);
        }
        if let Some(cmd) = console.bevel_request.take() {
            let msg = renderer.bevel.apply(cmd);
            console.log(&msg, [0.0, 1.0, 0.0]);
        }
        if let Some(request) = console.hw_shadows_request.take() {
            // no value: only show the current mode
            let on = request.unwrap_or(renderer.hw_shadows);
            let active = match request {
                Some(on) => renderer.set_hw_shadows(on),
                None => renderer.hw_shadows,
            };
            if on && !active {
                console.log(
                    "Hardware ray tracing is not supported on this GPU; using ray marching.",
                    [1.0, 0.5, 0.0],
                );
            } else {
                console.log(
                    if active {
                        "Shadows: hardware ray tracing"
                    } else {
                        "Shadows: ray marching"
                    },
                    [0.0, 1.0, 0.0],
                );
            }
            // ray-traced AO needs the hardware path; vertex AO otherwise
            renderer.apply_ao_state(planet, player.position);
        }
        if let Some(style) = console.style_request.take() {
            crate::meshing::lowpoly::set_style(style);
            planet.cells = crate::meshing::lowpoly::cells_for(style);
            renderer.force_reload_all(planet, player.position);
            console.log(&format!("Terrain style: {style:?}"), [0.0, 1.0, 0.0]);
        }

        // the console takes the keyboard (see window_event) and needs a free mouse cursor
        if console.is_open {
            let _ = renderer.window.set_cursor_grab(CursorGrabMode::None);
            renderer.window.set_cursor_visible(true);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let Self {
            renderer,
            controller,
            player,
            planet,
            console,
            ..
        } = self;

        // GLOBAL KEYS: work in every mode and even while the console is open (it captures all other
        // keyboard input, and in Space mode it isn't drawn, so an open console is easy to miss)
        if let WindowEvent::KeyboardInput {
            event: key_event, ..
        } = &event
        {
            if key_event.state == ElementState::Pressed && !key_event.repeat {
                match key_event.physical_key {
                    PhysicalKey::Code(KeyCode::F2) => {
                        let ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis();
                        let _ = std::fs::create_dir_all("screenshots");
                        renderer.request_screenshot(format!("screenshots/voxanet_{ms}.png"));
                        renderer.window.request_redraw();
                        return;
                    }
                    PhysicalKey::Code(KeyCode::Escape) => {
                        if jump_blocks_input(&self.jump) {
                            renderer.show_status("Jump in progress.");
                        } else if self.map.is_some() {
                            close_map(&mut self.map, controller);
                        } else if self.galaxy_view.is_some() {
                            close_map(&mut self.galaxy_view, controller);
                        } else if self.travel.is_some() {
                            self.cancel_travel();
                        } else {
                            controller.toggle_mouse_release();
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }

        // CONSOLE INPUT INTERCEPTION
        if console.is_open {
            match event {
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    if key_event.state == ElementState::Pressed {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::Backquote) => {
                                // the key is a dead key on some layouts (^ on German): reset it, or
                                // the next key is composed with it and dropped (^ + / composes nothing)
                                renderer.window.reset_dead_keys();
                                console.toggle();
                            }
                            PhysicalKey::Code(KeyCode::Enter) => console.submit(player),
                            PhysicalKey::Code(KeyCode::Backspace) => console.handle_backspace(),
                            PhysicalKey::Code(KeyCode::ArrowUp) => console.history_up(),
                            PhysicalKey::Code(KeyCode::ArrowDown) => console.history_down(),
                            _ => {
                                if let Some(txt) = &key_event.text {
                                    // Append text to console buffer
                                    for c in txt.chars() {
                                        console.handle_char(c);
                                    }
                                }
                            }
                        }
                    }
                    return;
                }
                _ => {}
            }
        }

        if let WindowEvent::KeyboardInput {
            event: key_event, ..
        } = &event
        {
            if key_event.state == ElementState::Pressed {
                if let PhysicalKey::Code(KeyCode::Backquote) = key_event.physical_key {
                    // a dead key on some layouts (^ on German): reset it so the first character typed
                    // into the console isn't composed with it (^ + / is dropped, ^ + a becomes â)
                    renderer.window.reset_dead_keys();
                    console.toggle();
                    return;
                }
            }
        }

        // a jump owns the controls while it climbs or flies (not cancellable, overlays blocked)
        if jump_blocks_input(&self.jump) && jump_swallows(&event) {
            return;
        }

        // the open map or galaxy view takes the keyboard and mouse (space/map.rs MapView)
        let galaxy_open = self.galaxy_view.is_some();
        if self.map.is_some() || galaxy_open {
            // M and G switch between the two views or close the open one
            if let WindowEvent::KeyboardInput {
                event: key_event, ..
            } = &event
            {
                if key_event.state == ElementState::Pressed && !key_event.repeat {
                    match key_event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyM) => {
                            close_map(&mut self.galaxy_view, controller);
                            if galaxy_open {
                                self.map = Some(open_map(controller));
                            } else {
                                close_map(&mut self.map, controller);
                            }
                            return;
                        }
                        PhysicalKey::Code(KeyCode::KeyG) => {
                            close_map(&mut self.map, controller);
                            if galaxy_open {
                                close_map(&mut self.galaxy_view, controller);
                            } else {
                                self.galaxy_view = Some(open_galaxy(controller));
                            }
                            return;
                        }
                        _ => {}
                    }
                }
            }
            let count = if galaxy_open {
                self.universe.galaxy.systems.len()
            } else {
                self.system.planets.len()
            };
            let m = match self.map.as_mut() {
                Some(m) => m,
                None => self.galaxy_view.as_mut().expect("an overlay is open"),
            };
            match &event {
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } if key_event.state == ElementState::Pressed => {
                    match key_event.physical_key {
                        PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter) => m.confirm(),
                        PhysicalKey::Code(code) => {
                            if let Some(n) = digit_key(code).filter(|&n| n >= 1 && n <= count) {
                                m.selected = Some(n - 1);
                            }
                        }
                        _ => {}
                    }
                    return;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    m.cursor_moved(glam::Vec2::new(position.x as f32, position.y as f32));
                    return;
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    match state {
                        ElementState::Pressed => m.press(),
                        ElementState::Released => m.release(),
                    }
                    return;
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let steps = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                    m.camera.zoom(steps);
                    return;
                }
                WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. } => return,
                _ => {} // redraw, resize, close: as usual
            }
        } else if let WindowEvent::KeyboardInput {
            event: key_event, ..
        } = &event
        {
            use KeyCode::*;
            if key_event.state == ElementState::Pressed
                && !key_event.repeat
                && key_event.physical_key == PhysicalKey::Code(KeyM)
            {
                self.map = Some(open_map(controller));
                return;
            }
            if key_event.state == ElementState::Pressed
                && !key_event.repeat
                && key_event.physical_key == PhysicalKey::Code(KeyG)
            {
                self.galaxy_view = Some(open_galaxy(controller));
                return;
            }
            // a movement key (or F, which would otherwise start a landing against the climb) takes over a
            // running trip; the key acts from its next press
            if self.travel.is_some()
                && key_event.state == ElementState::Pressed
                && matches!(
                    key_event.physical_key,
                    PhysicalKey::Code(
                        KeyW | KeyA | KeyS | KeyD | KeyQ | KeyE | Space | ShiftLeft | KeyF
                    )
                )
            {
                self.cancel_travel();
                return;
            }
        }

        controller.process_events(&event, player, planet);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width, size.height),

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if controller.first_person && controller.mouse_released {
                    // recapture takes priority (in every mode): don't also mine/place on the click
                    // that brings the mouse back
                    controller.mouse_released = false;
                    let _ = renderer.window.set_cursor_grab(CursorGrabMode::Locked);
                    renderer.window.set_cursor_visible(false);
                } else if matches!(self.mode, GameMode::Planet) {
                    let is_right = button == MouseButton::Right;
                    if controller.fly_mode && button != MouseButton::Middle {
                        // no block editing in fly mode (cursor_target): say so instead of ignoring it
                        renderer.show_status("Land to build.");
                    } else if let Some(id) = controller.cursor_id {
                        if button == MouseButton::Middle {
                            // pick the targeted block's type for placing (the bedrock core isn't placeable)
                            if let Some(ty) = planet.block_type(id).filter(|ty| {
                                crate::planet::material::placeable(
                                    &planet.planet_type.def().palette,
                                )
                                .contains(ty)
                            }) {
                                controller.selected_block = ty;
                            }
                        } else if is_right {
                            let place_info = controller.raycast(
                                player,
                                planet,
                                renderer.config.width as f32,
                                renderer.config.height as f32,
                                true,
                            );
                            if let Some((place_id, _)) = place_info {
                                // a block where the player stands would trap them
                                let result = if crate::player::physics::Physics::overlaps_block(
                                    player.position,
                                    place_id,
                                    planet,
                                ) {
                                    Err(None)
                                } else {
                                    planet
                                        .add_block(place_id, controller.selected_block)
                                        .map_err(Some)
                                };
                                match result {
                                    Ok(()) => renderer.refresh_neighbors(place_id, planet),
                                    Err(why) => renderer.show_status(edit_refused_message(why)),
                                }
                            }
                        } else if button == MouseButton::Left {
                            match planet.remove_block(id) {
                                Ok(()) => renderer.refresh_neighbors(id, planet),
                                Err(why) => renderer.show_status(edit_refused_message(Some(why))),
                            }
                        }
                        renderer.window.request_redraw();
                    } else {
                        if controller.first_person {
                            controller.mouse_released = false;
                            let _ = renderer.window.set_cursor_grab(CursorGrabMode::Locked);
                            renderer.window.set_cursor_visible(false);
                        }
                    }
                }
            }

            // planet-only keys: in Space mode these would silently regenerate the hidden planet. First
            // press only: holding F would otherwise auto-repeat, starting and cancelling a landing in turn
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && matches!(self.mode, GameMode::Planet) =>
            {
                // F: land while flying (refused over lava), take off while walking or swimming
                if let PhysicalKey::Code(KeyCode::KeyF) = event.physical_key {
                    if controller.first_person {
                        let over_lava =
                            crate::player::landing::over_damaging_liquid(planet, player.position);
                        match crate::player::landing::f_action(
                            controller.fly_mode,
                            player.landing,
                            player.taking_off.is_some(),
                            over_lava,
                        ) {
                            crate::player::landing::FAction::StartLanding => {
                                player.landing = true;
                                renderer.show_status("Landing...");
                            }
                            crate::player::landing::FAction::CancelLanding => {
                                player.landing = false;
                                renderer.show_status("Landing cancelled.");
                            }
                            crate::player::landing::FAction::RefuseLanding => {
                                let liquid =
                                    planet.planet_type.def().liquid.map_or("liquid", |l| l.name);
                                renderer.show_status(&format!("Can't land here: {liquid} below."));
                            }
                            crate::player::landing::FAction::TakeOff => {
                                controller.fly_mode = true;
                                player.landing = false;
                                let layer = crate::player::landing::takeoff_target_layer(
                                    planet,
                                    player.position,
                                );
                                player.taking_off =
                                    Some(crate::meshing::gen::CoordSystem::get_layer_radius(
                                        layer,
                                        planet.resolution,
                                    ));
                                player.velocity = glam::Vec3::ZERO;
                                renderer.show_status("Taking off...");
                            }
                            crate::player::landing::FAction::StopTakeOff => {
                                player.taking_off = None;
                                renderer.show_status("Hovering.");
                            }
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let t = self.clock.now();
                if let Some(g) = &mut self.galaxy_view {
                    let screen = (renderer.config.width as f32, renderer.config.height as f32);
                    g.hover = g.cursor.and_then(|c| {
                        crate::space::map::pick_system(&self.universe.galaxy, &g.camera, c, screen)
                    });
                    renderer.render_galaxy_map(&self.universe, &self.system, g, t, &self.console);
                    return;
                }
                if let Some(m) = &mut self.map {
                    let screen = (renderer.config.width as f32, renderer.config.height as f32);
                    let layout = crate::space::map::MapLayout::new(&self.system);
                    m.hover = m.cursor.and_then(|c| {
                        crate::space::map::pick_planet(
                            &self.system,
                            &layout,
                            &m.camera,
                            c,
                            screen,
                            t,
                        )
                    });
                    let (here, ship) = match self.mode {
                        GameMode::Planet => (Some(self.loaded_world), None),
                        GameMode::Space => (
                            None,
                            Some(match self.flight_frame {
                                FlightFrame::Free => self.space_flight.position,
                                FlightFrame::Captured(i) => self.system.planets[i]
                                    .from_planet_frame(self.space_flight.position, t),
                            }),
                        ),
                    };
                    // a running trip's path, drawn on the map
                    let line: Vec<glam::DVec3> = match &self.travel {
                        Some(Travel::Cruise(trip)) => {
                            let n = crate::space::render::TRAVEL_LINE_POINTS - 1;
                            (0..=n)
                                .map(|k| {
                                    trip.position(trip.t0 + trip.duration * k as f64 / n as f64)
                                })
                                .collect()
                        }
                        _ => Vec::new(),
                    };
                    renderer.render_map(
                        &self.system,
                        t,
                        m,
                        here,
                        ship,
                        &self
                            .visited
                            .iter()
                            .filter(|(s, _)| *s == self.universe.current)
                            .map(|(_, p)| *p)
                            .collect(),
                        &line,
                        &self.console,
                    );
                    return;
                }
                // where you are, in the HUD of both modes: the planet you're on or captured by
                let here = match (&self.mode, self.flight_frame) {
                    (GameMode::Planet, _) => Some(self.loaded_world),
                    (GameMode::Space, FlightFrame::Captured(i)) => Some(i),
                    (GameMode::Space, FlightFrame::Free) => None,
                };
                renderer.location_hud = self.universe.location_label(
                    here.map(|i| (i, self.system.planets[i].planet_type.def().name)),
                );
                match &self.mode {
                    GameMode::Planet => {
                        let i = self.loaded_world;
                        let sun = self.system.planets[i]
                            .sun_dir_in_planet_frame(self.system.star.position(), t);
                        // the solar system behind the voxel world, seen from exactly the planet camera
                        use crate::space::render::{Backdrop, SpaceCamera, SpaceContent};
                        let (eye, cam_rot) = controller.camera_pose(player);
                        let fov_y = controller.fov_y();
                        let camera = SpaceCamera::on_system_planet(
                            &self.system.planets[i],
                            eye,
                            cam_rot,
                            fov_y,
                            t,
                        );
                        let content = SpaceContent::AllButPlanet(i);
                        // the solar system's impostor fading out over the voxel world after a landing
                        let handover_overlay = self
                            .landed_at
                            .and_then(|at| {
                                crate::player::landing::handover_overlay_opacity(
                                    at.elapsed().as_secs_f32(),
                                )
                            })
                            .map(|opacity| (i, opacity));
                        let backdrop = Backdrop {
                            system: &self.system,
                            camera,
                            content,
                            t,
                            handover_overlay,
                        };
                        renderer.render(controller, player, planet, console, t, sun, &backdrop)
                    }
                    GameMode::Space => {
                        let camera = crate::space::render::SpaceCamera::from_flight_in_frame(
                            &self.space_flight,
                            self.flight_frame,
                            &self.system,
                            t,
                        );
                        renderer.render_space(&camera, &self.system, t, &self.console)
                    }
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.game.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("voxanet")
                        .with_fullscreen(
                            (!windowed(&std::env::args().collect::<Vec<_>>()))
                                .then_some(winit::window::Fullscreen::Borderless(None)),
                        ),
                )
                .unwrap(),
        );
        self.game = Some(Game::new(window, self.initial_biome, self.seed));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(game) = &mut self.game else { return };
        game.tick();
        if window_id == game.renderer.window.id() {
            game.window_event(event_loop, event);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(game) = &mut self.game else { return };
        game.tick();
        if let DeviceEvent::MouseMotion { delta } = event {
            // turning the view takes over a running trip: only motion that would turn it, summed past a
            // threshold after a short grace period (TripMouse; the tick drops the motion meanwhile)
            if game.travel.is_some()
                && mouse_turns_view(
                    game.controller.first_person,
                    game.controller.mouse_released,
                    game.console.is_open,
                    game.map.is_some() || game.galaxy_view.is_some(),
                )
                && game.trip_mouse.add(delta, Instant::now())
            {
                game.cancel_travel();
            }
            game.controller.process_mouse_motion(delta);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(game) = &mut self.game else { return };
        game.tick();
        if game.quit {
            event_loop.exit();
            return;
        }
        game.renderer.window.request_redraw();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Review Focus 5: the autopilot owns the flight frame while it runs, so the planets it passes can't
    // capture it mid-trip (next_flight_frame runs only in the manual arm, which asserts it's chosen)
    #[test]
    fn travel_owns_the_flight_frame() {
        assert_eq!(space_arm(&None), SpaceArm::Manual);
        assert_eq!(
            space_arm(&Some(Travel::Climb { target: 2 })),
            SpaceArm::Autopilot
        );
        let s = crate::space::solar_system::SolarSystem::generate(1);
        let trip = crate::space::travel::Trip::plan(
            &s,
            3,
            s.planets[0].position_at(0.0),
            glam::DVec3::Y,
            glam::DVec3::Y,
            0.0,
        );
        assert_eq!(space_arm(&Some(Travel::Cruise(trip))), SpaceArm::Autopilot);
    }

    // Important 1: only motion that turns the view counts, after a grace period, summed to a threshold
    #[test]
    fn mouse_cancels_a_trip_only_when_it_turns_the_view() {
        assert!(mouse_turns_view(true, false, false, false));
        assert!(!mouse_turns_view(false, false, false, false)); // third person
        assert!(!mouse_turns_view(true, true, false, false)); // cursor freed
        assert!(!mouse_turns_view(true, false, true, false)); // console open
        assert!(!mouse_turns_view(true, false, false, true)); // map open
        let t0 = Instant::now();
        let at = |s: f32| t0 + std::time::Duration::from_secs_f32(s);
        let mut m = TripMouse::new(t0);
        // within the grace period nothing counts, however large
        assert!(!m.add((500.0, 0.0), at(0.2)));
        // small drift adds up slowly
        for k in 0..9 {
            assert!(!m.add((3.0, 3.0), at(0.5 + k as f32 * 0.01)));
        }
        assert!(m.add((30.0, 0.0), at(0.7)));
    }

    // no block targeted (no selection box, no mining, placing or picking) while flying
    #[test]
    fn blocks_are_targeted_only_when_not_flying() {
        let id = crate::planet::data::BlockId {
            face: 1,
            layer: 20,
            u: 3,
            v: 4,
        };
        assert_eq!(cursor_target(false, Some(id)), Some(id));
        assert_eq!(cursor_target(true, Some(id)), None);
        assert_eq!(cursor_target(false, None), None);
    }

    fn mined_planet() -> PlanetData {
        let mut planet = PlanetData::new(32);
        let h = planet.terrain.get_height(0, 3, 3);
        planet
            .remove_block(crate::planet::data::BlockId {
                face: 0,
                layer: h,
                u: 3,
                v: 3,
            })
            .unwrap();
        planet
    }

    // installing a different planet stores the installed planet's edits under its number
    #[test]
    fn stash_stores_the_installed_planets_edits() {
        let mut store = std::collections::HashMap::new();
        let mut planet = mined_planet();
        stash_edits(&mut store, Some(2), 5, &mut planet);
        assert!(store.get(&2).is_some_and(|e| !e.is_empty()));
        assert!(planet.edits.is_empty());
    }

    #[test]
    fn the_loaded_world_compares_as_system_and_planet() {
        assert!(loaded_here((0, 3), 0, 3));
        assert!(!loaded_here((0, 3), 1, 3)); // planet #4 of another system
        assert!(!loaded_here((1, 3), 0, 3));
        assert!(!loaded_here((0, 3), 0, 2));
    }

    #[test]
    fn the_jump_owns_the_controls_until_it_has_flown() {
        let d = glam::Vec3::X;
        assert!(!jump_blocks_input(&None));
        let mut j = Jump::new(3, false, d);
        assert!(j.flies(), "climbing");
        assert!(jump_blocks_input(&Some(Jump::new(3, false, d))));
        j.stage = JumpStage::Fly {
            started: Instant::now(),
            duration: 4.0,
            prev_frac: 0.0,
            initial_fwd: d,
            initial_velocity: glam::DVec3::ZERO,
        };
        assert!(j.flies(), "flying");
        j.stage = JumpStage::HomeFadeIn(Instant::now());
        assert!(!j.flies(), "home's fade-in gives the controls back");
        assert!(!jump_blocks_input(&Some(j)));
    }

    #[test]
    fn the_console_waits_for_a_jump() {
        use JumpCommand as C;
        let all = [C::Home, C::Goto, C::Map, C::Galaxy, C::Travel, C::Jump];
        let d = glam::Vec3::X;
        // no jump: everything acts
        for c in all {
            assert!(!refused_during_jump(None, c, false), "{c:?}");
            assert!(!refused_during_jump(None, c, true), "{c:?}");
        }
        // climbing or flying: all refused, except home from another system (queue_home decides)
        let climb = Jump::new(3, false, d);
        let mut fly = Jump::new(3, false, d);
        fly.stage = JumpStage::Fly {
            started: Instant::now(),
            duration: 4.0,
            prev_frac: 0.3,
            initial_fwd: d,
            initial_velocity: glam::DVec3::ZERO,
        };
        for j in [&climb, &fly] {
            for c in all {
                assert!(refused_during_jump(Some(j), c, false), "{c:?}");
            }
            assert!(!refused_during_jump(Some(j), C::Home, true));
            assert!(refused_during_jump(Some(j), C::Goto, true));
        }
        // home's fade-in: the overlays are back, nothing moves the player off the start planet
        let mut fade = Jump::new(0, true, d);
        fade.stage = JumpStage::HomeFadeIn(Instant::now());
        assert!(!refused_during_jump(Some(&fade), C::Map, false));
        assert!(!refused_during_jump(Some(&fade), C::Galaxy, false));
        for c in [C::Home, C::Goto, C::Travel, C::Jump] {
            assert!(refused_during_jump(Some(&fade), c, false), "{c:?}");
        }
    }

    #[test]
    fn a_jump_swallows_clicks_but_not_releases() {
        let device_id = DeviceId::dummy();
        let click = |state| WindowEvent::MouseInput {
            device_id,
            state,
            button: MouseButton::Left,
        };
        assert!(jump_swallows(&click(ElementState::Pressed)));
        assert!(!jump_swallows(&click(ElementState::Released)));
        assert!(!jump_swallows(&WindowEvent::Focused(true)));
        assert!(!jump_swallows(&WindowEvent::CursorLeft { device_id }));
    }

    // a slow frame can step from before the switch straight past the end (space/warp.rs
    // a_long_frame_switches_and_arrives): the switch block must run before the end-of-flight block in
    // tick(), so the jump still lands in the new system
    #[test]
    fn the_switch_runs_before_the_end_of_the_flight() {
        let src = include_str!("main.rs");
        let tick = &src[src.find("fn tick(&mut self)").unwrap()..];
        let switch = tick.find("// the switch, decided by the flight").unwrap();
        let end = tick.find("// the end of a flight:").unwrap();
        assert!(switch < end);
    }

    #[test]
    fn home_during_a_jump_waits_for_it() {
        let d = glam::Vec3::X;
        // on its way to the start system: home on arrival
        let mut j = Jump::new(0, false, d);
        assert!(j.queue_home());
        assert!(j.then_home);
        // anywhere else, or already switched: left alone
        let mut j = Jump::new(4, false, d);
        assert!(!j.queue_home());
        assert!(!j.then_home);
        let mut j = Jump::new(0, false, d);
        j.switched = true;
        assert!(!j.queue_home());
        assert!(!j.then_home);
    }

    #[test]
    fn home_fades_in_from_black() {
        assert_eq!(home_fade(0.0), 1.0);
        assert!((home_fade(HOME_FADE_SECS * 0.5) - 0.5).abs() < 1e-6);
        assert_eq!(home_fade(HOME_FADE_SECS * 2.0), 0.0);
    }

    // the warp's streaks are drawn over the space scene, under the HUD
    #[test]
    fn the_warp_is_drawn_after_the_space_scene() {
        let r = include_str!("render/renderer.rs");
        let f = &r[r.find("pub fn render_space(").unwrap()..];
        let f = &f[..f.find("\n    pub fn ").unwrap_or(f.len())];
        let scene = f.find("self.system.render(").unwrap();
        let warp = f.find("self.system.draw_warp(").unwrap();
        let hud = f.find("self.render_space_overlay(").unwrap();
        assert!(scene < warp && warp < hud);
    }

    // Review Focus 2: edits are kept per (system, planet): A#3's edits come back for A#3, never for B#3
    #[test]
    fn edits_are_kept_per_system_and_planet() {
        let mut store = std::collections::HashMap::new();
        let mut planet = mined_planet();
        // A#3 installed, B#3 comes in: A#3's edits go to the store under (A, 3)
        stash_edits(&mut store, Some((0usize, 3usize)), (4, 3), &mut planet);
        assert!(store.contains_key(&(0, 3)));
        assert!(!store.contains_key(&(4, 3)), "B#3 has none");
    }

    // re-installing the planet that's already installed (or the first install at start-up) stores nothing
    #[test]
    fn stash_is_a_no_op_for_the_installed_planet() {
        let mut store = std::collections::HashMap::new();
        let mut planet = mined_planet();
        stash_edits(&mut store, Some(2), 2, &mut planet);
        stash_edits(&mut store, None, 2, &mut planet);
        assert!(store.is_empty());
        assert!(!planet.edits.is_empty());
    }

    // fullscreen unless --windowed
    #[test]
    fn windowed_flag_opts_out_of_fullscreen() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(!windowed(&args(&["voxanet"])));
        assert!(!windowed(&args(&["voxanet", "--seed", "3"])));
        assert!(windowed(&args(&["voxanet", "--windowed"])));
        assert!(windowed(&args(&["voxanet", "--seed", "3", "--windowed"])));
    }

    #[test]
    fn seed_flag_is_parsed_with_fallback() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_seed(&args(&["voxanet", "--seed", "42"])), 42);
        assert_eq!(parse_seed(&args(&["voxanet"])), 1);
        assert_eq!(parse_seed(&args(&["voxanet", "--seed", "x"])), 1);
        assert_eq!(parse_seed(&args(&["voxanet", "--seed"])), 1);
    }

    // the near impostor is outdated once the planet is edited after it was built
    #[test]
    fn near_impostor_is_outdated_only_after_an_edit() {
        let mut planet = PlanetData::new(32);
        let built_at = planet.edit_generation();
        assert!(!near_impostor_outdated(built_at, &planet));
        let h = planet.terrain.get_height(0, 3, 3);
        planet
            .remove_block(crate::planet::data::BlockId {
                face: 0,
                layer: h,
                u: 3,
                v: 3,
            })
            .unwrap();
        assert!(near_impostor_outdated(built_at, &planet));
    }

    // an unedited planet takes no room in the store
    #[test]
    fn stash_skips_unedited_planets() {
        let mut store = std::collections::HashMap::new();
        let mut planet = PlanetData::new(32);
        stash_edits(&mut store, Some(2), 5, &mut planet);
        assert!(store.is_empty());
    }

    // a bake finishes on another thread some time after capture; by then the player may have flown
    // out of orbit, gone home, landed elsewhere or been captured by another planet
    #[test]
    fn late_bakes_are_dropped_only_after_going_home_landing_or_another_capture() {
        assert!(bake_still_wanted(
            GameMode::Space,
            FlightFrame::Captured(2),
            2,
            0,
            0
        ));
        assert!(!bake_still_wanted(
            GameMode::Space,
            FlightFrame::Captured(3),
            2,
            0,
            0
        ));
        assert!(!bake_still_wanted(
            GameMode::Planet,
            FlightFrame::Captured(2),
            2,
            0,
            0
        ));
        // Review Focus 3: a bake for the solar system left by a jump is dropped
        assert!(!bake_still_wanted(
            GameMode::Space,
            FlightFrame::Captured(2),
            2,
            0,
            5
        ));
        // spec §1: leaving orbit before landing keeps the bake — a bake finishing while flight is
        // free again (nobody else captured it meanwhile) is still installed, so re-capture won't re-bake
        assert!(bake_still_wanted(
            GameMode::Space,
            FlightFrame::Free,
            2,
            0,
            0
        ));
    }
}
