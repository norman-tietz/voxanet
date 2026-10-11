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
    index: usize,
    edits: Option<crate::planet::data::PlanetEdits>,
) -> BakedPlanet {
    let data = system.planets[index].bake_with_edits(edits);
    let (near_verts, near_indices) = crate::space::impostor::near_impostor_mesh(&data);
    BakedPlanet {
        index,
        impostor_generation: data.edit_generation(),
        data,
        near_verts,
        near_indices,
    }
}

fn start_bake(
    system: &crate::space::solar_system::SolarSystem,
    index: usize,
    edits: Option<crate::planet::data::PlanetEdits>,
) -> std::sync::mpsc::Receiver<BakedPlanet> {
    let (tx, rx) = std::sync::mpsc::channel();
    let target = system.planets[index];
    std::thread::spawn(move || {
        let data = target.bake_with_edits(edits);
        let (near_verts, near_indices) = crate::space::impostor::near_impostor_mesh(&data);
        let _ = tx.send(BakedPlanet {
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
fn stash_edits(
    store: &mut std::collections::HashMap<usize, crate::planet::data::PlanetEdits>,
    installed: Option<usize>,
    incoming: usize,
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

// a near impostor rebuilt in the background: (planet index, edit generation it shows, mesh)
type ImpostorRebuild = (usize, u64, Vec<crate::planet::data::Vertex>, Vec<u32>);

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
fn bake_still_wanted(mode: GameMode, frame: FlightFrame, index: usize) -> bool {
    mode == GameMode::Space
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
    start_planet: usize,       // where the game starts and /system home returns to
    // each planet's edits while another planet is installed (stash_edits), handed to its next bake
    edits_by_planet: std::collections::HashMap<usize, crate::planet::data::PlanetEdits>,
    impostor_generation: u64, // the installed planet's edit_generation when its near impostor was built
    impostor_rx: Option<std::sync::mpsc::Receiver<ImpostorRebuild>>, // a near impostor rebuild in flight
    map: Option<crate::space::map::MapView>, // the solar-system map, while it's open (M)
    travel: Option<Travel>, // an automatic flight to a planet's orbit (space/travel.rs), while it runs
    trip_mouse: TripMouse,  // mouse motion during a trip, toward cancelling it
    visited: std::collections::HashSet<usize>, // planets whose voxel world has been installed this session
    landed_at: Option<Instant>, // the last landing handover, for its cross-fade (landing::handover_overlay_opacity)
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
        let system = crate::space::solar_system::SolarSystem::generate(seed);
        println!(
            "Solar system seed {seed}: {} star",
            system.star.star_type.def().name
        );
        let start_planet = system.start_planet_index(initial_biome);

        // every planet lives in the solar system: start on one of them, standing at local noon
        let mut planet = PlanetData::new(8); // placeholder, replaced right below
        let baked = bake_planet(&system, start_planet, None);
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
            visited,
            travel,
            trip_mouse,
            clock,
            loaded_world,
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
        let want_locked = controller.first_person && !controller.mouse_released && map.is_none();
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

        // during a trip the mouse turns nothing: its motion only counts toward cancelling (device_event),
        // and the count restarts (with its grace period) whenever the mouse can't turn the view
        if travel.is_some() {
            controller.mouse_delta = (0.0, 0.0);
            if !mouse_turns_view(
                controller.first_person,
                controller.mouse_released,
                console.is_open,
                map.is_some(),
            ) {
                *trip_mouse = TripMouse::new(now);
            }
        }

        // physics & player Update (once per tick: movement and turning speeds are per second)
        match mode {
            GameMode::Planet => {
                let climbing = matches!(travel, Some(Travel::Climb { .. }));
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
                let radius = system.planets[*loaded_world].radius;
                if crate::player::landing::should_lift_off(player.position.length() / radius) {
                    lift_off_from = Some(*loaded_world);
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
                    if *loaded_world != target && *baking != Some(target) {
                        *bake_rx = Some(start_bake(
                            system,
                            target,
                            edits_by_planet.get(&target).cloned(),
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
                    if *loaded_world != i && *baking != Some(i) {
                        // the stored edits stay in the store until this bake is installed, so a
                        // dropped bake loses nothing
                        *bake_rx = Some(start_bake(system, i, edits_by_planet.get(&i).cloned()));
                        *baking = Some(i);
                    }
                    // once loaded, the voxel engine streams from the flight's planet-frame position,
                    // so its meshes are ready by the time the landing handover needs them
                    if *loaded_world == i {
                        renderer.update_view(space_flight.position.as_vec3(), planet);
                        if !*voxel_ready_announced && renderer.view_covered() {
                            println!("Voxel world for #{} ready", i + 1);
                            *voxel_ready_announced = true;
                        }
                        let distance_radii =
                            space_flight.position.length() / system.planets[i].radius as f64;
                        if crate::player::landing::should_land(
                            distance_radii,
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
            if bake_still_wanted(*mode, *flight_frame, baked.index) {
                // the engine streams in the planet frame; free flight is in space
                let local = match *flight_frame {
                    FlightFrame::Captured(_) => space_flight.position,
                    FlightFrame::Free => system.planets[baked.index]
                        .to_planet_frame(space_flight.position, clock.now()),
                };
                let index = baked.index;
                let generation = baked.impostor_generation;
                // the installed planet keeps its edits for its next visit; the new one has its own now
                stash_edits(edits_by_planet, Some(*loaded_world), index, planet);
                edits_by_planet.remove(&index);
                install_world(renderer, controller, planet, baked, local.as_vec3());
                *loaded_world = index;
                *impostor_generation = generation;
                *voxel_ready_announced = false;
                println!("Baked #{} (res {})", index + 1, planet.resolution);
            }
        }

        // the solar system shows the installed planet's near impostor: rebuild it in the background when the
        // planet was edited since it was built (after liftoff, or leaving by /system goto)
        if *mode == GameMode::Space
            && impostor_rx.is_none()
            && near_impostor_outdated(*impostor_generation, planet)
        {
            let (tx, rx) = std::sync::mpsc::channel();
            let (index, generation, data) =
                (*loaded_world, planet.edit_generation(), planet.clone());
            std::thread::spawn(move || {
                let (v, i) = crate::space::impostor::near_impostor_mesh(&data);
                let _ = tx.send((index, generation, v, i));
            });
            *impostor_rx = Some(rx);
        }
        let rebuilt = impostor_rx.as_ref().map(|rx| rx.try_recv());
        if let Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) = rebuilt {
            *impostor_rx = None; // the rebuild thread died: allow a new one
        }
        if let Some(Ok((index, generation, v, i))) = rebuilt {
            *impostor_rx = None;
            // dropped if another planet was installed meanwhile
            if index == *loaded_world {
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
        if let Some(request) = console.system_request.take() {
            use crate::app::cmd::SystemRequest;
            let say = |console: &mut Console, text: &str| {
                console.log(text, [1.0, 1.0, 1.0]);
                println!("{text}");
            };
            match request {
                SystemRequest::Home => {
                    // a running bake was for wherever the player was flying: drop it, and the trip too
                    *bake_rx = None;
                    *baking = None;
                    *travel = None;
                    renderer.travel_hud = None;
                    let i = *start_planet;
                    if *loaded_world != i {
                        let baked = bake_planet(system, i, edits_by_planet.remove(&i));
                        let generation = baked.impostor_generation;
                        // the time after the bake: the planet keeps turning while it bakes
                        let t = clock.now();
                        let spawn = noon_spawn(&baked.data, system, i, t);
                        stash_edits(edits_by_planet, Some(*loaded_world), i, planet);
                        install_world(renderer, controller, planet, baked, spawn);
                        *impostor_generation = generation;
                        *loaded_world = i;
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
        visited.insert(*loaded_world);
        if let Some(request) = console.map_request.take() {
            match (request, map.is_some()) {
                (None, true) => close_map(map, controller),
                (selected, _) => {
                    let m = map.get_or_insert_with(|| open_map(controller));
                    if let Some(i) = selected.filter(|&i| i < system.planets.len()) {
                        m.selected = Some(i);
                    }
                }
            }
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
                    WeatherCommand::On | WeatherCommand::Off | WeatherCommand::Particles(_)
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
                        if self.map.is_some() {
                            close_map(&mut self.map, controller);
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

        // the open map takes the keyboard and mouse (space/map.rs MapView)
        if let Some(m) = &mut self.map {
            match &event {
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } if key_event.state == ElementState::Pressed => {
                    match key_event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyM) => close_map(&mut self.map, controller),
                        PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter) => m.confirm(),
                        PhysicalKey::Code(code) => {
                            if let Some(n) = digit_key(code)
                                .filter(|&n| n >= 1 && n <= self.system.planets.len())
                            {
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
                        &self.visited,
                        &line,
                        &self.console,
                    );
                    return;
                }
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
                    game.map.is_some(),
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
            2
        ));
        assert!(!bake_still_wanted(
            GameMode::Space,
            FlightFrame::Captured(3),
            2
        ));
        assert!(!bake_still_wanted(
            GameMode::Planet,
            FlightFrame::Captured(2),
            2
        ));
        // spec §1: leaving orbit before landing keeps the bake — a bake finishing while flight is
        // free again (nobody else captured it meanwhile) is still installed, so re-capture won't re-bake
        assert!(bake_still_wanted(GameMode::Space, FlightFrame::Free, 2));
    }
}
