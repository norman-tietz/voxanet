// engine main.rs

mod biome;
mod cmd;
mod common;
mod controller;
mod deferred;
mod entity;
mod flare;
mod galaxy;
mod galaxy_render;
mod galaxy_terrain;
mod gen;
mod gpu_timer;
mod hw_rt;
mod icosphere;
mod landing;
mod lod_animation;
mod lowpoly;
mod material;
mod mc_tables;
mod noise;
mod physics;
mod renderer;
mod rt_blur;
mod rt_shadow;
mod screenshot;
mod smoothing;
mod star_shading;
mod system_diagnostics;

use crate::cmd::Console;
use crate::common::PlanetData;
use crate::controller::Controller;
use crate::entity::Player;
use crate::galaxy::FlightFrame;
use crate::renderer::Renderer;
use crate::system_diagnostics::SystemDiagnostics;
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
fn parse_biome_arg() -> crate::biome::PlanetType {
    let Some(requested) = std::env::args().skip_while(|a| a != "--biome").nth(1) else {
        return crate::biome::PlanetType::EarthLike;
    };
    crate::biome::PlanetType::from_name(&requested).unwrap_or_else(|| {
        let names: Vec<_> = crate::biome::PlanetType::ALL
            .iter()
            .map(|t| t.def().name)
            .collect();
        eprintln!(
            "Unknown --biome '{requested}', valid options: {}. Using Earth-like.",
            names.join(", ")
        );
        crate::biome::PlanetType::EarthLike
    })
}

// --seed <n>: the galaxy seed (default 1): sets the star type and the planets
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
    if let Some(id) = crate::gen::CoordSystem::pos_to_id(dir * (res as f32 / 2.0), res) {
        crate::gen::CoordSystem::get_layer_radius(planet.effective_height(id.face, id.u, id.v), res)
            + margin
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
    initial_biome: crate::biome::PlanetType, // from --biome, see parse_biome_arg()
    seed: u64,                               // the galaxy seed, from --seed (parse_seed)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum GameMode {
    Planet,
    Galaxy,
}

// galaxy mode has no Q/E key-yaw, so mouse-look must be active even if the player entered while in
// third person (Controller::raw_input gates it on first_person && !mouse_released); and the console
// isn't drawn in galaxy mode, while an open one swallows all keyboard input (WASD included)
fn enter_galaxy_controls(controller: &mut Controller, console: &mut Console) {
    controller.first_person = true;
    controller.mouse_released = false;
    if console.is_open {
        console.toggle();
    }
}

// a galaxy planet baked on a background thread: its voxel world plus the near impostor built from it
struct BakedPlanet {
    index: usize,
    data: PlanetData,
    near_verts: Vec<crate::common::Vertex>,
    near_indices: Vec<u32>,
    impostor_generation: u64, // data.edit_generation() when the near impostor was built
}

// bakes a galaxy planet's voxel world and its near impostor: synchronously at start-up and for
// /galaxy home, on a background thread on capture (start_bake)
fn bake_planet(
    galaxy: &crate::galaxy::Galaxy,
    index: usize,
    edits: Option<crate::common::PlanetEdits>,
) -> BakedPlanet {
    let data = galaxy.planets[index].bake_with_edits(edits);
    let (near_verts, near_indices) = crate::galaxy_terrain::near_impostor_mesh(&data);
    BakedPlanet {
        index,
        impostor_generation: data.edit_generation(),
        data,
        near_verts,
        near_indices,
    }
}

fn start_bake(
    galaxy: &crate::galaxy::Galaxy,
    index: usize,
    edits: Option<crate::common::PlanetEdits>,
) -> std::sync::mpsc::Receiver<BakedPlanet> {
    let (tx, rx) = std::sync::mpsc::channel();
    let target = galaxy.planets[index];
    std::thread::spawn(move || {
        let data = target.bake_with_edits(edits);
        let (near_verts, near_indices) = crate::galaxy_terrain::near_impostor_mesh(&data);
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
    store: &mut std::collections::HashMap<usize, crate::common::PlanetEdits>,
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
type ImpostorRebuild = (usize, u64, Vec<crate::common::Vertex>, Vec<u32>);

// makes a baked galaxy planet the voxel engine's world: palette, full mesh reload streaming from
// `stream_from` (planet frame), near impostor for the galaxy renderer
fn install_world(
    renderer: &mut Renderer,
    controller: &mut Controller,
    planet: &mut PlanetData,
    baked: BakedPlanet,
    stream_from: glam::Vec3,
) {
    *planet = baked.data;
    controller.selected_block = crate::material::placeable(&planet.planet_type.def().palette)[0];
    renderer.force_reload_all(planet, stream_from);
    renderer.set_near_impostor(baked.index, &baked.near_verts, &baked.near_indices);
    renderer.log_memory(planet);
}

// the HUD status line for a refused edit; None: the block would overlap the player
fn edit_refused_message(why: Option<crate::common::EditRefused>) -> &'static str {
    use crate::common::EditRefused;
    match why {
        Some(EditRefused::BuildLimit) => "Build limit reached.",
        Some(EditRefused::MiningFloor) => "Bedrock: can't dig deeper.",
        Some(EditRefused::EditLimit) => "Too many edits on this planet.",
        None => "No room: you're in the way.",
    }
}

// where a player stands at local noon on galaxy planet `index` (whose voxel world is `planet`):
// the surface point facing the star, moved to dry land if that is lava
fn noon_spawn(
    planet: &PlanetData,
    galaxy: &crate::galaxy::Galaxy,
    index: usize,
    t: f64,
) -> glam::Vec3 {
    let noon = galaxy.planets[index].sun_dir_in_planet_frame(galaxy.star.position(), t);
    let dir = planet.safe_spawn_direction(noon);
    dir * spawn_radius(planet, dir, 10.0)
}

// a finished bake is installed unless the player has since gone home, landed (both leave galaxy
// mode) or been captured by another planet; having merely left orbit again keeps it (spec §1:
// re-capturing the same planet doesn't re-bake). Capturing another planet also replaces bake_rx,
// so an older bake can't arrive after that.
fn bake_still_wanted(mode: GameMode, frame: FlightFrame, index: usize) -> bool {
    mode == GameMode::Galaxy
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
    galaxy: crate::galaxy::Galaxy,
    galaxy_flight: crate::galaxy::GalaxyFlight,
    flight_frame: FlightFrame, // galaxy space, or captured by a planet (galaxy.rs)
    bake_rx: Option<std::sync::mpsc::Receiver<BakedPlanet>>, // the running background bake, if any
    baking: Option<usize>,     // which planet it is for
    voxel_ready_announced: bool, // "voxel world ready" printed for the loaded galaxy planet
    clock: Instant,            // shared game clock: galaxy orbits and planet day/night
    loaded_world: usize,       // the galaxy planet the voxel engine holds (always one)
    start_planet: usize,       // where the game starts and /galaxy home returns to
    // each planet's edits while another planet is installed (stash_edits), handed to its next bake
    edits_by_planet: std::collections::HashMap<usize, crate::common::PlanetEdits>,
    impostor_generation: u64, // the installed planet's edit_generation when its near impostor was built
    impostor_rx: Option<std::sync::mpsc::Receiver<ImpostorRebuild>>, // a near impostor rebuild in flight
    landed_at: Option<Instant>, // the last landing handover, for its cross-fade (landing::handover_overlay_opacity)
}

impl Game {
    fn new(window: Arc<Window>, initial_biome: crate::biome::PlanetType, seed: u64) -> Self {
        let mut renderer = pollster::block_on(Renderer::new(window));
        let mut controller = Controller::new();
        let mut player = Player::new();
        let galaxy = crate::galaxy::Galaxy::generate(seed);
        println!(
            "Galaxy seed {seed}: {} star",
            galaxy.star.star_type.def().name
        );
        let start_planet = galaxy.start_planet_index(initial_biome);

        // every planet lives in the galaxy: start on one of them, standing at local noon
        let mut planet = PlanetData::new(8); // placeholder, replaced right below
        let baked = bake_planet(&galaxy, start_planet, None);
        let impostor_generation = baked.impostor_generation;
        // the clock starts once the bake is done, so t = 0 really is now and the planet hasn't
        // turned on past noon while baking
        let clock = Instant::now();
        let spawn = noon_spawn(&baked.data, &galaxy, start_planet, 0.0);
        install_world(&mut renderer, &mut controller, &mut planet, baked, spawn);
        player.spawn(spawn);
        player.handover_altitude = Some(crate::landing::handover_altitude(
            galaxy.planets[start_planet].radius,
        ));
        player.spin_rate = galaxy.planets[start_planet].spin_rate() as f32;

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
            galaxy,
            galaxy_flight: crate::galaxy::GalaxyFlight::new(glam::DVec3::new(0.0, 0.0, 120_000.0)),
            flight_frame: FlightFrame::Free,
            bake_rx: None,
            baking: None,
            voxel_ready_announced: false,
            landed_at: None,
            clock,
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
            galaxy,
            galaxy_flight,
            flight_frame,
            bake_rx,
            baking,
            voxel_ready_announced,
            landed_at,
            clock,
            loaded_world,
            start_planet,
            edits_by_planet,
            impostor_generation,
            impostor_rx,
        } = self;

        let now = Instant::now();
        let dt = (now - *last_time).as_secs_f32();
        *last_time = now;

        // cursor locking logic: locked only in first person, and only while the mouse hasn't been
        // explicitly released (Escape) without leaving first person
        let want_locked = controller.first_person && !controller.mouse_released;
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

        // physics & player Update (once per tick: movement and turning speeds are per second)
        match mode {
            GameMode::Planet => {
                controller.update_player(player, planet, dt);

                // raycast & cursor Update
                let width = renderer.config.width as f32;
                let height = renderer.config.height as f32;
                let ray_result = controller.raycast(player, planet, width, height, false);
                controller.cursor_id = ray_result.map(|(id, _)| id);

                renderer.update_cursor(planet, controller.cursor_id);
                renderer.update_view(player.position, planet);

                // high enough above the planet: hand back to captured galaxy flight
                let radius = galaxy.planets[*loaded_world].radius;
                if crate::landing::should_lift_off(player.position.length() / radius) {
                    lift_off_from = Some(*loaded_world);
                }
            }
            GameMode::Galaxy => {
                let (_, jump, down, sprint, mouse_delta) = controller.raw_input();
                let (input, keys) = controller.galaxy_flight_input();
                // turning and climbing follow galaxy space's +Y in free flight, the local radial up
                // while captured (the planet's spin axis would pitch the view at its equator)
                let up = match *flight_frame {
                    FlightFrame::Free => None,
                    FlightFrame::Captured(_) => galaxy_flight.position.as_vec3().try_normalize(),
                };
                galaxy_flight.update(dt, input, jump, down, mouse_delta, sprint, keys, up);
                // captured flight turns with the planet only close to it (galaxy::co_rotation)
                if let FlightFrame::Captured(i) = *flight_frame {
                    let p = &galaxy.planets[i];
                    let distance = (galaxy_flight.position.length() / p.radius as f64) as f32;
                    let q =
                        crate::galaxy::counter_spin(p.spin_rate() as f32, distance, dt).as_dquat();
                    galaxy_flight.position = q * galaxy_flight.position;
                    galaxy_flight.velocity = q * galaxy_flight.velocity;
                    galaxy_flight.rotation = (q.as_quat() * galaxy_flight.rotation).normalize();
                }
                // the star's heat zone pushes free flight back out and can't be passed (galaxy.rs);
                // planets orbit outside it, so captured flight never meets it
                if *flight_frame == FlightFrame::Free {
                    let star = galaxy.star.position();
                    let r = galaxy.star.radius;
                    galaxy_flight.velocity = crate::galaxy::star_heat_push(
                        galaxy_flight.velocity,
                        galaxy_flight.position - star,
                        r,
                        dt,
                    );
                    let kept =
                        crate::galaxy::keep_off_star(galaxy_flight.position - star, r) + star;
                    if kept != galaxy_flight.position {
                        // at the limit: no further inward speed
                        let out = (kept - star).normalize();
                        let inward = galaxy_flight.velocity.dot(out).min(0.0);
                        galaxy_flight.velocity -= out * inward;
                        galaxy_flight.position = kept;
                    }
                }
                let t = clock.elapsed().as_secs_f64();
                // capture by / release from a planet's frame (galaxy.rs, CAPTURE_RADII)
                let next = crate::galaxy::next_flight_frame(
                    galaxy,
                    *flight_frame,
                    galaxy_flight.position,
                    t,
                );
                if next != *flight_frame {
                    galaxy_flight.change_frame(*flight_frame, next, galaxy, t);
                    match next {
                        FlightFrame::Captured(i) => println!("Entering the orbit of #{}", i + 1),
                        FlightFrame::Free => println!("Leaving orbit"),
                    }
                    *flight_frame = next;
                }
                if let FlightFrame::Captured(i) = *flight_frame {
                    // the impostor is all there is until the voxel world is ready: don't sink into it
                    galaxy_flight.position = crate::landing::keep_above_planet(
                        galaxy_flight.position,
                        galaxy.planets[i].radius as f64,
                    );
                    // capture bakes the planet's voxel world in the background, unless it's loaded
                    if *loaded_world != i && *baking != Some(i) {
                        // the stored edits stay in the store until this bake is installed, so a
                        // dropped bake loses nothing
                        *bake_rx = Some(start_bake(galaxy, i, edits_by_planet.get(&i).cloned()));
                        *baking = Some(i);
                    }
                    // once loaded, the voxel engine streams from the flight's planet-frame position,
                    // so its meshes are ready by the time the landing handover needs them
                    if *loaded_world == i {
                        renderer.update_view(galaxy_flight.position.as_vec3(), planet);
                        if !*voxel_ready_announced && renderer.view_covered() {
                            println!("Voxel world for #{} ready", i + 1);
                            *voxel_ready_announced = true;
                        }
                        let distance_radii =
                            galaxy_flight.position.length() / galaxy.planets[i].radius as f64;
                        if crate::landing::should_land(distance_radii, renderer.view_covered()) {
                            land_on = Some(i);
                        }
                    }
                }
            }
        }

        if let Some(i) = land_on {
            // galaxy flight -> voxel engine: same eye, same view direction and roll, in fly mode
            let (position, rotation, cam_pitch, cam_roll) = crate::landing::player_pose_from_flight(
                galaxy_flight.position.as_vec3(),
                galaxy_flight.rotation,
            );
            player.position = position;
            player.rotation = rotation;
            player.cam_pitch = cam_pitch;
            player.cam_roll = cam_roll;
            // both sides live in the planet frame: the flight carries on at its speed
            player.velocity = galaxy_flight.velocity.as_vec3();
            player.roll_rate = galaxy_flight.roll_rate; // a roll in progress carries on
            player.landing = false;
            player.taking_off = None;
            player.handover_altitude =
                Some(crate::landing::handover_altitude(galaxy.planets[i].radius));
            player.spin_rate = galaxy.planets[i].spin_rate() as f32;
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
            // voxel engine -> captured galaxy flight: same eye, same view direction and roll, same velocity
            *galaxy_flight = crate::landing::liftoff_flight(
                player.position,
                player.rotation,
                player.cam_pitch,
                player.cam_roll,
                player.velocity,
            );
            galaxy_flight.roll_rate = player.roll_rate; // a roll in progress carries on
            *flight_frame = FlightFrame::Captured(i);
            player.landing = false;
            *mode = GameMode::Galaxy;
            enter_galaxy_controls(controller, console);
            println!("Liftoff into the orbit of #{}", i + 1);
        }

        // a background bake finished: install it unless the player went home, landed or got
        // captured by another planet meanwhile
        let finished = bake_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(baked) = finished {
            *bake_rx = None;
            *baking = None;
            if bake_still_wanted(*mode, *flight_frame, baked.index) {
                // the engine streams in the planet frame; free flight is in galaxy space
                let local = match *flight_frame {
                    FlightFrame::Captured(_) => galaxy_flight.position,
                    FlightFrame::Free => galaxy.planets[baked.index]
                        .to_planet_frame(galaxy_flight.position, clock.elapsed().as_secs_f64()),
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

        // the galaxy shows the installed planet's near impostor: rebuild it in the background when the
        // planet was edited since it was built (after liftoff, or leaving by /galaxy goto)
        if *mode == GameMode::Galaxy
            && impostor_rx.is_none()
            && near_impostor_outdated(*impostor_generation, planet)
        {
            let (tx, rx) = std::sync::mpsc::channel();
            let (index, generation, data) =
                (*loaded_world, planet.edit_generation(), planet.clone());
            std::thread::spawn(move || {
                let (v, i) = crate::galaxy_terrain::near_impostor_mesh(&data);
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
        if let Some(fp) = console.view_request.take() {
            controller.first_person = fp;
            controller.cam_dist = if fp { 40.0 } else { 100.0 };
        }
        if let Some(request) = console.galaxy_request.take() {
            use crate::cmd::GalaxyRequest;
            let say = |console: &mut Console, text: &str| {
                console.log(text, [1.0, 1.0, 1.0]);
                println!("{text}");
            };
            match request {
                GalaxyRequest::Home => {
                    // a running bake was for wherever the player was flying: drop it
                    *bake_rx = None;
                    *baking = None;
                    let i = *start_planet;
                    if *loaded_world != i {
                        let baked = bake_planet(galaxy, i, edits_by_planet.remove(&i));
                        let generation = baked.impostor_generation;
                        // the time after the bake: the planet keeps turning while it bakes
                        let t = clock.elapsed().as_secs_f64();
                        let spawn = noon_spawn(&baked.data, galaxy, i, t);
                        stash_edits(edits_by_planet, Some(*loaded_world), i, planet);
                        install_world(renderer, controller, planet, baked, spawn);
                        *impostor_generation = generation;
                        *loaded_world = i;
                    }
                    let t = clock.elapsed().as_secs_f64();
                    player.spawn(noon_spawn(planet, galaxy, i, t));
                    player.handover_altitude =
                        Some(crate::landing::handover_altitude(galaxy.planets[i].radius));
                    player.spin_rate = galaxy.planets[i].spin_rate() as f32;
                    controller.fly_mode = false;
                    *flight_frame = FlightFrame::Free;
                    *mode = GameMode::Planet;
                    say(
                        console,
                        &format!("Back on #{} {}.", i + 1, planet.planet_type.def().name),
                    );
                }
                GalaxyRequest::Add {
                    planet_type,
                    radius,
                } => match galaxy.add_planet(planet_type, radius) {
                    Ok(i) => say(
                        console,
                        &format!(
                            "Added #{n} {} (radius {radius}). /galaxy goto {n} 3 to visit.",
                            planet_type.def().name,
                            n = i + 1
                        ),
                    ),
                    Err(message) => console.log(message, [1.0, 0.5, 0.0]),
                },
                GalaxyRequest::Goto { planet: n, radii } => match galaxy.planets.get(n - 1) {
                    None => console.log(
                        &format!(
                            "No planet #{n}: this galaxy has 1..={}",
                            galaxy.planets.len()
                        ),
                        [1.0, 0.5, 0.0],
                    ),
                    Some(target) => {
                        let t = clock.elapsed().as_secs_f64();
                        // over the day side, looking at the planet's centre (roll levels out by itself)
                        let dir = target.sun_dir_in_planet_frame(galaxy.star.position(), t);
                        *galaxy_flight = crate::galaxy::GalaxyFlight::new(
                            (dir * radii * target.radius).as_dvec3(),
                        );
                        galaxy_flight.rotation =
                            glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, -dir);
                        *flight_frame = FlightFrame::Captured(n - 1);
                        *mode = GameMode::Galaxy;
                        enter_galaxy_controls(controller, console);
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
        const COMMAND_TRIGGER: &str = "/tmp/voxanet_cmd.trigger";
        if let Ok(cmd) = std::fs::read_to_string(COMMAND_TRIGGER) {
            let _ = std::fs::remove_file(COMMAND_TRIGGER);
            console.exec(cmd.trim(), player);
        }

        if let Some(on) = console.hw_shadows_request.take() {
            let active = renderer.set_hw_shadows(on);
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
        // keyboard input, and in galaxy mode it isn't drawn, so an open console is easy to miss)
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
                        controller.toggle_mouse_release();
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
                            PhysicalKey::Code(KeyCode::Backquote) => console.toggle(),
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
                    console.toggle();
                    return;
                }
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
                    if let Some(id) = controller.cursor_id {
                        if button == MouseButton::Middle {
                            // pick the targeted block's type for placing (the bedrock core isn't placeable)
                            if let Some(ty) = planet.block_type(id).filter(|ty| {
                                crate::material::placeable(&planet.planet_type.def().palette)
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
                                let result = if crate::physics::Physics::overlaps_block(
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

            // planet-only keys: in galaxy mode these would silently regenerate the hidden planet. First
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
                            crate::landing::over_damaging_liquid(planet, player.position);
                        match crate::landing::f_action(
                            controller.fly_mode,
                            player.landing,
                            player.taking_off.is_some(),
                            over_lava,
                        ) {
                            crate::landing::FAction::StartLanding => {
                                player.landing = true;
                                renderer.show_status("Landing...");
                            }
                            crate::landing::FAction::CancelLanding => {
                                player.landing = false;
                                renderer.show_status("Landing cancelled.");
                            }
                            crate::landing::FAction::RefuseLanding => {
                                renderer.show_status("Can't land here: lava below.");
                            }
                            crate::landing::FAction::TakeOff => {
                                controller.fly_mode = true;
                                player.landing = false;
                                let layer =
                                    crate::landing::takeoff_target_layer(planet, player.position);
                                player.taking_off =
                                    Some(crate::gen::CoordSystem::get_layer_radius(
                                        layer,
                                        planet.resolution,
                                    ));
                                player.velocity = glam::Vec3::ZERO;
                                renderer.show_status("Taking off...");
                            }
                            crate::landing::FAction::StopTakeOff => {
                                player.taking_off = None;
                                renderer.show_status("Hovering.");
                            }
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let t = self.clock.elapsed().as_secs_f64();
                match &self.mode {
                    GameMode::Planet => {
                        let i = self.loaded_world;
                        let sun = self.galaxy.planets[i]
                            .sun_dir_in_planet_frame(self.galaxy.star.position(), t);
                        // the galaxy behind the voxel world, seen from exactly the planet camera
                        use crate::galaxy_render::{Backdrop, GalaxyCamera, GalaxyContent};
                        let (eye, cam_rot) = controller.camera_pose(player);
                        let fov_y = controller.fov_y();
                        let camera = GalaxyCamera::on_galaxy_planet(
                            &self.galaxy.planets[i],
                            eye,
                            cam_rot,
                            fov_y,
                            t,
                        );
                        let content = GalaxyContent::AllButPlanet(i);
                        // the galaxy's impostor fading out over the voxel world after a landing
                        let handover_overlay = self
                            .landed_at
                            .and_then(|at| {
                                crate::landing::handover_overlay_opacity(at.elapsed().as_secs_f32())
                            })
                            .map(|opacity| (i, opacity));
                        let backdrop = Backdrop {
                            galaxy: &self.galaxy,
                            camera,
                            content,
                            t,
                            handover_overlay,
                        };
                        renderer.render(controller, player, planet, console, t, sun, &backdrop)
                    }
                    GameMode::Galaxy => {
                        let camera = crate::galaxy_render::GalaxyCamera::from_flight_in_frame(
                            &self.galaxy_flight,
                            self.flight_frame,
                            &self.galaxy,
                            t,
                        );
                        renderer.render_galaxy(&camera, &self.galaxy, t)
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
                .create_window(Window::default_attributes().with_title("voxanet"))
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
            game.controller.process_mouse_motion(delta);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(game) = &mut self.game else { return };
        game.tick();
        game.renderer.window.request_redraw();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mined_planet() -> PlanetData {
        let mut planet = PlanetData::new(32);
        let h = planet.terrain.get_height(0, 3, 3);
        planet
            .remove_block(crate::common::BlockId {
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
            .remove_block(crate::common::BlockId {
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
            GameMode::Galaxy,
            FlightFrame::Captured(2),
            2
        ));
        assert!(!bake_still_wanted(
            GameMode::Galaxy,
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
        assert!(bake_still_wanted(GameMode::Galaxy, FlightFrame::Free, 2));
    }
}
