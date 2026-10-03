// engine main.rs

mod biome;
mod cmd;
mod common;
mod controller;
mod deferred;
mod entity;
mod galaxy;
mod galaxy_render;
mod galaxy_terrain;
mod gen;
mod gpu_timer;
mod hw_rt;
mod icosphere;
mod lod_animation;
mod material;
mod noise;
mod physics;
mod renderer;
mod rt_blur;
mod rt_shadow;
mod screenshot;
mod system_diagnostics;

use crate::cmd::Console;
use crate::common::PlanetData;
use crate::controller::Controller;
use crate::entity::Player;
use crate::renderer::Renderer;
use crate::system_diagnostics::SystemDiagnostics;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

// --biome <name> picks the starting planet type (default earth-like); matched case-insensitively
// against each PlanetType's def().name with spaces/hyphens stripped, so "Earth-like", "earthlike" and
// "EARTHLIKE" all match. Unrecognized values fall back to earth-like with a warning.
fn parse_biome_arg() -> crate::biome::PlanetType {
    let normalize = |s: &str| s.to_lowercase().replace(['-', ' '], "");
    let Some(requested) = std::env::args().skip_while(|a| a != "--biome").nth(1) else {
        return crate::biome::PlanetType::EarthLike;
    };
    let wanted = normalize(&requested);
    crate::biome::PlanetType::ALL
        .into_iter()
        .find(|t| normalize(t.def().name) == wanted)
        .unwrap_or_else(|| {
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
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        game: None,
        initial_biome,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    // created in `resumed`, once a window can be opened
    game: Option<Game>,
    initial_biome: crate::biome::PlanetType, // from --biome, see parse_biome_arg()
}

enum GameMode {
    Planet,
    Galaxy { return_point: glam::Vec3 },
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
    galaxy_start_time: Instant,
}

impl Game {
    fn new(window: Arc<Window>, initial_biome: crate::biome::PlanetType) -> Self {
        let renderer = pollster::block_on(Renderer::new(window));
        let mut controller = Controller::new();
        let mut player = Player::new();
        let mut planet = PlanetData::new(49); // Keep high resolution
        planet.switch_planet_type(initial_biome); // no-op if already Earth-like (the default)
        controller.selected_block =
            crate::material::placeable(&planet.planet_type.def().palette)[0];

        let mut console = Console::new();
        console.log("Welcome to voxanet.", [0.0, 1.0, 0.0]);
        console.log("Press ` to open console.", [1.0, 1.0, 1.0]);
        console.log(
            &format!("Planet type: {}", planet.planet_type.def().name),
            [1.0, 1.0, 1.0],
        );

        // initialize player spawn: search for dry land along +Y ("North Pole") first, same safety
        // rule the B key uses, so e.g. --biome volcanic never starts the player inside lava
        let spawn_dir = planet.safe_spawn_direction(glam::Vec3::Y);
        let spawn_h = spawn_radius(&planet, spawn_dir, 10.0);

        player.spawn(spawn_dir * spawn_h);

        Self {
            renderer,
            controller,
            player,
            planet,
            console,
            last_time: Instant::now(),
            current_cursor_locked: false,
            mode: GameMode::Planet,
            galaxy: crate::galaxy::Galaxy::generate(1),
            galaxy_flight: crate::galaxy::GalaxyFlight::new(glam::DVec3::new(0.0, 0.0, 120_000.0)),
            galaxy_start_time: Instant::now(),
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
            galaxy: _,
            galaxy_flight,
            galaxy_start_time: _,
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
            }
            GameMode::Galaxy { .. } => {
                let (input, jump, down, sprint, mouse_delta) = controller.raw_input();
                galaxy_flight.update(dt, input, jump, down, mouse_delta, sprint);
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
        if let Some(enter) = console.galaxy_request.take() {
            match (&*mode, enter) {
                (GameMode::Planet, true) => {
                    *mode = GameMode::Galaxy {
                        return_point: player.position,
                    };
                    *galaxy_flight =
                        crate::galaxy::GalaxyFlight::new(glam::DVec3::new(0.0, 0.0, 120_000.0));
                    // galaxy mode has no Q/E key-yaw, so make sure mouse-look is active even if the
                    // player entered while in third person (Controller::raw_input gates it on
                    // first_person && !mouse_released) — otherwise there's no way to turn at all.
                    controller.first_person = true;
                    controller.mouse_released = false;
                    console.log(
                        "Entered galaxy mode. /galaxy exit to return.",
                        [1.0, 1.0, 1.0],
                    );
                    println!("Entered galaxy mode. /galaxy exit to return.");
                }
                (GameMode::Galaxy { return_point }, false) => {
                    player.position = *return_point;
                    player.velocity = glam::Vec3::ZERO;
                    *mode = GameMode::Planet;
                    console.log("Returned to the planet.", [1.0, 1.0, 1.0]);
                    println!("Returned to the planet.");
                }
                _ => {} // already in the requested mode; no-op
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
                if matches!(self.mode, GameMode::Planet) {
                    if controller.first_person && controller.mouse_released {
                        // recapture takes priority: don't also mine/place on the click that brings the mouse back
                        controller.mouse_released = false;
                        let _ = renderer.window.set_cursor_grab(CursorGrabMode::Locked);
                        renderer.window.set_cursor_visible(false);
                    } else {
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
                                    planet.add_block(place_id, controller.selected_block);
                                    renderer.refresh_neighbors(place_id, planet);
                                }
                            } else if button == MouseButton::Left {
                                planet.remove_block(id);
                                renderer.refresh_neighbors(id, planet);
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
            }

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(KeyCode::F2) = event.physical_key {
                    let ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis();
                    let _ = std::fs::create_dir_all("screenshots");
                    renderer.request_screenshot(format!("screenshots/voxanet_{ms}.png"));
                    renderer.window.request_redraw();
                }
                if let PhysicalKey::Code(KeyCode::KeyB) = event.physical_key {
                    planet.switch_planet_type(planet.planet_type.next());
                    controller.selected_block =
                        crate::material::placeable(&planet.planet_type.def().palette)[0];

                    let current_dir = if player.position.length() > 0.1 {
                        player.position.normalize()
                    } else {
                        glam::Vec3::Y
                    };
                    let spawn_dir = planet.safe_spawn_direction(current_dir);
                    let radius = spawn_radius(planet, spawn_dir, 5.0);

                    player.spawn(spawn_dir * radius);

                    renderer.force_reload_all(planet, player.position);
                    renderer.log_memory(planet);
                    println!("Switched to planet type: {}", planet.planet_type.def().name);
                    renderer.window.request_redraw();
                }
                if let Key::Character(ref s) = event.logical_key {
                    if s == "]" || s == "[" {
                        if s == "]" {
                            planet.resize(true);
                        } else {
                            planet.resize(false);
                        }

                        let current_dir = if player.position.length() > 0.1 {
                            player.position.normalize()
                        } else {
                            glam::Vec3::Y
                        };
                        // same death-loop guard the B key uses: resizing while standing over
                        // damaging liquid must not respawn the player back inside it
                        let spawn_dir = planet.safe_spawn_direction(current_dir);
                        let radius = spawn_radius(planet, spawn_dir, 5.0);

                        player.position = spawn_dir * radius;
                        player.velocity = glam::Vec3::ZERO;

                        renderer.force_reload_all(planet, player.position);
                        renderer.log_memory(planet);
                        renderer.window.request_redraw();
                    }
                }
            }

            WindowEvent::RedrawRequested => match &self.mode {
                GameMode::Planet => renderer.render(controller, player, planet, console),
                GameMode::Galaxy { .. } => {
                    let t = self.galaxy_start_time.elapsed().as_secs_f64();
                    renderer.render_galaxy(&self.galaxy_flight, &self.galaxy, t);
                }
            },
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
        self.game = Some(Game::new(window, self.initial_biome));
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
