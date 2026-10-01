// engine main.rs

mod common;
mod gen;
mod physics;
mod entity;
mod controller;
mod renderer;
mod noise;
mod lod_animation;
mod cmd;
mod rt_shadow;
mod rt_blur;
mod gpu_timer;
mod material;
mod system_diagnostics;



use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};
use winit::keyboard::{Key, PhysicalKey, KeyCode};
use crate::common::PlanetData;
use crate::renderer::Renderer;
use crate::controller::Controller;
use crate::entity::Player;
use crate::cmd::Console;
use crate::system_diagnostics::SystemDiagnostics;
use std::time::Instant;



fn main() {

    SystemDiagnostics::print_startup_info();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { game: None };
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    // created in `resumed`, once a window can be opened
    game: Option<Game>,
}

struct Game {
    renderer: Renderer,
    controller: Controller,
    player: Player,
    planet: PlanetData,
    console: Console,
    last_time: Instant,
    current_mode_first_person: bool,
}

impl Game {
    fn new(window: Arc<Window>) -> Self {
        let renderer = pollster::block_on(Renderer::new(window));
        let controller = Controller::new();
        let mut player = Player::new();
        let planet = PlanetData::new(49); // Keep high resolution

        let mut console = Console::new();
        console.log("Welcome to voxanet.", [0.0, 1.0, 0.0]);
        console.log("Press ` to open console.", [1.0, 1.0, 1.0]);


        // initialize player spawn

        // we query the height at face 0, u=res/2, v=res/2 (roughly the "North Pole" of face 0)
        let center = planet.resolution / 2;
        let ground_level = planet.terrain.get_height(0, center, center);
        let spawn_h = crate::gen::CoordSystem::get_layer_radius(ground_level, planet.resolution) + 10.0;


        player.spawn(glam::Vec3::new(0.0, spawn_h, 0.0));

        Self { renderer, controller, player, planet, console, last_time: Instant::now(), current_mode_first_person: false }
    }

    // runs before every event is handled
    fn tick(&mut self) {
        let Self { renderer, controller, player, planet, console, last_time, current_mode_first_person } = self;

        let now = Instant::now();
        let dt = (now - *last_time).as_secs_f32();
        *last_time = now;

        // cursor locking logic
        if controller.first_person != *current_mode_first_person {
            *current_mode_first_person = controller.first_person;
            if *current_mode_first_person {
                let _ = renderer.window.set_cursor_grab(CursorGrabMode::Locked);
                renderer.window.set_cursor_visible(false);
            } else {
                let _ = renderer.window.set_cursor_grab(CursorGrabMode::None);
                renderer.window.set_cursor_visible(true);
            }
        }

        // physics & player Update (once per tick: movement and turning speeds are per second)
        controller.update_player(player, planet, dt);

        // raycast & cursor Update
        let width = renderer.config.width as f32;
        let height = renderer.config.height as f32;
        let ray_result = controller.raycast(player, planet, width, height, false);
        controller.cursor_id = ray_result.map(|(id, _)| id);

        renderer.update_cursor(planet, controller.cursor_id);
        renderer.update_view(player.position, planet);


        // UPDATE ANIMATION
        console.update_animation(dt);

        // the console takes the keyboard (see window_event) and needs a free mouse cursor
        if console.is_open {
             let _ = renderer.window.set_cursor_grab(CursorGrabMode::None);
             renderer.window.set_cursor_visible(true);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let Self { renderer, controller, player, planet, console, .. } = self;

        // CONSOLE INPUT INTERCEPTION
        if console.is_open {
            match event {
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                     if key_event.state == ElementState::Pressed {
                         match key_event.physical_key {
                             PhysicalKey::Code(KeyCode::Backquote) => console.toggle(),
                             PhysicalKey::Code(KeyCode::Enter) => console.submit(player),
                             PhysicalKey::Code(KeyCode::Backspace) => console.handle_backspace(),
                             _ => {
                                 if let Some(txt) = &key_event.text {
                                     // Append text to console buffer
                                     for c in txt.chars() { console.handle_char(c); }
                                 }
                             }
                         }
                     }
                     return;
                },
                 _ => {}
            }
        }

        if let WindowEvent::KeyboardInput { event: key_event, .. } = &event {
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

            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                let is_right = button == MouseButton::Right;
                if let Some(id) = controller.cursor_id {
                     if is_right {
                         let place_info = controller.raycast(player, planet, renderer.config.width as f32, renderer.config.height as f32, true);
                         if let Some((place_id, _)) = place_info {
                             planet.add_block(place_id, controller.selected_block);
                             renderer.refresh_neighbors(place_id, planet);
                         }
                     } else {
                         planet.remove_block(id);
                         renderer.refresh_neighbors(id, planet);
                     }
                    renderer.window.request_redraw();
                } else {
                    if controller.first_person {
                        let _ = renderer.window.set_cursor_grab(CursorGrabMode::Locked);
                        renderer.window.set_cursor_visible(false);
                    }
                }
            },

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                 if let Key::Character(ref s) = event.logical_key {
                    if s == "]" || s == "[" {
                        if s == "]" { planet.resize(true); }
                        else { planet.resize(false); }

                        let new_res = planet.resolution;
                        let current_dir = if player.position.length() > 0.1 { player.position.normalize() } else { glam::Vec3::Y };
                        let probe_dist = new_res as f32 / 2.0;
                        let dummy_pos = current_dir * probe_dist;

                        let spawn_radius = if let Some(id) = crate::gen::CoordSystem::pos_to_id(dummy_pos, new_res) {
                            let h = planet.terrain.get_height(id.face, id.u, id.v);
                            crate::gen::CoordSystem::get_layer_radius(h, new_res) + 5.0
                        } else {
                            (new_res as f32 / 2.0) + 20.0
                        };

                        player.position = current_dir * spawn_radius;
                        player.velocity = glam::Vec3::ZERO;

                        renderer.force_reload_all(planet, player.position);
                        renderer.log_memory(planet);
                        renderer.window.request_redraw();
                    }
                }
            },

            WindowEvent::RedrawRequested => {
                    renderer.render(controller, player, planet, console);

                },
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.game.is_some() { return; }
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("voxanet")).unwrap());
        self.game = Some(Game::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let Some(game) = &mut self.game else { return };
        game.tick();
        if window_id == game.renderer.window.id() {
            game.window_event(event_loop, event);
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
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
