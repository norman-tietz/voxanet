//engine controller

use crate::common::*;
use crate::entity::Player;
use crate::gen::CoordSystem;
use crate::material::BlockType;
use crate::physics::Physics;
use glam::{Mat4, Quat, Vec2, Vec3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// keyboard turning speed (Q/E), radians per second
const TURN_SPEED: f32 = 2.0;

// flight keys as axes from -1 to 1: A/D roll (A left), Q/E turn (Q left). Planet fly mode in first
// person, orbit and galaxy flight; their rates are eased (smoothing.rs)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlightKeys {
    pub roll: f32,
    pub turn: f32,
}

fn key_axis(positive: bool, negative: bool) -> f32 {
    (positive as i32 - negative as i32) as f32
}

pub struct Controller {
    pub cam_dist: f32,

    // input State
    pub mouse_pos: Vec2,
    pub mouse_delta: (f32, f32),
    pub is_orbiting: bool,
    pub is_wireframe: bool,
    pub show_collisions: bool,
    pub fly_mode: bool,
    pub sprint: bool,
    pub move_down: bool, // Left Shift: fly down, or dive while swimming
    pub freeze_culling: bool,
    pub cursor_id: Option<BlockId>,

    pub first_person: bool,
    pub mouse_released: bool, // Escape toggles this: free the cursor without leaving first person

    keys: [bool; 7],               // W, A, S, D, Space, Q, E
    pub selected_block: BlockType, // what right-click places
}

impl Controller {
    pub fn new() -> Self {
        Self {
            cam_dist: 200.0,
            mouse_pos: Vec2::ZERO,
            mouse_delta: (0.0, 0.0),
            is_orbiting: false,
            cursor_id: None,
            is_wireframe: false,
            show_collisions: false,
            fly_mode: false,
            freeze_culling: false,
            sprint: false,
            move_down: false,
            first_person: true,
            mouse_released: false,
            keys: [false; 7],
            selected_block: BlockType::Dirt,
        }
    }

    // gathers raw per-tick input, independent of which physics mode consumes it (planet-surface
    // Player::update or galaxy_mode's GalaxyFlight::update); resets mouse_delta as a side effect,
    // exactly as update_player did before this was extracted, so planet-mode behavior is unchanged
    pub fn raw_input(&mut self) -> (Vec3, bool, bool, bool, (f32, f32)) {
        let mut input = Vec3::ZERO;
        if self.keys[0] {
            input.z -= 1.0;
        } // W
        if self.keys[1] {
            input.x -= 1.0;
        } // A
        if self.keys[2] {
            input.z += 1.0;
        } // S
        if self.keys[3] {
            input.x += 1.0;
        } // D
        let jump = self.keys[4]; // space

        let rotation_delta = if self.first_person && !self.mouse_released {
            self.mouse_delta
        } else {
            (0.0, 0.0)
        };
        self.mouse_delta = (0.0, 0.0);

        (input, jump, self.move_down, self.sprint, rotation_delta)
    }

    // movement input and flight keys: in flight A/D roll and Q/E turn instead of A/D moving sideways
    fn movement_and_keys(&self, flight: bool) -> (Vec3, FlightKeys) {
        let input = Vec3::new(
            if flight {
                0.0
            } else {
                key_axis(self.keys[3], self.keys[1])
            },
            0.0,
            key_axis(self.keys[2], self.keys[0]),
        );
        let keys = if flight {
            FlightKeys {
                roll: key_axis(self.keys[1], self.keys[3]),
                turn: key_axis(self.keys[5], self.keys[6]),
            }
        } else {
            FlightKeys::default()
        };
        (input, keys)
    }

    // planet mode: flight keys while flying in first person, else walking keys (A/D strafe, Q/E turn)
    pub fn flight_input(&self) -> (Vec3, FlightKeys) {
        self.movement_and_keys(self.fly_mode && self.first_person)
    }

    // galaxy flight (free or in orbit): always flight keys
    pub fn galaxy_flight_input(&self) -> (Vec3, FlightKeys) {
        self.movement_and_keys(true)
    }

    pub fn update_player(&mut self, player: &mut Player, planet: &PlanetData, dt: f32) {
        let (_, jump, down, sprint, rotation_delta) = self.raw_input();
        let (input, keys) = self.flight_input();
        // on foot (and in third person) Q/E turn directly; in flight they're eased flight keys
        let in_flight = self.fly_mode && self.first_person;
        let turn = if in_flight {
            0.0
        } else {
            key_axis(self.keys[5], self.keys[6]) * TURN_SPEED * dt
        };

        let touched_down = player.update(
            dt,
            planet,
            input,
            jump,
            down,
            rotation_delta,
            turn,
            keys,
            self.fly_mode,
            sprint,
        );
        if touched_down {
            self.fly_mode = false; // walking (or swimming) from here
        }
    }

    pub fn get_camera_pos(&self, player: &Player) -> Vec3 {
        if self.first_person {
            // first person: Camera is at player position + eye height
            player.position + (Physics::get_up_vector(player.position) * 1.6)
        } else {
            let up = Physics::get_up_vector(player.position);
            player.position + (up * self.cam_dist)
        }
    }

    // Escape: free the cursor without leaving first person (or capture it again). A global key,
    // handled in main.rs before the console sees input, so it works in every mode.
    pub fn toggle_mouse_release(&mut self) {
        if self.first_person {
            self.mouse_released = !self.mouse_released;
        }
    }

    pub fn process_mouse_motion(&mut self, delta: (f64, f64)) {
        if self.first_person && !self.mouse_released {
            // accumulate raw mouse delta
            self.mouse_delta.0 += delta.0 as f32;
            self.mouse_delta.1 += delta.1 as f32;
        }
    }

    pub fn process_events(
        &mut self,
        event: &WindowEvent,
        _player: &mut Player,
        planet: &PlanetData,
    ) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                let d = new_pos - self.mouse_pos;
                self.mouse_pos = new_pos;
                self.mouse_delta = (d.x, d.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Middle {
                    self.is_orbiting = *state == ElementState::Pressed;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !self.first_person {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                    };

                    self.cam_dist = (self.cam_dist - y * 50.0).clamp(10.0, 10000.0);
                    return true;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                // toggles act on the first press only: held down, the OS repeats the key and they'd flip back and forth
                let toggled = pressed && !event.repeat;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.keys[0] = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.keys[1] = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.keys[2] = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.keys[3] = pressed,
                    PhysicalKey::Code(KeyCode::Space) => self.keys[4] = pressed,
                    PhysicalKey::Code(KeyCode::KeyQ) => self.keys[5] = pressed,
                    PhysicalKey::Code(KeyCode::KeyE) => self.keys[6] = pressed,

                    PhysicalKey::Code(KeyCode::ControlLeft) => self.sprint = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => self.move_down = pressed,

                    PhysicalKey::Code(
                        code @ (KeyCode::Digit1
                        | KeyCode::Digit2
                        | KeyCode::Digit3
                        | KeyCode::Digit4
                        | KeyCode::Digit5),
                    ) if pressed => {
                        let i = match code {
                            KeyCode::Digit1 => 0,
                            KeyCode::Digit2 => 1,
                            KeyCode::Digit3 => 2,
                            KeyCode::Digit4 => 3,
                            _ => 4,
                        };
                        self.selected_block =
                            crate::material::placeable(&planet.planet_type.def().palette)[i];
                        return true;
                    }

                    PhysicalKey::Code(KeyCode::KeyP) if toggled => {
                        if _player.debug_mode {
                            self.is_wireframe = !self.is_wireframe;
                        }
                        return true;
                    }

                    PhysicalKey::Code(KeyCode::KeyO) if toggled => {
                        if _player.debug_mode {
                            self.show_collisions = !self.show_collisions;
                            println!("Show Collisions: {}", self.show_collisions);
                        }
                        return true;
                    }

                    PhysicalKey::Code(KeyCode::Quote) if toggled => {
                        if _player.debug_mode {
                            self.freeze_culling = !self.freeze_culling;
                        }
                        return true;
                    }

                    PhysicalKey::Code(KeyCode::KeyK) if toggled => {
                        self.first_person = !self.first_person;
                        self.mouse_released = false; // always re-enter first person with the mouse captured

                        if self.first_person {
                            self.cam_dist = 40.0;
                        } else {
                            self.cam_dist = 100.0;
                        }
                        return true;
                    }

                    _ => {}
                }
            }
            _ => {}
        }
        false
    }

    // vertical field of view: 80 degrees in first person, 45 in orbit mode for less distortion
    pub fn fov_y(&self) -> f32 {
        let fov_degrees: f32 = if self.first_person { 80.0 } else { 45.0 };
        fov_degrees.to_radians()
    }

    // world -> camera, in the planet frame
    pub fn view_matrix(&self, player: &Player) -> Mat4 {
        if self.first_person {
            player.get_view_matrix()
        } else {
            let up = Physics::get_up_vector(player.position);
            let cam_pos = player.position + (up * self.cam_dist);
            let target = player.position;

            let player_forward = player.rotation * Vec3::NEG_Z;

            glam::camera::rh::view::look_at_mat4(cam_pos, target, player_forward)
        }
    }

    // eye position and orientation (looking down local -Z) of the camera view_matrix describes; the
    // galaxy backdrop is drawn from exactly this pose (Renderer::render)
    pub fn camera_pose(&self, player: &Player) -> (Vec3, Quat) {
        let camera_to_world = self.view_matrix(player).inverse();
        (
            camera_to_world.w_axis.truncate(),
            Quat::from_mat4(&camera_to_world).normalize(),
        )
    }

    pub fn get_matrix(&self, player: &Player, width: f32, height: f32) -> Mat4 {
        // far plane increased to 20,000 for massive zoom out
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fov_y(),
            width / height,
            0.1,
            20000.0,
        );
        proj * self.view_matrix(player)
    }

    pub fn raycast(
        &self,
        player: &Player,
        planet: &PlanetData,
        width: f32,
        height: f32,
        place_mode: bool,
    ) -> Option<(BlockId, f32)> {
        let mvp = self.get_matrix(player, width, height);
        let inv = mvp.inverse();

        let (ndc_x, ndc_y) = if self.first_person {
            (0.0, 0.0)
        } else {
            (
                (2.0 * self.mouse_pos.x / width) - 1.0,
                1.0 - (2.0 * self.mouse_pos.y / height),
            )
        };

        let start = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let end = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (end - start).normalize();

        let mut dist = 0.0;
        let mut last_empty = None;

        let reach = if self.first_person {
            8.0
        } else {
            self.cam_dist + 100.0
        };
        // stop raycast if we hit the absolute math center (radius < 0.5)
        let min_radius = 0.5;

        while dist < reach {
            let p = start + dir * dist;
            if p.length() < min_radius {
                break;
            }

            // since blocks are now approx 1.0 unit thick/wide, 0.25 is a safe step.
            let step = 0.25;

            if let Some(id) = CoordSystem::pos_to_id(p, planet.resolution) {
                let exists = planet.exists(id);
                if place_mode {
                    if exists {
                        return last_empty.map(|i| (i, dist));
                    } else {
                        last_empty = Some(id);
                    }
                } else {
                    if exists {
                        return Some((id, dist));
                    }
                }
            }
            dist += step;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller_with(keys: [bool; 7], fly_mode: bool, first_person: bool) -> Controller {
        let mut c = Controller::new();
        c.keys = keys;
        c.fly_mode = fly_mode;
        c.first_person = first_person;
        c
    }

    // keys: [W, A, S, D, Space, Q, E]
    const A: [bool; 7] = [false, true, false, false, false, false, false];
    const D_AND_Q: [bool; 7] = [false, false, false, true, false, true, false];

    // first-person flight: A/D roll (A left), Q/E turn (Q left), no sideways movement
    #[test]
    fn flight_keys_roll_with_a_d_and_turn_with_q_e() {
        let c = controller_with(A, true, true);
        let (input, keys) = c.flight_input();
        assert_eq!(input.x, 0.0, "A strafed in flight");
        assert_eq!(
            keys,
            FlightKeys {
                roll: 1.0,
                turn: 0.0
            }
        );
        let c = controller_with(D_AND_Q, true, true);
        assert_eq!(
            c.flight_input().1,
            FlightKeys {
                roll: -1.0,
                turn: 1.0
            }
        );
    }

    // walking (and third-person flying): A/D strafe and Q/E turn, no flight keys
    #[test]
    fn on_foot_a_d_strafe_and_q_e_turn() {
        for (fly, first) in [(false, true), (true, false)] {
            let c = controller_with(A, fly, first);
            let (input, keys) = c.flight_input();
            assert_eq!(input.x, -1.0);
            assert_eq!(keys, FlightKeys::default());
        }
    }

    fn player_on_a_slope() -> Player {
        let mut player = Player::new();
        player.spawn(Vec3::new(3.0, 40.0, -5.0));
        player.cam_pitch = 0.3;
        player
    }

    // the galaxy backdrop is drawn from camera_pose(); it must be exactly the camera the voxel
    // engine renders with, in both views, or stars would drift against the terrain
    #[test]
    fn camera_pose_reproduces_the_view_matrix() {
        let player = player_on_a_slope();
        for first_person in [true, false] {
            let mut c = Controller::new();
            c.first_person = first_person;
            let (eye, rot) = c.camera_pose(&player);
            let rebuilt = Mat4::from_rotation_translation(rot, eye).inverse();
            let view = c.view_matrix(&player);
            let max_diff = (rebuilt - view)
                .to_cols_array()
                .iter()
                .fold(0.0f32, |m, d| m.max(d.abs()));
            assert!(
                max_diff < 1e-4,
                "first_person={first_person}: off by {max_diff}"
            );
        }
    }

    #[test]
    fn fov_is_80_degrees_first_person_and_45_third_person() {
        let mut c = Controller::new();
        c.first_person = true;
        assert!((c.fov_y() - 80f32.to_radians()).abs() < 1e-6);
        c.first_person = false;
        assert!((c.fov_y() - 45f32.to_radians()).abs() < 1e-6);
    }
}
