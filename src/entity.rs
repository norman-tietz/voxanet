use glam::{Vec3, Quat, Mat4};
use crate::physics::Physics;
use crate::common::PlanetData;

// swimming (Player::swim); depths are of the feet below the sea surface
const SWIM_DEPTH: f32 = 0.5;     // shallower than this the player wades (walks)
const FLOAT_DEPTH: f32 = 1.3;    // floating at rest: eyes (1.6) just above the surface
const MAX_BUOYANCY: f32 = 1.15;  // fully submerged: rise slowly when not swimming
const WATER_DRAG: f32 = 3.0;     // 1/s
const SURFACE_HOP: f32 = 6.5;    // upward speed of a hop out of the water

pub struct Player {
    // State
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat, 
    pub cam_pitch: f32, 
    pub grounded: bool,
    pub debug_mode: bool,

    // Configuration
    pub move_speed: f32, 
    pub jump_force: f32, 
    pub mouse_sens: f32,
}

impl Player {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 200.0, 0.0), 
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            cam_pitch: 0.0,
            grounded: false,
            debug_mode: false, 
            move_speed: 5.0,
            jump_force: 8.0,     
            mouse_sens: 0.002,   
        }
    }

    pub fn spawn(&mut self, pos: Vec3) {
        self.position = pos;
        self.velocity = Vec3::ZERO;
        self.grounded = false;
        let up = Physics::get_up_vector(self.position);
        self.rotation = Quat::from_rotation_arc(Vec3::Y, up);
    }

    // turn: keyboard yaw in radians for this step, positive turns left
    pub fn update(&mut self, dt: f32, planet: &PlanetData, input: Vec3, jump: bool, mouse_delta: (f32, f32), turn: f32, flying: bool, sprint: bool) {
        let up = Physics::get_up_vector(self.position);
        
        // --- ROTATION (YAW) ---
        let yaw_delta = -mouse_delta.0 * self.mouse_sens + turn;
        if yaw_delta.abs() > 1e-6 {
            let yaw_rot = Quat::from_axis_angle(up, yaw_delta);
            self.rotation = yaw_rot * self.rotation;
        }
        
        // --- PITCH ---
        if mouse_delta.1.abs() > 0.001 {
            self.cam_pitch = (self.cam_pitch - mouse_delta.1 * self.mouse_sens)
                .clamp(-1.5, 1.5);
        }

        
        // --- SWIMMING ---
        // deeper than the knees in the ocean: buoyancy, water drag, half speed; W moves where you look,
        // Space swims up (or hops out at the surface), Left Ctrl dives
        let depth = planet.water_depth(self.position).unwrap_or(f32::MIN); // of the feet
        if !flying && depth > SWIM_DEPTH {
            self.swim(dt, planet, input, jump, sprint, depth, up);
            self.rotation = Physics::align_to_planet(self.rotation, up);
            return;
        }

        let effective_speed = if sprint {
            if flying { self.move_speed * 10.0 } else { self.move_speed * 2.0 }
        } else {
            self.move_speed
        };
        
        // --- MOVEMENT INPUT ---
        if flying {
            
            if input.length() > 0.01 {
                let input_normalized = input.normalize();
                let pitch_rot = Quat::from_axis_angle(Vec3::X, self.cam_pitch);
                let fly_dir = self.rotation * pitch_rot * Vec3::new(input_normalized.x, 0.0, input_normalized.z);                
                // self.velocity = fly_dir * 1.5;
                self.velocity = fly_dir * effective_speed;
            } else {                
                self.velocity = Vec3::ZERO;
            }
        } else {
            // walk
            if input.length() > 0.01 {
                let input_normalized = input.normalize();
                let move_dir = self.rotation * Vec3::new(input_normalized.x, 0.0, input_normalized.z);
                let current_horz = self.velocity - (up * self.velocity.dot(up));
                
                
                let target_horz = move_dir * effective_speed;
                
                // acceleration
                let accel = 25.0;
                let new_horz = current_horz + (target_horz - current_horz).clamp_length_max(accel * dt);
                
                self.velocity = new_horz + (up * self.velocity.dot(up));
            } else {
                
                let horz_vel = self.velocity - (up * self.velocity.dot(up));
                
                let friction = if self.grounded { 15.0 } else { 0.5 }; 
                
                let reduced = horz_vel * (1.0 - friction * dt).max(0.0);
                self.velocity = reduced + (up * self.velocity.dot(up));
            }
        }
        
        // --- JUMP ---
        if jump && self.grounded && !flying {
            self.velocity += up * self.jump_force;
            self.grounded = false;
        }
        
        // --- GRAVITY ---
        if !flying {
            self.velocity -= up * Physics::GRAVITY * dt;
        }
        
        // --- PHYSICS SOLVE ---
        let (new_pos, new_vel, grounded) = Physics::solve_movement(
            self.position, 
            self.velocity, 
            dt, 
            planet, 
            flying
        );
        
        self.position = new_pos;
        self.velocity = new_vel;
        self.grounded = grounded;
        
        // --- ALIGN TO SURFACE ---
        self.rotation = Physics::align_to_planet(self.rotation, up);
    }

    fn swim(&mut self, dt: f32, planet: &PlanetData, input: Vec3, up_key: bool, down_key: bool, depth: f32, up: Vec3) {
        let speed = self.move_speed * 0.5;
        let mut desired = Vec3::ZERO;
        if input.length() > 0.01 {
            let pitch_rot = Quat::from_axis_angle(Vec3::X, self.cam_pitch);
            let n = input.normalize();
            desired = self.rotation * pitch_rot * Vec3::new(n.x, 0.0, n.z) * speed;
        }
        let at_surface = depth < FLOAT_DEPTH + 0.4;
        if up_key && at_surface {
            // hop out of the water, e.g. onto a beach that is level with the sea surface
            let vert = self.velocity.dot(up);
            if vert < SURFACE_HOP * 0.5 { self.velocity += up * (SURFACE_HOP - vert); }
        } else if up_key {
            desired += up * speed;
        }
        if down_key { desired -= up * speed; }

        // buoyancy: balances gravity when floating with the head above water, slightly stronger below
        let lift = (depth / FLOAT_DEPTH).min(MAX_BUOYANCY);
        self.velocity += up * Physics::GRAVITY * (lift - 1.0) * dt;
        // water drag towards the swimming velocity
        self.velocity += (desired - self.velocity) * (WATER_DRAG * dt).min(1.0);

        let (new_pos, new_vel, grounded) = Physics::solve_movement(self.position, self.velocity, dt, planet, false);
        self.position = new_pos;
        self.velocity = new_vel;
        self.grounded = grounded;
    }

    pub fn get_model_matrix(&self) -> Mat4 {
        Mat4::from_translation(self.position) * Mat4::from_quat(self.rotation)
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let up = Physics::get_up_vector(self.position);
        let cam_pos = self.position + (up * Physics::EYE_HEIGHT); 
        
        let pitch_rot = Quat::from_axis_angle(Vec3::X, self.cam_pitch);
        let final_rot = self.rotation * pitch_rot;
        
        let forward = final_rot * Vec3::NEG_Z; 
        
        glam::camera::rh::view::look_at_mat4(cam_pos, cam_pos + forward, up)
    }
}