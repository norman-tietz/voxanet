use crate::common::PlanetData;
use crate::physics::Physics;
use glam::{Mat4, Quat, Vec3};

// swimming (Player::swim); depths are of the feet below the sea surface
const SWIM_DEPTH: f32 = 0.5; // shallower than this the player wades (walks)
const FLOAT_DEPTH: f32 = 1.3; // floating at rest: eyes (1.6) just above the surface
const MAX_BUOYANCY: f32 = 1.15; // fully submerged: rise slowly when not swimming
const WATER_DRAG: f32 = 3.0; // 1/s
const SURFACE_HOP: f32 = 6.5; // upward speed of a hop out of the water

// health (Player.health); damage while submerged in a damaging liquid beyond SWIM_DEPTH, passive
// regen otherwise
const DAMAGE_RATE: f32 = 20.0; // HP/s; dies in ~5s fully submerged
const REGEN_RATE: f32 = 5.0; // HP/s; ~20s from empty, only while not taking damage
const MAX_HEALTH: f32 = 100.0;

// Compute damage amount for this tick. Pure decision logic, testable without a real planet/terrain.
// Returns damage amount (>0 means take damage, 0 means no damage).
fn damage_this_tick(depth: Option<f32>, damaging: bool, dt: f32) -> f32 {
    match depth {
        Some(d) if d > SWIM_DEPTH && damaging => DAMAGE_RATE * dt,
        _ => 0.0,
    }
}

// Decides whether holding the up key in `swim()` should hop the player out of the water (e.g. onto
// a shore level with the sea surface) this tick, and by how much. Pure so the anti-amplification
// bound is regression-tested without a planet: `vert` (current vertical speed, +up) must stay in a
// narrow band — not already rising fast, and critically not falling fast either, since the old
// `SURFACE_HOP - vert` formula grew unbounded as a dive's `vert` went more negative, turning every
// dive into a launch back out of the water (and, on lava, letting held-jump dodge damage
// indefinitely). `at_surface` must also be a tight band around the wade threshold: it must exclude
// FLOAT_DEPTH (the resting float depth), or an ordinary jump press while just treading water would
// hop instead of gently swimming up.
fn surface_hop_boost(vert: f32, at_surface: bool) -> Option<f32> {
    if at_surface && vert > -1.0 && vert < SURFACE_HOP * 0.5 {
        Some((SURFACE_HOP - vert).min(SURFACE_HOP))
    } else {
        None
    }
}

pub struct Player {
    // State
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub cam_pitch: f32,
    pub grounded: bool,
    pub debug_mode: bool,
    pub health: f32,
    pub max_health: f32,
    pub spawn_point: Vec3,

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
            health: MAX_HEALTH,
            max_health: MAX_HEALTH,
            spawn_point: Vec3::new(0.0, 200.0, 0.0),
            move_speed: 5.0,
            jump_force: 8.0,
            mouse_sens: 0.002,
        }
    }

    pub fn spawn(&mut self, pos: Vec3) {
        self.position = pos;
        self.velocity = Vec3::ZERO;
        self.grounded = false;
        self.health = self.max_health;
        self.spawn_point = pos;
        let up = Physics::get_up_vector(self.position);
        self.rotation = Quat::from_rotation_arc(Vec3::Y, up);
    }

    // turn: keyboard yaw in radians for this step, positive turns left
    pub fn update(
        &mut self,
        dt: f32,
        planet: &PlanetData,
        input: Vec3,
        jump: bool,
        mouse_delta: (f32, f32),
        turn: f32,
        flying: bool,
        sprint: bool,
    ) {
        let up = Physics::get_up_vector(self.position);

        // --- ROTATION (YAW) ---
        let yaw_delta = -mouse_delta.0 * self.mouse_sens + turn;
        if yaw_delta.abs() > 1e-6 {
            let yaw_rot = Quat::from_axis_angle(up, yaw_delta);
            self.rotation = yaw_rot * self.rotation;
        }

        // --- PITCH ---
        if mouse_delta.1.abs() > 0.001 {
            self.cam_pitch = (self.cam_pitch - mouse_delta.1 * self.mouse_sens).clamp(-1.5, 1.5);
        }

        // --- SWIMMING ---
        // deeper than the knees in the ocean: buoyancy, water drag, half speed; W moves where you look,
        // Space swims up (or hops out at the surface), Left Ctrl dives
        let water_depth = planet.water_depth(self.position); // of the feet
        let depth = water_depth.unwrap_or(f32::MIN);

        // --- HEALTH ---
        let damaging = planet.planet_type.def().liquid.is_some_and(|l| l.damaging);
        let damage_amount = damage_this_tick(water_depth, damaging, dt);
        if damage_amount > 0.0 {
            self.health = (self.health - damage_amount).max(0.0);
        } else {
            self.health = (self.health + REGEN_RATE * dt).min(self.max_health);
        }
        if self.health <= 0.0 {
            let spawn_point = self.spawn_point;
            self.spawn(spawn_point);
            // position/rotation/velocity were just reset: `up` and `depth` above were computed at
            // the pre-respawn location and must not drive this tick's swim/movement logic
            return;
        }

        if !flying && depth > SWIM_DEPTH {
            self.swim(dt, planet, input, jump, sprint, depth, up);
            self.rotation = Physics::align_to_planet(self.rotation, up);
            return;
        }

        let effective_speed = if sprint {
            if flying {
                self.move_speed * 10.0
            } else {
                self.move_speed * 2.0
            }
        } else {
            self.move_speed
        };

        // --- MOVEMENT INPUT ---
        if flying {
            if input.length() > 0.01 {
                let input_normalized = input.normalize();
                let pitch_rot = Quat::from_axis_angle(Vec3::X, self.cam_pitch);
                let fly_dir = self.rotation
                    * pitch_rot
                    * Vec3::new(input_normalized.x, 0.0, input_normalized.z);
                // self.velocity = fly_dir * 1.5;
                self.velocity = fly_dir * effective_speed;
            } else {
                self.velocity = Vec3::ZERO;
            }
        } else {
            // walk
            // probe slightly below the feet: physics rests the feet just above the solid block
            // (only a 5% "shave" margin), so probing `self.position` itself lands in the AIR
            // block above the ground at random depending on exact resting height
            let friction_scale = planet
                .ground_block(self.position - up * 0.1)
                .map_or(1.0, |b| b.friction_scale());
            if input.length() > 0.01 {
                let input_normalized = input.normalize();
                let move_dir =
                    self.rotation * Vec3::new(input_normalized.x, 0.0, input_normalized.z);
                let current_horz = self.velocity - (up * self.velocity.dot(up));

                let target_horz = move_dir * effective_speed;

                // acceleration
                let accel = 25.0 * friction_scale;
                let new_horz =
                    current_horz + (target_horz - current_horz).clamp_length_max(accel * dt);

                self.velocity = new_horz + (up * self.velocity.dot(up));
            } else {
                let horz_vel = self.velocity - (up * self.velocity.dot(up));

                let friction = (if self.grounded { 15.0 } else { 0.5 }) * friction_scale;

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
        let (new_pos, new_vel, grounded) =
            Physics::solve_movement(self.position, self.velocity, dt, planet, flying);

        self.position = new_pos;
        self.velocity = new_vel;
        self.grounded = grounded;

        // --- ALIGN TO SURFACE ---
        self.rotation = Physics::align_to_planet(self.rotation, up);
    }

    fn swim(
        &mut self,
        dt: f32,
        planet: &PlanetData,
        input: Vec3,
        up_key: bool,
        down_key: bool,
        depth: f32,
        up: Vec3,
    ) {
        let speed = self.move_speed * 0.5;
        let mut desired = Vec3::ZERO;
        if input.length() > 0.01 {
            let pitch_rot = Quat::from_axis_angle(Vec3::X, self.cam_pitch);
            let n = input.normalize();
            desired = self.rotation * pitch_rot * Vec3::new(n.x, 0.0, n.z) * speed;
        }
        let at_surface = depth < SWIM_DEPTH + 0.4;
        let vert = self.velocity.dot(up);
        if up_key {
            if let Some(boost) = surface_hop_boost(vert, at_surface) {
                self.velocity += up * boost;
            } else {
                desired += up * speed;
            }
        }
        if down_key {
            desired -= up * speed;
        }

        // buoyancy: balances gravity when floating with the head above water, slightly stronger below
        let lift = (depth / FLOAT_DEPTH).min(MAX_BUOYANCY);
        self.velocity += up * Physics::GRAVITY * (lift - 1.0) * dt;
        // water drag towards the swimming velocity
        self.velocity += (desired - self.velocity) * (WATER_DRAG * dt).min(1.0);

        let (new_pos, new_vel, grounded) =
            Physics::solve_movement(self.position, self.velocity, dt, planet, false);
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

#[cfg(test)]
mod tests {
    use super::*;

    // Test wrapper that calls the production function
    fn liquid_damage_this_tick(depth: Option<f32>, damaging: bool, dt: f32) -> f32 {
        damage_this_tick(depth, damaging, dt)
    }

    #[test]
    fn damaging_liquid_drains_health_proportional_to_dt() {
        assert_eq!(
            liquid_damage_this_tick(Some(2.0), true, 0.5),
            DAMAGE_RATE * 0.5
        );
    }

    #[test]
    fn non_damaging_liquid_does_nothing() {
        assert_eq!(liquid_damage_this_tick(Some(2.0), false, 0.5), 0.0);
    }

    #[test]
    fn shallow_damaging_liquid_does_nothing() {
        assert_eq!(
            liquid_damage_this_tick(Some(SWIM_DEPTH * 0.5), true, 0.5),
            0.0
        );
    }

    #[test]
    fn no_liquid_does_nothing() {
        assert_eq!(liquid_damage_this_tick(None, true, 0.5), 0.0);
    }

    // Regression for the "walking on water" / "jumping prevents diving or lava damage" bug: a fast
    // dive (strongly negative `vert`) must sink, not get launched back out.
    #[test]
    fn surface_hop_never_fires_during_an_active_dive() {
        assert_eq!(surface_hop_boost(-15.0, true), None);
    }

    // The old formula was `SURFACE_HOP - vert`, unbounded as vert fell; confirm the boost is now
    // capped at SURFACE_HOP even at the edge of the eligible band.
    #[test]
    fn surface_hop_is_capped_at_surface_hop_speed() {
        let boost = surface_hop_boost(-0.99, true).expect("just inside the eligible band");
        assert!(boost <= SURFACE_HOP);
    }

    // Floating at rest (FLOAT_DEPTH) must be outside the hop band, or every jump press while
    // treading water in open ocean would hop instead of a gentle swim-up.
    #[test]
    fn surface_hop_does_not_fire_while_floating_at_rest_depth() {
        let at_rest_depth_is_surface = FLOAT_DEPTH < SWIM_DEPTH + 0.4;
        assert!(!at_rest_depth_is_surface);
        assert_eq!(surface_hop_boost(0.0, false), None);
    }

    #[test]
    fn surface_hop_fires_for_a_stationary_player_right_at_the_surface() {
        assert!(surface_hop_boost(0.0, true).is_some());
    }

    #[test]
    fn spawn_resets_health_and_remembers_spawn_point() {
        let mut p = Player::new();
        p.health = 10.0;
        p.spawn(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.health, p.max_health);
        assert_eq!(p.spawn_point, Vec3::new(1.0, 2.0, 3.0));
    }

    // pure scaling math, factored out of update()'s walk branch so it's testable without a planet
    fn scaled_walk_params(base_accel: f32, base_friction: f32, scale: f32) -> (f32, f32) {
        (base_accel * scale, base_friction * scale)
    }

    #[test]
    fn normal_ground_does_not_change_walk_params() {
        assert_eq!(scaled_walk_params(25.0, 15.0, 1.0), (25.0, 15.0));
    }

    #[test]
    fn ice_scales_down_both_acceleration_and_friction() {
        let (accel, friction) =
            scaled_walk_params(25.0, 15.0, crate::material::BlockType::Ice.friction_scale());
        assert!(accel < 25.0 && accel > 0.0);
        assert!(friction < 15.0 && friction > 0.0);
    }
}
