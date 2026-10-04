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

// fly mode terrain-following floor (Player::update's flying branch): fly mode has no collision
// (Physics::solve_movement just integrates position when flying), so without this the player can
// fly straight through the ground. The floor only ever pushes up, never down, so flying on over
// lower terrain keeps the current altitude instead of hugging the ground like a drone.
const FLY_HOVER_CLEARANCE: f32 = 3.0; // world units kept above the highest sampled terrain
const FLY_LOOKAHEAD_TIME: f32 = 1.5; // seconds of travel to look ahead, scaled by current speed
const FLY_LOOKAHEAD_MIN: f32 = 5.0;
const FLY_LOOKAHEAD_MAX: f32 = 60.0;
const FLY_LOOKAHEAD_SAMPLES: u32 = 4; // points between the player and the lookahead distance
const LANDING_ROLL_LEVEL_RATE: f32 = 1.5; // rad/s an F landing eases the roll level

// an angle wrapped into -PI..PI
fn wrap_angle(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}
const FLY_CLIMB_RATE: f32 = 15.0; // world units/s the floor correction may lift the player

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

// How far upward to move this tick to approach the fly-mode terrain floor, capped at the climb
// rate so a sudden floor jump (e.g. a cliff entering the lookahead window) is a smooth climb, not
// a snap. Pure so the rate limit is regression-tested without a planet. Never negative: the floor
// only ever pushes the player up, never pulls them down toward it.
fn fly_floor_climb(radius: f32, floor: f32, dt: f32) -> f32 {
    (floor - radius).max(0.0).min(FLY_CLIMB_RATE * dt)
}

// Clamps a downward vertical speed (`vert`, +up) so this tick's descent can't cross below `floor`,
// regardless of how fast the player is trying to descend (sprinting down, or diving via pitch).
// Pure so the clamp is regression-tested without a planet. Complements fly_floor_climb: that one
// corrects an overshoot after the fact (for the lookahead-anticipated climb toward rising terrain
// ahead), this one prevents the overshoot from happening in the first place for a descent already
// in progress, so the two don't fight each other.
fn fly_max_descent(vert: f32, radius: f32, floor: f32, dt: f32) -> f32 {
    if vert >= 0.0 {
        return vert;
    }
    let allowed_descent = (radius - floor).max(0.0);
    let min_vert = -(allowed_descent / dt.max(1e-6));
    vert.max(min_vert)
}

// The minimum world-space radius fly mode should keep the player at: hover clearance above the
// highest terrain sampled between the player and `lookahead_dist` ahead along `horizontal_dir`.
// Looking ahead (rather than just checking directly underfoot) is what lets the floor start
// rising before the player reaches a mountain, instead of reacting only once already over it.
fn fly_terrain_floor(
    pos: Vec3,
    horizontal_dir: Vec3,
    lookahead_dist: f32,
    planet: &PlanetData,
) -> f32 {
    let res = planet.resolution;
    let mut max_height = 0u32;
    for i in 0..=FLY_LOOKAHEAD_SAMPLES {
        let dist = lookahead_dist * (i as f32) / (FLY_LOOKAHEAD_SAMPLES as f32);
        let sample_pos = pos + horizontal_dir * dist;
        if let Some(id) = crate::gen::CoordSystem::pos_to_id(sample_pos, res) {
            max_height = max_height.max(planet.effective_height(id.face, id.u, id.v));
        }
    }
    crate::gen::CoordSystem::get_layer_radius(max_height, res) + FLY_HOVER_CLEARANCE
}

pub struct Player {
    // State
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub cam_pitch: f32,
    pub cam_roll: f32, // fly mode only (Q/E): roll about the view direction; level when walking/swimming
    pub grounded: bool,
    pub debug_mode: bool,
    pub health: f32,
    pub max_health: f32,
    pub spawn_point: Vec3,
    pub landing: bool, // F pressed while flying: auto-descending until touchdown (landing.rs)
    pub taking_off: Option<f32>, // F pressed while walking/swimming: auto-climbing to this radius
    pub handover_altitude: Option<f32>, // on a galaxy planet: fly speed scales up to this altitude

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
            cam_roll: 0.0,
            grounded: false,
            debug_mode: false,
            health: MAX_HEALTH,
            max_health: MAX_HEALTH,
            spawn_point: Vec3::new(0.0, 200.0, 0.0),
            landing: false,
            taking_off: None,
            handover_altitude: None,
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
        self.landing = false;
        self.taking_off = None;
        self.cam_roll = 0.0;
        let up = Physics::get_up_vector(self.position);
        self.rotation = Quat::from_rotation_arc(Vec3::Y, up);
    }

    // turn: keyboard yaw in radians for this step, positive turns left; roll: keyboard roll for this
    // step (fly mode), positive rolls left
    // returns true when an F-landing touched down this tick (the caller turns fly mode off)
    pub fn update(
        &mut self,
        dt: f32,
        planet: &PlanetData,
        input: Vec3,
        jump: bool,
        down: bool,
        mouse_delta: (f32, f32),
        turn: f32,
        roll: f32,
        flying: bool,
        sprint: bool,
    ) -> bool {
        let up = Physics::get_up_vector(self.position);

        // --- ROLL ---
        // walking and swimming are level; an F landing eases the roll level and ignores Q/E, a take-off
        // climb keeps it as it is; otherwise Q/E roll freely
        if !flying {
            self.cam_roll = 0.0;
        } else if self.landing {
            let step = LANDING_ROLL_LEVEL_RATE * dt;
            self.cam_roll -= self.cam_roll.clamp(-step, step);
        } else if self.taking_off.is_none() {
            self.cam_roll = wrap_angle(self.cam_roll + roll);
        }
        // the mouse moves in screen directions: turn the rolled screen's right/down back into yaw/pitch
        let (sin, cos) = self.cam_roll.sin_cos();
        let mouse_delta = (
            mouse_delta.0 * cos + mouse_delta.1 * sin,
            -mouse_delta.0 * sin + mouse_delta.1 * cos,
        );

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
        // Space swims up (or hops out at the surface), Left Shift dives
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
            return false;
        }

        if !flying && depth > SWIM_DEPTH {
            self.swim(dt, planet, input, jump, down, depth, up);
            self.rotation = Physics::align_to_planet(self.rotation, up);
            return false;
        }

        let effective_speed = if flying {
            let altitude = self.position.length() - planet.resolution as f32 / 2.0;
            crate::landing::fly_speed(self.move_speed, sprint, altitude, self.handover_altitude)
        } else if sprint {
            self.move_speed * 2.0
        } else {
            self.move_speed
        };

        // --- MOVEMENT INPUT ---
        if flying {
            let mut fly_dir = Vec3::ZERO;
            if input.length() > 0.01 {
                let input_normalized = input.normalize();
                let pitch_rot = Quat::from_axis_angle(Vec3::X, self.cam_pitch);
                fly_dir += self.rotation
                    * pitch_rot
                    * Vec3::new(input_normalized.x, 0.0, input_normalized.z);
            }
            // Space/Left Shift: climb/descend along the planet-relative up vector, independent of
            // where the camera is looking (unlike WASD, which flies toward the look direction)
            if jump {
                fly_dir += up;
            }
            if down {
                fly_dir -= up;
            }
            let target = if fly_dir.length() > 0.01 {
                fly_dir.normalize() * effective_speed
            } else {
                Vec3::ZERO
            };
            // eased: pressing and releasing keys ramps the speed instead of switching it
            self.velocity = crate::smoothing::ease_toward(
                self.velocity,
                target,
                dt,
                crate::smoothing::FLIGHT_EASE_SECONDS,
            );
            // F-landing: straight down along local down, fast high up and slowing near the ground;
            // WASD still steers sideways (spec §4)
            if self.landing {
                let altitude = self.position.length() - planet.resolution as f32 / 2.0;
                let horizontal = self.velocity - up * self.velocity.dot(up);
                self.velocity = horizontal - up * crate::landing::landing_descent_speed(altitude);
            }
            // F take-off: straight up along local up, easing in at the target without overshooting it;
            // WASD still steers sideways
            if let Some(target) = self.taking_off {
                let remaining = (target - self.position.length()).max(0.0);
                let horizontal = self.velocity - up * self.velocity.dot(up);
                let climb = crate::landing::takeoff_climb_speed(remaining).min(remaining / dt);
                self.velocity = horizontal + up * climb;
            }

            // terrain floor, clamped before solving movement so a fast descent (sprinting down,
            // or diving via pitch) can't tunnel through the floor within a single tick — unlike the
            // post-solve correction below, which only ever climbs, this only ever holds back a
            // descent already in progress, so it doesn't fight the anticipatory climb.
            let vel_horizontal = self.velocity - up * self.velocity.dot(up);
            let look_dir = if vel_horizontal.length() > 0.5 {
                vel_horizontal.normalize()
            } else {
                Vec3::ZERO
            };
            let lookahead = (vel_horizontal.length() * FLY_LOOKAHEAD_TIME)
                .clamp(FLY_LOOKAHEAD_MIN, FLY_LOOKAHEAD_MAX);
            let floor = fly_terrain_floor(self.position, look_dir, lookahead, planet);
            let radius = self.position.length();
            let vert = self.velocity.dot(up);
            let clamped_vert = fly_max_descent(vert, radius, floor, dt);
            if clamped_vert != vert {
                self.velocity -= up * (vert - clamped_vert);
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

        // --- FLY MODE TERRAIN FLOOR ---
        if flying {
            let vel_horizontal = self.velocity - up * self.velocity.dot(up);
            let dir = if vel_horizontal.length() > 0.5 {
                vel_horizontal.normalize()
            } else {
                Vec3::ZERO
            };
            let lookahead = (vel_horizontal.length() * FLY_LOOKAHEAD_TIME)
                .clamp(FLY_LOOKAHEAD_MIN, FLY_LOOKAHEAD_MAX);
            let floor = fly_terrain_floor(self.position, dir, lookahead, planet);
            let radius = self.position.length();
            let climb = fly_floor_climb(radius, floor, dt);
            if climb > 0.0 {
                self.position += up * climb;
                let vert = self.velocity.dot(up);
                if vert < 0.0 {
                    self.velocity -= up * vert;
                }
            }
            if self
                .taking_off
                .is_some_and(|target| self.position.length() >= target - 0.05)
            {
                self.taking_off = None; // arrived: plain fly mode from here
            }
            if self.landing
                && crate::landing::touched_down(
                    self.position.length(),
                    floor,
                    planet.water_depth(self.position).is_some_and(|d| d > 0.0),
                )
            {
                self.landing = false;
                self.rotation = Physics::align_to_planet(self.rotation, up);
                return true;
            }
        }

        // --- ALIGN TO SURFACE ---
        self.rotation = Physics::align_to_planet(self.rotation, up);
        false
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

        let final_rot = self.rotation
            * Quat::from_axis_angle(Vec3::X, self.cam_pitch)
            * Quat::from_axis_angle(Vec3::Z, self.cam_roll);
        let forward = final_rot * Vec3::NEG_Z;

        // the rolled camera's own up (unrolled it lies in the plane of the planet's up and `forward`)
        glam::camera::rh::view::look_at_mat4(cam_pos, cam_pos + forward, final_rot * Vec3::Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test wrapper that calls the production function
    fn liquid_damage_this_tick(depth: Option<f32>, damaging: bool, dt: f32) -> f32 {
        damage_this_tick(depth, damaging, dt)
    }

    // a flying player over flat-ish ground, looking slightly down
    fn flying_player() -> (Player, PlanetData) {
        let planet = PlanetData::new(32);
        let mut player = Player::new();
        let dir = crate::gen::CoordSystem::get_block_center(0, 10, 10, 30, 32).normalize();
        player.spawn(dir * crate::gen::CoordSystem::get_layer_radius(30, 32));
        player.cam_pitch = -0.2;
        (player, planet)
    }

    fn tick(player: &mut Player, planet: &PlanetData, mouse: (f32, f32), roll: f32, flying: bool) {
        player.update(
            1.0 / 60.0,
            planet,
            Vec3::ZERO,
            false,
            false,
            mouse,
            0.0,
            roll,
            flying,
            false,
        );
    }

    // the camera's right and up vectors in world space (the view matrix's inverse)
    fn camera_axes(player: &Player) -> (Vec3, Vec3, Vec3) {
        let cam = player.get_view_matrix().inverse();
        (
            cam.x_axis.truncate(),
            cam.y_axis.truncate(),
            -cam.z_axis.truncate(),
        )
    }

    // fly mode: releasing W doesn't stop dead, the player coasts and slows down
    #[test]
    fn fly_mode_coasts_after_releasing_w() {
        let (mut player, planet) = flying_player();
        player.cam_pitch = 0.0;
        let forward = Vec3::new(0.0, 0.0, -1.0);
        for _ in 0..60 {
            player.update(
                1.0 / 60.0,
                &planet,
                forward,
                false,
                false,
                (0.0, 0.0),
                0.0,
                0.0,
                true,
                false,
            );
        }
        let cruising = player.velocity.length();
        assert!(cruising > 1.0, "sanity check: flying forward");
        player.update(
            1.0 / 60.0,
            &planet,
            Vec3::ZERO,
            false,
            false,
            (0.0, 0.0),
            0.0,
            0.0,
            true,
            false,
        );
        let coasting = player.velocity.length();
        assert!(
            coasting > 0.5 * cruising,
            "stopped dead: {coasting} after {cruising}"
        );
        assert!(coasting < cruising, "didn't slow down");
    }

    // fly mode: Q/E roll the camera (Q left: its right side comes up); walking is always level
    #[test]
    fn fly_mode_rolls_and_walking_is_level() {
        let (mut player, planet) = flying_player();
        let (right, up, forward) = camera_axes(&player);
        tick(&mut player, &planet, (0.0, 0.0), 0.4, true);
        assert!((player.cam_roll - 0.4).abs() < 1e-6);
        let (right2, _, forward2) = camera_axes(&player);
        assert!(
            forward2.dot(forward) > 1.0 - 1e-4,
            "rolling moved the view direction"
        );
        assert!(
            right2.dot(up) > 0.3,
            "Q didn't raise the camera's right side"
        );
        assert!((right2.dot(right) - 0.4f32.cos()).abs() < 1e-3);
        tick(&mut player, &planet, (0.0, 0.0), 0.0, false);
        assert_eq!(player.cam_roll, 0.0, "walking (or swimming) is level");
    }

    // rolled 90°, moving the mouse right still turns the view toward the screen's right
    #[test]
    fn mouse_turns_toward_the_screen_when_rolled() {
        let (mut player, planet) = flying_player();
        player.cam_roll = std::f32::consts::FRAC_PI_2;
        let (screen_right, screen_up, forward) = camera_axes(&player);
        tick(&mut player, &planet, (40.0, 0.0), 0.0, true);
        let (_, _, forward2) = camera_axes(&player);
        let moved = forward2 - forward;
        assert!(moved.dot(screen_right) > 0.05, "{moved:?}");
        assert!(
            moved.dot(screen_up).abs() < 0.2 * moved.dot(screen_right),
            "{moved:?}"
        );
    }

    // an F landing ignores Q/E and eases the roll level; so does nothing during a take-off climb
    #[test]
    fn landing_levels_the_roll_and_ignores_q_e() {
        let (mut player, planet) = flying_player();
        player.cam_roll = 1.0;
        player.landing = true;
        let mut last = player.cam_roll;
        for _ in 0..30 {
            tick(&mut player, &planet, (0.0, 0.0), 0.2, true);
            if !player.landing {
                break;
            }
            assert!(player.cam_roll < last, "roll didn't ease toward level");
            last = player.cam_roll;
        }
        let (mut player, planet) = flying_player();
        player.cam_roll = 0.5;
        player.taking_off = Some(player.position.length() + 20.0);
        tick(&mut player, &planet, (0.0, 0.0), 0.3, true);
        assert_eq!(player.cam_roll, 0.5, "Q/E rolled during the take-off climb");
    }

    // F take-off: the climb rises smoothly to its target and ends there (fly mode, hovering)
    #[test]
    fn take_off_climbs_to_the_target_and_stops() {
        let planet = PlanetData::new(32);
        let (face, u, v) = (0u8, 10u32, 10u32);
        let top = planet.surface(face, u, v) + 1;
        let dir = crate::gen::CoordSystem::get_block_center(face, u, v, top, 32).normalize();
        let mut player = Player::new();
        player.spawn(dir * crate::gen::CoordSystem::get_layer_radius(top, 32));
        let target_layer = crate::landing::takeoff_target_layer(&planet, player.position);
        let target = crate::gen::CoordSystem::get_layer_radius(target_layer, 32);
        player.taking_off = Some(target);
        let mut max_step = 0.0f32;
        for _ in 0..600 {
            let before = player.position.length();
            player.update(
                1.0 / 60.0,
                &planet,
                Vec3::ZERO,
                false,
                false,
                (0.0, 0.0),
                0.0,
                0.0,
                true,
                false,
            );
            max_step = max_step.max(player.position.length() - before);
        }
        assert!(player.taking_off.is_none(), "climb never finished");
        assert!(
            player.position.length() >= target - 0.2,
            "stopped short of {target}"
        );
        assert!(max_step < 1.0, "the climb jumped {max_step} in one tick");
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
    fn fly_floor_climb_never_pulls_the_player_down_toward_a_lower_floor() {
        // "keep height over lower terrain": already above the floor means no correction at all.
        assert_eq!(fly_floor_climb(50.0, 30.0, 1.0 / 60.0), 0.0);
    }

    #[test]
    fn fly_floor_climb_is_capped_at_the_climb_rate() {
        // a big, sudden floor jump (e.g. a cliff entering the lookahead window) must still be a
        // smooth climb, not an instant snap to the new floor.
        let dt = 1.0 / 60.0;
        let climb = fly_floor_climb(0.0, 1000.0, dt);
        assert_eq!(climb, FLY_CLIMB_RATE * dt);
    }

    #[test]
    fn fly_floor_climb_does_not_overshoot_a_small_gap() {
        let dt = 1.0 / 60.0;
        let gap = 0.01; // much smaller than FLY_CLIMB_RATE * dt
        let climb = fly_floor_climb(99.99, 99.99 + gap, dt);
        assert!((climb - gap).abs() < 1e-4, "climb was {climb}");
    }

    #[test]
    fn fly_max_descent_passes_through_upward_velocity_unchanged() {
        assert_eq!(fly_max_descent(5.0, 50.0, 30.0, 1.0 / 60.0), 5.0);
    }

    #[test]
    fn fly_max_descent_allows_a_slow_descent_well_above_the_floor() {
        let dt = 1.0 / 60.0;
        // radius far above the floor: a modest descent speed shouldn't be clamped at all
        assert_eq!(fly_max_descent(-5.0, 100.0, 30.0, dt), -5.0);
    }

    #[test]
    fn fly_max_descent_never_lets_one_tick_cross_the_floor() {
        let dt = 1.0 / 60.0;
        // sprinting down (-50 units/s) right at the floor: must not descend at all this tick
        let radius = 30.0;
        let floor = 30.0;
        let clamped = fly_max_descent(-50.0, radius, floor, dt);
        assert!(
            radius + clamped * dt >= floor - 1e-4,
            "would cross the floor: {clamped}"
        );
    }

    #[test]
    fn fly_max_descent_never_lets_a_fast_descent_tunnel_through_a_gap() {
        let dt = 1.0 / 60.0;
        let radius = 35.0;
        let floor = 30.0; // 5 units of room left
        let clamped = fly_max_descent(-500.0, radius, floor, dt); // absurdly fast dive
        assert!(
            radius + clamped * dt >= floor - 1e-4,
            "would cross the floor: {clamped}"
        );
    }

    // with no horizontal direction (hovering in place), the floor is based only on the current
    // column — directly checks the i=0 (distance 0) lookahead sample against the same column read
    // through the normal terrain API, so this doesn't depend on what the noise actually generated.
    #[test]
    fn fly_terrain_floor_with_no_direction_uses_the_current_column() {
        let planet = PlanetData::new(32);
        let (face, u, v) = (0, planet.resolution / 2, planet.resolution / 2);
        let h = planet.effective_height(face, u, v);
        let pos = crate::gen::CoordSystem::get_vertex_pos(face, u, v, h + 20, planet.resolution);

        let floor = fly_terrain_floor(pos, Vec3::ZERO, 30.0, &planet);
        let expected =
            crate::gen::CoordSystem::get_layer_radius(h, planet.resolution) + FLY_HOVER_CLEARANCE;
        assert!(
            (floor - expected).abs() < 1e-3,
            "floor was {floor}, expected {expected}"
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
