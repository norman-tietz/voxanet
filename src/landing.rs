// landing.rs
// Galaxy landing, milestone 4: the handover between captured galaxy flight and the voxel engine,
// and flying/landing on a planet. Pure functions, unit-tested without a window or GPU.

use crate::physics::Physics;
use glam::{Mat3, Quat, Vec3};

// captured galaxy flight hands over to the voxel engine below this distance from the planet's
// centre (in its radii), once the voxel world is ready; the voxel engine hands back above
// LIFTOFF_HANDOVER_RADII — the gap is hysteresis against ping-pong (spec §1)
pub const LAND_HANDOVER_RADII: f64 = 3.0;
pub const LIFTOFF_HANDOVER_RADII: f32 = 3.5;

// the planet camera's pitch limit (Player::update clamps cam_pitch to ±1.5)
pub const MAX_PITCH: f32 = 1.5;

pub fn should_land(distance_radii: f64, voxel_world_ready: bool) -> bool {
    voxel_world_ready && distance_radii < LAND_HANDOVER_RADII
}

pub fn should_lift_off(distance_radii: f32) -> bool {
    distance_radii > LIFTOFF_HANDOVER_RADII
}

// after the landing handover the galaxy's near impostor stays over the voxel world this long, fading
// out (alpha-blended, GalaxyRenderer::draw_handover_overlay), so the switch between the two renderings
// is a cross-fade instead of a cut
pub const HANDOVER_FADE_SECONDS: f32 = 1.0;

// the impostor's opacity `elapsed` seconds after the landing handover; None once it has faded out
pub fn handover_overlay_opacity(elapsed: f32) -> Option<f32> {
    let t = (elapsed / HANDOVER_FADE_SECONDS).max(0.0);
    (t < 1.0).then_some(1.0 - t * t * (3.0 - 2.0 * t))
}

// galaxy flight (eye position and orientation, planet frame) -> the planet engine's player: feet
// position, upright rotation (local Y = up, local -Z = heading), camera pitch and roll. The eye stays
// where it is and a rolled view stays rolled; pitch is clamped to MAX_PITCH.
pub fn player_pose_from_flight(eye: Vec3, rotation: Quat) -> (Vec3, Quat, f32, f32) {
    let up = eye.normalize();
    let (player_rotation, pitch, roll) = camera_angles(up, rotation, MAX_PITCH);
    (eye - up * Physics::EYE_HEIGHT, player_rotation, pitch, roll)
}

// a camera orientation split into the planet player's upright rotation (local Y = `up`, local -Z =
// heading), pitch (clamped to ±`max_pitch`) and roll, so rotation * pitch * roll gives it back
// (Player::get_view_matrix) whenever the pitch isn't clamped
pub fn camera_angles(up: Vec3, rotation: Quat, max_pitch: f32) -> (Quat, f32, f32) {
    let forward = rotation * Vec3::NEG_Z;
    let pitch = forward
        .dot(up)
        .clamp(-1.0, 1.0)
        .asin()
        .clamp(-max_pitch, max_pitch);
    // heading: the view direction flattened onto the horizon; looking (almost) straight down or up
    // it's undefined, so take where the top of the screen points (flipped when looking up)
    let mut heading = forward - up * forward.dot(up);
    if heading.length_squared() < 1e-6 {
        let screen_up = rotation * Vec3::Y;
        heading = screen_up - up * screen_up.dot(up);
        if forward.dot(up) > 0.0 {
            heading = -heading;
        }
    }
    let heading = heading.normalize();
    let player_rotation = Quat::from_mat3(&Mat3::from_cols(heading.cross(up), up, -heading));
    // roll: how far the camera's right axis is turned about the view from the unrolled camera's
    // (Player::get_view_matrix applies it after the pitch)
    let unrolled = player_rotation * Quat::from_axis_angle(Vec3::X, pitch);
    let right = rotation * Vec3::X;
    let roll = right
        .dot(unrolled * Vec3::Y)
        .atan2(right.dot(unrolled * Vec3::X));
    (player_rotation.normalize(), pitch, roll)
}

// the planet engine's player -> galaxy flight: eye position and the camera's orientation (the same
// rotation * pitch * roll the first-person view matrix uses, Player::get_view_matrix), so the view
// doesn't move on liftoff
pub fn flight_pose_from_player(
    position: Vec3,
    rotation: Quat,
    cam_pitch: f32,
    cam_roll: f32,
) -> (Vec3, Quat) {
    let up = position.normalize();
    let eye = position + up * Physics::EYE_HEIGHT;
    let camera = rotation
        * Quat::from_axis_angle(Vec3::X, cam_pitch)
        * Quat::from_axis_angle(Vec3::Z, cam_roll);
    (eye, camera.normalize())
}

// the galaxy flight a liftoff hands over to: the planet camera's eye and orientation
// (flight_pose_from_player) and the player's velocity, so a climb doesn't stall at the handover —
// both sides live in the planet frame, so the velocity carries over unchanged
pub fn liftoff_flight(
    position: Vec3,
    rotation: Quat,
    cam_pitch: f32,
    cam_roll: f32,
    velocity: Vec3,
) -> crate::galaxy::GalaxyFlight {
    let (eye, flight_rotation) = flight_pose_from_player(position, rotation, cam_pitch, cam_roll);
    let mut flight = crate::galaxy::GalaxyFlight::new(eye.as_dvec3());
    flight.rotation = flight_rotation;
    flight.velocity = velocity.as_dvec3();
    flight
}

// altitude above sea level at which the landing/liftoff handover happens (LAND_HANDOVER_RADII from
// the centre is this many radii above the surface)
pub fn handover_altitude(planet_radius: f32) -> f32 {
    (LAND_HANDOVER_RADII as f32 - 1.0) * planet_radius
}

// fly speed on a planet: today's (move_speed, or ×10 with boost) near the ground, growing with
// altitude to exactly galaxy flight's cruise/boost speed at the handover altitude, so speed doesn't
// jump at the handover. `handover_altitude` is None on the home planet (no handover, no scaling).
pub fn fly_speed(base: f32, sprint: bool, altitude: f32, handover_altitude: Option<f32>) -> f32 {
    let (near_ground, at_handover) = if sprint {
        (base * 10.0, crate::galaxy::GALAXY_BOOST_SPEED)
    } else {
        (base, crate::galaxy::GALAXY_CRUISE_SPEED)
    };
    match handover_altitude {
        None => near_ground,
        Some(h) => near_ground.max(at_handover * (altitude / h).max(0.0)),
    }
}

// F-landing descent speed: proportional to altitude (fast high up, easing in near the ground),
// never below LAND_MIN_DESCENT so it still finishes
const LAND_DESCENT_RATE: f32 = 0.8; // 1/s: altitude roughly halves every ~0.9 s
const LAND_MIN_DESCENT: f32 = 3.0; // world units/s

pub fn landing_descent_speed(altitude: f32) -> f32 {
    (altitude * LAND_DESCENT_RATE).max(LAND_MIN_DESCENT)
}

// the fly terrain floor (FLY_HOVER_CLEARANCE above the ground, entity.rs) counts as touched down
// within this margin; so do feet in water (the player then swims)
const TOUCHDOWN_MARGIN: f32 = 0.5;

pub fn touched_down(radius: f32, floor: f32, feet_in_water: bool) -> bool {
    feet_in_water || radius - floor < TOUCHDOWN_MARGIN
}

// F take-off climbs above the ground by this share of the planet's relief (its highest natural peak
// above sea level), at least TAKEOFF_MIN_CLEARANCE layers, and not above the highest peak: bigger
// planets have more relief and climb higher, and valleys and canyons stay flyable
const TAKEOFF_RELIEF_SHARE: f32 = 0.3;
pub const TAKEOFF_MIN_CLEARANCE: u32 = 3;

// the layer an F take-off from `position` (the feet) climbs to: measured from the water surface when
// swimming, else from the top of the column below
pub fn takeoff_target_layer(planet: &crate::common::PlanetData, position: Vec3) -> u32 {
    let sea = planet.terrain.sea_level();
    let (_, peak) = planet.terrain.height_range();
    let in_water = planet.water_depth(position).is_some_and(|d| d > 0.0);
    let ground = match crate::gen::CoordSystem::pos_to_id(position, planet.resolution) {
        Some(id) if !in_water => planet.surface(id.face, id.u, id.v) + 1,
        Some(id) => planet.terrain.water_level(id.face, id.u, id.v) + 1,
        None => sea + 1,
    };
    let relief = peak.saturating_sub(sea) as f32;
    let climb = ((relief * TAKEOFF_RELIEF_SHARE).round() as u32).max(TAKEOFF_MIN_CLEARANCE);
    (ground + climb)
        .min(peak + 1)
        .max(ground + TAKEOFF_MIN_CLEARANCE)
}

// F take-off climb speed: the landing descent mirrored — fast while far below the target, easing in
// near it, never below LAND_MIN_DESCENT so it still finishes
pub fn takeoff_climb_speed(remaining: f32) -> f32 {
    (remaining * LAND_DESCENT_RATE).max(LAND_MIN_DESCENT)
}

// glide assist (planet fly mode): a descent coming down below the glide ceiling (twice the cloud
// layer's altitude, glide_ceiling; `Player::glide`) is brought onto a line at the take-off height
// (takeoff_target_layer, fixed when it engages) and follows the planet's curve there instead of
// flying on straight and out again. The path's climb angle is steered toward GLIDE_TAU seconds of
// the remaining height at the current speed (at most GLIDE_MAX_DESCENT down, GLIDE_MAX_CLIMB up),
// turning velocity and view together, at GLIDE_SOFT_RATE — harder (up to GLIDE_HARD_RATE) only when
// the pull-out wouldn't fit into GLIDE_ROOM of the remaining height otherwise. Fly speed grows with
// altitude, so a soft curve needs the ship slower than that: while pulling out it brakes toward
// GLIDE_BRAKE × the remaining height per second (GlidePull::max_speed).
// A steep pull-out goes toward the ship's own up (the camera's, so it follows the roll): over the
// top, even when the flight line passes a little below the planet's centre as seen from the ship
// and the way over leads through a vertical dive; only when the line hits the planet's lower
// GLIDE_UNDER_SHARE of its diameter (along the ship's up) does it bend down and pass under. The
// side is kept until the pull-out is done (GlidePull::axis).
pub const GLIDE_TAU: f32 = 3.0; // s
pub const GLIDE_MAX_DESCENT: f32 = 75.0 * std::f32::consts::PI / 180.0;
pub const GLIDE_MAX_CLIMB: f32 = 20.0 * std::f32::consts::PI / 180.0;
const GLIDE_SOFT_RATE: f32 = 0.6; // rad/s
const GLIDE_HARD_RATE: f32 = 4.0; // rad/s
const GLIDE_ROOM: f32 = 0.7; // share of the remaining height a pull-out may use
const GLIDE_BRAKE: f32 = 0.5; // 1/s: pulling out, at most this × the remaining height per second
const GLIDE_CEILING_CLOUDS: f32 = 2.0; // the glide ceiling in cloud-layer altitudes
const GLIDE_MIN_SPEED: f32 = 2.0; // world units/s: slower (W released, coasting out) isn't steered
                                  // a path this much steeper than wanted is a pull-out (side chosen by the ship's up); closer, it's
                                  // tracked in the vertical plane
const GLIDE_PULL_OUT: f32 = 5.0 * std::f32::consts::PI / 180.0;
pub const GLIDE_UNDER_SHARE: f32 = 0.3; // over 70 %, under 30 %

// the radius below which a descent engages the glide assist, on a planet of `radius` (sea level)
pub fn glide_ceiling(radius: f32) -> f32 {
    radius * (1.0 + GLIDE_CEILING_CLOUDS * (crate::common::CLOUD_ALT - 1.0))
}

// one step of the glide assist: the world rotation it applies to the velocity and the camera
pub struct GlidePull {
    pub rotation: Quat,
    // the pull-out's turn axis, passed back in on the next step so it keeps its side while the ship
    // rolls; None once the path is on track
    pub axis: Option<Vec3>,
    // still pulling over the top, before the vertical: the ship's up points away from the horizon
    // the flight is turned toward, so levelling the roll now would flip it upside down and back
    pub over_the_top: bool,
    // pulling out: the speed to brake to, so the curve stays soft
    pub max_speed: Option<f32>,
}

// the climb angle (radians, negative: descending) the glide assist steers toward, `height` above its
// line at `speed`
pub fn glide_path_angle(height: f32, speed: f32) -> f32 {
    (-height / (GLIDE_TAU * speed.max(1e-3)))
        .clamp(-GLIDE_MAX_DESCENT.sin(), GLIDE_MAX_CLIMB.sin())
        .asin()
}

// the glide assist's step for a velocity `velocity` at `position` (planet centre at the origin),
// steering onto the sphere `target_radius`, with the camera's up `camera_up`; `axis`: the previous
// step's (None: a pull-out chooses its side afresh). None when there's nothing to turn
pub fn glide_assist(
    velocity: Vec3,
    position: Vec3,
    target_radius: f32,
    camera_up: Vec3,
    axis: Option<Vec3>,
    dt: f32,
) -> Option<GlidePull> {
    let speed = velocity.length();
    if speed < GLIDE_MIN_SPEED {
        return None;
    }
    let up = position.normalize();
    let dir = velocity / speed;
    let climb = dir.dot(up).clamp(-1.0, 1.0).asin();
    let height = position.length() - target_radius;
    let wanted = glide_path_angle(height, speed);
    // soft, unless levelling out at this speed wouldn't fit into the room left above the line
    let steeper = (wanted - climb).max(0.0);
    let needed = speed * (1.0 - steeper.cos()) / (GLIDE_ROOM * height.max(0.5));
    let step = needed.clamp(GLIDE_SOFT_RATE, GLIDE_HARD_RATE) * dt;
    let max_speed = (steeper > 0.05).then(|| GLIDE_BRAKE * height.max(0.0));
    // toward the nearest horizon
    let horizon = (up - dir * up.dot(dir)).try_normalize()?;

    let latched = axis.and_then(|axis| axis.cross(dir).try_normalize());
    if latched.is_none() && climb > wanted - GLIDE_PULL_OUT {
        // on track: turn up or down in the vertical plane
        let turn = (wanted - climb).clamp(-step, step);
        if turn.abs() < 1e-6 {
            return None;
        }
        return Some(GlidePull {
            rotation: Quat::from_axis_angle(dir.cross(horizon).normalize(), turn),
            axis: None,
            over_the_top: false,
            max_speed,
        });
    }
    // pull-out
    let bend = latched.unwrap_or_else(|| {
        // the ship's up across the flight line; with the camera looking across the flight, fall
        // back to the nearest horizon
        let bend = (camera_up - dir * camera_up.dot(dir))
            .try_normalize()
            .unwrap_or(horizon);
        // where the line passes the planet's centre, along the ship's up, in radii of the glide's
        // sphere (-1: its lower edge as seen from the ship, 1 its upper edge)
        let offset = (position - dir * position.dot(dir)).dot(bend) / target_radius;
        if offset < 2.0 * GLIDE_UNDER_SHARE - 1.0 {
            -bend
        } else {
            bend
        }
    });
    // the turn still needed in the plane of `dir` and `bend` until the climb angle is `wanted`: the
    // direction turned by θ rises along up as a·cos θ + b·sin θ = A·cos(θ − φ)
    let (a, b) = (dir.dot(up), bend.dot(up));
    let (amplitude, phase) = ((a * a + b * b).sqrt(), b.atan2(a));
    let spread = (wanted.sin() / amplitude).clamp(-1.0, 1.0).acos();
    // the nearer of the two crossings ahead; one just passed (a step landing a little beyond it)
    // counts as reached rather than as a full turn away
    let (pi, tau) = (std::f32::consts::PI, std::f32::consts::TAU);
    let remaining = [phase - spread, phase + spread]
        .map(|t| {
            let t = (t + pi).rem_euclid(tau) - pi;
            if t < -GLIDE_PULL_OUT {
                t + tau
            } else {
                t.max(0.0)
            }
        })
        .into_iter()
        .fold(f32::MAX, f32::min);
    let axis = dir.cross(bend).normalize();
    Some(GlidePull {
        rotation: Quat::from_axis_angle(axis, step.min(remaining)),
        axis: (remaining > step).then_some(axis),
        over_the_top: bend.dot(horizon) < 0.0,
        max_speed,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FAction {
    StartLanding,
    CancelLanding,
    RefuseLanding,
    TakeOff,
    StopTakeOff,
}

// F: flying -> land (or cancel a landing in progress; refused over damaging liquid; during the
// take-off climb: stop it and hover); walking/swimming -> take off
pub fn f_action(
    flying: bool,
    landing: bool,
    taking_off: bool,
    over_damaging_liquid: bool,
) -> FAction {
    match (flying, landing, taking_off) {
        (true, true, _) => FAction::CancelLanding,
        (true, false, true) => FAction::StopTakeOff,
        (true, false, false) if over_damaging_liquid => FAction::RefuseLanding,
        (true, false, false) => FAction::StartLanding,
        (false, _, _) => FAction::TakeOff,
    }
}

// whether the column straight below `position` is under a damaging liquid (lava): landing there
// would kill the player
pub fn over_damaging_liquid(planet: &crate::common::PlanetData, position: Vec3) -> bool {
    let damaging = planet.planet_type.def().liquid.is_some_and(|l| l.damaging);
    if !damaging {
        return false;
    }
    let res = planet.resolution;
    let probe = position.normalize_or_zero() * (res as f32 / 2.0);
    // the same water rule as swimming and the rendered surface, so a dug, lava-filled hole counts too
    crate::gen::CoordSystem::pos_to_id(probe, res)
        .is_some_and(|id| planet.holds_water(id.face, id.u, id.v))
}

// captured galaxy flight can't go below this distance from the planet's centre (in its radii):
// it only matters while the voxel world isn't ready yet — once it is, the landing handover happens
// at LAND_HANDOVER_RADII, above this
pub const MIN_CAPTURED_RADII: f64 = 2.0;

pub fn keep_above_planet(pos: glam::DVec3, planet_radius: f64) -> glam::DVec3 {
    let min = MIN_CAPTURED_RADII * planet_radius;
    let dist = pos.length();
    if dist < min && dist > 1e-9 {
        pos * (min / dist)
    } else {
        pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the handover overlay starts as the impostor alone and fades out smoothly, then stops
    #[test]
    fn handover_overlay_fades_out() {
        assert_eq!(handover_overlay_opacity(0.0), Some(1.0));
        let mid = handover_overlay_opacity(HANDOVER_FADE_SECONDS / 2.0).unwrap();
        assert!((mid - 0.5).abs() < 1e-6);
        let late = handover_overlay_opacity(HANDOVER_FADE_SECONDS * 0.9).unwrap();
        assert!(late > 0.0 && late < mid);
        assert_eq!(handover_overlay_opacity(HANDOVER_FADE_SECONDS), None);
    }

    // liftoff must keep the climb going: galaxy flight starts with the player's velocity, not
    // from rest (both are in the planet frame)
    #[test]
    fn liftoff_keeps_the_climb_speed() {
        let position = Vec3::new(0.0, 600.0, 0.0);
        let rotation = Quat::IDENTITY; // upright at +Y, looking along -Z
        let velocity = Vec3::new(10.0, 620.0, -5.0);
        let flight = liftoff_flight(position, rotation, -0.3, 0.0, velocity);
        assert!((flight.velocity - velocity.as_dvec3()).length() < 1e-6);
        let (eye, flight_rot) = flight_pose_from_player(position, rotation, -0.3, 0.0);
        assert!((flight.position - eye.as_dvec3()).length() < 1e-6);
        assert!(flight.rotation.dot(flight_rot).abs() > 1.0 - 1e-6);
    }

    // until the voxel world is ready the impostor is all there is: captured flight must not sink
    // into (or through) the planet while waiting
    #[test]
    fn captured_flight_is_kept_above_two_radii() {
        let r = 168.5;
        let inside = glam::DVec3::new(0.0, 1.2 * r, 0.0);
        let kept = keep_above_planet(inside, r);
        assert!((kept.length() - MIN_CAPTURED_RADII * r).abs() < 1e-9);
        assert!(
            (kept.normalize() - inside.normalize()).length() < 1e-12,
            "same direction"
        );
        let fine = glam::DVec3::new(2.5 * r, 0.0, 0.0);
        assert_eq!(keep_above_planet(fine, r), fine);
    }

    #[test]
    fn fly_speed_matches_galaxy_cruise_at_the_handover_altitude() {
        let h = handover_altitude(168.5);
        assert!((h - 337.0).abs() < 1e-3, "2 radii above sea level: {h}");
        assert!(
            (fly_speed(5.0, false, h, Some(h)) - crate::galaxy::GALAXY_CRUISE_SPEED).abs() < 1e-3
        );
        assert!(
            (fly_speed(5.0, true, h, Some(h)) - crate::galaxy::GALAXY_BOOST_SPEED).abs() < 1e-3
        );
        // near the ground: today's speeds
        assert_eq!(fly_speed(5.0, false, 1.0, Some(h)), 5.0);
        assert_eq!(fly_speed(5.0, true, 1.0, Some(h)), 50.0);
        // grows with altitude
        assert!(fly_speed(5.0, false, 200.0, Some(h)) > fly_speed(5.0, false, 100.0, Some(h)));
    }

    #[test]
    fn home_planet_fly_speed_is_unchanged() {
        assert_eq!(fly_speed(5.0, false, 300.0, None), 5.0);
        assert_eq!(fly_speed(5.0, true, 300.0, None), 50.0);
    }

    #[test]
    fn landing_descends_fast_when_high_and_slow_near_the_ground() {
        assert!(landing_descent_speed(300.0) > 100.0);
        assert!(landing_descent_speed(300.0) > landing_descent_speed(30.0));
        assert_eq!(
            landing_descent_speed(0.5),
            landing_descent_speed(0.0),
            "minimum near ground"
        );
        assert!(landing_descent_speed(0.0) > 0.0);
    }

    #[test]
    fn touchdown_at_the_fly_floor_or_in_water() {
        assert!(touched_down(103.3, 103.0, false));
        assert!(!touched_down(110.0, 103.0, false));
        assert!(touched_down(110.0, 103.0, true));
    }

    #[test]
    fn f_lands_cancels_refuses_or_takes_off() {
        assert_eq!(f_action(true, false, false, false), FAction::StartLanding);
        assert_eq!(f_action(true, true, false, false), FAction::CancelLanding);
        assert_eq!(
            f_action(true, true, false, true),
            FAction::CancelLanding,
            "cancel is always allowed"
        );
        assert_eq!(f_action(true, false, false, true), FAction::RefuseLanding);
        assert_eq!(f_action(false, false, false, false), FAction::TakeOff);
        assert_eq!(f_action(false, false, false, true), FAction::TakeOff);
    }

    // how far below the horizon a velocity points, in degrees
    fn dive_deg(v: Vec3, up: Vec3) -> f32 {
        (-v.normalize().dot(up)).asin().to_degrees()
    }

    // a flight on a radius-100 planet from `position` along `v`, at its fly speed (fly_speed: 2.5 ×
    // the altitude per second) braked as the glide asks, eased like Player::update, with the glide
    // assist turning velocity and camera together toward the sphere `target`, for `seconds`; returns
    // the final position, velocity, camera, the lowest radius and the hardest turn (rad/s)
    fn glide(
        mut position: Vec3,
        mut v: Vec3,
        mut camera: Quat,
        target: f32,
        seconds: f32,
    ) -> (Vec3, Vec3, Quat, f32, f32) {
        let dt = 1.0 / 60.0;
        let (mut axis, mut lowest, mut hardest) = (None, f32::MAX, 0.0f32);
        for _ in 0..(seconds / dt) as u32 {
            let mut speed = fly_speed(5.0, false, position.length() - 100.0, Some(200.0));
            if let Some(pull) = glide_assist(v, position, target, camera * Vec3::Y, axis, dt) {
                v = pull.rotation * v;
                camera = pull.rotation * camera;
                axis = pull.axis;
                hardest = hardest.max(pull.rotation.to_axis_angle().1 / dt);
                speed = pull.max_speed.map_or(speed, |max| speed.min(max.max(5.0)));
            }
            let target_v = v.normalize() * speed;
            v = crate::smoothing::ease_toward(
                v,
                target_v,
                dt,
                crate::smoothing::FLIGHT_EASE_SECONDS,
            );
            position += v * dt;
            lowest = lowest.min(position.length());
        }
        (position, v, camera, lowest, hardest)
    }

    // a camera at `position` looking along `v`, rolled by `roll` from level
    fn camera_along(v: Vec3, position: Vec3, roll: f32) -> Quat {
        let (forward, up) = (v.normalize(), position.normalize());
        let right = forward.cross(up).normalize();
        let level = Quat::from_mat3(&Mat3::from_cols(right, right.cross(forward), -forward));
        level * Quat::from_axis_angle(Vec3::Z, roll)
    }

    // a steep dive through the clouds levels off onto the glide's line and then follows the planet's
    // curve along it, instead of flying on straight and out of the atmosphere again
    #[test]
    fn a_dive_levels_off_onto_the_line_and_follows_the_curve() {
        let target = 110.0;
        let position = Vec3::new(0.0, glide_ceiling(100.0), 0.0);
        let v = Vec3::new(0.0, -1.0, -0.12).normalize() * 160.0; // ~83 degrees down at fly speed
        let camera = camera_along(v, position, 0.0);
        let (end, v2, camera2, lowest, _) = glide(position, v, camera, target, 20.0);
        assert!(lowest > target - 2.0, "dropped to {lowest}");
        // 20 s on the line is radians of arc on this sphere: it stayed on it
        assert!(
            (end.length() - target).abs() < 1.0,
            "ended at radius {}",
            end.length()
        );
        assert!(dive_deg(v2, end.normalize()).abs() < 2.0, "not level");
        assert!(
            (camera2 * Vec3::Y).dot(end.normalize()) > 0.9,
            "camera not upright"
        );
    }

    // a 45° approach is a soft, long curve: on this small planet (radius 100) it needs only a little
    // more than the soft rate, briefly, to fit above the line
    #[test]
    fn a_shallow_approach_curves_softly() {
        let position = Vec3::new(0.0, glide_ceiling(250.0) * 100.0 / 250.0, 0.0);
        let v = Vec3::new(0.0, -1.0, -1.0).normalize() * 160.0; // 45 degrees down
        let camera = camera_along(v, position, 0.0);
        let (_, _, _, lowest, hardest) = glide(position, v, camera, 110.0, 20.0);
        assert!(hardest < 1.0, "turned at {hardest} rad/s");
        assert!(lowest > 108.0, "dropped to {lowest}");
    }

    // below the line it climbs back up to it, gently
    #[test]
    fn below_the_line_it_climbs_back() {
        let position = Vec3::new(0.0, 104.0, 0.0);
        let v = Vec3::new(0.0, 0.0, -30.0);
        let (end, _, _, _, _) = glide(position, v, camera_along(v, position, 0.0), 110.0, 10.0);
        assert!(
            (end.length() - 110.0).abs() < 1.0,
            "ended at radius {}",
            end.length()
        );
    }

    // upside down (rolled 180°), steeply diving: the flight line passes the centre a little below it
    // as seen from the ship, so the pull-out still goes toward the ship's up, over the top through a
    // vertical dive, and comes out upright heading the other way (a split-S)
    #[test]
    fn upside_down_a_steep_dive_is_pulled_over_the_top() {
        let (target, position) = (110.0, Vec3::new(0.0, 132.0, 0.0));
        let v = Vec3::new(0.0, -80.0, -8.0); // ~84 degrees down: offset about -0.1
        let camera = camera_along(v, position, std::f32::consts::PI);
        let dir = v.normalize();
        let offset = (position - dir * position.dot(dir)).dot(camera * Vec3::Y) / target;
        assert!(offset < 0.0 && offset > -0.4, "offset {offset}");
        let first = glide_assist(v, position, target, camera * Vec3::Y, None, 1.0 / 60.0).unwrap();
        assert!(first.over_the_top);
        let (end, v2, camera2, _, _) = glide(position, v, camera, target, 1.0);
        assert!(v2.z > 0.0, "went under instead of over: {v2:?}");
        assert!(
            (camera2 * Vec3::Y).dot(end.normalize()) > 0.5,
            "not upright after the split-S"
        );
    }

    // upside down in a shallower dive the line hits the lower 30 % (seen from the ship): too far
    // below to go over, so the pull-out bends toward the ship's down (the near horizon)
    #[test]
    fn upside_down_a_shallow_dive_passes_under() {
        let (target, position) = (110.0, Vec3::new(0.0, 132.0, 0.0));
        let v = Vec3::new(0.0, -40.0, -40.0); // 45 degrees down: offset about -0.8
        let camera = camera_along(v, position, std::f32::consts::PI);
        let first = glide_assist(v, position, target, camera * Vec3::Y, None, 1.0 / 60.0).unwrap();
        assert!(!first.over_the_top);
        let (_, v2, camera2, _, _) = glide(position, v, camera, target, 1.0);
        assert!(v2.z < 0.0, "went over instead of under: {v2:?}");
        assert!((camera2 * Vec3::Y).y < -0.5, "rolled during the push");
    }

    // the glide assist's view, split into the player's rotation, pitch and roll every step, can turn
    // through a vertical dive (with MAX_PITCH it would stick at the clamp)
    #[test]
    fn camera_angles_follow_a_view_through_the_vertical() {
        let up = Vec3::Y;
        let mut camera = Quat::from_rotation_x(-1.45);
        let step = Quat::from_rotation_x(-0.02); // nose down about the camera's right axis
        let expected = Quat::from_rotation_x(-1.45 - 0.02 * 20.0);
        for _ in 0..20 {
            let (rotation, pitch, roll) =
                camera_angles(up, step * camera, std::f32::consts::FRAC_PI_2);
            camera = rotation
                * Quat::from_axis_angle(Vec3::X, pitch)
                * Quat::from_axis_angle(Vec3::Z, roll);
        }
        let (got, want) = (camera * Vec3::NEG_Z, expected * Vec3::NEG_Z);
        assert!(got.dot(want) > 1.0 - 1e-4, "{got:?} vs {want:?}");
        assert!((camera * Vec3::Y).dot(expected * Vec3::Y) > 1.0 - 1e-4);
    }

    // nothing to steer: coasting to a stop, or already on the line
    #[test]
    fn slow_or_on_track_flight_is_left_alone() {
        let position = Vec3::new(0.0, 110.0, 0.0);
        assert!(glide_assist(
            Vec3::new(0.0, -1.0, -1.0),
            position,
            100.0,
            Vec3::Y,
            None,
            0.1
        )
        .is_none());
        assert!(glide_assist(
            Vec3::new(0.0, 0.0, -30.0),
            position,
            110.0,
            Vec3::Y,
            None,
            0.1
        )
        .is_none());
    }

    // F during the take-off climb stops it: the player hovers where they are
    #[test]
    fn f_during_take_off_stops_the_climb() {
        assert_eq!(f_action(true, false, true, false), FAction::StopTakeOff);
        assert_eq!(f_action(true, false, true, true), FAction::StopTakeOff);
    }

    // feet resting on top of the column's surface
    fn standing_on(planet: &crate::common::PlanetData, face: u8, u: u32, v: u32) -> (Vec3, u32) {
        let top = planet.surface(face, u, v) + 1;
        let res = planet.resolution;
        let dir = crate::gen::CoordSystem::get_block_center(face, u, v, top, res).normalize();
        (
            dir * crate::gen::CoordSystem::get_layer_radius(top, res),
            top,
        )
    }

    // the lowest land column (at or above sea level): a valley floor or beach
    fn lowest_land(planet: &crate::common::PlanetData) -> (u8, u32, u32) {
        let sea = planet.terrain.sea_level();
        let res = planet.resolution;
        (0..6u8)
            .flat_map(|f| (0..res).flat_map(move |u| (0..res).map(move |v| (f, u, v))))
            .filter(|&(f, u, v)| planet.terrain.get_height(f, u, v) >= sea)
            .min_by_key(|&(f, u, v)| planet.terrain.get_height(f, u, v))
            .unwrap()
    }

    // the climb above the ground grows with the planet's relief, at least TAKEOFF_MIN_CLEARANCE
    #[test]
    fn take_off_climbs_higher_on_planets_with_more_relief() {
        let mut climbs = Vec::new();
        for res in [80u32, 160, 337] {
            let planet = crate::common::PlanetData::new(res);
            let (f, u, v) = lowest_land(&planet);
            let (feet, ground) = standing_on(&planet, f, u, v);
            let target = takeoff_target_layer(&planet, feet);
            assert!(target >= ground + TAKEOFF_MIN_CLEARANCE, "res {res}");
            climbs.push(target - ground);
        }
        assert!(climbs[0] < climbs[1] && climbs[1] < climbs[2], "{climbs:?}");
    }

    // from a valley the target stays at or below the highest peak (canyons stay flyable); on the
    // peak itself the minimum clearance applies
    #[test]
    fn take_off_stays_below_the_highest_peak() {
        for res in [80u32, 160, 337] {
            let planet = crate::common::PlanetData::new(res);
            let (_, peak) = planet.terrain.height_range();
            let (f, u, v) = lowest_land(&planet);
            let (feet, _) = standing_on(&planet, f, u, v);
            assert!(takeoff_target_layer(&planet, feet) <= peak + 1, "res {res}");
            let summit = (0..6u8)
                .flat_map(|f| (0..res).flat_map(move |u| (0..res).map(move |v| (f, u, v))))
                .find(|&(f, u, v)| planet.terrain.get_height(f, u, v) == peak)
                .unwrap();
            let (feet, ground) = standing_on(&planet, summit.0, summit.1, summit.2);
            assert_eq!(
                takeoff_target_layer(&planet, feet),
                ground + TAKEOFF_MIN_CLEARANCE
            );
        }
    }

    // swimming: the climb is measured from the water surface
    #[test]
    fn take_off_from_water_starts_at_the_surface() {
        let planet = crate::common::PlanetData::new(160);
        let sea = planet.terrain.sea_level();
        let res = planet.resolution;
        let (f, u, v) = (0..6u8)
            .flat_map(|f| (0..res).flat_map(move |u| (0..res).map(move |v| (f, u, v))))
            .find(|&(f, u, v)| planet.terrain.get_height(f, u, v) + 3 < sea)
            .unwrap();
        let dir = crate::gen::CoordSystem::get_block_center(f, u, v, sea, res).normalize();
        let floating = dir * (crate::gen::CoordSystem::get_layer_radius(sea + 1, res) - 0.5);
        let (land_feet, land_ground) = standing_on(
            &planet,
            lowest_land(&planet).0,
            lowest_land(&planet).1,
            lowest_land(&planet).2,
        );
        let land_climb = takeoff_target_layer(&planet, land_feet) - land_ground;
        assert_eq!(
            takeoff_target_layer(&planet, floating),
            sea + 1 + land_climb
        );
    }

    // landing straight down into lava would kill the player; into water is fine (they swim)
    #[test]
    fn only_damaging_liquid_below_refuses_a_landing() {
        use crate::biome::PlanetType;
        let res = 32;
        let find = |planet: &crate::common::PlanetData, want_sea: bool| {
            let sea = planet.terrain.sea_level();
            (0..6u8)
                .flat_map(|f| (0..res).flat_map(move |u| (0..res).map(move |v| (f, u, v))))
                .find(|&(f, u, v)| (planet.terrain.get_height(f, u, v) < sea) == want_sea)
                // a column's centre (get_direction gives its corner, which can resolve to a neighbour)
                .map(|(f, u, v)| {
                    crate::gen::CoordSystem::get_block_center(f, u, v, res / 2, res).normalize()
                        * 40.0
                })
                .unwrap()
        };
        let mut volcanic = crate::common::PlanetData::new(res);
        volcanic.switch_planet_type(PlanetType::Volcanic);
        assert!(
            over_damaging_liquid(&volcanic, find(&volcanic, true)),
            "over lava"
        );
        assert!(
            !over_damaging_liquid(&volcanic, find(&volcanic, false)),
            "over dry land"
        );
        let earth = crate::common::PlanetData::new(res);
        assert!(
            !over_damaging_liquid(&earth, find(&earth, true)),
            "over water"
        );
    }

    fn view_dir(rotation: Quat, cam_pitch: f32) -> Vec3 {
        (rotation * Quat::from_axis_angle(Vec3::X, cam_pitch)) * Vec3::NEG_Z
    }

    #[test]
    fn handover_has_hysteresis_between_three_and_three_and_a_half_radii() {
        assert!(should_land(2.9, true));
        assert!(
            !should_land(2.9, false),
            "never before the voxel world is ready"
        );
        assert!(!should_land(3.2, true));
        assert!(
            !should_lift_off(3.2),
            "between 3 and 3.5 radii nothing happens either way"
        );
        assert!(should_lift_off(3.6));
    }

    // landing then lifting off (or the reverse) must not move the eye or turn the view
    #[test]
    fn player_and_flight_poses_round_trip() {
        let up = Vec3::new(0.3, 0.9, -0.2).normalize();
        let position = up * 400.0;
        let heading = up.cross(Vec3::X).normalize(); // some horizontal direction
        let rotation = Quat::from_mat3(&glam::Mat3::from_cols(heading.cross(up), up, -heading));
        let cam_pitch = -0.6;
        let cam_roll = 0.7; // a rolled view stays rolled across the handovers

        let (eye, flight_rot) = flight_pose_from_player(position, rotation, cam_pitch, cam_roll);
        assert!((eye - (position + up * crate::physics::Physics::EYE_HEIGHT)).length() < 1e-4);
        assert!(((flight_rot * Vec3::NEG_Z) - view_dir(rotation, cam_pitch)).length() < 1e-4);

        let (p2, r2, pitch2, roll2) = player_pose_from_flight(eye, flight_rot);
        assert!((p2 - position).length() < 1e-3, "{p2:?} vs {position:?}");
        assert!((pitch2 - cam_pitch).abs() < 1e-4);
        assert!(
            (roll2 - cam_roll).abs() < 1e-4,
            "roll {roll2} vs {cam_roll}"
        );
        assert!((view_dir(r2, pitch2) - view_dir(rotation, cam_pitch)).length() < 1e-4);
        assert!(
            ((r2 * Vec3::Y) - up).length() < 1e-4,
            "player stands upright"
        );
    }

    // /galaxy goto looks straight at the planet's centre: heading is undefined there, and the
    // planet camera can't pitch past MAX_PITCH
    #[test]
    fn straight_down_flight_becomes_the_steepest_planet_view() {
        let eye = Vec3::new(0.0, 0.0, 500.0);
        let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, -eye.normalize());
        let (position, player_rot, pitch, _roll) = player_pose_from_flight(eye, rotation);
        assert!(position.is_finite() && player_rot.is_finite() && pitch.is_finite());
        assert!((pitch + MAX_PITCH).abs() < 1e-6, "pitch {pitch}");
        let dir = view_dir(player_rot, pitch);
        assert!(
            dir.dot(-eye.normalize()) > MAX_PITCH.sin() - 1e-4,
            "still looking down: {dir:?}"
        );
    }
    #[test]
    fn lava_lakes_refuse_landing() {
        use crate::biome::PlanetType;
        let (planet, (face, u, v)) = crate::common::tests::lake_planet(PlanetType::Volcanic);
        let res = planet.resolution;
        let level = planet.terrain.water_level(face, u, v);
        let above = crate::gen::CoordSystem::get_block_center(face, u, v, level + 5, res);
        assert!(over_damaging_liquid(&planet, above));
    }
}
