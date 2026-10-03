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

// galaxy flight (eye position and orientation, planet frame) -> the planet engine's player: feet
// position, upright rotation (local Y = up, local -Z = heading) and camera pitch. The eye stays where
// it is; roll is dropped (captured flight is levelled anyway); pitch is clamped to MAX_PITCH.
pub fn player_pose_from_flight(eye: Vec3, rotation: Quat) -> (Vec3, Quat, f32) {
    let up = eye.normalize();
    let forward = rotation * Vec3::NEG_Z;
    let pitch = forward
        .dot(up)
        .clamp(-1.0, 1.0)
        .asin()
        .clamp(-MAX_PITCH, MAX_PITCH);
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
    (
        eye - up * Physics::EYE_HEIGHT,
        player_rotation.normalize(),
        pitch,
    )
}

// the planet engine's player -> galaxy flight: eye position and the camera's orientation (the same
// rotation * pitch the first-person view matrix uses, Player::get_view_matrix), so the view doesn't
// move on liftoff
pub fn flight_pose_from_player(position: Vec3, rotation: Quat, cam_pitch: f32) -> (Vec3, Quat) {
    let up = position.normalize();
    let eye = position + up * Physics::EYE_HEIGHT;
    (
        eye,
        (rotation * Quat::from_axis_angle(Vec3::X, cam_pitch)).normalize(),
    )
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

// how far take-off lifts the player off the ground (fly mode velocity follows input every tick, so
// the "small upward push" is a position nudge, not a velocity impulse)
pub const TAKEOFF_LIFT: f32 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FAction {
    StartLanding,
    CancelLanding,
    RefuseLanding,
    TakeOff,
}

// F: flying -> land (or cancel a landing in progress; refused over damaging liquid);
// walking/swimming -> take off
pub fn f_action(flying: bool, landing: bool, over_damaging_liquid: bool) -> FAction {
    match (flying, landing) {
        (true, true) => FAction::CancelLanding,
        (true, false) if over_damaging_liquid => FAction::RefuseLanding,
        (true, false) => FAction::StartLanding,
        (false, _) => FAction::TakeOff,
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
    crate::gen::CoordSystem::pos_to_id(probe, res).is_some_and(|id| {
        planet.terrain.get_height(id.face, id.u, id.v) < planet.terrain.sea_level()
    })
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
        assert_eq!(f_action(true, false, false), FAction::StartLanding);
        assert_eq!(f_action(true, true, false), FAction::CancelLanding);
        assert_eq!(
            f_action(true, true, true),
            FAction::CancelLanding,
            "cancel is always allowed"
        );
        assert_eq!(f_action(true, false, true), FAction::RefuseLanding);
        assert_eq!(f_action(false, false, false), FAction::TakeOff);
        assert_eq!(f_action(false, false, true), FAction::TakeOff);
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

        let (eye, flight_rot) = flight_pose_from_player(position, rotation, cam_pitch);
        assert!((eye - (position + up * crate::physics::Physics::EYE_HEIGHT)).length() < 1e-4);
        assert!(((flight_rot * Vec3::NEG_Z) - view_dir(rotation, cam_pitch)).length() < 1e-4);

        let (p2, r2, pitch2) = player_pose_from_flight(eye, flight_rot);
        assert!((p2 - position).length() < 1e-3, "{p2:?} vs {position:?}");
        assert!((pitch2 - cam_pitch).abs() < 1e-4);
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
        let (position, player_rot, pitch) = player_pose_from_flight(eye, rotation);
        assert!(position.is_finite() && player_rot.is_finite() && pitch.is_finite());
        assert!((pitch + MAX_PITCH).abs() < 1e-6, "pitch {pitch}");
        let dir = view_dir(player_rot, pitch);
        assert!(
            dir.dot(-eye.normalize()) > MAX_PITCH.sin() - 1e-4,
            "still looking down: {dir:?}"
        );
    }
}
