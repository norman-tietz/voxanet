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

#[cfg(test)]
mod tests {
    use super::*;

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
