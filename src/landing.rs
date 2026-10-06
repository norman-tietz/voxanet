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
    // roll: how far the camera's right axis is turned about the view from the unrolled camera's
    // (Player::get_view_matrix applies it after the pitch)
    let unrolled = player_rotation * Quat::from_axis_angle(Vec3::X, pitch);
    let right = rotation * Vec3::X;
    let roll = right
        .dot(unrolled * Vec3::Y)
        .atan2(right.dot(unrolled * Vec3::X));
    (
        eye - up * Physics::EYE_HEIGHT,
        player_rotation.normalize(),
        pitch,
        roll,
    )
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

// dive assist (planet fly mode): racing steeply at the surface, the flight is bent toward the horizon
// so it becomes a sweeping descent instead of a crash. It engages when the surface (ground or water)
// is less than DIVE_ASSIST_SECONDS away at the current downward speed, pulls harder the closer the
// impact, and stops at DIVE_ASSIST_TARGET below the horizon; slow descents are left alone
pub const DIVE_ASSIST_SECONDS: f32 = 4.0;
pub const DIVE_ASSIST_TARGET: f32 = 10.0 * std::f32::consts::PI / 180.0;
const DIVE_ASSIST_MIN_SPEED: f32 = 15.0; // world units/s downward
const DIVE_ASSIST_MAX_RATE: f32 = 1.5; // rad/s of pitch-up, approached as the impact nears

// the pitch-up (radians, toward the horizon) the dive assist applies this step, for a velocity
// `velocity` at `altitude` above the surface (`up`: the local radial up); 0 when not engaged
pub fn dive_assist(velocity: Vec3, up: Vec3, altitude: f32, dt: f32) -> f32 {
    let down = -velocity.dot(up);
    if down < DIVE_ASSIST_MIN_SPEED || altitude <= 0.0 {
        return 0.0;
    }
    let time_to_impact = altitude / down;
    let dive = (down / velocity.length()).clamp(-1.0, 1.0).asin();
    if time_to_impact >= DIVE_ASSIST_SECONDS || dive <= DIVE_ASSIST_TARGET {
        return 0.0;
    }
    let pull = DIVE_ASSIST_MAX_RATE * (1.0 - time_to_impact / DIVE_ASSIST_SECONDS);
    (pull * dt).min(dive - DIVE_ASSIST_TARGET)
}

// the radius of the surface below `position`: the top of its column, or the water surface above it
pub fn surface_radius(planet: &crate::common::PlanetData, position: Vec3) -> f32 {
    let res = planet.resolution;
    let sea = planet.terrain.sea_level();
    // the column's water level: its lake's, else the sea (dry land stands above it anyway)
    let top = crate::gen::CoordSystem::pos_to_id(position, res).map_or(sea + 1, |id| {
        planet
            .surface(id.face, id.u, id.v)
            .max(planet.terrain.water_level(id.face, id.u, id.v))
            + 1
    });
    crate::gen::CoordSystem::get_layer_radius(top, res)
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

    // a fast, steep dive at the surface (impact within DIVE_ASSIST_SECONDS) is bent up toward a
    // shallow descent: the assist pitches up, harder the closer the impact, and stops at the target
    #[test]
    fn a_fast_dive_is_bent_toward_a_shallow_descent() {
        let up = Vec3::Y;
        let mut v = Vec3::new(0.0, -200.0, -20.0); // ~84 degrees down, fast
        let mut altitude = 600.0; // 3 s to impact
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            let pitch = dive_assist(v, up, altitude, dt);
            if pitch > 0.0 {
                let axis = v.cross(up).normalize();
                v = Quat::from_axis_angle(axis, pitch) * v;
            }
            altitude = (altitude + v.dot(up) * dt).max(1.0);
        }
        let dive = dive_deg(v, up);
        assert!(
            (dive - DIVE_ASSIST_TARGET.to_degrees()).abs() < 0.5,
            "ended at {dive} degrees"
        );
    }

    // the closer the impact, the harder the pull
    #[test]
    fn the_pull_grows_as_impact_nears() {
        let (up, v) = (Vec3::Y, Vec3::new(0.0, -200.0, -20.0));
        let far = dive_assist(v, up, 700.0, 0.1); // 3.5 s
        let near = dive_assist(v, up, 200.0, 0.1); // 1 s
        assert!(far > 0.0 && near > far, "{far} {near}");
    }

    // left alone: slow descents, shallow dives, impacts further than DIVE_ASSIST_SECONDS away, climbs
    #[test]
    fn slow_shallow_distant_or_climbing_flight_is_not_assisted() {
        let up = Vec3::Y;
        assert_eq!(
            dive_assist(Vec3::new(0.0, -10.0, -1.0), up, 20.0, 0.1),
            0.0,
            "slow"
        );
        assert_eq!(
            dive_assist(Vec3::new(0.0, -15.0, -200.0), up, 30.0, 0.1),
            0.0,
            "shallow"
        );
        assert_eq!(
            dive_assist(Vec3::new(0.0, -200.0, -20.0), up, 1000.0, 0.1),
            0.0,
            "5 s away"
        );
        assert_eq!(
            dive_assist(Vec3::new(0.0, 200.0, -20.0), up, 50.0, 0.1),
            0.0,
            "climbing"
        );
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
    fn lava_lakes_refuse_landing_and_surfaces_use_the_lake_level() {
        use crate::biome::PlanetType;
        let (planet, (face, u, v)) = crate::common::tests::lake_planet(PlanetType::Volcanic);
        let res = planet.resolution;
        let level = planet.terrain.water_level(face, u, v);
        let above = crate::gen::CoordSystem::get_block_center(face, u, v, level + 5, res);
        assert!(over_damaging_liquid(&planet, above));
        let lake_surface = crate::gen::CoordSystem::get_layer_radius(level + 1, res);
        assert!((surface_radius(&planet, above) - lake_surface).abs() < 1e-3);
    }
}
