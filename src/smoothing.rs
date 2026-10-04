// smoothing.rs
// Easing for flight controls: key presses and releases ramp speeds and rates instead of switching
// them instantly (planet fly mode, orbit and galaxy flight).

use glam::Vec3;

// time constant of eased flight movement: velocity covers 63 % of the way to its target in this many
// seconds and ~95 % in three times that, at any speed (so cruise, boost and altitude-scaled speeds all
// respond alike)
pub const FLIGHT_EASE_SECONDS: f32 = 0.25;

// flight keys (A/D roll, Q/E turn) set a target rate; the actual rate ramps toward it at the accel,
// so a key reaches full rate in 0.3 s and coasts to a stop in 0.3 s after release
pub const ROLL_SPEED: f32 = 1.5; // rad/s
pub const ROLL_ACCEL: f32 = 5.0; // rad/s^2
pub const FLIGHT_TURN_SPEED: f32 = 1.5; // rad/s
pub const TURN_ACCEL: f32 = 5.0; // rad/s^2

// `current` eased toward `target` over `dt` with time constant `tau` (frame-rate independent)
pub fn ease_toward(current: Vec3, target: Vec3, dt: f32, tau: f32) -> Vec3 {
    current + (target - current) * (1.0 - (-dt / tau).exp())
}

// a rate moved toward `target` by at most `max_delta` (rate ramps for roll and turn keys)
pub fn approach(current: f32, target: f32, max_delta: f32) -> f32 {
    current + (target - current).clamp(-max_delta, max_delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    // easing toward a target is the same whatever the frame rate
    #[test]
    fn ease_toward_is_frame_rate_independent() {
        let target = Vec3::new(100.0, 0.0, -40.0);
        let run = |fps: u32| {
            let mut v = Vec3::ZERO;
            for _ in 0..fps / 2 {
                v = ease_toward(v, target, 1.0 / fps as f32, 0.25);
            }
            v
        };
        assert!(
            (run(30) - run(120)).length() < 1e-2,
            "{:?} vs {:?}",
            run(30),
            run(120)
        );
        // half a second is two time constants: 1 - e^-2 of the way
        let expected = target * (1.0 - (-2.0f32).exp());
        assert!((run(60) - expected).length() < 1e-2);
    }

    // a rate approaches its target by at most max_delta per step and never overshoots
    #[test]
    fn approach_steps_toward_the_target_without_overshooting() {
        assert_eq!(approach(0.0, 1.5, 0.5), 0.5);
        assert_eq!(approach(1.4, 1.5, 0.5), 1.5);
        assert_eq!(approach(1.5, 0.0, 0.5), 1.0);
        assert_eq!(approach(-0.2, 0.0, 0.5), 0.0);
    }
}
