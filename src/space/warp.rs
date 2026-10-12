// warp.rs
// The jump between solar systems as a light-speed flight (spec docs/superpowers/specs/2026-10-12-warp-jump-design.md):
// its duration by distance, the phases (align, departure, light speed, arrival), the speed of the star
// streaks and how far the scene is faded under them per moment, the unseen switch point, and the uniform
// of warp.wgsl, the full-screen streak pass drawn over the finished space frame.

use bytemuck::{Pod, Zeroable};

pub const ALIGN: f32 = 0.2;
pub const DEPARTURE: f32 = 0.2;
pub const LIGHT_SPEED: f32 = 0.3;
pub const ARRIVAL: f32 = 0.3;
pub const SWITCH_AT: f32 = 0.55; // inside the light-speed phase: nothing but streaks on screen

pub(crate) const WARP_SHADER: &str = include_str!("warp.wgsl");

// 3 s to a neighbour, 8 s across the galaxy (100 ly)
pub fn jump_duration(ly: f64) -> f32 {
    (3.0 + 5.0 * ly / 100.0).clamp(3.0, 8.0) as f32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Align,
    Departure,
    LightSpeed,
    Arrival,
    Done,
}

pub fn phase(frac: f32) -> Phase {
    if frac < ALIGN {
        Phase::Align
    } else if frac < ALIGN + DEPARTURE {
        Phase::Departure
    } else if frac < ALIGN + DEPARTURE + LIGHT_SPEED {
        Phase::LightSpeed
    } else if frac < 1.0 {
        Phase::Arrival
    } else {
        Phase::Done
    }
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

// the ramp 0 -> 1 over the departure, 1 at light speed, 1 -> 0 over the arrival
fn ramp(frac: f32) -> f32 {
    let up = smoothstep((frac - ALIGN) / DEPARTURE);
    let down = 1.0 - smoothstep((frac - (ALIGN + DEPARTURE + LIGHT_SPEED)) / ARRIVAL);
    up.min(down)
}

pub fn speed(frac: f32) -> f32 {
    ramp(frac)
}

pub fn scene_fade(frac: f32) -> f32 {
    ramp(frac)
}

// the streaks' outward flow (warp.wgsl's vanish.w), advanced by `dt` seconds at `speed`: integrated, so a
// change of speed changes how fast the stars stream, not where they are (time × speed made them jump
// every frame while the speed ramped)
pub fn advance_flow(flow: f32, dt: f32, speed: f32) -> f32 {
    flow + dt.max(0.0) * (0.4 + 2.6 * speed)
}

pub fn crosses_switch(prev_frac: f32, frac: f32) -> bool {
    prev_frac < SWITCH_AT && frac >= SWITCH_AT
}

// warp.wgsl's Warp uniform
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WarpUniform {
    pub ray_dirs: [[f32; 4]; 3], // the camera's rays: top-left corner, per screen width, per screen height
    pub vanish: [f32; 4], // xyz: the travel direction (the vanishing point), w: the flow (advance_flow)
    pub params: [f32; 4], // x: speed 0..1, y: scene fade 0..1, zw: screen size in pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_follows_the_distance() {
        assert_eq!(jump_duration(0.0), 3.0);
        assert!((jump_duration(50.0) - 5.5).abs() < 1e-6);
        assert_eq!(jump_duration(100.0), 8.0);
        assert_eq!(jump_duration(500.0), 8.0);
        assert!(jump_duration(30.0) < jump_duration(40.0));
    }

    #[test]
    fn phases_cover_the_jump_in_order() {
        assert!((ALIGN + DEPARTURE + LIGHT_SPEED + ARRIVAL - 1.0).abs() < 1e-6);
        assert_eq!(phase(0.0), Phase::Align);
        assert_eq!(phase(0.19), Phase::Align);
        assert_eq!(phase(0.25), Phase::Departure);
        assert_eq!(phase(0.5), Phase::LightSpeed);
        assert_eq!(phase(0.8), Phase::Arrival);
        assert_eq!(phase(1.0), Phase::Done);
        assert_eq!(phase(SWITCH_AT), Phase::LightSpeed, "the switch is unseen");
    }

    // Review Focus 3
    #[test]
    fn speed_and_fade_are_continuous() {
        let mut prev = (speed(0.0), scene_fade(0.0));
        for i in 1..=1000 {
            let f = i as f32 / 1000.0;
            let now = (speed(f), scene_fade(f));
            assert!(
                (now.0 - prev.0).abs() < 0.02 && (now.1 - prev.1).abs() < 0.02,
                "pop at {f}"
            );
            prev = now;
        }
        assert_eq!(
            (speed(0.1), scene_fade(0.1)),
            (0.0, 0.0),
            "align: still, scene visible"
        );
        assert_eq!(
            (speed(0.55), scene_fade(0.55)),
            (1.0, 1.0),
            "light speed: streaks only"
        );
        assert_eq!(
            (speed(1.0), scene_fade(1.0)),
            (0.0, 0.0),
            "arrived: still, scene visible"
        );
    }

    // Review Focus 4
    #[test]
    fn the_switch_happens_once() {
        // 60 fps over 4 s, and a single 1 s frame straight over the switch point
        let mut switches = 0;
        let mut prev = 0.0;
        for i in 1..=240 {
            let f = i as f32 / 240.0;
            if crosses_switch(prev, f) {
                switches += 1;
            }
            prev = f;
        }
        assert_eq!(switches, 1);
        assert!(crosses_switch(0.3, 0.9));
        assert!(!crosses_switch(0.6, 0.9));
        assert!(!crosses_switch(0.1, 0.5));
    }

    #[test]
    fn the_flow_follows_the_speed_without_jumps() {
        assert_eq!(advance_flow(1.0, 0.0, 1.0), 1.0);
        assert!(
            (advance_flow(0.0, 1.0, 0.0) - 0.4).abs() < 1e-6,
            "slow drift while aligning"
        );
        assert!((advance_flow(0.0, 1.0, 1.0) - 3.0).abs() < 1e-6);
        assert_eq!(advance_flow(2.0, -1.0, 1.0), 2.0, "never backward");
        // a frame's step stays small while the speed ramps, wherever the flow already is
        let (mut flow, mut prev_speed) = (0.0, 0.0);
        for i in 1..=600 {
            let s = speed(i as f32 / 600.0);
            let next = advance_flow(flow, 1.0 / 60.0, s);
            assert!(next - flow <= 3.0 / 60.0 + 1e-6 && next >= flow);
            (flow, prev_speed) = (next, s);
        }
        assert_eq!(prev_speed, 0.0);
    }

    // a slow frame from before the switch straight past the end: main.rs runs the switch block first
    // (the_switch_runs_before_the_end_of_the_flight), then sees the flight done
    #[test]
    fn a_long_frame_switches_and_arrives() {
        assert!(crosses_switch(0.3, 1.2));
        assert_eq!(phase(1.2), Phase::Done);
    }

    #[test]
    fn uniform_layout_and_shader() {
        assert_eq!(std::mem::size_of::<WarpUniform>(), 5 * 16);
        crate::render::renderer::tests::assert_valid_wgsl(WARP_SHADER);
        assert!(WARP_SHADER.contains("fn vs_warp(") && WARP_SHADER.contains("fn fs_warp("));
        // Review Focus 5: no streaks while still, and no acos of an out-of-range value
        assert!(WARP_SHADER.contains("if (warp.params.x <= 0.001)"));
        assert!(WARP_SHADER.contains("acos(clamp("));
        // Important 1 of the final review: the loop walks the cells whose streaks can reach the pixel
        // (heads outward of it), enough of them for the longest streak
        assert!(WARP_SHADER.contains("let cell = cell0 + f32(k);"));
        let constant = |name: &str| -> f32 {
            let line = WARP_SHADER
                .lines()
                .find(|l| l.starts_with(&format!("const {name}:")))
                .unwrap();
            let value = line.split('=').nth(1).unwrap().split(';').next().unwrap();
            value.trim().parse().unwrap()
        };
        let cells = constant("MAX_CELLS");
        assert!(cells >= (constant("MAX_LEN") / constant("RINGS")).ceil() + 1.0);
    }
}
