// travel.rs
// The automatic flight to a planet's orbit (spec §2): the climb speed from the surface, the trip through
// solar-system space — exact arrival point from the analytic orbits, a cubic Bézier bent around the star's
// heat zone, smootherstep along the arc length — and the cancel speed cap. main.rs runs the stages.

use glam::{DQuat, DVec3, Quat, Vec3};

use crate::space::solar_system::{
    Planet, SolarSystem, HEAT_ZONE_RADII, SPACE_BOOST_SPEED, SYSTEM_SCALE,
};

pub const ORBIT_RADII: f64 = 3.5;
const LUT: usize = 256;
const HEAT_MARGIN: f64 = 1.2;
// a moon's trip keeps this many parent radii from the parent (around its cloud shell)
const PARENT_MARGIN: f64 = 1.4;
const LOOK_SHARE: f64 = 0.25; // the last quarter turns the view toward the planet
const START_SHARE: f64 = 0.15; // the first 15 % turns it from where it looked into the flight direction
const CHECK_SAMPLES: usize = 256; // samples of the final heat-zone check

pub fn duration(distance: f64) -> f64 {
    (6.0 + 1.5 * (distance / (1000.0 * SYSTEM_SCALE)).max(1e-9).log10()).clamp(6.0, 12.0)
}

// the automatic climb from the surface: faster with altitude (~2-3 s to the liftoff handover)
pub fn climb_speed(altitude: f32) -> f32 {
    (altitude * 1.2).max(15.0)
}

pub fn cancel_velocity(v: DVec3) -> DVec3 {
    let cap = SPACE_BOOST_SPEED as f64;
    if v.length() > cap {
        v.normalize() * cap
    } else {
        v
    }
}

fn smootherstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

fn bezier(c: &[DVec3; 4], u: f64) -> DVec3 {
    let v = 1.0 - u;
    c[0] * (v * v * v) + c[1] * (3.0 * v * v * u) + c[2] * (3.0 * v * u * u) + c[3] * (u * u * u)
}

pub struct Trip {
    pub target: usize,
    pub t0: f64,
    pub duration: f64,
    pub arrival_local: DVec3, // the orbit point in the target's frame
    planet: Planet,           // the target's orbit, for the co-moving arrival
    start_forward: DVec3,     // the view direction when the trip was planned
    ctrl: [DVec3; 4],
    lut: Vec<f64>, // cumulative arc length at u = i / LUT, normalised to 0..1
}

// cumulative arc length of the Bézier at u = i / LUT, normalised to 0..1
fn arc_lut(ctrl: &[DVec3; 4]) -> Vec<f64> {
    let mut lut = vec![0.0; LUT + 1];
    let mut prev = ctrl[0];
    for i in 1..=LUT {
        let q = bezier(ctrl, i as f64 / LUT as f64);
        lut[i] = lut[i - 1] + (q - prev).length();
        prev = q;
    }
    let total = lut[LUT];
    if total > 0.0 {
        lut.iter_mut().for_each(|l| *l /= total);
    } else {
        lut.iter_mut()
            .enumerate()
            .for_each(|(i, l)| *l = i as f64 / LUT as f64);
    }
    lut
}

// turns direction a toward b by the share f of the angle between them (about the axis `fallback`
// when they're opposite, so the turn never passes through a zero vector)
fn turn_toward(a: DVec3, b: DVec3, f: f64, fallback: DVec3) -> DVec3 {
    let cos = a.dot(b).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle * f < 1e-9 {
        return a;
    }
    let axis = a
        .cross(b)
        .try_normalize()
        .or_else(|| a.cross(fallback).try_normalize())
        .or_else(|| a.any_orthogonal_vector().try_normalize())
        .unwrap_or(DVec3::Y);
    DQuat::from_axis_angle(axis, angle * f) * a
}

impl Trip {
    // start_forward: the view direction now, which the view leaves over the first START_SHARE of the trip
    pub fn plan(
        system: &SolarSystem,
        target: usize,
        start: DVec3,
        start_dir: DVec3,
        start_forward: DVec3,
        t0: f64,
    ) -> Trip {
        let p = system.planets[target];
        let star = system.star.position();
        // the planet moves during the trip: settle the duration on the distance to where it will be
        let mut dur = duration((p.position_at(t0) - start).length());
        for _ in 0..4 {
            dur = duration((p.position_at(t0 + dur) - start).length());
        }
        let t1 = t0 + dur;
        let sun = p.sun_dir_in_planet_frame(star, t1).as_dvec3().normalize();
        let arrival_local = sun * ORBIT_RADII * p.radius as f64;
        let arrival = p.from_planet_frame(arrival_local, t1);
        let centre = p.position_at(t1);
        let len = (arrival - start).length().max(1.0);
        let out = (arrival - centre).normalize_or(DVec3::Y);
        let mut ctrl = [
            start,
            start + start_dir.normalize_or(out) * len / 3.0,
            arrival + out * len / 3.0,
            arrival,
        ];
        // bend the curve around the star's heat zone
        let limit = HEAT_ZONE_RADII * system.star.radius * HEAT_MARGIN;
        for _ in 0..12 {
            let (mut closest, mut at) = (f64::MAX, DVec3::ZERO);
            for i in 0..=64 {
                let q = bezier(&ctrl, i as f64 / 64.0);
                let d = (q - star).length();
                if d < closest {
                    closest = d;
                    at = q;
                }
            }
            if closest > limit {
                break;
            }
            let away =
                (at - star).normalize_or((ctrl[1] - star).cross(DVec3::Y).normalize_or(DVec3::X));
            let push = away * (limit - closest + limit * 0.5) * 1.5;
            ctrl[1] += push;
            ctrl[2] += push;
        }
        let mut trip = Trip {
            target,
            t0,
            duration: dur,
            arrival_local,
            planet: p,
            start_forward: start_forward.normalize_or(start_dir.normalize_or(out)),
            ctrl,
            lut: arc_lut(&ctrl),
        };
        // the final check on the path actually flown (the co-moving term included): if it still
        // touches the zone, swing both inner control points out to one side of the star, wider each
        // time, until it clears
        let zone = HEAT_ZONE_RADII * system.star.radius;
        let mid = (start + arrival) * 0.5 - star;
        let chord = (arrival - start).normalize_or(DVec3::X);
        let side = (mid - chord * mid.dot(chord))
            .try_normalize()
            .or_else(|| chord.cross(DVec3::Y).try_normalize())
            .unwrap_or(DVec3::Z);
        let mut reach = limit.max(start.distance(star)).max(arrival.distance(star)) * 1.5;
        for _ in 0..16 {
            if trip.clears(|_| star, (zone * 1.05).min(start.distance(star) * 0.999)) {
                break;
            }
            trip.ctrl[1] = star + side * reach + (start - star) * 0.5;
            trip.ctrl[2] = star + side * reach + (arrival - star) * 0.5;
            trip.lut = arc_lut(&trip.ctrl);
            reach *= 1.5;
        }
        // a moon's parent is an obstacle too: the same swing-out around the parent's sphere
        if let Some(parent) = p.parent.map(|i| system.planets[i]) {
            let tm = t0 + dur * 0.5;
            let pc = parent.position_at(tm);
            let limit = parent.radius as f64 * PARENT_MARGIN;
            let mid = (start + arrival) * 0.5 - pc;
            let chord = (arrival - start).normalize_or(DVec3::X);
            let side = (mid - chord * mid.dot(chord))
                .try_normalize()
                .or_else(|| chord.cross(DVec3::Y).try_normalize())
                .unwrap_or(DVec3::Z);
            let star_limit = (zone * 1.05).min(start.distance(star) * 0.999);
            let parent_ok = |t: &Trip| t.clears(|x| parent.position_at(x), limit);
            let star_ok = |t: &Trip| t.clears(|_| star, star_limit);
            // push both inner control points further out to one side, wider each time (either side,
            // whichever keeps the star's zone clear too); if the parent can't be cleared, keep the last
            // configuration that clears the star (the base one at worst)
            let base = trip.ctrl;
            let mut best = base;
            let mut done = parent_ok(&trip) && star_ok(&trip);
            let mut reach = limit * 0.5;
            'swing: for _ in 0..16 {
                if done {
                    break;
                }
                for sign in [1.0, -1.0] {
                    trip.ctrl[1] = base[1] + side * sign * reach;
                    trip.ctrl[2] = base[2] + side * sign * reach;
                    trip.lut = arc_lut(&trip.ctrl);
                    if star_ok(&trip) {
                        best = trip.ctrl;
                        if parent_ok(&trip) {
                            done = true;
                            break 'swing;
                        }
                    }
                }
                reach *= 1.5;
            }
            if !done {
                trip.ctrl = best;
                trip.lut = arc_lut(&best);
            }
        }
        trip
    }

    // the flown path stays farther than `limit` from the obstacle (its position as a function of time)
    fn clears(&self, obstacle: impl Fn(f64) -> DVec3, limit: f64) -> bool {
        (0..=CHECK_SAMPLES).all(|i| {
            let t = self.t0 + self.duration * i as f64 / CHECK_SAMPLES as f64;
            self.position(t).distance(obstacle(t)) > limit
        })
    }

    // curve parameter u for an arc-length fraction s
    fn u_at(&self, s: f64) -> f64 {
        let i = self.lut.partition_point(|&l| l < s).clamp(1, LUT);
        let (a, b) = (self.lut[i - 1], self.lut[i]);
        let f = if b > a { (s - a) / (b - a) } else { 0.0 };
        ((i - 1) as f64 + f) / LUT as f64
    }

    // along the curve, plus the target's motion blended in (0 at both ends), so the ship arrives
    // co-moving with the planet instead of at rest in solar-system space
    pub fn position(&self, t: f64) -> DVec3 {
        let x = (t - self.t0) / self.duration;
        let s = smootherstep(x);
        let t1 = self.t0 + self.duration;
        let follow = self.planet.position_at(t) - self.planet.position_at(t1);
        bezier(&self.ctrl, self.u_at(s)) + follow * s
    }

    pub fn velocity(&self, t: f64) -> DVec3 {
        let h = 1e-3;
        (self.position(t + h) - self.position(t - h)) / (2.0 * h)
    }

    pub fn done(&self, t: f64) -> bool {
        t >= self.t0 + self.duration
    }

    pub fn remaining(&self, t: f64) -> f64 {
        (self.t0 + self.duration - t).max(0.0)
    }

    // where the view should look at t: from the view direction at plan time toward the flight
    // direction over the first START_SHARE, toward the planet over the last LOOK_SHARE
    fn desired_forward(&self, system: &SolarSystem, t: f64) -> DVec3 {
        let x = (t - self.t0) / self.duration;
        let pos = self.position(t);
        let along = self
            .velocity(t)
            .try_normalize()
            .unwrap_or_else(|| (self.ctrl[3] - pos).normalize_or(self.start_forward));
        let to_planet = (system.planets[self.target].position_at(t) - pos).normalize_or(along);
        let f_start = smootherstep(x / START_SHARE);
        let f_end = smootherstep((x - (1.0 - LOOK_SHARE)) / LOOK_SHARE);
        let fallback = self.start_forward.cross(along).normalize_or(DVec3::Y);
        let base = turn_toward(self.start_forward, along, f_start, fallback);
        turn_toward(base, to_planet, f_end, fallback)
    }

    // the view at t, turned from the current one by the shortest arc to the desired direction: no
    // fixed reference axis, so the player's roll is kept and no heading is singular
    pub fn view_rotation(&self, system: &SolarSystem, t: f64, current: Quat) -> Quat {
        let cur = (current * Vec3::NEG_Z).normalize();
        let want = self.desired_forward(system, t).as_vec3().normalize_or(cur);
        if cur.dot(want) < -0.9999 {
            // exactly opposite: turn about the camera's own up rather than an arbitrary axis
            let up = current * Vec3::Y;
            return (Quat::from_axis_angle(up, std::f32::consts::PI) * current).normalize();
        }
        (Quat::from_rotation_arc(cur, want) * current).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::solar_system::{SolarSystem, HEAT_ZONE_RADII};
    use glam::DVec3;

    fn sys() -> SolarSystem {
        SolarSystem::generate(1)
    }

    // a trip to a moon never passes through its parent, from either side
    #[test]
    fn trips_to_a_moon_avoid_the_parent_sphere() {
        let s = crate::space::solar_system::tests::with_moon();
        let m = s.planets.len() - 1;
        let parent = s.planets[2];
        for side in [DVec3::X, DVec3::NEG_X, DVec3::Z, DVec3::NEG_Z] {
            let start = parent.position_at(10.0) + side * 30.0 * parent.radius as f64;
            let trip = Trip::plan(&s, m, start, -side, -side, 10.0);
            for k in 0..=200 {
                let t = trip.t0 + trip.duration * k as f64 / 200.0;
                let d = (trip.position(t) - parent.position_at(t)).length();
                assert!(
                    d >= parent.radius as f64 * 1.1,
                    "{side:?} {k}: {d} vs {}",
                    parent.radius
                );
            }
        }
    }

    // a start inside the parent's margin cannot clear it by any swing: the trip must still keep the star's
    // zone clear
    #[test]
    fn a_failed_parent_swing_keeps_the_star_clearance() {
        let s = crate::space::solar_system::tests::with_moon();
        let m = s.planets.len() - 1;
        let parent = s.planets[2];
        let star = s.star.position();
        let zone = HEAT_ZONE_RADII * s.star.radius;
        for side in [DVec3::X, DVec3::NEG_X, DVec3::Z, DVec3::NEG_Z] {
            let start = parent.position_at(10.0) + side * 1.05 * parent.radius as f64;
            let trip = Trip::plan(&s, m, start, side, side, 10.0);
            assert!(!trip.clears(
                |x| parent.position_at(x),
                parent.radius as f64 * PARENT_MARGIN
            ));
            for k in 0..=200 {
                let t = trip.t0 + trip.duration * k as f64 / 200.0;
                assert!((trip.position(t) - star).length() > zone, "{side:?} {k}");
            }
        }
    }

    #[test]
    fn duration_grows_with_distance_within_bounds() {
        assert_eq!(duration(10.0), 6.0);
        assert!(duration(1e5) > duration(1e4));
        assert_eq!(duration(1e12), 12.0);
    }

    // Review Focus 1
    #[test]
    fn trip_ends_at_the_orbit_point_at_arrival() {
        let s = sys();
        let start = s.planets[0].position_at(0.0) + DVec3::new(0.0, 500.0 * SYSTEM_SCALE, 0.0);
        let trip = Trip::plan(&s, 3, start, DVec3::Y, DVec3::X, 0.0);
        let t1 = trip.t0 + trip.duration;
        let p = &s.planets[3];
        let expected = p.from_planet_frame(trip.arrival_local, t1);
        assert!((trip.position(t1) - expected).length() < 1e-3);
        assert!((trip.position(trip.t0) - start).length() < 1e-6);
        assert!((trip.arrival_local.length() - ORBIT_RADII * p.radius as f64).abs() < 1e-6);
        // over the day side
        let sun = p.sun_dir_in_planet_frame(s.star.position(), t1).as_dvec3();
        assert!(trip.arrival_local.normalize().dot(sun) > 0.999);
        assert!(trip.done(t1) && !trip.done(trip.t0 + 0.5 * trip.duration));
    }

    #[test]
    fn speed_is_smooth_and_eases_in_and_out() {
        let s = sys();
        let trip = Trip::plan(
            &s,
            4,
            s.planets[0].position_at(0.0),
            DVec3::X,
            DVec3::X,
            0.0,
        );
        let n = 200;
        let mut prev = trip.position(0.0);
        let mut speeds = Vec::new();
        for i in 1..=n {
            let t = trip.duration * i as f64 / n as f64;
            let p = trip.position(t);
            speeds.push((p - prev).length());
            prev = p;
        }
        let max = speeds.iter().cloned().fold(0.0, f64::max);
        assert!(
            speeds[0] < 0.05 * max && speeds[n - 1] < 0.05 * max,
            "eases in and out"
        );
        for w in speeds.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.1 * max, "no jumps");
        }
    }

    // Review Focus 2
    #[test]
    fn trip_avoids_the_heat_zone() {
        let s = sys();
        // start opposite the target across the star
        let target = 2;
        let tp = s.planets[target].position_at(0.0) - s.star.position();
        let start = s.star.position() - tp;
        let trip = Trip::plan(&s, target, start, DVec3::Y, DVec3::Y, 0.0);
        let limit = HEAT_ZONE_RADII * s.star.radius;
        for i in 0..=400 {
            let t = trip.duration * i as f64 / 400.0;
            assert!(
                (trip.position(t) - s.star.position()).length() > limit,
                "inside the heat zone at {t}"
            );
        }
    }

    // Review Focus 3
    #[test]
    fn short_trip_is_well_defined() {
        let s = sys();
        let p = &s.planets[1];
        let t1_guess = 6.0;
        let start = p.from_planet_frame(
            p.sun_dir_in_planet_frame(s.star.position(), t1_guess)
                .as_dvec3()
                * ORBIT_RADII
                * p.radius as f64,
            t1_guess,
        );
        let trip = Trip::plan(&s, 1, start, DVec3::Z, DVec3::Z, 0.0);
        assert_eq!(trip.duration, 6.0);
        for i in 0..=20 {
            let t = trip.duration * i as f64 / 20.0;
            assert!(trip.position(t).is_finite() && trip.velocity(t).is_finite());
        }
    }

    #[test]
    fn cancel_caps_the_speed_and_climb_speeds_up() {
        let v = cancel_velocity(DVec3::new(1e6, 0.0, 0.0));
        assert!((v.length() - crate::space::solar_system::SPACE_BOOST_SPEED as f64).abs() < 1e-6);
        assert_eq!(
            cancel_velocity(DVec3::new(3.0, 0.0, 0.0)),
            DVec3::new(3.0, 0.0, 0.0)
        );
        assert!(climb_speed(100.0) > climb_speed(10.0));
        assert!(climb_speed(0.0) >= 15.0);
    }

    // flies a whole trip at 60 Hz the way main.rs does, returning the view at every frame
    fn fly(s: &SolarSystem, trip: &Trip, start: Quat) -> Vec<(f64, Quat)> {
        let mut q = start;
        let mut out = vec![(trip.t0, q)];
        let frames = (trip.duration * 60.0).ceil() as usize;
        for k in 1..frames {
            let t = trip.t0 + k as f64 / 60.0;
            q = trip.view_rotation(s, t, q);
            out.push((t, q));
        }
        out
    }

    #[test]
    fn view_turns_toward_the_planet_at_the_end() {
        let s = sys();
        let trip = Trip::plan(
            &s,
            5,
            s.planets[0].position_at(0.0),
            DVec3::X,
            DVec3::NEG_Z,
            0.0,
        );
        let views = fly(&s, &trip, Quat::IDENTITY);
        let (t, q) = *views.last().unwrap();
        let fwd = (q * Vec3::NEG_Z).as_dvec3();
        let to_planet = (s.planets[5].position_at(t) - trip.position(t)).normalize();
        assert!(fwd.dot(to_planet) > 0.99);
        assert!(trip.remaining(trip.t0) > trip.remaining(t));
    }

    // Important 2: the view starts where the player looked, with the player's roll, and turns only by
    // shortest arcs: no twist about the view axis, no snaps, no singular heading
    #[test]
    fn view_keeps_the_roll_and_never_jumps() {
        let s = sys();
        let rolled = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.2);
        let start_fwd = (rolled * Vec3::NEG_Z).as_dvec3();
        let trip = Trip::plan(
            &s,
            4,
            s.planets[0].position_at(0.0),
            DVec3::Y,
            start_fwd,
            0.0,
        );
        let views = fly(&s, &trip, rolled);
        // the first frame is the player's own view, roll included
        assert!(views[1].1.angle_between(rolled) < 1e-3);
        let mut twist = 0.0f32;
        for w in views.windows(2) {
            let (a, b) = (w[0].1, w[1].1);
            assert!(
                a.angle_between(b) < 10f32.to_radians(),
                "a jump at {}",
                w[1].0
            );
            // the turn's component about the view axis (shortest arcs have none to first order)
            let (axis, angle) = (b * a.inverse()).to_axis_angle();
            twist += axis.dot(b * Vec3::NEG_Z) * angle;
        }
        assert!(twist.abs() < 2f32.to_radians(), "twisted by {twist}");
    }

    #[test]
    fn view_heading_through_plus_z_is_calm() {
        let s = sys();
        let target = 3;
        let guess = s.planets[target].position_at(9.0);
        // start behind the target along -Z and to its side, heading +X+Z: the heading has to swing
        // across +Z (with a small vertical component) to come round to the planet
        let start =
            guess - DVec3::Z * (30000.0 * SYSTEM_SCALE) - DVec3::X * (6000.0 * SYSTEM_SCALE)
                + DVec3::Y * (50.0 * SYSTEM_SCALE);
        let heading = DVec3::new(1.0, 0.02, 1.0).normalize();
        let trip = Trip::plan(&s, target, start, heading, heading, 0.0);
        let look = Quat::from_rotation_arc(Vec3::NEG_Z, heading.as_vec3());
        let views = fly(&s, &trip, look);
        let mut crossed = false;
        for w in views.windows(2) {
            let (a, b) = (w[0].1, w[1].1);
            assert!(
                a.angle_between(b) < 10f32.to_radians(),
                "a jump at {}",
                w[1].0
            );
            assert!((a * Vec3::Y).dot(b * Vec3::Y) > 10f32.to_radians().cos());
            let (fa, fb) = (a * Vec3::NEG_Z, b * Vec3::NEG_Z);
            crossed |= fa.x.signum() != fb.x.signum() && fb.z > 0.5;
        }
        assert!(crossed, "the heading never crossed +Z");
    }

    // Important 3: the trip ends co-moving with the planet
    #[test]
    fn arrival_moves_with_the_planet() {
        let s = sys();
        for target in [0, 3, 6] {
            let start =
                s.planets[(target + 2) % 7].position_at(0.0) + DVec3::Y * (300.0 * SYSTEM_SCALE);
            let trip = Trip::plan(&s, target, start, DVec3::Y, DVec3::Y, 0.0);
            let t1 = trip.t0 + trip.duration;
            let p = &s.planets[target];
            let planet_v = (p.position_at(t1 + 1e-3) - p.position_at(t1 - 1e-3)) / 2e-3;
            let v = trip.velocity(t1 - 0.02);
            assert!(
                (v - planet_v).length() < 0.05 * planet_v.length(),
                "#{target}: {v} vs {planet_v}"
            );
            // over the last frame or so the ship barely moves relative to the planet
            let rel = |t: f64| trip.position(t) - p.position_at(t);
            let d = (rel(t1) - rel(t1 - 0.02)).length();
            assert!(d < 0.5, "#{target}: {d}");
        }
    }

    // Minor 6: whatever the start, the flown path never enters the heat zone
    #[test]
    fn no_start_enters_the_heat_zone() {
        let s = sys();
        let star = s.star.position();
        let zone = HEAT_ZONE_RADII * s.star.radius;
        for target in 0..s.planets.len() {
            for k in 0..12 {
                let ang = std::f64::consts::TAU * k as f64 / 12.0;
                let r = zone * (1.05 + 0.5 * (k % 3) as f64);
                let start = star + DVec3::new(ang.cos(), 0.05 * (k as f64 - 6.0), ang.sin()) * r;
                // heading straight at the star: the hardest start
                let dir = (star - start).normalize();
                let trip = Trip::plan(&s, target, start, dir, dir, 0.0);
                for i in 1..=400 {
                    let t = trip.duration * i as f64 / 400.0;
                    assert!(
                        trip.position(t).distance(star) > zone,
                        "#{target} from {k}: inside the heat zone at {t}"
                    );
                }
            }
        }
    }
}
