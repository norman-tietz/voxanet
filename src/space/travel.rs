// travel.rs
// The automatic flight to a planet's orbit (spec §2): the climb speed from the surface, the trip through
// solar-system space — exact arrival point from the analytic orbits, a cubic Bézier bent around the star's
// heat zone, smootherstep along the arc length — and the cancel speed cap. main.rs runs the stages.

use glam::{DVec3, Quat, Vec3};

use crate::space::solar_system::{SolarSystem, HEAT_ZONE_RADII, SPACE_BOOST_SPEED};

pub const ORBIT_RADII: f64 = 3.5;
const LUT: usize = 256;
const HEAT_MARGIN: f64 = 1.2;
const LOOK_SHARE: f64 = 0.25; // the last quarter turns the view toward the planet

pub fn duration(distance: f64) -> f64 {
    (6.0 + 1.5 * (distance / 1000.0).max(1e-9).log10()).clamp(6.0, 12.0)
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
    ctrl: [DVec3; 4],
    lut: Vec<f64>, // cumulative arc length at u = i / LUT, normalised to 0..1
}

impl Trip {
    pub fn plan(
        system: &SolarSystem,
        target: usize,
        start: DVec3,
        start_dir: DVec3,
        t0: f64,
    ) -> Trip {
        let p = &system.planets[target];
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
        let mut lut = vec![0.0; LUT + 1];
        let mut prev = ctrl[0];
        for i in 1..=LUT {
            let q = bezier(&ctrl, i as f64 / LUT as f64);
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
        Trip {
            target,
            t0,
            duration: dur,
            arrival_local,
            ctrl,
            lut,
        }
    }

    // curve parameter u for an arc-length fraction s
    fn u_at(&self, s: f64) -> f64 {
        let i = self.lut.partition_point(|&l| l < s).clamp(1, LUT);
        let (a, b) = (self.lut[i - 1], self.lut[i]);
        let f = if b > a { (s - a) / (b - a) } else { 0.0 };
        ((i - 1) as f64 + f) / LUT as f64
    }

    pub fn position(&self, t: f64) -> DVec3 {
        let s = smootherstep((t - self.t0) / self.duration);
        bezier(&self.ctrl, self.u_at(s))
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

    // along the flight, turning toward the planet over the last quarter (roll: shortest arc from -Z)
    pub fn view_rotation(&self, system: &SolarSystem, t: f64) -> Quat {
        let pos = self.position(t);
        let along = self
            .velocity(t)
            .normalize_or((self.ctrl[3] - pos).normalize_or(DVec3::NEG_Z));
        let to_planet = (system.planets[self.target].position_at(t) - pos).normalize_or(along);
        let f = smootherstep(((t - self.t0) / self.duration - (1.0 - LOOK_SHARE)) / LOOK_SHARE);
        let dir = along.lerp(to_planet, f).normalize_or(to_planet).as_vec3();
        Quat::from_rotation_arc(Vec3::NEG_Z, dir)
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
        let start = s.planets[0].position_at(0.0) + DVec3::new(0.0, 500.0, 0.0);
        let trip = Trip::plan(&s, 3, start, DVec3::Y, 0.0);
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
        let trip = Trip::plan(&s, 4, s.planets[0].position_at(0.0), DVec3::X, 0.0);
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
        let trip = Trip::plan(&s, target, start, DVec3::Y, 0.0);
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
        let trip = Trip::plan(&s, 1, start, DVec3::Z, 0.0);
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

    #[test]
    fn view_turns_toward_the_planet_at_the_end() {
        let s = sys();
        let trip = Trip::plan(&s, 5, s.planets[0].position_at(0.0), DVec3::X, 0.0);
        let t1 = trip.t0 + trip.duration - 1e-3;
        let fwd = (trip.view_rotation(&s, t1) * glam::Vec3::NEG_Z).as_dvec3();
        let to_planet = (s.planets[5].position_at(t1) - trip.position(t1)).normalize();
        assert!(fwd.dot(to_planet) > 0.99);
        assert!(trip.remaining(trip.t0) > trip.remaining(t1));
    }
}
