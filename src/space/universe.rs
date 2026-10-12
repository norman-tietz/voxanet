// universe.rs
// The solar systems you've been to (spec §2): the galaxy catalogue, the current system's index, and the
// parked systems — generated on the first jump there and kept for the session. Game.system is the current
// system itself; switch() parks it and hands back the target. Planet edits and visited flags live in
// Game, keyed by (system, planet).

use std::collections::HashMap;

use glam::{DVec3, Quat, Vec3};

use crate::space::galaxy::Galaxy;
use crate::space::solar_system::SolarSystem;

const ARRIVAL_ORBITS: f64 = 1.3; // the arrival distance, in units of the outermost orbit

pub struct Universe {
    pub galaxy: Galaxy,
    pub current: usize,
    parked: HashMap<usize, SolarSystem>,
}

impl Universe {
    pub fn new(galaxy: Galaxy) -> Self {
        Self {
            galaxy,
            current: 0,
            parked: HashMap::new(),
        }
    }

    // parks `current_system` and returns the target (from the parked systems, or generated now);
    // the current system as target hands it back untouched
    pub fn switch(
        &mut self,
        target: usize,
        current_system: SolarSystem,
    ) -> Result<SolarSystem, (SolarSystem, &'static str)> {
        if target == self.current {
            return Err((current_system, "You are here."));
        }
        let next = self
            .parked
            .remove(&target)
            .unwrap_or_else(|| SolarSystem::generate(self.galaxy.systems[target].seed));
        self.parked.insert(self.current, current_system);
        self.current = target;
        Ok(next)
    }

    // a system generated earlier (parked); None: never visited (the current one is Game.system)
    pub fn known(&self, i: usize) -> Option<&SolarSystem> {
        self.parked.get(&i)
    }

    // where a jump arrives: at the system's edge on the side facing `from_dir` (the direction toward the
    // previous system, galaxy axes = system axes), looking at the star
    pub fn arrival_pose(system: &SolarSystem, from_dir: DVec3) -> (DVec3, Quat) {
        let outer = system
            .planets
            .iter()
            .map(|p| p.orbit_radius)
            .fold(1.0, f64::max);
        let dir = from_dir.normalize_or(DVec3::Z);
        let pos = system.star.position() + dir * outer * ARRIVAL_ORBITS;
        let rot = Quat::from_rotation_arc(Vec3::NEG_Z, (-dir).as_vec3());
        (pos, rot)
    }

    // the HUD's location line (both modes): the current solar system's name, and the planet (`index`,
    // 0-based) when you're on it or captured by it, labelled like the compass ("#7 Shattered")
    pub fn location_label(&self, planet: Option<(usize, &str)>) -> String {
        let system = &self.galaxy.systems[self.current].name;
        match planet {
            Some((i, kind)) => format!("{system} · #{} {kind}", i + 1),
            None => system.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::galaxy::Galaxy;

    #[test]
    fn location_label_names_the_system_and_the_planet() {
        let (u, _) = uni();
        let name = u.galaxy.systems[0].name.clone();
        assert_eq!(u.location_label(None), name);
        assert_eq!(
            u.location_label(Some((6, "Shattered"))),
            format!("{name} · #7 Shattered")
        );
    }

    fn uni() -> (Universe, SolarSystem) {
        let g = Galaxy::generate(1);
        let start = SolarSystem::generate(g.systems[0].seed);
        (Universe::new(g), start)
    }

    #[test]
    fn a_jump_parks_the_current_system_and_generates_the_target() {
        let (mut u, start) = uni();
        let start_seed = start.seed;
        let b = u.switch(4, start).unwrap();
        assert_eq!(u.current, 4);
        assert_eq!(b.seed, u.galaxy.systems[4].seed);
        assert_eq!(u.known(0).map(|s| s.seed), Some(start_seed));
        assert!(u.known(5).is_none(), "never visited: unexplored");
    }

    // Review Focus 1
    #[test]
    fn jumping_back_returns_the_parked_system() {
        let (mut u, mut start) = uni();
        start
            .add_planet(crate::planet::biome::PlanetType::Ice, 120.0)
            .unwrap();
        let n = start.planets.len();
        let b = u.switch(4, start).unwrap();
        let a = u.switch(0, b).unwrap();
        assert_eq!(a.planets.len(), n, "the added planet is still there");
        assert_eq!(u.known(4).map(|s| s.seed), Some(u.galaxy.systems[4].seed));
    }

    // Review Focus 4
    #[test]
    fn you_are_here() {
        let (mut u, start) = uni();
        let seed = start.seed;
        let (back, why) = u.switch(0, start).unwrap_err();
        assert_eq!(back.seed, seed);
        assert_eq!(why, "You are here.");
        assert_eq!(u.current, 0);
    }

    #[test]
    fn arrival_is_at_the_edge_facing_where_you_came_from() {
        let s = SolarSystem::generate(3);
        let outer = s.planets.iter().map(|p| p.orbit_radius).fold(0.0, f64::max);
        let from = DVec3::new(1.0, 0.2, -0.5).normalize();
        let (pos, rot) = Universe::arrival_pose(&s, from);
        assert!(((pos - s.star.position()).length() - 1.3 * outer).abs() < 1e-6);
        assert!((pos - s.star.position()).normalize().dot(from) > 0.9999);
        let fwd = (rot * glam::Vec3::NEG_Z).as_dvec3();
        assert!(
            fwd.dot((s.star.position() - pos).normalize()) > 0.9999,
            "looking at the star"
        );
    }
}
