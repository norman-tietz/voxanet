// Hardware ray-traced ambient occlusion (docs/superpowers/specs/2026-10-09-rt-ao-design.md): settings,
// the /ao console command, and the rule that picks between it and the mesher's vertex AO. No GPU code.

pub const DEFAULT_RADIUS: f32 = 3.0; // world units (blocks): gaps, wall feet, steps, narrow gorges
pub const DEFAULT_RAYS: u32 = 4; // per shadow texel
pub const DEFAULT_STRENGTH: f32 = 1.0;
pub const AO_USAGE: &str = "Usage: /ao get|on|off|radius <0.5-8>|rays <1-16>|strength <0-1>";

// the AO look, tuned with /ao (cmd.rs); HwParams.ao = uniform()
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoSettings {
    pub radius: f32,
    pub rays: u32,
    pub strength: f32,
    pub enabled: bool, // off: no AO rays, the mesher's vertex AO instead (vertex_ao_wanted)
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            radius: DEFAULT_RADIUS,
            rays: DEFAULT_RAYS,
            strength: DEFAULT_STRENGTH,
            enabled: true,
        }
    }
}

impl AoSettings {
    // rt_hw.wgsl HwParams.ao: radius, ray count (0 while off: no AO rays, ao = 1), strength
    pub fn uniform(&self) -> [f32; 4] {
        let rays = if self.enabled { self.rays as f32 } else { 0.0 };
        [self.radius, rays, self.strength, 0.0]
    }

    // applies an /ao command; returns the line to log
    pub fn apply(&mut self, cmd: AoCommand) -> String {
        match cmd {
            AoCommand::Get => {}
            AoCommand::On => self.enabled = true,
            AoCommand::Off => self.enabled = false,
            AoCommand::Radius(r) => {
                self.radius = r;
                self.enabled = true;
            }
            AoCommand::Rays(n) => {
                self.rays = n;
                self.enabled = true;
            }
            AoCommand::Strength(s) => {
                self.strength = s;
                self.enabled = true;
            }
        }
        format!(
            "AO: {}, radius {:.2}, rays {}, strength {:.2}",
            if self.enabled {
                "ray traced"
            } else {
                "off (vertex AO)"
            },
            self.radius,
            self.rays,
            self.strength
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AoCommand {
    Get,
    On,
    Off,
    Radius(f32),
    Rays(u32),
    Strength(f32),
}

impl AoCommand {
    // the words after /ao
    pub fn parse(args: &[&str]) -> Result<AoCommand, &'static str> {
        let float = |s: &str, lo: f32, hi: f32| {
            s.parse::<f32>()
                .ok()
                .filter(|v| (lo..=hi).contains(v))
                .ok_or(AO_USAGE)
        };
        match args {
            ["get"] => Ok(AoCommand::Get),
            ["on"] => Ok(AoCommand::On),
            ["off"] => Ok(AoCommand::Off),
            ["radius", v] => float(v, 0.5, 8.0).map(AoCommand::Radius),
            ["rays", v] => v
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=16).contains(n))
                .map(AoCommand::Rays)
                .ok_or(AO_USAGE),
            ["strength", v] => float(v, 0.0, 1.0).map(AoCommand::Strength),
            _ => Err(AO_USAGE),
        }
    }
}

// exactly one kind of AO: the mesher's vertex AO unless ray-traced AO runs (hardware ray queries,
// hardware shadows in use, /ao on)
pub fn vertex_ao_wanted(hw_available: bool, hw_shadows: bool, ao_enabled: bool) -> bool {
    !(hw_available && hw_shadows && ao_enabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_feed_the_uniform() {
        let s = AoSettings::default();
        assert_eq!(
            (DEFAULT_RADIUS, DEFAULT_RAYS, DEFAULT_STRENGTH),
            (3.0, 4, 1.0)
        );
        assert_eq!(s.uniform(), [3.0, 4.0, 1.0, 0.0]);
    }

    #[test]
    fn ao_command_parses_and_checks_ranges() {
        assert_eq!(AoCommand::parse(&["get"]), Ok(AoCommand::Get));
        assert_eq!(AoCommand::parse(&["on"]), Ok(AoCommand::On));
        assert_eq!(AoCommand::parse(&["off"]), Ok(AoCommand::Off));
        assert_eq!(
            AoCommand::parse(&["radius", "2.5"]),
            Ok(AoCommand::Radius(2.5))
        );
        assert_eq!(AoCommand::parse(&["rays", "8"]), Ok(AoCommand::Rays(8)));
        assert_eq!(
            AoCommand::parse(&["strength", "0.5"]),
            Ok(AoCommand::Strength(0.5))
        );
        for bad in [
            &["radius", "0.2"][..],
            &["radius", "9"],
            &["rays", "0"],
            &["rays", "17"],
            &["rays", "2.5"],
            &["strength", "1.5"],
            &["strength", "x"],
            &["radius"],
            &[],
            &["nope"],
        ] {
            assert_eq!(AoCommand::parse(bad), Err(AO_USAGE), "{bad:?}");
        }
    }

    // off casts no AO rays (ray count 0) but keeps the values; on and setting a value bring them back
    #[test]
    fn off_keeps_the_values_for_on() {
        let mut s = AoSettings::default();
        s.apply(AoCommand::Rays(8));
        s.apply(AoCommand::Off);
        assert_eq!(s.uniform()[1], 0.0);
        assert!(!s.enabled);
        s.apply(AoCommand::On);
        assert_eq!(s.uniform(), [3.0, 8.0, 1.0, 0.0]);
        s.apply(AoCommand::Off);
        s.apply(AoCommand::Radius(1.5));
        assert!(s.enabled);
        assert_eq!(s.uniform()[0], 1.5);
        assert!(s.apply(AoCommand::Get).contains("radius 1.50"));
    }

    // exactly one kind of AO: vertex AO unless hardware rays are available, used and AO is on
    #[test]
    fn vertex_ao_is_used_exactly_when_rt_ao_is_not() {
        for hw in [false, true] {
            for shadows in [false, true] {
                for on in [false, true] {
                    assert_eq!(
                        vertex_ao_wanted(hw, shadows, on),
                        !(hw && shadows && on),
                        "{hw} {shadows} {on}"
                    );
                }
            }
        }
    }
}
