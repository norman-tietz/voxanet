// Fill-light tuning: the ground's bounce light and wrap lighting (/light), shared by the voxel scene
// (shade()) and the galaxy impostors (planet_color) so the landing handover keeps the look. A Rust mirror
// of atmosphere.wgsl's closed-form terms (atmo_wrap_diffuse) for tests; the WGSL is what renders.

// how much of the sunlit ground's light bounces up into the lower half of a surface's sky light, in the
// surface's own colour (atmo_ground_bounce): walls and overhangs pick up the colour around them
pub const DEFAULT_BOUNCE: f32 = 0.5;
// how far diffuse sunlight wraps past the terminator (atmo_wrap_diffuse); 0 = plain Lambert, crisp
pub const DEFAULT_WRAP: f32 = 0.0;
const MAX_WRAP: f32 = 0.5;
pub const LIGHT_USAGE: &str = "Usage: /light [bounce <0-1>|wrap <0-0.5>]";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightSettings {
    pub bounce: f32,
    pub wrap: f32,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            bounce: DEFAULT_BOUNCE,
            wrap: DEFAULT_WRAP,
        }
    }
}

impl LightSettings {
    // shader.wgsl Global.light and galaxy.wgsl PlanetUniform.light: x = bounce, y = wrap
    pub fn uniform(&self) -> [f32; 4] {
        [self.bounce, self.wrap, 0.0, 0.0]
    }

    // applies a /light command; returns the line to log
    pub fn apply(&mut self, cmd: LightCommand) -> String {
        match cmd {
            LightCommand::Get => {}
            LightCommand::Bounce(v) => self.bounce = v,
            LightCommand::Wrap(v) => self.wrap = v,
        }
        format!("Light: bounce {:.2}, wrap {:.2}", self.bounce, self.wrap)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightCommand {
    Get,
    Bounce(f32),
    Wrap(f32),
}

impl LightCommand {
    // the words after /light
    pub fn parse(args: &[&str]) -> Result<LightCommand, &'static str> {
        let value = |s: &str, max: f32| {
            s.parse::<f32>()
                .ok()
                .filter(|v| (0.0..=max).contains(v))
                .ok_or(LIGHT_USAGE)
        };
        match args {
            [] => Ok(LightCommand::Get),
            ["bounce", v] => value(v, 1.0).map(LightCommand::Bounce),
            ["wrap", v] => value(v, MAX_WRAP).map(LightCommand::Wrap),
            _ => Err(LIGHT_USAGE),
        }
    }
}

// atmosphere.wgsl atmo_wrap_diffuse: the diffuse sun term with wrap, (N·L + w) / (1 + w), clamped at 0
#[cfg_attr(not(test), allow(dead_code))]
pub fn wrap_diffuse(n_l: f32, wrap: f32) -> f32 {
    ((n_l + wrap) / (1.0 + wrap)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_feed_the_uniform() {
        assert_eq!(LightSettings::default().uniform(), [0.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn light_commands_parse_and_apply() {
        assert_eq!(LightCommand::parse(&[]), Ok(LightCommand::Get));
        assert_eq!(
            LightCommand::parse(&["bounce", "0.8"]),
            Ok(LightCommand::Bounce(0.8))
        );
        assert_eq!(
            LightCommand::parse(&["wrap", "0.25"]),
            Ok(LightCommand::Wrap(0.25))
        );
        assert_eq!(LightCommand::parse(&["wrap", "0.6"]), Err(LIGHT_USAGE));
        assert_eq!(LightCommand::parse(&["bounce", "-1"]), Err(LIGHT_USAGE));
        assert_eq!(LightCommand::parse(&["glow"]), Err(LIGHT_USAGE));
        let mut s = LightSettings::default();
        assert_eq!(
            s.apply(LightCommand::Wrap(0.3)),
            "Light: bounce 0.50, wrap 0.30"
        );
        assert_eq!(s.uniform(), [0.5, 0.3, 0.0, 0.0]);
    }

    // no wrap is Lambert; with wrap the lit side stays fully lit facing the sun, and light reaches
    // past the terminator, fading out at N·L = −wrap
    #[test]
    fn wrap_diffuse_extends_lambert() {
        for n_l in [-1.0, -0.2, 0.0, 0.4, 1.0f32] {
            assert_eq!(wrap_diffuse(n_l, 0.0), n_l.max(0.0));
        }
        assert_eq!(wrap_diffuse(1.0, 0.3), 1.0);
        assert!(wrap_diffuse(0.0, 0.3) > 0.2);
        assert_eq!(wrap_diffuse(-0.3, 0.3), 0.0);
        assert!(wrap_diffuse(0.5, 0.3) > wrap_diffuse(0.4, 0.3));
    }
}
