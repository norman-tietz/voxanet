// Film look: animated grain and a vignette over the finished frame (fs_film, shader.wgsl), drawn in
// both render paths right after the lens flare, under the HUD. Settings and the /film console command;
// no GPU code.

pub const DEFAULT_GRAIN: f32 = 0.04; // ± per-pixel brightness noise, as a fraction
pub const DEFAULT_VIGNETTE: f32 = 0.25; // how much darker the corners get
const MAX_GRAIN: f32 = 0.2;
pub const FILM_USAGE: &str = "Usage: /film get|on|off|grain <0-0.2>|vignette <0-1>";

// the film look, tuned with /film (cmd.rs); FilmUniform = uniform()
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmSettings {
    pub grain: f32,
    pub vignette: f32,
    pub enabled: bool, // off draws nothing but keeps the values for /film on
}

impl Default for FilmSettings {
    fn default() -> Self {
        Self {
            grain: DEFAULT_GRAIN,
            vignette: DEFAULT_VIGNETTE,
            enabled: true,
        }
    }
}

impl FilmSettings {
    // shader.wgsl FilmParams: grain, vignette (0 while off), time in seconds (a new grain pattern every
    // frame), screen width / height (a round vignette)
    pub fn uniform(&self, time: f32, aspect: f32) -> [f32; 4] {
        let on = if self.enabled { 1.0 } else { 0.0 };
        [self.grain * on, self.vignette * on, time, aspect]
    }

    // applies a /film command; returns the line to log
    pub fn apply(&mut self, cmd: FilmCommand) -> String {
        match cmd {
            FilmCommand::Get => {}
            FilmCommand::On => self.enabled = true,
            FilmCommand::Off => self.enabled = false,
            FilmCommand::Grain(g) => {
                self.grain = g;
                self.enabled = true;
            }
            FilmCommand::Vignette(v) => {
                self.vignette = v;
                self.enabled = true;
            }
        }
        format!(
            "Film: {}, grain {:.2}, vignette {:.2}",
            if self.enabled { "on" } else { "off" },
            self.grain,
            self.vignette
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilmCommand {
    Get,
    On,
    Off,
    Grain(f32),
    Vignette(f32),
}

impl FilmCommand {
    // the words after /film
    pub fn parse(args: &[&str]) -> Result<FilmCommand, &'static str> {
        let value = |s: &str, max: f32| {
            s.parse::<f32>()
                .ok()
                .filter(|v| (0.0..=max).contains(v))
                .ok_or(FILM_USAGE)
        };
        match args {
            ["get"] => Ok(FilmCommand::Get),
            ["on"] => Ok(FilmCommand::On),
            ["off"] => Ok(FilmCommand::Off),
            ["grain", v] => value(v, MAX_GRAIN).map(FilmCommand::Grain),
            ["vignette", v] => value(v, 1.0).map(FilmCommand::Vignette),
            _ => Err(FILM_USAGE),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_feed_the_uniform() {
        let s = FilmSettings::default();
        assert_eq!((DEFAULT_GRAIN, DEFAULT_VIGNETTE), (0.04, 0.25));
        assert_eq!(
            s.uniform(12.5, 1.6),
            [DEFAULT_GRAIN, DEFAULT_VIGNETTE, 12.5, 1.6]
        );
    }

    #[test]
    fn film_command_parses_and_checks_ranges() {
        assert_eq!(FilmCommand::parse(&["get"]), Ok(FilmCommand::Get));
        assert_eq!(FilmCommand::parse(&["on"]), Ok(FilmCommand::On));
        assert_eq!(FilmCommand::parse(&["off"]), Ok(FilmCommand::Off));
        assert_eq!(
            FilmCommand::parse(&["grain", "0.1"]),
            Ok(FilmCommand::Grain(0.1))
        );
        assert_eq!(
            FilmCommand::parse(&["vignette", "0.5"]),
            Ok(FilmCommand::Vignette(0.5))
        );
        for bad in [
            &["grain", "0.3"][..],
            &["vignette", "1.5"],
            &["grain", "-0.1"],
            &["grain", "x"],
            &["grain"],
            &[],
            &["nope"],
        ] {
            assert_eq!(FilmCommand::parse(bad), Err(FILM_USAGE), "{bad:?}");
        }
    }

    // off draws nothing (grain and vignette 0) but keeps the values; on and setting a value bring it back
    #[test]
    fn off_keeps_the_values_for_on() {
        let mut s = FilmSettings::default();
        s.apply(FilmCommand::Grain(0.1));
        s.apply(FilmCommand::Off);
        assert_eq!(s.uniform(0.0, 1.0)[..2], [0.0, 0.0]);
        s.apply(FilmCommand::On);
        assert_eq!(s.uniform(0.0, 1.0)[..2], [0.1, DEFAULT_VIGNETTE]);
        s.apply(FilmCommand::Off);
        s.apply(FilmCommand::Vignette(0.5));
        assert_eq!(s.uniform(0.0, 1.0)[..2], [0.1, 0.5]);
        assert!(s.apply(FilmCommand::Get).contains("vignette 0.50"));
    }
}
