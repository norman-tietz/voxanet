// Plastic shading (docs/superpowers/specs/2026-10-10-plastic-shading-design.md): the settings and the
// /plastic console command, and a Rust mirror of atmosphere.wgsl's closed-form BRDF terms (atmo_plastic)
// for tests; the WGSL is what renders.

pub const PLASTIC_F0: f32 = 0.04; // Fresnel reflectance at normal incidence: dielectric plastic (atmosphere.wgsl)
pub const DEFAULT_ROUGHNESS: f32 = 0.35; // satin plastic
pub const DEFAULT_SPECULAR: f32 = 1.0; // scales highlight and reflection; 0 = the old diffuse look
pub const DEFAULT_SMUDGE: f32 = 0.3; // roughness added in smudged patches
pub const DEFAULT_SCRATCH: f32 = 0.2; // roughness added along scratches
                                      // weight of the sky reflection (the sun highlight keeps the full specular strength): physically plastic
                                      // reflects ~13 % at typical viewing angles, which under a bright bluish sky washed the block colours out
pub const DEFAULT_REFLECTION: f32 = 0.15;
pub const PLASTIC_USAGE: &str =
    "Usage: /plastic [on|off|roughness <0.05-1>|specular <0-2>|smudge <0-1>|scratch <0-1>|reflection <0-1>|env [on|off]]";

// the plastic look, tuned with /plastic (cmd.rs); GlobalUniform.plastic = uniform()
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticSettings {
    pub roughness: f32,
    pub specular: f32,
    pub smudge: f32,
    pub scratch: f32,
    pub reflection: f32, // weight of the sky reflection
    pub enabled: bool,   // off: no specular (the old diffuse look), the values kept for /plastic on
    pub env: bool, // reflect the pre-filtered environment map (env_map.rs); off: the analytic sky, today's look
}

impl Default for PlasticSettings {
    fn default() -> Self {
        Self {
            roughness: DEFAULT_ROUGHNESS,
            specular: DEFAULT_SPECULAR,
            smudge: DEFAULT_SMUDGE,
            scratch: DEFAULT_SCRATCH,
            reflection: DEFAULT_REFLECTION,
            enabled: true,
            env: true,
        }
    }
}

impl PlasticSettings {
    // shader.wgsl Global.plastic: base roughness, specular strength (0 while off), smudge, scratch
    pub fn uniform(&self) -> [f32; 4] {
        let specular = if self.enabled { self.specular } else { 0.0 };
        [self.roughness, specular, self.smudge, self.scratch]
    }

    // galaxy.wgsl PlanetUniform.plastic for the impostors: base roughness, specular strength (0 while
    // off), reflection weight; no imperfections
    pub fn impostor_uniform(&self) -> [f32; 4] {
        let u = self.uniform();
        [u[0], u[1], self.reflection, 0.0]
    }

    // shader.wgsl Global.plastic_ext: x = sky reflection weight, y = 1 when the environment map is reflected
    pub fn uniform_ext(&self) -> [f32; 4] {
        [self.reflection, if self.env { 1.0 } else { 0.0 }, 0.0, 0.0]
    }

    // applies a /plastic command; returns the line to log
    pub fn apply(&mut self, cmd: PlasticCommand) -> String {
        match cmd {
            PlasticCommand::Get => {}
            PlasticCommand::On => self.enabled = true,
            PlasticCommand::Off => self.enabled = false,
            PlasticCommand::Roughness(v) => (self.roughness, self.enabled) = (v, true),
            PlasticCommand::Specular(v) => (self.specular, self.enabled) = (v, true),
            PlasticCommand::Smudge(v) => (self.smudge, self.enabled) = (v, true),
            PlasticCommand::Scratch(v) => (self.scratch, self.enabled) = (v, true),
            PlasticCommand::Reflection(v) => (self.reflection, self.enabled) = (v, true),
            PlasticCommand::Env(v) => self.env = v,
        }
        format!(
            "Plastic: {}, roughness {:.2}, specular {:.2}, smudge {:.2}, scratch {:.2}, reflection {:.2}, env {}",
            if self.enabled { "on" } else { "off" },
            self.roughness,
            self.specular,
            self.smudge,
            self.scratch,
            self.reflection,
            if self.env { "on" } else { "off" }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlasticCommand {
    Get,
    On,
    Off,
    Roughness(f32),
    Specular(f32),
    Smudge(f32),
    Scratch(f32),
    Reflection(f32),
    Env(bool),
}

impl PlasticCommand {
    // the words after /plastic
    pub fn parse(args: &[&str]) -> Result<PlasticCommand, &'static str> {
        let value = |s: &str, lo: f32, hi: f32| {
            s.parse::<f32>()
                .ok()
                .filter(|v| (lo..=hi).contains(v))
                .ok_or(PLASTIC_USAGE)
        };
        match args {
            [] => Ok(PlasticCommand::Get),
            ["on"] => Ok(PlasticCommand::On),
            ["off"] => Ok(PlasticCommand::Off),
            ["roughness", v] => value(v, 0.05, 1.0).map(PlasticCommand::Roughness),
            ["specular", v] => value(v, 0.0, 2.0).map(PlasticCommand::Specular),
            ["smudge", v] => value(v, 0.0, 1.0).map(PlasticCommand::Smudge),
            ["scratch", v] => value(v, 0.0, 1.0).map(PlasticCommand::Scratch),
            ["reflection", v] => value(v, 0.0, 1.0).map(PlasticCommand::Reflection),
            ["env", "on"] => Ok(PlasticCommand::Env(true)),
            ["env", "off"] => Ok(PlasticCommand::Env(false)),
            _ => Err(PLASTIC_USAGE),
        }
    }
}

// --- mirror of shader.wgsl plastic_roughness's face-coordinate choice ---

#[cfg_attr(not(test), allow(dead_code))]
const NO_EDGE_F: f32 = crate::bevel::NO_EDGE;

// the two edge slots (Vertex::edge, bevel.rs) whose distances serve as a face's scratch coordinates:
// two neighbouring edges. Fans of five or more sides (slot 4 used: hex cells) take sides 0 and 1; a
// square takes 0 and 3 (its slot 1 is parallel to 3), or 1 where 3 is masked (a hex wall piece's joint).
#[cfg_attr(not(test), allow(dead_code))]
pub fn scratch_slots(edge: [f32; 8]) -> Option<(usize, usize)> {
    let real = |d: f32| d < 1e3;
    if !real(edge[0]) {
        None
    } else if real(edge[4]) {
        Some((0, 1))
    } else if real(edge[3]) {
        Some((0, 3))
    } else if real(edge[1]) {
        Some((0, 1))
    } else {
        None
    }
}

// --- mirror of atmosphere.wgsl atmo_plastic's closed-form terms ---

// GGX (Trowbridge-Reitz) normal distribution at cos(theta_h) = n_dot_h, roughness r (alpha = r²)
#[cfg_attr(not(test), allow(dead_code))]
pub fn ggx_d(n_dot_h: f32, r: f32) -> f32 {
    let a = r * r;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * d * d)
}

// Schlick's Fresnel at cos(theta) = `cos`
#[cfg_attr(not(test), allow(dead_code))]
pub fn fresnel(cos: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos).powi(5)
}

// Schlick's Fresnel with the roughness correction, for the sky reflection
#[cfg_attr(not(test), allow(dead_code))]
pub fn fresnel_rough(n_dot_v: f32, r: f32, f0: f32) -> f32 {
    f0 + ((1.0 - r).max(f0) - f0) * (1.0 - n_dot_v).powi(5)
}

// weight of the diffuse terms: what the reflection doesn't take (1 at strength 0)
#[cfg_attr(not(test), allow(dead_code))]
pub fn diffuse_weight(strength: f32, n_dot_v: f32, r: f32) -> f32 {
    1.0 - strength * fresnel_rough(n_dot_v, r, PLASTIC_F0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_feed_the_uniform() {
        let s = PlasticSettings::default();
        assert_eq!(
            (
                DEFAULT_ROUGHNESS,
                DEFAULT_SPECULAR,
                DEFAULT_SMUDGE,
                DEFAULT_SCRATCH
            ),
            (0.35, 1.0, 0.3, 0.2)
        );
        assert_eq!(s.uniform(), [0.35, 1.0, 0.3, 0.2]);
    }

    #[test]
    fn plastic_command_parses_and_checks_ranges() {
        assert_eq!(PlasticCommand::parse(&[]), Ok(PlasticCommand::Get));
        assert_eq!(PlasticCommand::parse(&["on"]), Ok(PlasticCommand::On));
        assert_eq!(PlasticCommand::parse(&["off"]), Ok(PlasticCommand::Off));
        assert_eq!(
            PlasticCommand::parse(&["roughness", "0.5"]),
            Ok(PlasticCommand::Roughness(0.5))
        );
        assert_eq!(
            PlasticCommand::parse(&["specular", "1.5"]),
            Ok(PlasticCommand::Specular(1.5))
        );
        assert_eq!(
            PlasticCommand::parse(&["smudge", "0.7"]),
            Ok(PlasticCommand::Smudge(0.7))
        );
        assert_eq!(
            PlasticCommand::parse(&["scratch", "0"]),
            Ok(PlasticCommand::Scratch(0.0))
        );
        for bad in [
            &["roughness", "0.01"][..],
            &["roughness", "1.5"],
            &["specular", "3"],
            &["smudge", "-0.1"],
            &["scratch", "x"],
            &["roughness"],
            &["nope"],
        ] {
            assert_eq!(PlasticCommand::parse(bad), Err(PLASTIC_USAGE), "{bad:?}");
        }
    }

    // off: no specular (strength 0 in the uniform) but the values stay; on and setting a value bring it back
    #[test]
    fn off_keeps_the_values_for_on() {
        let mut s = PlasticSettings::default();
        s.apply(PlasticCommand::Roughness(0.6));
        s.apply(PlasticCommand::Off);
        assert_eq!(s.uniform()[1], 0.0);
        assert_eq!(s.uniform()[0], 0.6);
        s.apply(PlasticCommand::On);
        assert_eq!(s.uniform(), [0.6, 1.0, 0.3, 0.2]);
        s.apply(PlasticCommand::Off);
        s.apply(PlasticCommand::Specular(1.5));
        assert_eq!(s.uniform()[1], 1.5);
        assert!(s.apply(PlasticCommand::Get).contains("roughness 0.60"));
    }

    // the GGX distribution is normalised: integrated over the hemisphere, weighted by cos(theta_h), it is 1
    #[test]
    fn ggx_distribution_integrates_to_one() {
        for r in [0.2f32, 0.35, 0.8] {
            let steps = 4000;
            let mut sum = 0.0f64;
            for i in 0..steps {
                let theta = (i as f64 + 0.5) / steps as f64 * std::f64::consts::FRAC_PI_2;
                let d = ggx_d(theta.cos() as f32, r) as f64;
                // dω = 2π sin θ dθ (rotationally symmetric), weighted by cos θ
                sum += d
                    * theta.cos()
                    * 2.0
                    * std::f64::consts::PI
                    * theta.sin()
                    * (std::f64::consts::FRAC_PI_2 / steps as f64);
            }
            assert!((sum - 1.0).abs() < 0.02, "roughness {r}: {sum}");
        }
    }

    #[test]
    fn fresnel_goes_from_f0_to_one() {
        assert!((fresnel(1.0, PLASTIC_F0) - PLASTIC_F0).abs() < 1e-6);
        assert!((fresnel(0.0, PLASTIC_F0) - 1.0).abs() < 1e-6);
        for r in [0.05f32, 0.35, 1.0] {
            assert!((fresnel_rough(1.0, r, PLASTIC_F0) - PLASTIC_F0).abs() < 1e-6);
            assert!(fresnel_rough(0.0, r, PLASTIC_F0) <= 1.0);
        }
    }

    // /plastic off (strength 0) leaves the diffuse light untouched at every angle
    #[test]
    fn strength_zero_keeps_the_diffuse_light() {
        for n_v in [0.0f32, 0.3, 1.0] {
            assert_eq!(diffuse_weight(0.0, n_v, 0.35), 1.0);
            assert!(diffuse_weight(1.0, n_v, 0.35) < 1.0);
        }
    }
    // the WGSL is what renders: same F0, and the shared function exists (atmosphere.wgsl is prepended to
    // the scene and the galaxy shaders)
    #[test]
    fn wgsl_uses_the_same_constants() {
        let src = include_str!("atmosphere.wgsl");
        assert!(src.contains(&format!("const PLASTIC_F0: f32 = {PLASTIC_F0};")));
        assert!(src.contains("fn atmo_plastic("));
    }
    // the sky reflection has its own weight: physically ~13 % at typical angles, which washed the block
    // colours out under a bright sky; the sun highlight keeps the full strength
    #[test]
    fn reflection_has_its_own_weight() {
        let mut s = PlasticSettings::default();
        assert_eq!(DEFAULT_REFLECTION, 0.15);
        assert_eq!(s.uniform_ext(), [DEFAULT_REFLECTION, 1.0, 0.0, 0.0]);
        assert_eq!(
            PlasticCommand::parse(&["reflection", "0.5"]),
            Ok(PlasticCommand::Reflection(0.5))
        );
        assert_eq!(
            PlasticCommand::parse(&["reflection", "1.5"]),
            Err(PLASTIC_USAGE)
        );
        s.apply(PlasticCommand::Off);
        s.apply(PlasticCommand::Reflection(0.5));
        assert!(s.enabled);
        assert_eq!(s.uniform_ext()[0], 0.5);
        assert!(s.apply(PlasticCommand::Get).contains("reflection 0.50"));
    }
    // the galaxy impostors get the base roughness, the strength and the reflection weight (no
    // imperfections), so the landing handover doesn't jump in brightness
    #[test]
    fn impostors_get_the_base_look() {
        let mut s = PlasticSettings::default();
        assert_eq!(
            s.impostor_uniform(),
            [DEFAULT_ROUGHNESS, DEFAULT_SPECULAR, DEFAULT_REFLECTION, 0.0]
        );
        s.apply(PlasticCommand::Off);
        assert_eq!(s.impostor_uniform()[1], 0.0);
    }
    // scratches need two neighbouring edges as face coordinates: a square's slots 0 and 3 (1 is
    // parallel to 3), a hex fan's sides 0 and 1 (3 is opposite 0), a wall piece whose slot 3 is a
    // joint (masked) its slot 1; none without slot 0
    #[test]
    fn scratch_axes_are_neighbouring_edges() {
        let n = NO_EDGE_F;
        assert_eq!(
            scratch_slots([0.2, 0.5, 0.8, 0.3, n, n, n, n]),
            Some((0, 3))
        ); // square
        assert_eq!(
            scratch_slots([0.2, 0.5, 0.8, 0.9, 0.7, 0.4, n, n]),
            Some((0, 1))
        ); // hexagon
        assert_eq!(scratch_slots([0.2, 0.5, 0.8, n, n, n, n, n]), Some((0, 1))); // wall piece, joint at a
        assert_eq!(scratch_slots([0.2, n, 0.8, n, n, n, n, n]), None); // both ends joints
        assert_eq!(scratch_slots([n; 8]), None); // low-poly facet, LOD, water
    }

    // the WGSL implements the same choice and the other review fixes
    #[test]
    fn wgsl_plastic_follows_the_review_fixes() {
        let atmo = include_str!("atmosphere.wgsl");
        // the reflected sky without the sun disc (the sun is the GGX highlight); the sky now comes in
        // as `env`, from the env map's bake (shader.wgsl) or atmo_env_analytic
        let f = &atmo[atmo.find("fn atmo_env_analytic(").unwrap()..];
        let f = &f[..f.find("fn atmo_plastic(").unwrap()];
        assert!(
            !f.contains("atmo_sky_over_black"),
            "sun disc in the reflection"
        );
        assert!(f.contains("atmo_sky_dome(R"));
        let shader = include_str!("shader.wgsl");
        let r = &shader[shader.find("fn plastic_roughness(").unwrap()..];
        assert!(r.contains("edge1"), "polygons pick sides 0 and 1");
        assert!(!r.contains("floor(world_pos"), "seed from the world grid");
    }

    #[test]
    fn env_toggle_parses_and_rides_in_the_uniform() {
        assert_eq!(
            PlasticCommand::parse(&["env", "off"]),
            Ok(PlasticCommand::Env(false))
        );
        assert_eq!(
            PlasticCommand::parse(&["env", "on"]),
            Ok(PlasticCommand::Env(true))
        );
        assert_eq!(PlasticCommand::parse(&["env", "maybe"]), Err(PLASTIC_USAGE));
        let mut s = PlasticSettings::default();
        assert!(s.env);
        assert_eq!(s.uniform_ext()[1], 1.0);
        s.apply(PlasticCommand::Env(false));
        assert_eq!(s.uniform_ext()[1], 0.0);
        assert!(PLASTIC_USAGE.contains("env [on|off]"));
    }
}
