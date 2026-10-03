// biome.rs
// The whole planet's type: Earth-like (today's planet), Volcanic (lava, damaging, glowing),
// Ice (no liquid at all, walkable and slippery). Switched at runtime (main.rs, `B` key),
// regenerating edits but reusing the same terrain shape in every case — only which
// material/liquid/atmosphere fills that shape changes. Phase 1 deliberately keeps terrain
// shape (noise, continents, mountains) identical across types and only swaps what fills it;
// more types (Desert/Toxic/Ocean) can be added later as pure data in the `def()` table below.
// (Internal design note, not part of this repo: 2026-10-02-planet-types-design.md.)

use crate::material::BlockType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanetType {
    EarthLike,
    Volcanic,
    Ice,
}

impl PlanetType {
    pub const ALL: [PlanetType; 3] = [PlanetType::EarthLike, PlanetType::Volcanic, PlanetType::Ice];

    // matches a planet type's name case-insensitively, ignoring spaces and hyphens ("Earth-like",
    // "earthlike", "EARTHLIKE"); used by --biome and /galaxy add
    pub fn from_name(name: &str) -> Option<PlanetType> {
        let normalize = |s: &str| s.to_lowercase().replace(['-', ' '], "");
        let wanted = normalize(name);
        PlanetType::ALL
            .into_iter()
            .find(|t| normalize(t.def().name) == wanted)
    }

    pub fn def(self) -> PlanetTypeDef {
        match self {
            PlanetType::EarthLike => PlanetTypeDef {
                name: "Earth-like",
                day_length_secs: 120.0,
                liquid: Some(LiquidDef {
                    shallow_color: [0.020, 0.150, 0.170],
                    deep_color: [0.002, 0.030, 0.090],
                    behavior: LiquidBehavior::Reflective,
                    damaging: false,
                }),
                palette: Palette {
                    peak: BlockType::Snow,
                    rock: BlockType::Stone,
                    beach: BlockType::Sand,
                    ground: BlockType::Grass,
                    subsurface: BlockType::Dirt,
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.15, 0.3, 0.6],
                    sky_horizon_warm: [0.88, 1.08, 1.48],
                    cloud_light: [0.92, 0.94, 0.98],
                    cloud_dark: [0.16, 0.18, 0.24],
                    space_color: [0.010, 0.015, 0.030],
                },
            },
            PlanetType::Volcanic => PlanetTypeDef {
                name: "Volcanic",
                day_length_secs: 60.0,
                liquid: Some(LiquidDef {
                    shallow_color: [1.4, 0.5, 0.05],
                    deep_color: [0.35, 0.05, 0.02],
                    behavior: LiquidBehavior::Glowing,
                    damaging: true,
                }),
                palette: Palette {
                    peak: BlockType::Ember,
                    rock: BlockType::Obsidian,
                    beach: BlockType::Ash,
                    ground: BlockType::Basalt,
                    subsurface: BlockType::Basalt,
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.25, 0.08, 0.04],
                    sky_horizon_warm: [0.9, 0.35, 0.1],
                    cloud_light: [0.55, 0.45, 0.4],
                    cloud_dark: [0.12, 0.09, 0.08],
                    space_color: [0.03, 0.01, 0.015],
                },
            },
            PlanetType::Ice => PlanetTypeDef {
                name: "Ice",
                day_length_secs: 240.0,
                liquid: None,
                palette: Palette {
                    peak: BlockType::Snow,
                    rock: BlockType::Stone,
                    beach: BlockType::Ice,
                    ground: BlockType::Stone,
                    subsurface: BlockType::Stone,
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.55, 0.65, 0.78],
                    sky_horizon_warm: [0.85, 0.9, 1.0],
                    cloud_light: [0.96, 0.97, 1.0],
                    cloud_dark: [0.55, 0.62, 0.72],
                    space_color: [0.02, 0.025, 0.04],
                },
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LiquidBehavior {
    Reflective,
    Glowing,
}

#[derive(Clone, Copy, Debug)]
pub struct LiquidDef {
    pub shallow_color: [f32; 3],
    pub deep_color: [f32; 3],
    pub behavior: LiquidBehavior,
    pub damaging: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct AtmosphereDef {
    pub sky_zenith: [f32; 3],
    pub sky_horizon_warm: [f32; 3],
    pub cloud_light: [f32; 3],
    pub cloud_dark: [f32; 3],
    pub space_color: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Palette {
    pub peak: BlockType,
    pub rock: BlockType,
    pub beach: BlockType,
    pub ground: BlockType,
    pub subsurface: BlockType,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanetTypeDef {
    pub name: &'static str,
    // seconds for one full day/night cycle (the planet's axial spin period); each type's own,
    // since a faster or slower spin is just more data, same as its palette or atmosphere
    pub day_length_secs: f32,
    pub liquid: Option<LiquidDef>,
    pub palette: Palette,
    pub atmosphere: AtmosphereDef,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --biome and /galaxy add accept the same spellings
    #[test]
    fn planet_type_names_parse_like_the_biome_flag() {
        assert_eq!(
            PlanetType::from_name("Earth-like"),
            Some(PlanetType::EarthLike)
        );
        assert_eq!(
            PlanetType::from_name("earthlike"),
            Some(PlanetType::EarthLike)
        );
        assert_eq!(
            PlanetType::from_name("VOLCANIC"),
            Some(PlanetType::Volcanic)
        );
        assert_eq!(PlanetType::from_name("ice"), Some(PlanetType::Ice));
        assert_eq!(PlanetType::from_name("lava"), None);
    }

    #[test]
    fn earth_like_def_matches_todays_values() {
        let def = PlanetType::EarthLike.def();
        assert_eq!(def.name, "Earth-like");
        let liquid = def.liquid.expect("Earth-like has water");
        assert!(matches!(liquid.behavior, LiquidBehavior::Reflective));
        assert!(!liquid.damaging);
        assert_eq!(def.palette.ground, crate::material::BlockType::Grass);
        assert_eq!(def.palette.peak, crate::material::BlockType::Snow);
    }

    #[test]
    fn volcanic_liquid_is_glowing_and_damaging() {
        let def = PlanetType::Volcanic.def();
        let liquid = def.liquid.expect("Volcanic has lava");
        assert!(matches!(liquid.behavior, LiquidBehavior::Glowing));
        assert!(liquid.damaging);
    }

    #[test]
    fn each_planet_type_has_a_positive_day_length() {
        for t in PlanetType::ALL {
            assert!(t.def().day_length_secs > 0.0, "{}", t.def().name);
        }
    }

    #[test]
    fn ice_has_no_liquid_and_no_grass_anywhere() {
        let def = PlanetType::Ice.def();
        assert!(def.liquid.is_none());
        assert_ne!(def.palette.ground, crate::material::BlockType::Grass);
        assert_ne!(def.palette.peak, crate::material::BlockType::Grass);
        assert_ne!(def.palette.rock, crate::material::BlockType::Grass);
        assert_ne!(def.palette.beach, crate::material::BlockType::Grass);
    }
}
