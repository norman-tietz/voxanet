// biome.rs
// The whole planet's type, as pure data in the `def()` table: Earth-like (water), Volcanic (lava),
// Ice (no liquid, slippery ice shores), Desert (no liquid), Toxic (glowing sludge), Ocean (water,
// sandy islands) and Shattered (glowing plasma). Every galaxy planet has one (galaxy.rs cycles
// through `ALL`). Each type has its own landscape (noise.rs TerrainShape, one profile per type on a
// shared continent mask) and fills it with its own blocks, liquid (look and physics, `LiquidPhysics`),
// atmosphere and day length.
// (Internal design note, not part of this repo: 2026-10-02-planet-types-design.md.)

use crate::material::BlockType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanetType {
    EarthLike,
    Volcanic,
    Ice,
    Desert,
    Toxic,
    Ocean,
    Shattered,
}

impl PlanetType {
    pub const ALL: [PlanetType; 7] = [
        PlanetType::EarthLike,
        PlanetType::Volcanic,
        PlanetType::Ice,
        PlanetType::Desert,
        PlanetType::Toxic,
        PlanetType::Ocean,
        PlanetType::Shattered,
    ];

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
                    name: "water",
                    physics: WATER,
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
                    name: "lava",
                    physics: LAVA,
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
            PlanetType::Desert => PlanetTypeDef {
                name: "Desert",
                day_length_secs: 300.0, // very long, hot days
                liquid: None,
                palette: Palette {
                    peak: BlockType::Stone,
                    rock: BlockType::Stone,
                    beach: BlockType::Sand,
                    ground: BlockType::Sand,
                    subsurface: BlockType::Sand, // no sandstone block (yet)
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.85, 0.65, 0.40],
                    sky_horizon_warm: [1.2, 0.9, 0.6],
                    cloud_light: [0.95, 0.90, 0.85],
                    cloud_dark: [0.70, 0.65, 0.60],
                    space_color: [0.04, 0.03, 0.02],
                },
            },
            PlanetType::Toxic => PlanetTypeDef {
                name: "Toxic",
                day_length_secs: 150.0,
                liquid: Some(LiquidDef {
                    shallow_color: [0.2, 1.2, 0.1], // poisonous neon green
                    deep_color: [0.05, 0.3, 0.02],
                    behavior: LiquidBehavior::Glowing,
                    name: "toxic sludge",
                    physics: SLUDGE,
                }),
                palette: Palette {
                    peak: BlockType::Stone,
                    rock: BlockType::Obsidian,
                    beach: BlockType::Ash, // no mud block (yet)
                    ground: BlockType::Dirt,
                    subsurface: BlockType::Dirt,
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.3, 0.5, 0.3], // sickly yellow-green light
                    sky_horizon_warm: [0.6, 0.7, 0.2],
                    cloud_light: [0.4, 0.5, 0.3],
                    cloud_dark: [0.15, 0.2, 0.1],
                    space_color: [0.015, 0.025, 0.015],
                },
            },
            PlanetType::Ocean => PlanetTypeDef {
                name: "Ocean",
                day_length_secs: 180.0,
                liquid: Some(LiquidDef {
                    shallow_color: [0.05, 0.4, 0.6],
                    deep_color: [0.005, 0.05, 0.15],
                    behavior: LiquidBehavior::Reflective,
                    name: "water",
                    physics: WATER,
                }),
                palette: Palette {
                    peak: BlockType::Stone,
                    rock: BlockType::Stone,
                    beach: BlockType::Sand,
                    ground: BlockType::Sand, // islands are mostly beach
                    subsurface: BlockType::Dirt,
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.1, 0.25, 0.5], // heavy, stormy blue sky
                    sky_horizon_warm: [0.5, 0.6, 0.8],
                    cloud_light: [0.6, 0.65, 0.7],
                    cloud_dark: [0.2, 0.25, 0.35],
                    space_color: [0.01, 0.01, 0.02],
                },
            },
            PlanetType::Shattered => PlanetTypeDef {
                name: "Shattered",
                day_length_secs: 90.0, // fast, chaotic rotation
                liquid: Some(LiquidDef {
                    shallow_color: [1.2, 0.1, 1.5], // glowing plasma (magenta/purple)
                    deep_color: [0.3, 0.02, 0.4],
                    behavior: LiquidBehavior::Glowing,
                    name: "plasma",
                    physics: PLASMA,
                }),
                palette: Palette {
                    peak: BlockType::Ember, // glowing peaks
                    rock: BlockType::Obsidian,
                    beach: BlockType::Ash,
                    ground: BlockType::Basalt,
                    subsurface: BlockType::Basalt,
                },
                atmosphere: AtmosphereDef {
                    sky_zenith: [0.1, 0.05, 0.2], // alien neon sky
                    sky_horizon_warm: [0.5, 0.1, 0.8],
                    cloud_light: [0.7, 0.3, 0.9],
                    cloud_dark: [0.1, 0.02, 0.15],
                    space_color: [0.04, 0.01, 0.05],
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
    pub name: &'static str, // for messages ("Can't land here: lava below.")
    pub physics: LiquidPhysics,
}

impl LiquidDef {
    pub fn damaging(&self) -> bool {
        self.physics.damage_rate > 0.0
    }
}

// how the player moves in a liquid deeper than the knees (entity.rs Player::update)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiquidPhysics {
    pub swimmable: bool, // buoyancy and swimming (Player::swim); otherwise wading on the bottom
    pub damage_rate: f32, // HP/s while submerged beyond the wading depth; 0 = harmless
    pub wade_speed: f32, // walking speed factor when not swimmable (thick liquid)
    pub sink_speed: f32, // the fastest a body sinks through it when not swimmable, units/s
}

// water: swim
pub const WATER: LiquidPhysics = LiquidPhysics {
    swimmable: true,
    damage_rate: 0.0,
    wade_speed: 1.0,
    sink_speed: 0.0,
};
// lava: far too thick to swim; you sink slowly, wade at a crawl and burn (dead in ~5 s)
pub const LAVA: LiquidPhysics = LiquidPhysics {
    swimmable: false,
    damage_rate: 20.0,
    wade_speed: 0.3,
    sink_speed: 0.8,
};
// toxic sludge: viscous, no swimming; less deadly than lava, so wading out of a shore is survivable
pub const SLUDGE: LiquidPhysics = LiquidPhysics {
    swimmable: false,
    damage_rate: 12.0,
    wade_speed: 0.4,
    sink_speed: 1.2,
};
// plasma: harmless, buoyant like water
pub const PLASMA: LiquidPhysics = LiquidPhysics {
    swimmable: true,
    damage_rate: 0.0,
    wade_speed: 1.0,
    sink_speed: 0.0,
};

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
    pub day_length_secs: f32,
    pub liquid: Option<LiquidDef>,
    pub palette: Palette,
    pub atmosphere: AtmosphereDef,
}

#[cfg(test)]
mod tests {
    use super::*;

    // water and plasma swim and are harmless; lava and toxic sludge can't be swum in and hurt
    #[test]
    fn liquids_have_their_physics() {
        let physics = |t: PlanetType| t.def().liquid.map(|l| l.physics);
        assert_eq!(physics(PlanetType::EarthLike), Some(WATER));
        assert_eq!(physics(PlanetType::Ocean), Some(WATER));
        assert_eq!(physics(PlanetType::Shattered), Some(PLASMA));
        assert_eq!(physics(PlanetType::Volcanic), Some(LAVA));
        assert_eq!(physics(PlanetType::Toxic), Some(SLUDGE));
        assert_eq!(physics(PlanetType::Ice), None);
        assert_eq!(physics(PlanetType::Desert), None);
        for t in PlanetType::ALL {
            if let Some(l) = t.def().liquid {
                // a liquid you can't swim in must slow you down and let you sink
                assert_eq!(l.physics.swimmable, l.physics.sink_speed == 0.0, "{t:?}");
                assert!(l.physics.wade_speed > 0.0 && l.physics.wade_speed <= 1.0);
                assert_eq!(
                    l.damaging(),
                    !matches!(
                        t,
                        PlanetType::EarthLike | PlanetType::Ocean | PlanetType::Shattered
                    )
                );
            }
        }
        assert!(SLUDGE.damage_rate < LAVA.damage_rate);
    }

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
        assert_eq!(PlanetType::from_name("desert"), Some(PlanetType::Desert));
        assert_eq!(PlanetType::from_name("TOXIC"), Some(PlanetType::Toxic));
        assert_eq!(PlanetType::from_name("Ocean"), Some(PlanetType::Ocean));
        assert_eq!(
            PlanetType::from_name("shattered"),
            Some(PlanetType::Shattered)
        );

        assert_eq!(PlanetType::from_name("lava"), None);
    }

    #[test]
    fn earth_like_def_matches_todays_values() {
        let def = PlanetType::EarthLike.def();
        assert_eq!(def.name, "Earth-like");
        let liquid = def.liquid.expect("Earth-like has water");
        assert!(matches!(liquid.behavior, LiquidBehavior::Reflective));
        assert!(!liquid.damaging());
        assert_eq!(def.palette.ground, crate::material::BlockType::Grass);
        assert_eq!(def.palette.peak, crate::material::BlockType::Snow);
    }

    #[test]
    fn volcanic_liquid_is_glowing_and_damaging() {
        let def = PlanetType::Volcanic.def();
        let liquid = def.liquid.expect("Volcanic has lava");
        assert!(matches!(liquid.behavior, LiquidBehavior::Glowing));
        assert!(liquid.damaging());
    }

    #[test]
    fn toxic_liquid_is_glowing_and_damaging() {
        let def = PlanetType::Toxic.def();
        let liquid = def.liquid.expect("Toxic has acid");
        assert!(matches!(liquid.behavior, LiquidBehavior::Glowing));
        assert!(liquid.damaging());
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
