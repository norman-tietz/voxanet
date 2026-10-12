// per-planet variation on top of the planet type (spec 2026-10-12-system-variety-design.md):
// generated with the planet (space/variety.rs), carried on space::solar_system::Planet and copied into
// PlanetData at bake time. NEUTRAL reproduces the type exactly (the start system's planets).
use crate::planet::noise::TerrainTweaks;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetTraits {
    pub day_scale: f32,  // × the type's day length (0.6–1.6)
    pub relief: f32,     // × the terrain relief R (0.7–1.4)
    pub land_shift: f32, // added to the continent mask's land_offset (−0.15..+0.15; 0 without liquid)
    pub sky_hue: f32,    // hue rotation of the sky and cloud colours, radians (−0.25..+0.25)
    pub storminess: f32, // × how stormy the weather is (0.5–1.5)
    pub gravity: f32,    // × Physics::GRAVITY (0.54–1.43)
}

impl PlanetTraits {
    pub const NEUTRAL: Self = Self {
        day_scale: 1.0,
        relief: 1.0,
        land_shift: 0.0,
        sky_hue: 0.0,
        storminess: 1.0,
        gravity: 1.0,
    };

    // the planet type's definition with this planet's sky tint applied; the one place the engine
    // (Renderer::render) and the impostors (planet_uniform, lod/impostor water) get a tinted atmosphere from
    pub fn tinted_def(
        &self,
        planet_type: crate::planet::biome::PlanetType,
    ) -> crate::planet::biome::PlanetTypeDef {
        let mut def = planet_type.def();
        def.atmosphere = def.atmosphere.tinted(self.sky_hue);
        def
    }

    // the part the terrain bake and the impostors need
    pub fn terrain(&self) -> TerrainTweaks {
        TerrainTweaks {
            relief: self.relief,
            land_shift: self.land_shift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_terrain_tweaks_are_neutral() {
        assert_eq!(PlanetTraits::NEUTRAL.terrain(), TerrainTweaks::NEUTRAL);
    }

    #[test]
    fn tinted_def_is_the_type_at_hue_zero_and_shifts_the_sky_otherwise() {
        use crate::planet::biome::PlanetType;
        let base = PlanetType::EarthLike.def();
        let same = PlanetTraits::NEUTRAL.tinted_def(PlanetType::EarthLike);
        assert_eq!(same.atmosphere.sky_zenith, base.atmosphere.sky_zenith);
        assert_eq!(same.atmosphere.cloud_light, base.atmosphere.cloud_light);
        let t = PlanetTraits {
            sky_hue: 0.2,
            ..PlanetTraits::NEUTRAL
        }
        .tinted_def(PlanetType::EarthLike);
        assert_ne!(t.atmosphere.sky_zenith, base.atmosphere.sky_zenith);
        assert_eq!(t.atmosphere.density, base.atmosphere.density);
        assert_eq!(t.atmosphere.space_color, base.atmosphere.space_color);
        assert_eq!(t.palette.ground.color(), base.palette.ground.color());
    }
}
