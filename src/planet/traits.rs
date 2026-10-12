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
}
