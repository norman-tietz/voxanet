// star_shading.rs
// Rust mirror of star.wgsl's closed-form shading terms, for tests (the WGSL is what renders).
#![cfg_attr(not(test), allow(dead_code))]

pub const LIMB_DARKENING: f32 = 0.6; // star.wgsl STAR_LIMB_DARKENING

// limb darkening: brightness at a point whose normal makes cos `mu` with the view ray
pub fn limb(mu: f32) -> f32 {
    1.0 - LIMB_DARKENING * (1.0 - mu.clamp(0.0, 1.0))
}

// corona brightness `d` star radii outside the limb, for a corona_size `size`
pub fn corona_falloff(d: f32, size: f32) -> f32 {
    let d = d.max(0.0);
    (-d / size).exp() / (1.0 + 8.0 * d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limb_darkening_is_darker_at_the_edge() {
        assert_eq!(limb(1.0), 1.0);
        assert!((limb(0.0) - 0.4).abs() < 1e-6);
        assert!(limb(0.3) < limb(0.7));
    }

    #[test]
    fn corona_falls_off_from_the_limb() {
        assert_eq!(corona_falloff(0.0, 0.45), 1.0);
        let (a, b, c) = (
            corona_falloff(0.1, 0.45),
            corona_falloff(0.5, 0.45),
            corona_falloff(2.0, 0.45),
        );
        assert!(1.0 > a && a > b && b > c && c >= 0.0);
        assert!(c < 0.02, "still bright 2 radii out: {c}");
    }
}
