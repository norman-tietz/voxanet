// env_map.rs
// Pre-filtered environment map for the plastic's sky reflection (docs/superpowers/specs/
// 2026-10-11-env-map-design.md): the sky around the camera (dome × opacity with the clouds, no sun
// disc), premultiplied with its coverage above the local horizon, in a 2-slot cube-map array whose
// levels are GGX-filtered for roughness level / 5. Each frame one step: a face of the source cube
// (fs_env_bake, shader.wgsl), or one filtered level of the back slot (env_filter.wgsl); after the last
// level the slots swap, so nothing reads a half-updated map. shade() samples the read slot.

pub const ENV_SIZE: u32 = 128; // texels per face edge (shader.wgsl ENV_SIZE, env_filter.wgsl)
pub const ENV_LEVELS: u32 = 6; // 128 … 4 (shader.wgsl ENV_LEVELS)
pub const SRC_MIPS: u32 = 8; // the source cube's box mips, 128 … 1 (the filter's undersampling guard)
pub const SLOTS: u32 = 2;
pub const FACES: u32 = 6;
pub const FILTER_SAMPLES: u32 = 32; // env_filter.wgsl SAMPLES
pub const STEPS: u32 = FACES + ENV_LEVELS; // one renewal

// one frame's GPU work
#[derive(Debug, PartialEq)]
pub enum EnvBake {
    Whole(u32), // every face, the source mips and every level into this slot (start-up, reset)
    Face(u32),  // one face of the source cube
    Level(u32, u32), // (slot, level): one filtered level; level 0 first builds the source's mips
}

// which slot is read and where the renewal of the other one stands; pure bookkeeping (tested)
#[derive(Debug, Default)]
pub struct Schedule {
    read: u32,
    step: u32,
    valid: bool, // the read slot holds a complete map
}

impl Schedule {
    // this frame's work; `reset`: the map no longer fits (planet switch, back from galaxy mode)
    pub fn update(&mut self, reset: bool) -> Vec<EnvBake> {
        if reset || !self.valid {
            self.valid = true;
            self.step = 0;
            return vec![EnvBake::Whole(self.read)];
        }
        let back = 1 - self.read;
        let work = if self.step < FACES {
            EnvBake::Face(self.step)
        } else {
            EnvBake::Level(back, self.step - FACES)
        };
        self.step += 1;
        if self.step == STEPS {
            self.read = back;
            self.step = 0;
        }
        vec![work]
    }

    pub fn read_slot(&self) -> u32 {
        self.read
    }
}

// the level shade() samples for a roughness (shader.wgsl env_sample)
#[cfg_attr(not(test), allow(dead_code))]
pub fn roughness_to_level(r: f32) -> f32 {
    r.clamp(0.0, 1.0) * (ENV_LEVELS - 1) as f32
}

// the GGX roughness level `level` is filtered with (env_filter.wgsl)
pub fn level_roughness(level: u32) -> f32 {
    level as f32 / (ENV_LEVELS - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // start-up / reset: the whole map into the read slot in that very frame (Review Focus 1)
    #[test]
    fn reset_bakes_whole_into_the_read_slot() {
        let mut s = Schedule::default();
        assert_eq!(s.update(false), vec![EnvBake::Whole(0)]);
        assert_eq!(s.read_slot(), 0);
        // a later reset bakes whole again, into whatever slot is read then
        for _ in 0..5 {
            s.update(false);
        }
        assert_eq!(s.update(true), vec![EnvBake::Whole(s.read_slot())]);
    }

    // after the whole bake: faces 0–5 into the source, then levels 0–5 into the back slot, then swap
    #[test]
    fn renewal_is_six_faces_then_six_levels() {
        let mut s = Schedule::default();
        s.update(false);
        let mut steps = Vec::new();
        for _ in 0..STEPS {
            steps.extend(s.update(false));
        }
        let mut want: Vec<EnvBake> = (0..FACES).map(EnvBake::Face).collect();
        want.extend((0..ENV_LEVELS).map(|l| EnvBake::Level(1, l)));
        assert_eq!(steps, want);
    }

    // Review Focus 5
    #[test]
    fn read_slot_changes_only_after_a_full_renewal() {
        let mut s = Schedule::default();
        s.update(false);
        for i in 0..STEPS {
            assert_eq!(s.read_slot(), 0, "step {i}");
            s.update(false);
        }
        assert_eq!(s.read_slot(), 1);
        for _ in 0..STEPS {
            s.update(false);
        }
        assert_eq!(s.read_slot(), 0);
    }

    // Review Focus 4
    #[test]
    fn roughness_maps_to_levels() {
        assert_eq!(roughness_to_level(0.0), 0.0);
        assert_eq!(roughness_to_level(1.0), (ENV_LEVELS - 1) as f32);
        assert!((roughness_to_level(0.35) - 1.75).abs() < 1e-6);
        assert_eq!(roughness_to_level(-1.0), 0.0);
        assert_eq!(roughness_to_level(3.0), (ENV_LEVELS - 1) as f32);
        for l in 0..ENV_LEVELS {
            assert!((roughness_to_level(level_roughness(l)) - l as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn sizes_are_consistent() {
        assert_eq!(ENV_SIZE >> (ENV_LEVELS - 1), 4);
        assert_eq!(ENV_SIZE >> (SRC_MIPS - 1), 1);
    }
}
