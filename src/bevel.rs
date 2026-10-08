// Block bevels (docs/superpowers/specs/2026-10-09-block-bevels-design.md): every cube/hex block face
// carries its distance to up to four of its edges (Vertex::edge), which fs_geom turns into rounded
// edges and darkened joints. Pure geometry and the console settings; no GPU code.

use glam::Vec3;

// Vertex::edge component for "no bevelled edge here"
pub const NO_EDGE: f32 = 1e4;

// distance from p to the infinite line through a and b
pub fn line_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    (p - a).cross((b - a).normalize()).length()
}

// p's distances to up to four lines, NO_EDGE for the unused components
pub fn edge_distances(p: Vec3, lines: &[(Vec3, Vec3)]) -> [f32; 4] {
    debug_assert!(lines.len() <= 4);
    let mut d = [NO_EDGE; 4];
    for (slot, &(a, b)) in d.iter_mut().zip(lines) {
        *slot = line_distance(p, a, b);
    }
    d
}

// per corner of quad `pos`, its distance to each edge pos[k] → pos[k + 1] that `bevel[k]` marks
// (NO_EDGE for the others); a corner lies on its own two edges
pub fn quad_edges(pos: [Vec3; 4], bevel: [bool; 4]) -> [[f32; 4]; 4] {
    pos.map(|p| {
        let mut d = [NO_EDGE; 4];
        for k in 0..4 {
            if bevel[k] {
                d[k] = line_distance(p, pos[k], pos[(k + 1) % 4]);
            }
        }
        d
    })
}

// the straight sides of a closed counter-clockwise outline given as pieces a → b (sixths of a column):
// consecutive collinear pieces (a hex cell's border side, cut into one-sixth pieces) form one line.
// returns each piece's line; lines are numbered in outline order
pub fn polygon_lines(pieces: &[((i64, i64), (i64, i64))]) -> Vec<usize> {
    let n = pieces.len();
    let dir = |i: usize| {
        let (a, b) = pieces[i];
        (b.0 - a.0, b.1 - a.1)
    };
    let collinear = |i: usize, j: usize| {
        let (p, q) = (dir(i), dir(j));
        p.0 * q.1 - p.1 * q.0 == 0 && p.0 * q.0 + p.1 * q.1 > 0
    };
    // start at a piece that begins a line, so a run wrapping around the list's end stays one line
    let start = (0..n)
        .find(|&i| !collinear((i + n - 1) % n, i))
        .unwrap_or(0);
    let mut line_of = vec![0; n];
    let mut line = 0;
    for k in 0..n {
        let i = (start + k) % n;
        if k > 0 && !collinear((i + n - 1) % n, i) {
            line += 1;
        }
        line_of[i] = line;
    }
    line_of
}

pub const DEFAULT_WIDTH: f32 = 0.06; // world units (about a block's 6 %)
pub const DEFAULT_CAVITY: f32 = 0.2; // albedo darkening right at an edge

// the bevel look, tuned with /bevel (cmd.rs); GlobalUniform.bevel = uniform()
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BevelSettings {
    pub width: f32, // 0 = off
    pub cavity: f32,
    last_width: f32, // the width /bevel on restores
}

impl Default for BevelSettings {
    fn default() -> Self {
        Self {
            width: DEFAULT_WIDTH,
            cavity: DEFAULT_CAVITY,
            last_width: DEFAULT_WIDTH,
        }
    }
}

impl BevelSettings {
    pub fn uniform(&self) -> [f32; 4] {
        [self.width, self.cavity, 0.0, 0.0]
    }
}

pub const BEVEL_USAGE: &str = "Usage: /bevel get|on|off|width <0-0.25>|cavity <0-1>";
const MAX_WIDTH: f32 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BevelCommand {
    Get,
    Width(f32),
    Cavity(f32),
    On,
    Off,
}

impl BevelCommand {
    // the words after /bevel
    pub fn parse(args: &[&str]) -> Result<BevelCommand, &'static str> {
        let value = |s: &str, max: f32| {
            s.parse::<f32>()
                .ok()
                .filter(|v| (0.0..=max).contains(v))
                .ok_or(BEVEL_USAGE)
        };
        match args {
            ["get"] => Ok(BevelCommand::Get),
            ["on"] => Ok(BevelCommand::On),
            ["off"] => Ok(BevelCommand::Off),
            ["width", v] => value(v, MAX_WIDTH).map(BevelCommand::Width),
            ["cavity", v] => value(v, 1.0).map(BevelCommand::Cavity),
            _ => Err(BEVEL_USAGE),
        }
    }
}

impl BevelSettings {
    // applies a /bevel command; returns the line to log
    pub fn apply(&mut self, cmd: BevelCommand) -> String {
        match cmd {
            BevelCommand::Get => {}
            BevelCommand::Width(w) => {
                self.width = w;
                if w > 0.0 {
                    self.last_width = w;
                }
            }
            BevelCommand::Cavity(c) => self.cavity = c,
            BevelCommand::On => self.width = self.last_width,
            BevelCommand::Off => self.width = 0.0,
        }
        format!("Bevel: width {:.2}, cavity {:.2}", self.width, self.cavity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_distance_is_perpendicular_and_unbounded() {
        let (a, b) = (Vec3::ZERO, Vec3::X);
        assert!((line_distance(Vec3::new(0.5, 2.0, 0.0), a, b) - 2.0).abs() < 1e-6);
        // beyond the segment's end: still the distance to the infinite line
        assert!((line_distance(Vec3::new(5.0, 0.0, 3.0), a, b) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn edge_distances_pads_with_no_edge() {
        let lines = [(Vec3::ZERO, Vec3::X), (Vec3::ZERO, Vec3::Y)];
        let d = edge_distances(Vec3::new(0.25, 0.5, 0.0), &lines);
        assert!((d[0] - 0.5).abs() < 1e-6);
        assert!((d[1] - 0.25).abs() < 1e-6);
        assert_eq!(d[2], NO_EDGE);
        assert_eq!(d[3], NO_EDGE);
    }

    #[test]
    fn quad_corners_sit_on_their_two_edges() {
        // a slightly trapezoidal face, like a cube-sphere block side
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.05, 1.0, 0.0),
            Vec3::new(-0.05, 1.0, 0.0),
        ];
        let e = quad_edges(pos, [true; 4]);
        for (i, d) in e.iter().enumerate() {
            // corner i starts edge i and ends edge i - 1
            assert!(d[i].abs() < 1e-5, "corner {i} on edge {i}: {d:?}");
            assert!(
                d[(i + 3) % 4].abs() < 1e-5,
                "corner {i} on edge {}: {d:?}",
                (i + 3) % 4
            );
            for k in [(i + 1) % 4, (i + 2) % 4] {
                assert!(
                    d[k] > 0.9 && d[k] < 1.1,
                    "corner {i} far from edge {k}: {d:?}"
                );
            }
        }
    }

    #[test]
    fn quad_edges_respects_the_mask() {
        let pos = [Vec3::ZERO, Vec3::X, Vec3::X + Vec3::Y, Vec3::Y];
        let e = quad_edges(pos, [true, false, true, false]);
        for d in e {
            assert_eq!(d[1], NO_EDGE);
            assert_eq!(d[3], NO_EDGE);
            assert!(d[0] < NO_EDGE && d[2] < NO_EDGE);
        }
    }

    #[test]
    fn polygon_lines_keeps_a_plain_hexagon() {
        let pts = [(0, 0), (6, -1), (12, 0), (12, 5), (6, 6), (0, 5)];
        let pieces: Vec<_> = (0..6).map(|i| (pts[i], pts[(i + 1) % 6])).collect();
        let l = polygon_lines(&pieces);
        // every piece is its own line, numbered in order (any rotation)
        for i in 0..6 {
            assert_eq!(l[(i + 1) % 6], (l[i] + 1) % 6);
        }
    }

    // a border cell: its border side comes in one-sixth pieces (hex::edges), which form one line;
    // the run wraps around the end of the list, so merging must be circular
    #[test]
    fn polygon_lines_merges_border_pieces() {
        let pieces = vec![
            ((2, 0), (3, 0)),
            ((3, 0), (4, 0)),
            ((4, 0), (4, 6)),
            ((4, 6), (0, 6)),
            ((0, 6), (0, 0)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
        ];
        let l = polygon_lines(&pieces);
        let bottom = l[0];
        for i in [0, 1, 5, 6] {
            assert_eq!(l[i], bottom);
        }
        // four sides, numbered around the outline
        let m = 4;
        assert_eq!(l.iter().max(), Some(&(m - 1)));
        assert_eq!(l[2], (bottom + 1) % m);
        assert_eq!(l[3], (bottom + 2) % m);
        assert_eq!(l[4], (bottom + m - 1) % m);
    }
    #[test]
    fn default_settings_feed_the_uniform() {
        let s = BevelSettings::default();
        assert_eq!(s.uniform(), [DEFAULT_WIDTH, DEFAULT_CAVITY, 0.0, 0.0]);
        assert_eq!((DEFAULT_WIDTH, DEFAULT_CAVITY), (0.06, 0.2));
    }
    #[test]
    fn bevel_command_parses_and_checks_ranges() {
        assert_eq!(BevelCommand::parse(&["get"]), Ok(BevelCommand::Get));
        assert_eq!(BevelCommand::parse(&["on"]), Ok(BevelCommand::On));
        assert_eq!(BevelCommand::parse(&["off"]), Ok(BevelCommand::Off));
        assert_eq!(
            BevelCommand::parse(&["width", "0.1"]),
            Ok(BevelCommand::Width(0.1))
        );
        assert_eq!(
            BevelCommand::parse(&["cavity", "0.5"]),
            Ok(BevelCommand::Cavity(0.5))
        );
        for bad in [
            &["width", "0.3"][..],
            &["cavity", "2"],
            &["width", "x"],
            &["width"],
            &[],
            &["nope"],
        ] {
            assert_eq!(BevelCommand::parse(bad), Err(BEVEL_USAGE), "{bad:?}");
        }
    }

    #[test]
    fn off_and_on_restore_the_last_width() {
        let mut s = BevelSettings::default();
        s.apply(BevelCommand::Width(0.1));
        s.apply(BevelCommand::Off);
        assert_eq!(s.width, 0.0);
        s.apply(BevelCommand::On);
        assert_eq!(s.width, 0.1);
        // width 0 is "off": on afterwards brings back the last non-zero width, not 0
        s.apply(BevelCommand::Width(0.0));
        s.apply(BevelCommand::On);
        assert_eq!(s.width, 0.1);
        s.apply(BevelCommand::Cavity(0.5));
        assert_eq!(s.cavity, 0.5);
        assert!(s.apply(BevelCommand::Get).contains("0.10"));
    }
}
