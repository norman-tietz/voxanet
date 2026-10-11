// hex.rs
// Hex columns: the honeycomb a cube face is divided into in the hex terrain style (/style hex).
// Column (u, v) of the square grid is hex cell (u, v): rows run along u, every odd row (v odd) shifted
// by half a column, so the height map, edits and chunks are the same in both styles. Cells are 1 wide,
// their vertical sides span the middle ⅔ of the row and their points reach ⅙ into the rows above and
// below (pointy-top hexagons about 13 % flatter than regular ones, area 1). At the u borders of the four
// ring faces (around Z, see ring) the honeycomb continues onto the next face: faces 1 and 2 shift their
// odd rows the other way (mirrored), so at each of those seams one face's odd rows end with a cell
// reaching half a column across and the other's with a half-column gap, which that cell fills. Elsewhere
// the honeycomb is cut off straight: the cells of the first and last row are squared off to the border,
// and on faces 4 and 5 the first cell of an odd row also takes the half-cell gap at u = 0 while the last
// is cut at u = res.
//
// Pure 2D geometry in face space (u, v in 0..=res), no world positions. Corners are exact integers in
// sixths of a column (u6 = 6u, v6 = 6v): every corner of every cell lies on that lattice, so shared
// corners and the cube-face seams compare exactly.

// whether a face's odd rows are shifted left instead of right (faces 1 and 2: their u runs the other way
// round the ring, and only then do all four ring seams pair a reaching cell with a gap)
pub fn mirrored(face: u8) -> bool {
    face == 1 || face == 2
}

// a row's shift along u, in sixths: odd rows half a column to the right (left on mirrored faces)
fn shift6(face: u8, row: i64) -> i64 {
    let s = 3 * row.rem_euclid(2);
    if mirrored(face) {
        -s
    } else {
        s
    }
}

// the row boundaries only need the shift modulo a column (−3 ≡ 3), so they are the same on every face
fn zig_shift6(row: i64) -> i64 {
    3 * row.rem_euclid(2)
}

// the boundary between row `row` and the row below it at u6 = x: a zigzag between the bottom points of
// row's cells (6·row − 1) and the corners between them (6·row + 1)
fn zig6(row: i64, x: f64) -> f64 {
    let t = (x - zig_shift6(row) as f64).rem_euclid(6.0);
    let d = (t - 3.0).abs(); // 0 under a cell's centre, 3 between two cells
    (6 * row - 1) as f64 + d * 2.0 / 3.0
}

// zig6 at a lattice column (x a multiple of 3), exactly
fn zig6_at(row: i64, x: i64) -> i64 {
    debug_assert!(x % 3 == 0);
    let t = (x - zig_shift6(row)).rem_euclid(6);
    6 * row - 1 + 2 * (t - 3).abs() / 3
}

// the ring: the four faces around the Z axis (+Y, −Y, +X, −X). Their v runs along Z in the same
// direction, so their rows meet across the four seams between them with the same row index, and the
// honeycomb continues across those seams (faces 1 and 2 mirrored, see shift6)
pub fn ring(face: u8) -> bool {
    face < 4
}

// the ring face across `face`'s u border (hi: beyond u = res, else below u = 0) and how its u follows
// from this face's: u' = a·u + b·res (exact in sixths with 6·res), from CoordSystem's face layout
pub fn ring_neighbour(face: u8, hi: bool) -> (u8, i64, i64) {
    match (face, hi) {
        (0, true) => (2, -1, 2),
        (0, false) => (3, 1, 1),
        (1, true) => (2, 1, -1),
        (1, false) => (3, -1, 0),
        (2, true) => (0, -1, 2),
        (2, false) => (1, 1, 1),
        (3, true) => (0, 1, -1),
        (3, false) => (1, -1, 0),
        _ => unreachable!("face {face} is not on the ring"),
    }
}

// a ring-face u beyond its border (by at most a column) on the face it lies on; v is unchanged. In
// range, or off the ring: unchanged
pub fn ring_wrap(face: u8, u: f64, res: u32) -> (u8, f64) {
    let r = res as f64;
    if !ring(face) || (0.0..=r).contains(&u) {
        return (face, u);
    }
    let (f, a, b) = ring_neighbour(face, u > r);
    (f, a as f64 * u + b as f64 * r)
}

// u of a point on `face` in the extended coordinates of the neighbouring ring face `owner` (the
// inverse of ring_wrap: u = a·(u' − b·res), a = ±1)
pub fn ring_extend(owner: u8, face: u8, u: f64, res: u32) -> f64 {
    let hi = ring_neighbour(owner, true).0 == face;
    let (f, a, b) = ring_neighbour(owner, hi);
    debug_assert_eq!(f, face, "faces {owner} and {face} don't share a ring seam");
    a as f64 * (u - b as f64 * res as f64)
}

// the cell containing face-space point (u, v) (columns; the face is [0, res]²) as (face, u, v): on a
// ring face a point in the half-column gap at a seam belongs to the neighbouring face's reaching cell,
// and a point beyond the u border is wrapped onto the next face first; other points outside are clamped
// onto the border cells
pub fn cell_at(face: u8, u: f64, v: f64, res: u32) -> (u8, u32, u32) {
    let (n, r) = (res as i64, res as f64);
    if ring(face) && !(0.0..=r).contains(&u) {
        let (f, w) = ring_wrap(face, u, res);
        return cell_at(f, w.clamp(0.0, r), v, res);
    }
    let (x, y) = (u * 6.0, v * 6.0);
    let r0 = v.floor() as i64;
    let row = if y < zig6(r0, x) {
        r0 - 1
    } else if y >= zig6(r0 + 1, x) {
        r0 + 1
    } else {
        r0
    };
    let row = row.clamp(0, n - 1);
    let col = ((x - shift6(face, row) as f64) / 6.0).floor() as i64;
    if !ring(face) && col < 0 {
        // the gap in front of an odd row's first cell (border_gap): above or below the neighbouring
        // hexagons' extended edges it belongs to the first cell of the row above / below
        let (dy, slope) = (y - (6 * row + 3) as f64, x.max(0.0) * 2.0 / 3.0);
        let row = if dy > slope && row + 1 < n {
            row + 1
        } else if dy < -slope && row > 0 {
            row - 1
        } else {
            row
        };
        return (face, 0, row as u32);
    }
    if ring(face) && !(0..n).contains(&col) {
        // the gap: the neighbour's cell reaches into it
        let (f, ..) = ring_neighbour(face, col >= n);
        let x = ring_extend(f, face, u, res) * 6.0;
        let col = ((x - shift6(f, row) as f64) / 6.0).floor() as i64;
        // a point exactly on the gap's far edge lands on the reaching cell's open end where the seam
        // reflects u (0 ↔ 2: u' = 2·res − u), one past it; boundary points may go either way, so the
        // clamp keeps the reaching cell
        debug_assert!(
            (-1..=n).contains(&col),
            "gap at {face} ({u}, {v}) not filled by {f}"
        );
        return (f, col.clamp(0, n - 1) as u32, row as u32);
    }
    (face, col.clamp(0, n - 1) as u32, row as u32)
}

// cell (u, v)'s polygon on `face`, counter-clockwise (v up), in sixths of the face's own (extended)
// coordinates: 6 corners; first/last rows squared off at the v border; on a ring face an odd row's cell
// may reach half a column beyond the u border, on faces 4 and 5 the u border clips (widens / cuts) them
pub fn outline(face: u8, u: u32, v: u32, res: u32) -> Vec<(i64, i64)> {
    let (col, row, n) = (u as i64, v as i64, res as i64);
    let s = shift6(face, row);
    let (lo, hi) = if ring(face) {
        (6 * col + s, 6 * col + 6 + s)
    } else {
        // the last cell is cut at u = res; an odd row's first cell starts as a whole hexagon, the gap
        // before it is shared out below (border_gap)
        (
            if col == 0 && s == 0 { 0 } else { 6 * col + s },
            if col == n - 1 { 6 * n } else { 6 * col + 6 + s },
        )
    };
    let bottom = |x: i64| if row == 0 { 0 } else { zig6_at(row, x) };
    let top = |x: i64| {
        if row == n - 1 {
            6 * n
        } else {
            zig6_at(row + 1, x)
        }
    };
    let xs: Vec<i64> = (lo..=hi).step_by(3).collect();
    let mut pts: Vec<(i64, i64)> = xs.iter().map(|&x| (x, bottom(x))).collect();
    pts.extend(xs.iter().rev().map(|&x| (x, top(x))));
    if !ring(face) && col == 0 {
        border_gap(&mut pts, row, n);
    }
    drop_collinear(pts)
}

// faces 4 and 5 (off the ring): the half-column gap at u = 0 in front of an odd row's first cell, cut
// into three triangles along the neighbouring hexagons' edges extended to the border: the upper one to
// the first cell of the row above, the lower one to the row below's, the middle one to the odd row's own
// first cell, which gets a point at the border. The only split that keeps all three cells convex (any
// other leaves a notch where a neighbour's point sits); cell_at follows the same lines
fn border_gap(pts: &mut Vec<(i64, i64)>, row: i64, n: i64) {
    if row % 2 == 1 {
        // the hexagon's outline ends at its top-left corner: on to the border (via the squared-off top
        // corner in the last row) and the point at 6·row + 3; its left corners fall in line and go
        if row == n - 1 {
            pts.push((0, 6 * n));
        }
        pts.push((0, 6 * row + 3));
    } else {
        // the even row's left side, from the row below's gap point to the row above's
        if row > 0 {
            pts[0] = (0, 6 * row - 3);
        }
        if row < n - 1 {
            let last = pts.len() - 1;
            pts[last] = (0, 6 * row + 9);
        }
    }
}

// removes corners lying on the straight line through their neighbours (and repeated corners)
fn drop_collinear(mut pts: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    loop {
        let n = pts.len();
        let straight = (0..n).find(|&i| {
            let (a, b, c) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0) == 0
        });
        match straight {
            Some(i) => {
                pts.remove(i);
            }
            None => return pts,
        }
    }
}

// the centroid of an outline, in columns (the top and bottom faces' fan centre)
pub fn centroid(pts: &[(i64, i64)]) -> (f64, f64) {
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..pts.len() {
        let (p, q) = (pts[i], pts[(i + 1) % pts.len()]);
        let cross = (p.0 * q.1 - q.0 * p.1) as f64;
        a += cross;
        cx += (p.0 + q.0) as f64 * cross;
        cy += (p.1 + q.1) as f64 * cross;
    }
    (cx / (3.0 * a) / 6.0, cy / (3.0 * a) / 6.0)
}

// the area of an outline in columns² (1 for an interior cell; tests)
#[cfg(test)]
pub fn area(pts: &[(i64, i64)]) -> f64 {
    let twice: i64 = (0..pts.len())
        .map(|i| {
            let (p, q) = (pts[i], pts[(i + 1) % pts.len()]);
            p.0 * q.1 - q.0 * p.1
        })
        .sum();
    twice as f64 / 2.0 / 36.0
}

// what lies across an outline edge
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Across {
    Cell(u8, u32, u32),
    // the border to a ±Z face (or faces 4/5's u borders): the next cube face, found in world space
    // (PlanetData::hex_walls)
    Border,
}

// edge a → b of a cell's outline (sixths) and what lies across it. Border edges come in pieces one sixth
// long, since the next face's cells change along the border on the sixths lattice
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub a: (i64, i64),
    pub b: (i64, i64),
    pub across: Across,
}

// the edges of cell (u, v) on `face`, counter-clockwise
pub fn edges(face: u8, u: u32, v: u32, res: u32) -> Vec<Edge> {
    let pts = outline(face, u, v, res);
    let n6 = 6 * res as i64;
    let on_border = |a: (i64, i64), b: (i64, i64)| {
        (a.0 == b.0 && (a.0 == 0 || a.0 == n6) && !ring(face))
            || (a.1 == b.1 && (a.1 == 0 || a.1 == n6))
    };
    let mut out = Vec::with_capacity(pts.len());
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        if on_border(a, b) {
            let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs());
            let (du, dv) = ((b.0 - a.0).signum(), (b.1 - a.1).signum());
            for k in 0..steps {
                out.push(Edge {
                    a: (a.0 + du * k, a.1 + dv * k),
                    b: (a.0 + du * (k + 1), a.1 + dv * (k + 1)),
                    across: Across::Border,
                });
            }
        } else {
            // a point just outside the edge's midpoint (outward = right of a → b, counter-clockwise)
            let (mx, my) = ((a.0 + b.0) as f64 / 12.0, (a.1 + b.1) as f64 / 12.0);
            let (dx, dy) = ((b.0 - a.0) as f64, (b.1 - a.1) as f64);
            let len = (dx * dx + dy * dy).sqrt();
            let (nx, ny) = (dy / len * 1e-3, -dx / len * 1e-3);
            let (cf, cu, cv) = cell_at(face, mx + nx, my + ny, res);
            out.push(Edge {
                a,
                b,
                across: Across::Cell(cf, cu, cv),
            });
        }
    }
    out
}

// a point's distance to edge a → b (sixths) in columns
pub fn distance_to_edge(u: f64, v: f64, a: (i64, i64), b: (i64, i64)) -> f64 {
    let (ax, ay) = (a.0 as f64 / 6.0, a.1 as f64 / 6.0);
    let (bx, by) = (b.0 as f64 / 6.0, b.1 as f64 / 6.0);
    let (dx, dy) = (bx - ax, by - ay);
    let t = (((u - ax) * dx + (v - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
    let (px, py) = (ax + dx * t - u, ay + dy * t - v);
    (px * px + py * py).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // whether point (u, v) (columns) lies inside an outline (sixths; even-odd rule: an odd row's
    // widened first cell is not convex)
    fn inside(pts: &[(i64, i64)], u: f64, v: f64) -> bool {
        let (x, y) = (u * 6.0, v * 6.0);
        let mut odd = false;
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            let (ax, ay, bx, by) = (a.0 as f64, a.1 as f64, b.0 as f64, b.1 as f64);
            if (ay > y) != (by > y) && x < ax + (y - ay) / (by - ay) * (bx - ax) {
                odd = !odd;
            }
        }
        odd
    }

    // a cheap deterministic point sequence
    fn points(n: usize, res: u32) -> impl Iterator<Item = (f64, f64)> {
        let mut s = 0x9e3779b97f4a7c15u64;
        (0..n).map(move |_| {
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 * res as f64
            };
            (next(), next())
        })
    }

    // the ring table undoes itself: a point beyond a ring border, wrapped to its face and extended back
    #[test]
    fn ring_wrap_and_extend_undo_each_other() {
        for res in [8u32, 9] {
            let r = res as f64;
            for face in 0..4u8 {
                for u in [-0.5, -0.25, -0.01, r + 0.01, r + 0.25, r + 0.5] {
                    let (f, w) = ring_wrap(face, u, res);
                    assert_ne!(f, face);
                    assert!((0.0..=r).contains(&w), "{face} {u} -> {f} {w}");
                    assert!((ring_extend(face, f, w, res) - u).abs() < 1e-12);
                }
                // in range: unchanged
                assert_eq!(ring_wrap(face, 3.5, res), (face, 3.5));
            }
            assert_eq!(ring_wrap(4, -0.5, res), (4, -0.5));
            // the table is symmetric: crossing back over the same seam
            for face in 0..4u8 {
                for hi in [false, true] {
                    let (f, ..) = ring_neighbour(face, hi);
                    assert!([true, false]
                        .iter()
                        .any(|&h| ring_neighbour(f, h).0 == face));
                }
            }
        }
    }

    // points exactly on a gap's edges belong to a cell next to them: the reflected seam 0 ↔ 2 maps the
    // gap's far edge onto the reaching cell's open end (u' = 2·res − u)
    #[test]
    fn gap_edges_have_an_owner() {
        for res in [8u32, 9] {
            let r = res as f64;
            for face in 0..4u8 {
                for u in [0.0, 0.5, r - 0.5, r, -0.5, r + 0.5] {
                    let (f, cu, cv) = cell_at(face, u, 5.5, res);
                    assert!(
                        ring(f) && cu < res && cv == 5,
                        "{face} {u} -> {f} {cu} {cv}"
                    );
                }
            }
        }
    }

    // faces 1 and 2 shift their odd rows the other way (left by half a column)
    #[test]
    fn faces_1_and_2_are_mirrored() {
        for face in 0..6u8 {
            let (cx, _) = centroid(&outline(face, 3, 5, 16));
            let want = if face == 1 || face == 2 { 3.0 } else { 4.0 };
            assert_eq!(cx, want, "face {face}");
        }
    }

    // every ring seam pairs a cell reaching half a column across with a half-column gap on the other
    // side, in every odd row; even rows end exactly at the border
    #[test]
    fn ring_seams_pair_a_reaching_cell_with_a_gap() {
        for res in [8u32, 9] {
            let n6 = 6 * res as i64;
            for face in 0..4u8 {
                for hi in [false, true] {
                    let (f, ..) = ring_neighbour(face, hi);
                    let back_hi = ring_neighbour(f, true).0 == face;
                    let col = |hi: bool| if hi { res - 1 } else { 0 };
                    let reaches = |face: u8, hi: bool, v: u32| {
                        let p = outline(face, col(hi), v, res);
                        if hi {
                            p.iter().any(|q| q.0 > n6)
                        } else {
                            p.iter().any(|q| q.0 < 0)
                        }
                    };
                    for v in 0..res {
                        let (here, there) = (reaches(face, hi, v), reaches(f, back_hi, v));
                        if v % 2 == 1 {
                            assert!(here != there, "res {res} face {face} hi {hi} row {v}");
                        } else {
                            assert!(!here && !there, "res {res} face {face} hi {hi} row {v}");
                        }
                    }
                }
            }
        }
    }

    // ring faces have no half or widened cells: every cell off the first and last row is a whole
    // hexagon (6 corners, area 1); faces 4 and 5 keep cut border cells (border_gap)
    #[test]
    fn ring_cells_are_whole_hexagons() {
        for res in [8u32, 9] {
            for face in 0..4u8 {
                for v in 1..res - 1 {
                    for u in 0..res {
                        let p = outline(face, u, v, res);
                        assert_eq!((p.len(), area(&p)), (6, 1.0), "face {face} ({u}, {v})");
                    }
                }
            }
            assert_eq!(area(&outline(4, 0, 1, res)), 7.0 / 6.0);
            assert_eq!(area(&outline(4, res - 1, 1, res)), 0.5);
        }
    }

    #[test]
    fn interior_cell_is_the_squashed_hexagon() {
        // even row: centre (3.5, 4.5) in columns
        assert_eq!(
            outline(0, 3, 4, 16),
            vec![(18, 25), (21, 23), (24, 25), (24, 29), (21, 31), (18, 29)]
        );
        // odd row: shifted half a column right
        assert_eq!(
            outline(0, 3, 5, 16),
            vec![(21, 31), (24, 29), (27, 31), (27, 35), (24, 37), (21, 35)]
        );
        assert_eq!(area(&outline(0, 3, 4, 16)), 1.0);
        assert_eq!(centroid(&outline(0, 3, 4, 16)), (3.5, 4.5));
    }

    // a point on `face` in `owner`'s (extended) coordinates
    fn in_owner(owner: u8, face: u8, u: f64, res: u32) -> f64 {
        if owner == face {
            u
        } else {
            ring_extend(owner, face, u, res)
        }
    }

    #[test]
    fn every_point_lies_in_the_outline_of_its_cell() {
        for res in [8u32, 9, 32] {
            let r = res as f64;
            // uniform points plus points close to the u borders (the ring seams)
            let near: Vec<(f64, f64)> = points(4_000, res)
                .map(|(u, v)| {
                    (
                        if u < r / 2.0 {
                            u * 0.01
                        } else {
                            r - (r - u) * 0.01
                        },
                        v,
                    )
                })
                .collect();
            for face in 0..6u8 {
                for &(u, v) in points(20_000, res).collect::<Vec<_>>().iter().chain(&near) {
                    let (f, cu, cv) = cell_at(face, u, v, res);
                    assert!(cu < res && cv < res);
                    assert!(f == face || (ring(face) && ring(f)), "{face} -> {f}");
                    let pts = outline(f, cu, cv, res);
                    let ou = in_owner(f, face, u, res);
                    // points on a boundary may go either way; skip those within a hair of one
                    let near_edge = edges(f, cu, cv, res)
                        .iter()
                        .any(|e| distance_to_edge(ou, v, e.a, e.b) < 1e-9);
                    assert!(
                        near_edge || inside(&pts, ou, v),
                        "res {res} face {face}: ({u}, {v}) -> ({f}, {cu}, {cv}) {pts:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cells_cover_the_face_exactly() {
        // per face: the cells' areas sum to the face's (a reaching cell counted on its owner, whose
        // half beyond the seam fills the neighbour's gap)
        for res in [1u32, 2, 7, 8, 32] {
            for face in 0..6u8 {
                let mut total = 0.0;
                for v in 0..res {
                    let mut row = 0.0;
                    for u in 0..res {
                        let a = area(&outline(face, u, v, res));
                        assert!(a > 0.0);
                        if u > 0 && u + 1 < res && v > 0 && v + 1 < res {
                            assert_eq!(a, 1.0, "interior cell ({u}, {v})");
                        }
                        row += a;
                    }
                    // off the ring the gap at u = 0 is shared with the rows above and below
                    if ring(face) {
                        assert_eq!(row, res as f64, "face {face} row {v} of res {res}");
                    }
                    total += row;
                }
                assert!(
                    (total - (res * res) as f64).abs() < 1e-9,
                    "face {face} res {res}"
                );
            }
        }
    }

    // every cell on every face is convex (the bevel's distances to its sides interpolate exactly only
    // over a convex fan, and a notch reads as a broken cell)
    #[test]
    fn every_cell_is_convex() {
        let convex = |p: &[(i64, i64)]| {
            let n = p.len();
            (0..n).all(|i| {
                let (a, b, c) = (p[(i + n - 1) % n], p[i], p[(i + 1) % n]);
                (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0) > 0
            })
        };
        for res in [2u32, 7, 8, 9] {
            for face in 0..6u8 {
                for v in 0..res {
                    for u in 0..res {
                        let p = outline(face, u, v, res);
                        assert!(convex(&p), "res {res} face {face} ({u}, {v}) {p:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn border_cells_are_cut_off_straight() {
        let res = 8;
        // faces 4 and 5, odd row, first cell: a hexagon with a point at the border (the gap's middle
        // triangle, area 1 + 1/6); the even rows' first cells take the gap's other triangles: the
        // border side runs from one gap point to the next (area 1 + 1/3)
        let first = outline(4, 0, 3, res);
        assert_eq!(area(&first), 7.0 / 6.0);
        assert_eq!(
            first.iter().filter(|p| p.0 == 0).collect::<Vec<_>>(),
            vec![&(0, 21)]
        );
        let even = outline(4, 0, 2, res);
        assert_eq!(area(&even), 4.0 / 3.0);
        assert_eq!(even.iter().filter(|p| p.0 == 0).count(), 2);
        assert!(even.contains(&(0, 9)) && even.contains(&(0, 21)));
        // odd row, last cell: cut at u = res (area ½)
        assert_eq!(area(&outline(4, res - 1, 1, res)), 0.5);
        // first row: squared off at v = 0, on every face
        for face in 0..6u8 {
            assert!(outline(face, 3, 0, res).iter().filter(|p| p.1 == 0).count() == 2);
        }
        // the cell centres map to their own cells
        for face in 0..6u8 {
            for v in 0..res {
                for u in 0..res {
                    let (cx, cy) = centroid(&outline(face, u, v, res));
                    assert_eq!(cell_at(face, cx, cy, res), (face, u, v), "face {face}");
                }
            }
        }
    }

    // `p` (sixths on `face`) on neighbouring ring face `f`
    fn six_on(face: u8, f: u8, p: (i64, i64), res: u32) -> (i64, i64) {
        if face == f {
            return p;
        }
        let hi = ring_neighbour(face, true).0 == f;
        let (_, a, b) = ring_neighbour(face, hi);
        (a * p.0 + b * 6 * res as i64, p.1)
    }

    #[test]
    fn neighbours_are_symmetric_and_edges_match() {
        for res in [7u32, 8] {
            for face in 0..6u8 {
                for v in 0..res {
                    for u in 0..res {
                        let es = edges(face, u, v, res);
                        let cells = es.iter().filter(|e| matches!(e.across, Across::Cell(..)));
                        if (ring(face) || (u > 0 && u + 1 < res)) && v > 0 && v + 1 < res {
                            assert_eq!(cells.count(), 6, "interior cell {face} ({u}, {v})");
                        }
                        for e in &es {
                            if let Across::Cell(nf, nu, nv) = e.across {
                                assert_ne!((nf, nu, nv), (face, u, v));
                                // the neighbour has the same edge pointing back here (reversed on
                                // the same face; either way round across a seam, where a face's u
                                // may run the other way)
                                let (a, b) =
                                    (six_on(face, nf, e.a, res), six_on(face, nf, e.b, res));
                                let back = edges(nf, nu, nv, res);
                                assert!(
                                    back.iter().any(|g| g.across == Across::Cell(face, u, v)
                                        && ((g.a == b && g.b == a)
                                            || (nf != face && g.a == a && g.b == b))),
                                    "{face} ({u}, {v}) edge {e:?} not mirrored by {nf} ({nu}, {nv})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn border_edges_lie_on_the_border_in_sixths() {
        let res = 8;
        let n6 = 6 * res as i64;
        for face in 0..6u8 {
            let mut border_len = 0;
            for v in 0..res {
                for u in 0..res {
                    for e in edges(face, u, v, res) {
                        if e.across == Across::Border {
                            let len = (e.b.0 - e.a.0).abs() + (e.b.1 - e.a.1).abs();
                            assert_eq!(len, 1);
                            let on_u = [e.a.0, e.b.0].iter().all(|&x| x == 0 || x == n6);
                            let on_v = [e.a.1, e.b.1].iter().all(|&y| y == 0 || y == n6);
                            // the ring faces' u borders are seams within the honeycomb
                            assert!(on_v || (on_u && !ring(face)), "{face} {e:?}");
                            border_len += len;
                        }
                    }
                }
            }
            // ring faces: only the two v borders; faces 4 and 5: all four
            assert_eq!(
                border_len,
                if ring(face) { 2 * n6 } else { 4 * n6 },
                "face {face}"
            );
        }
    }
}
