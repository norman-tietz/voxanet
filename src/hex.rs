// hex.rs
// Hex columns: the honeycomb a cube face is divided into in the hex terrain style (/terrain_style hex).
// Column (u, v) of the square grid is hex cell (u, v): rows run along u, every odd row (v odd) shifted
// by half a column, so the height map, edits and chunks are the same in both styles. Cells are 1 wide,
// their vertical sides span the middle ⅔ of the row and their points reach ⅙ into the rows above and
// below (pointy-top hexagons about 13 % flatter than regular ones, area 1). At the face border the
// honeycomb is cut off straight: the cells of the first and last row are squared off to the border, and
// in odd rows the first cell also takes the half-cell gap at u = 0 while the last is cut at u = res.
//
// Pure 2D geometry in face space (u, v in 0..=res), no world positions. Corners are exact integers in
// sixths of a column (u6 = 6u, v6 = 6v): every corner of every cell lies on that lattice, so shared
// corners and the cube-face seams compare exactly.

// face-space sixths per column
pub const SIXTHS: i64 = 6;

// a row's shift along u, in sixths: odd rows are half a column to the right
fn shift6(row: i64) -> i64 {
    3 * row.rem_euclid(2)
}

// the boundary between row `row` and the row below it at u6 = x: a zigzag between the bottom points of
// row's cells (6·row − 1) and the corners between them (6·row + 1)
fn zig6(row: i64, x: f64) -> f64 {
    let t = (x - shift6(row) as f64).rem_euclid(6.0);
    let d = (t - 3.0).abs(); // 0 under a cell's centre, 3 between two cells
    (6 * row - 1) as f64 + d * 2.0 / 3.0
}

// zig6 at a lattice column (x a multiple of 3), exactly
fn zig6_at(row: i64, x: i64) -> i64 {
    debug_assert!(x % 3 == 0);
    let t = (x - shift6(row)).rem_euclid(6);
    6 * row - 1 + 2 * (t - 3).abs() / 3
}

// the cell containing face-space point (u, v) (both in columns, the face is [0, res]²; points outside
// are clamped onto the border cells)
pub fn cell_at(u: f64, v: f64, res: u32) -> (u32, u32) {
    let (x, y) = (u * 6.0, v * 6.0);
    let r0 = v.floor() as i64;
    let row = if y < zig6(r0, x) {
        r0 - 1
    } else if y >= zig6(r0 + 1, x) {
        r0 + 1
    } else {
        r0
    };
    let row = row.clamp(0, res as i64 - 1);
    let col = ((x - shift6(row) as f64) / 6.0).floor() as i64;
    (col.clamp(0, res as i64 - 1) as u32, row as u32)
}

// cell (u, v)'s polygon, counter-clockwise (v up), in sixths. 6 corners inside the face; border cells
// have their outer point replaced by the border (and odd rows' end cells are widened / cut there)
pub fn outline(u: u32, v: u32, res: u32) -> Vec<(i64, i64)> {
    let (col, row, n) = (u as i64, v as i64, res as i64);
    let s = shift6(row);
    // the face border clamps cell_at's column: the first cell reaches u = 0, the last u = res
    let lo = if col == 0 { 0 } else { 6 * col + s };
    let hi = if col == n - 1 { 6 * n } else { 6 * col + 6 + s };
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
    drop_collinear(pts)
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

// the area of an outline in columns² (1 for an interior cell)
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
    Cell(u32, u32),
    // the face border: the next cube face (found in world space, PlanetData::hex_walls)
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

// the edges of cell (u, v), counter-clockwise
pub fn edges(u: u32, v: u32, res: u32) -> Vec<Edge> {
    let pts = outline(u, v, res);
    let n6 = 6 * res as i64;
    let on_border = |a: (i64, i64), b: (i64, i64)| {
        (a.0 == b.0 && (a.0 == 0 || a.0 == n6)) || (a.1 == b.1 && (a.1 == 0 || a.1 == n6))
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
            let (cu, cv) = cell_at(mx + nx, my + ny, res);
            out.push(Edge {
                a,
                b,
                across: Across::Cell(cu, cv),
            });
        }
    }
    out
}

// a point's distance to segment a → b (sixths) in columns
pub fn distance_to_edge(u: f64, v: f64, e: &Edge) -> f64 {
    let (ax, ay) = (e.a.0 as f64 / 6.0, e.a.1 as f64 / 6.0);
    let (bx, by) = (e.b.0 as f64 / 6.0, e.b.1 as f64 / 6.0);
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

    #[test]
    fn interior_cell_is_the_squashed_hexagon() {
        // even row: centre (3.5, 4.5) in columns
        assert_eq!(
            outline(3, 4, 16),
            vec![(18, 25), (21, 23), (24, 25), (24, 29), (21, 31), (18, 29)]
        );
        // odd row: shifted half a column right
        assert_eq!(
            outline(3, 5, 16),
            vec![(21, 31), (24, 29), (27, 31), (27, 35), (24, 37), (21, 35)]
        );
        assert_eq!(area(&outline(3, 4, 16)), 1.0);
        assert_eq!(centroid(&outline(3, 4, 16)), (3.5, 4.5));
    }

    #[test]
    fn every_point_lies_in_the_outline_of_its_cell() {
        for res in [8u32, 9, 32] {
            for (u, v) in points(20_000, res) {
                let (cu, cv) = cell_at(u, v, res);
                assert!(cu < res && cv < res);
                let pts = outline(cu, cv, res);
                // points on a boundary may go either way; skip those within a hair of one
                let near_edge = edges(cu, cv, res)
                    .iter()
                    .any(|e| distance_to_edge(u, v, e) < 1e-9);
                assert!(
                    near_edge || inside(&pts, u, v),
                    "res {res}: ({u}, {v}) -> ({cu}, {cv}) {pts:?}"
                );
            }
        }
    }

    #[test]
    fn cells_cover_the_face_exactly() {
        for res in [1u32, 2, 7, 8, 32] {
            let mut total = 0.0;
            for v in 0..res {
                let mut row = 0.0;
                for u in 0..res {
                    let a = area(&outline(u, v, res));
                    assert!(a > 0.0);
                    if u > 0 && u + 1 < res && v > 0 && v + 1 < res {
                        assert_eq!(a, 1.0, "interior cell ({u}, {v})");
                    }
                    row += a;
                }
                assert_eq!(row, res as f64, "row {v} of res {res}");
                total += row;
            }
            assert_eq!(total, (res * res) as f64);
        }
    }

    #[test]
    fn border_cells_are_cut_off_straight() {
        let res = 8;
        // odd row, first cell: widened to u = 0 (8 corners, area 1½)
        let first = outline(0, 1, res);
        assert_eq!(area(&first), 1.5);
        assert_eq!(first.iter().filter(|p| p.0 == 0).count(), 2);
        // odd row, last cell: cut at u = res (area ½)
        assert_eq!(area(&outline(res - 1, 1, res)), 0.5);
        // first row: squared off at v = 0
        assert!(outline(3, 0, res).iter().filter(|p| p.1 == 0).count() == 2);
        // the cell centres map to their own cells
        for v in 0..res {
            for u in 0..res {
                let (cx, cy) = centroid(&outline(u, v, res));
                assert_eq!(cell_at(cx, cy, res), (u, v));
            }
        }
    }

    #[test]
    fn neighbours_are_symmetric_and_edges_match() {
        for res in [7u32, 8] {
            for v in 0..res {
                for u in 0..res {
                    let es = edges(u, v, res);
                    let cells: Vec<_> = es
                        .iter()
                        .filter_map(|e| match e.across {
                            Across::Cell(a, b) => Some((a, b)),
                            Across::Border => None,
                        })
                        .collect();
                    if u > 0 && u + 1 < res && v > 0 && v + 1 < res {
                        assert_eq!(cells.len(), 6, "interior cell ({u}, {v})");
                    }
                    for e in &es {
                        if let Across::Cell(nu, nv) = e.across {
                            assert_ne!((nu, nv), (u, v));
                            // the neighbour has the same edge, reversed, pointing back here
                            let back = edges(nu, nv, res);
                            assert!(
                                back.iter().any(|f| f.a == e.b
                                    && f.b == e.a
                                    && f.across == Across::Cell(u, v)),
                                "({u}, {v}) edge {e:?} not mirrored by ({nu}, {nv})"
                            );
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
        let mut border_len = 0;
        for v in 0..res {
            for u in 0..res {
                for e in edges(u, v, res) {
                    if e.across == Across::Border {
                        let len = (e.b.0 - e.a.0).abs() + (e.b.1 - e.a.1).abs();
                        assert_eq!(len, 1);
                        assert!(
                            [e.a.0, e.b.0].iter().all(|&x| x == 0 || x == n6)
                                || [e.a.1, e.b.1].iter().all(|&y| y == 0 || y == n6)
                        );
                        border_len += len;
                    }
                }
            }
        }
        assert_eq!(border_len, 4 * n6);
    }
}
