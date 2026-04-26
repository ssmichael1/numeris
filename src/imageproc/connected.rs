//! Connected components labeling via the SAUF two-pass algorithm
//! (Scan + Array-based Union-Find, Wu/Otoo/Suzuki 2009) with union-find
//! path compression and union by rank.
//!
//! Operates on any [`MatrixRef<T>`](crate::MatrixRef) where foreground is
//! defined as "element != background". Connectivity is selectable between
//! 4- and 8-connectivity via [`Connectivity`].

use alloc::vec;
use alloc::vec::Vec;

use crate::dynmatrix::DynMatrix;
use crate::traits::{MatrixRef, Scalar};

/// Neighborhood connectivity for connected-components labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connectivity: a pixel is adjacent to its N, S, E, W neighbors.
    Four,
    /// 8-connectivity: a pixel is adjacent to its N, S, E, W, and diagonal neighbors.
    Eight,
}

/// A single connected component.
///
/// Pixel coordinates use `(row, col)` with the origin at the top-left of the
/// image. All floating-point statistics use `f64` regardless of the input
/// element type.
///
/// To recover the per-pixel coordinates of a component, pair the component
/// list with [`connected_components_labeled`] or
/// [`connected_components_with_label_buffer`] and iterate the bbox window
/// against the labels buffer. This avoids a per-component `Vec` allocation
/// when the pixel list is not needed (often the case for downstream stats).
///
/// # Moments
///
/// `mu20`, `mu02`, `mu11` are the **central second moments**, unnormalized:
///
/// ```text
/// mu20 = Σ (r - r̄)²
/// mu02 = Σ (c - c̄)²
/// mu11 = Σ (r - r̄)·(c - c̄)
/// ```
///
/// where `(r̄, c̄)` is [`centroid`](Self::centroid). To get the orientation,
/// eccentricity, or principal axes of the component, the eigen-decomposition
/// of `[[mu20, mu11], [mu11, mu02]]` is the standard tool. Divide by `area`
/// to get variance/covariance (the normalized central moments `η_pq`).
#[derive(Debug, Clone)]
pub struct Component {
    /// Number of foreground pixels in the component.
    pub area: u32,
    /// Inclusive bounding box minimum `(row, col)`.
    pub bbox_min: (u32, u32),
    /// Inclusive bounding box maximum `(row, col)`.
    pub bbox_max: (u32, u32),
    /// Geometric centroid `(row, col)` — mean of pixel coordinates.
    pub centroid: (f64, f64),
    /// Central second moment `Σ(r − r̄)²` (unnormalized).
    pub mu20: f64,
    /// Central second moment `Σ(c − c̄)²` (unnormalized).
    pub mu02: f64,
    /// Central second moment `Σ(r − r̄)·(c − c̄)` (unnormalized).
    pub mu11: f64,
}

/// Union-find with path compression and union by rank.
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new() -> Self {
        // Reserve label 0 for background; add a sentinel so label IDs
        // start at 1 and indexing is direct.
        Self {
            parent: vec![0],
            rank: vec![0],
        }
    }

    fn make_set(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        // Iterative path compression (halving).
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (rhi, rlo) = match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            core::cmp::Ordering::Less => (rb, ra),
            core::cmp::Ordering::Greater => (ra, rb),
            core::cmp::Ordering::Equal => {
                self.rank[ra as usize] += 1;
                (ra, rb)
            }
        };
        self.parent[rlo as usize] = rhi;
        rhi
    }
}

/// First pass: scan row-major, assign provisional labels, record equivalences.
///
/// Returns the provisional-labels buffer (zero = background) and the populated
/// union-find structure. Neighbors examined per pixel: `(r, c-1)` and
/// `(r-1, c)` for 4-conn, plus `(r-1, c-1)` and `(r-1, c+1)` for 8-conn.
fn first_pass<T, I>(
    img: &I,
    connectivity: Connectivity,
    background: T,
) -> (Vec<u32>, UnionFind)
where
    T: Scalar + PartialEq,
    I: MatrixRef<T> + ?Sized,
{
    let h = img.nrows();
    let w = img.ncols();
    let mut labels: Vec<u32> = vec![0; h * w];
    let mut uf = UnionFind::new();
    let eight = matches!(connectivity, Connectivity::Eight);

    for r in 0..h {
        for c in 0..w {
            if *img.get(r, c) == background {
                continue;
            }

            // Gather labels of already-visited neighbors.
            let mut n: [u32; 4] = [0; 4];
            let mut k: usize = 0;
            // West
            if c > 0 {
                let l = labels[r * w + (c - 1)];
                if l != 0 {
                    n[k] = l;
                    k += 1;
                }
            }
            // North
            if r > 0 {
                let l = labels[(r - 1) * w + c];
                if l != 0 {
                    n[k] = l;
                    k += 1;
                }
                if eight {
                    if c > 0 {
                        let l = labels[(r - 1) * w + (c - 1)];
                        if l != 0 {
                            n[k] = l;
                            k += 1;
                        }
                    }
                    if c + 1 < w {
                        let l = labels[(r - 1) * w + (c + 1)];
                        if l != 0 {
                            n[k] = l;
                            k += 1;
                        }
                    }
                }
            }

            let assigned = if k == 0 {
                uf.make_set()
            } else {
                let mut min_label = n[0];
                for i in 1..k {
                    min_label = uf.union(min_label, n[i]);
                }
                for i in 0..k {
                    uf.union(min_label, n[i]);
                }
                min_label
            };
            labels[r * w + c] = assigned;
        }
    }

    (labels, uf)
}

/// Resolve union-find roots to compact label IDs `1..=num_components`.
///
/// Returns a map `provisional_id -> canonical_id` of length
/// `uf.parent.len()`; index 0 is reserved for background (maps to 0).
fn resolve_roots(uf: &mut UnionFind) -> (Vec<u32>, usize) {
    let n = uf.parent.len();
    let mut canonical: Vec<u32> = vec![0; n];
    let mut num = 0u32;
    for pid in 1..n as u32 {
        let root = uf.find(pid);
        if canonical[root as usize] == 0 {
            num += 1;
            canonical[root as usize] = num;
        }
        canonical[pid as usize] = canonical[root as usize];
    }
    (canonical, num as usize)
}

/// Second pass: rewrite labels to canonical IDs and accumulate per-component
/// statistics. Writes `write_labels` into `out` only if `Some`.
#[allow(clippy::type_complexity)]
fn second_pass(
    labels: &mut [u32],
    canonical: &[u32],
    h: usize,
    w: usize,
    num_components: usize,
) -> Vec<Component> {
    // Accumulators per component (raw moments — converted at the end).
    // Index 0 is unused (background).
    let n = num_components + 1;
    let mut areas: Vec<u32> = vec![0; n];
    let mut sum_r: Vec<f64> = vec![0.0; n];
    let mut sum_c: Vec<f64> = vec![0.0; n];
    let mut sum_rr: Vec<f64> = vec![0.0; n];
    let mut sum_cc: Vec<f64> = vec![0.0; n];
    let mut sum_rc: Vec<f64> = vec![0.0; n];
    let mut r_min: Vec<u32> = vec![u32::MAX; n];
    let mut r_max: Vec<u32> = vec![0; n];
    let mut c_min: Vec<u32> = vec![u32::MAX; n];
    let mut c_max: Vec<u32> = vec![0; n];

    for r in 0..h {
        for c in 0..w {
            let prov = labels[r * w + c];
            if prov == 0 {
                continue;
            }
            let id = canonical[prov as usize];
            labels[r * w + c] = id;
            let i = id as usize;
            let rf = r as f64;
            let cf = c as f64;
            areas[i] += 1;
            sum_r[i] += rf;
            sum_c[i] += cf;
            sum_rr[i] += rf * rf;
            sum_cc[i] += cf * cf;
            sum_rc[i] += rf * cf;
            let ru = r as u32;
            let cu = c as u32;
            if ru < r_min[i] {
                r_min[i] = ru;
            }
            if ru > r_max[i] {
                r_max[i] = ru;
            }
            if cu < c_min[i] {
                c_min[i] = cu;
            }
            if cu > c_max[i] {
                c_max[i] = cu;
            }
        }
    }

    let mut out = Vec::with_capacity(num_components);
    for i in 1..n {
        let a = areas[i] as f64;
        let rbar = sum_r[i] / a;
        let cbar = sum_c[i] / a;
        // Central moments from raw moments:
        //   Σ(r - r̄)² = Σr² − n·r̄²  (and similarly for cc, rc).
        let mu20 = sum_rr[i] - a * rbar * rbar;
        let mu02 = sum_cc[i] - a * cbar * cbar;
        let mu11 = sum_rc[i] - a * rbar * cbar;
        out.push(Component {
            area: areas[i],
            bbox_min: (r_min[i], c_min[i]),
            bbox_max: (r_max[i], c_max[i]),
            centroid: (rbar, cbar),
            mu20,
            mu02,
            mu11,
        });
    }
    out
}

/// Connected-components labeling — returns the list of components.
///
/// A pixel is treated as **foreground** when `*img.get(r, c) != background`,
/// and components are the maximal sets of foreground pixels reachable under
/// the chosen [`Connectivity`].
///
/// Runs in `O(H·W·α(H·W))` using two-pass SAUF with union-find path
/// compression and union by rank; unlike the `_labeled` variant, this
/// function does **not** allocate a full `H × W` labels image, so it is
/// cheaper when only the component list is needed.
///
/// Components are returned in the order they are first encountered in
/// row-major scan order.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{connected_components, Connectivity};
///
/// // Two non-touching blobs of `1`s on a background of `0`s.
/// let img = DynMatrix::<u8>::from_rows(
///     4, 5,
///     &[1, 1, 0, 0, 0,
///       1, 1, 0, 0, 0,
///       0, 0, 0, 1, 1,
///       0, 0, 0, 1, 1],
/// );
/// let comps = connected_components(&img, Connectivity::Four, 0u8);
/// assert_eq!(comps.len(), 2);
/// assert_eq!(comps[0].area, 4);
/// assert_eq!(comps[1].area, 4);
/// ```
pub fn connected_components<T, I>(
    img: &I,
    connectivity: Connectivity,
    background: T,
) -> Vec<Component>
where
    T: Scalar + PartialEq,
    I: MatrixRef<T> + ?Sized,
{
    let h = img.nrows();
    let w = img.ncols();
    if h == 0 || w == 0 {
        return Vec::new();
    }
    let (mut labels, mut uf) = first_pass(img, connectivity, background);
    let (canonical, num) = resolve_roots(&mut uf);
    second_pass(&mut labels, &canonical, h, w, num)
}

/// Connected-components labeling — returns a row-major flat labels buffer
/// and the list of components.
///
/// The labels buffer has length `nrows * ncols`, indexed as `labels[r * w + c]`.
/// Background pixels have label `0`; foreground pixels hold a component ID in
/// `1..=components.len()`. Component `i` (0-indexed) has label `i + 1`.
///
/// Prefer this variant over [`connected_components_labeled`] when downstream
/// processing iterates rows in scan order (e.g. per-bbox sweeps), since this
/// avoids the column-major repack into [`DynMatrix`].
pub fn connected_components_with_label_buffer<T, I>(
    img: &I,
    connectivity: Connectivity,
    background: T,
) -> (Vec<u32>, Vec<Component>)
where
    T: Scalar + PartialEq,
    I: MatrixRef<T> + ?Sized,
{
    let h = img.nrows();
    let w = img.ncols();
    if h == 0 || w == 0 {
        return (Vec::new(), Vec::new());
    }
    let (mut labels, mut uf) = first_pass(img, connectivity, background);
    let (canonical, num) = resolve_roots(&mut uf);
    let comps = second_pass(&mut labels, &canonical, h, w, num);
    (labels, comps)
}

/// Connected-components labeling — returns both the labels image (as a
/// column-major [`DynMatrix`]) and the list of components.
///
/// The labels image has the same shape as `img`. Background pixels have
/// label `0`; foreground pixels hold a component ID in `1..=components.len()`.
/// Component `i` (0-indexed) has label `i + 1`.
///
/// Useful when a downstream step needs to mask or index pixels by component
/// (e.g. per-label filtering, re-rendering). If only the component list is
/// needed, prefer [`connected_components`] which skips the labels allocation.
/// If you plan to iterate in row-major order, prefer
/// [`connected_components_with_label_buffer`] which avoids the transpose.
///
/// # Example
///
/// ```
/// use numeris::DynMatrix;
/// use numeris::imageproc::{connected_components_labeled, Connectivity};
///
/// let img = DynMatrix::<u8>::from_rows(
///     3, 3,
///     &[1, 0, 2,
///       1, 0, 2,
///       0, 0, 0],
/// );
/// let (labels, comps) = connected_components_labeled(&img, Connectivity::Four, 0u8);
/// assert_eq!(comps.len(), 2);
/// assert_eq!(labels[(0, 0)], 1);
/// assert_eq!(labels[(0, 2)], 2);
/// assert_eq!(labels[(2, 1)], 0);
/// ```
pub fn connected_components_labeled<T, I>(
    img: &I,
    connectivity: Connectivity,
    background: T,
) -> (DynMatrix<u32>, Vec<Component>)
where
    T: Scalar + PartialEq,
    I: MatrixRef<T> + ?Sized,
{
    let h = img.nrows();
    let w = img.ncols();
    if h == 0 || w == 0 {
        return (DynMatrix::<u32>::zeros(h, w), Vec::new());
    }
    let (mut labels, mut uf) = first_pass(img, connectivity, background);
    let (canonical, num) = resolve_roots(&mut uf);
    let comps = second_pass(&mut labels, &canonical, h, w, num);

    // Pack row-major `labels` (scratch buffer) into a column-major DynMatrix.
    let mut out = DynMatrix::<u32>::zeros(h, w);
    for r in 0..h {
        for c in 0..w {
            out[(r, c)] = labels[r * w + c];
        }
    }
    (out, comps)
}
