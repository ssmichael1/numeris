//! Numerical quadrature (integration).
//!
//! All algorithms are no-alloc compatible, using fixed-size stack-allocated
//! arrays. Requires [`FloatScalar`] bound (real-valued only).
//!
//! # Gauss-Legendre quadrature
//!
//! - [`gauss_legendre`] — N-point Gauss-Legendre quadrature on `[a, b]`.
//!   Exact for polynomials of degree ≤ 2N − 1. Supports N = 1..10, 15, 20.
//!
//! # Adaptive quadrature
//!
//! - [`adaptive_simpson`] — Adaptive Simpson's rule with automatic subdivision.
//!   Uses an explicit stack (no recursion) for no-std compatibility.
//!
//! # Composite rules
//!
//! - [`trapezoid`] — Composite trapezoidal rule with `n` subintervals.
//! - [`simpson`] — Composite Simpson's 1/3 rule with `n` subintervals (`n` must be even).
//!
//! # Examples
//!
//! ```
//! use numeris::quad::{gauss_legendre, adaptive_simpson, trapezoid, simpson};
//!
//! // Gauss-Legendre: integrate x^2 from 0 to 1 (exact for degree ≤ 2*3-1 = 5)
//! let result = gauss_legendre::<f64, 3>(|x| x * x, 0.0, 1.0);
//! assert!((result - 1.0 / 3.0).abs() < 1e-15);
//!
//! // Adaptive Simpson: integrate sin(x) from 0 to pi
//! let result = adaptive_simpson(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 1e-12).unwrap();
//! assert!((result - 2.0).abs() < 1e-12);
//!
//! // Composite trapezoid
//! let result = trapezoid(|x: f64| x * x, 0.0, 1.0, 1000);
//! assert!((result - 1.0 / 3.0).abs() < 1e-5);
//!
//! // Composite Simpson
//! let result = simpson(|x: f64| x * x, 0.0, 1.0, 100);
//! assert!((result - 1.0 / 3.0).abs() < 1e-12);
//! ```

use crate::traits::FloatScalar;

#[cfg(test)]
mod tests;

/// Errors from quadrature algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuadError {
    /// Adaptive subdivision exceeded the maximum recursion depth without
    /// meeting the requested tolerance.
    MaxDepthExceeded,
    /// Invalid input parameters (e.g., `n` is zero or odd for Simpson's rule).
    InvalidInput,
}

impl core::fmt::Display for QuadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuadError::MaxDepthExceeded => write!(f, "maximum subdivision depth exceeded"),
            QuadError::InvalidInput => write!(f, "invalid input parameters"),
        }
    }
}

// ---------------------------------------------------------------------------
// Gauss-Legendre nodes and weights on [-1, 1]
// ---------------------------------------------------------------------------

/// 1-point
const GL1_NODES: [f64; 1] = [0.0];
const GL1_WEIGHTS: [f64; 1] = [2.0];

/// 2-point
const GL2_NODES: [f64; 2] = [
    -0.5773502691896257645091489841,
     0.5773502691896257645091489841,
];
const GL2_WEIGHTS: [f64; 2] = [1.0, 1.0];

/// 3-point
const GL3_NODES: [f64; 3] = [
    -0.7745966692414833770358530800,
     0.0,
     0.7745966692414833770358530800,
];
const GL3_WEIGHTS: [f64; 3] = [
    0.5555555555555555555555555556,
    0.8888888888888888888888888889,
    0.5555555555555555555555555556,
];

/// 4-point
const GL4_NODES: [f64; 4] = [
    -0.8611363115940525752239464889,
    -0.3399810435848562648026657591,
     0.3399810435848562648026657591,
     0.8611363115940525752239464889,
];
const GL4_WEIGHTS: [f64; 4] = [
    0.3478548451374538573730639492,
    0.6521451548625461426269360508,
    0.6521451548625461426269360508,
    0.3478548451374538573730639492,
];

/// 5-point
const GL5_NODES: [f64; 5] = [
    -0.9061798459386639927976268782,
    -0.5384693101056830910363144207,
     0.0,
     0.5384693101056830910363144207,
     0.9061798459386639927976268782,
];
const GL5_WEIGHTS: [f64; 5] = [
    0.2369268850561890875142640407,
    0.4786286704993664680412915148,
    0.5688888888888888888888888889,
    0.4786286704993664680412915148,
    0.2369268850561890875142640407,
];

/// 6-point
const GL6_NODES: [f64; 6] = [
    -0.9324695142031520278123015545,
    -0.6612093864662645136613995950,
    -0.2386191860831969086305017216,
     0.2386191860831969086305017216,
     0.6612093864662645136613995950,
     0.9324695142031520278123015545,
];
const GL6_WEIGHTS: [f64; 6] = [
    0.1713244923791703450402961421,
    0.3607615730481386075698335138,
    0.4679139345726910473898703440,
    0.4679139345726910473898703440,
    0.3607615730481386075698335138,
    0.1713244923791703450402961421,
];

/// 7-point
const GL7_NODES: [f64; 7] = [
    -0.9491079123427585245261896840,
    -0.7415311855993944398638647733,
    -0.4058451513773971669066064121,
     0.0,
     0.4058451513773971669066064121,
     0.7415311855993944398638647733,
     0.9491079123427585245261896840,
];
const GL7_WEIGHTS: [f64; 7] = [
    0.1294849661688696932706114326,
    0.2797053914892766679014677714,
    0.3818300505051189449503697754,
    0.4179591836734693877551020408,
    0.3818300505051189449503697754,
    0.2797053914892766679014677714,
    0.1294849661688696932706114326,
];

/// 8-point
const GL8_NODES: [f64; 8] = [
    -0.9602898564975362316835608686,
    -0.7966664774136267395915539365,
    -0.5255324099163289858177390492,
    -0.1834346424956498049394761424,
     0.1834346424956498049394761424,
     0.5255324099163289858177390492,
     0.7966664774136267395915539365,
     0.9602898564975362316835608686,
];
const GL8_WEIGHTS: [f64; 8] = [
    0.1012285362903762591525313543,
    0.2223810344533744705443559944,
    0.3137066458778872873379622020,
    0.3626837833783619829651504493,
    0.3626837833783619829651504493,
    0.3137066458778872873379622020,
    0.2223810344533744705443559944,
    0.1012285362903762591525313543,
];

/// 9-point
const GL9_NODES: [f64; 9] = [
    -0.9681602395076260898355762030,
    -0.8360311073266357942994297880,
    -0.6133714327005903973087020393,
    -0.3242534234038089290385380146,
     0.0,
     0.3242534234038089290385380146,
     0.6133714327005903973087020393,
     0.8360311073266357942994297880,
     0.9681602395076260898355762030,
];
const GL9_WEIGHTS: [f64; 9] = [
    0.0812743883615744119718921581,
    0.1806481606948574040584720312,
    0.2606106964029354623187428694,
    0.3123470770400028400686304065,
    0.3302393550012597631645250693,
    0.3123470770400028400686304065,
    0.2606106964029354623187428694,
    0.1806481606948574040584720312,
    0.0812743883615744119718921581,
];

/// 10-point
const GL10_NODES: [f64; 10] = [
    -0.9739065285171717200779640120,
    -0.8650633666889845107320966884,
    -0.6794095682990244062343273651,
    -0.4333953941292471907992659432,
    -0.1488743389816312108848260012,
     0.1488743389816312108848260012,
     0.4333953941292471907992659432,
     0.6794095682990244062343273651,
     0.8650633666889845107320966884,
     0.9739065285171717200779640120,
];
const GL10_WEIGHTS: [f64; 10] = [
    0.0666713443086881375935688098,
    0.1494513491505805931457763400,
    0.2190863625159820439955349342,
    0.2692667193099963550912269216,
    0.2955242247147528701738929999,
    0.2955242247147528701738929999,
    0.2692667193099963550912269216,
    0.2190863625159820439955349342,
    0.1494513491505805931457763400,
    0.0666713443086881375935688098,
];

/// 15-point
const GL15_NODES: [f64; 15] = [
    -0.9879925180204854284895657186,
    -0.9372733924007059043077589477,
    -0.8482065834104272162006483207,
    -0.7244177313601700474161860547,
    -0.5709721726085388475372267373,
    -0.3941513470775633698972073710,
    -0.2011940939974345223006283034,
     0.0,
     0.2011940939974345223006283034,
     0.3941513470775633698972073710,
     0.5709721726085388475372267373,
     0.7244177313601700474161860547,
     0.8482065834104272162006483207,
     0.9372733924007059043077589477,
     0.9879925180204854284895657186,
];
const GL15_WEIGHTS: [f64; 15] = [
    0.0307532419961172683546283935,
    0.0703660474881081247092674164,
    0.1071592204671719350118695471,
    0.1395706779261543144478047946,
    0.1662692058169939335532008605,
    0.1861610000155622110268005619,
    0.1984314853271115764561183264,
    0.2025782419255612728806201999,
    0.1984314853271115764561183264,
    0.1861610000155622110268005619,
    0.1662692058169939335532008605,
    0.1395706779261543144478047946,
    0.1071592204671719350118695471,
    0.0703660474881081247092674164,
    0.0307532419961172683546283935,
];

/// 20-point
const GL20_NODES: [f64; 20] = [
    -0.9931285991850949247861223884,
    -0.9639719272779137912676661312,
    -0.9122344282513259058677524413,
    -0.8391169718222188233945290617,
    -0.7463319064601507926143050704,
    -0.6360536807265150254528366962,
    -0.5108670019508270980043640510,
    -0.3737060887154195606725481771,
    -0.2277858511416450780804961953,
    -0.0765265211334973337546404093,
     0.0765265211334973337546404093,
     0.2277858511416450780804961953,
     0.3737060887154195606725481771,
     0.5108670019508270980043640510,
     0.6360536807265150254528366962,
     0.7463319064601507926143050704,
     0.8391169718222188233945290617,
     0.9122344282513259058677524413,
     0.9639719272779137912676661312,
     0.9931285991850949247861223884,
];
const GL20_WEIGHTS: [f64; 20] = [
    0.0176140071391521183118619624,
    0.0406014298003869413310399522,
    0.0626720483341090635584173116,
    0.0832767415767047487247581432,
    0.1019301198172404350367501354,
    0.1181945319615184173123773777,
    0.1316886384491766268984944997,
    0.1420961093183820513292983251,
    0.1491729864726037467900547586,
    0.1527533871307258506980843320,
    0.1527533871307258506980843320,
    0.1491729864726037467900547586,
    0.1420961093183820513292983251,
    0.1316886384491766268984944997,
    0.1181945319615184173123773777,
    0.1019301198172404350367501354,
    0.0832767415767047487247581432,
    0.0626720483341090635584173116,
    0.0406014298003869413310399522,
    0.0176140071391521183118619624,
];

// ---------------------------------------------------------------------------
// Helper: evaluate GL quadrature given nodes/weights on [-1,1]
// ---------------------------------------------------------------------------

#[inline]
fn gl_eval<T: FloatScalar>(f: &impl Fn(T) -> T, a: T, b: T, nodes: &[f64], weights: &[f64]) -> T {
    let half = T::from(0.5).unwrap();
    let mid = half * (b + a);
    let half_len = half * (b - a);
    let mut sum = T::zero();
    for i in 0..nodes.len() {
        let t = T::from(nodes[i]).unwrap();
        let w = T::from(weights[i]).unwrap();
        sum = sum + w * f(mid + half_len * t);
    }
    sum * half_len
}

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature
// ---------------------------------------------------------------------------

/// N-point Gauss-Legendre quadrature of `f` over `[a, b]`.
///
/// Exact for polynomials of degree ≤ 2N − 1.
/// Supported values of `N`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20.
///
/// # Panics
///
/// Panics if `N` is not one of the supported values.
///
/// # Examples
///
/// ```
/// use numeris::quad::gauss_legendre;
///
/// // 5-point GL integrates x^9 exactly (degree 9 ≤ 2*5-1 = 9)
/// let result = gauss_legendre::<f64, 5>(|x| x.powi(9), -1.0, 1.0);
/// assert!(result.abs() < 1e-14); // odd function → integral is 0
///
/// // Integrate e^x from 0 to 1 (exact answer: e - 1)
/// let result = gauss_legendre::<f64, 10>(|x| x.exp(), 0.0, 1.0);
/// assert!((result - (1.0_f64.exp() - 1.0)).abs() < 1e-14);
/// ```
pub fn gauss_legendre<T: FloatScalar, const N: usize>(
    f: impl Fn(T) -> T,
    a: T,
    b: T,
) -> T {
    match N {
        1 => gl_eval(&f, a, b, &GL1_NODES, &GL1_WEIGHTS),
        2 => gl_eval(&f, a, b, &GL2_NODES, &GL2_WEIGHTS),
        3 => gl_eval(&f, a, b, &GL3_NODES, &GL3_WEIGHTS),
        4 => gl_eval(&f, a, b, &GL4_NODES, &GL4_WEIGHTS),
        5 => gl_eval(&f, a, b, &GL5_NODES, &GL5_WEIGHTS),
        6 => gl_eval(&f, a, b, &GL6_NODES, &GL6_WEIGHTS),
        7 => gl_eval(&f, a, b, &GL7_NODES, &GL7_WEIGHTS),
        8 => gl_eval(&f, a, b, &GL8_NODES, &GL8_WEIGHTS),
        9 => gl_eval(&f, a, b, &GL9_NODES, &GL9_WEIGHTS),
        10 => gl_eval(&f, a, b, &GL10_NODES, &GL10_WEIGHTS),
        15 => gl_eval(&f, a, b, &GL15_NODES, &GL15_WEIGHTS),
        20 => gl_eval(&f, a, b, &GL20_NODES, &GL20_WEIGHTS),
        _ => panic!("gauss_legendre: unsupported N = {}; supported: 1..10, 15, 20", N),
    }
}

// ---------------------------------------------------------------------------
// Composite trapezoidal rule
// ---------------------------------------------------------------------------

/// Composite trapezoidal rule for `f` over `[a, b]` with `n` subintervals.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Examples
///
/// ```
/// use numeris::quad::trapezoid;
///
/// let result = trapezoid(|x: f64| x * x, 0.0, 1.0, 10000);
/// assert!((result - 1.0 / 3.0).abs() < 1e-7);
/// ```
pub fn trapezoid<T: FloatScalar>(f: impl Fn(T) -> T, a: T, b: T, n: usize) -> T {
    assert!(n > 0, "trapezoid: n must be > 0");
    let n_t = T::from(n).unwrap();
    let h = (b - a) / n_t;
    let mut sum = (f(a) + f(b)) * T::from(0.5).unwrap();
    for i in 1..n {
        let x = a + T::from(i).unwrap() * h;
        sum = sum + f(x);
    }
    sum * h
}

// ---------------------------------------------------------------------------
// Composite Simpson's 1/3 rule
// ---------------------------------------------------------------------------

/// Composite Simpson's 1/3 rule for `f` over `[a, b]` with `n` subintervals.
///
/// `n` must be even and positive.
///
/// # Panics
///
/// Panics if `n` is zero or odd.
///
/// # Examples
///
/// ```
/// use numeris::quad::simpson;
///
/// let result = simpson(|x: f64| x * x, 0.0, 1.0, 100);
/// assert!((result - 1.0 / 3.0).abs() < 1e-14);
/// ```
pub fn simpson<T: FloatScalar>(f: impl Fn(T) -> T, a: T, b: T, n: usize) -> T {
    assert!(n > 0 && n % 2 == 0, "simpson: n must be even and > 0");
    let n_t = T::from(n).unwrap();
    let h = (b - a) / n_t;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + T::from(i).unwrap() * h;
        let coeff = if i % 2 == 0 {
            T::from(2.0).unwrap()
        } else {
            T::from(4.0).unwrap()
        };
        sum = sum + coeff * f(x);
    }
    sum * h / T::from(3.0).unwrap()
}

// ---------------------------------------------------------------------------
// Adaptive Simpson's rule (iterative, no recursion)
// ---------------------------------------------------------------------------

/// Maximum stack depth for adaptive Simpson subdivision.
const MAX_DEPTH: usize = 50;

/// Adaptive Simpson's rule for `f` over `[a, b]` with tolerance `tol`.
///
/// Uses an explicit stack (no recursion) for no-std compatibility.
/// Subdivides intervals until the local error estimate (comparing the
/// whole-interval Simpson approximation with the two-half-interval sum)
/// is below `tol`.
///
/// Returns [`QuadError::MaxDepthExceeded`] if the maximum subdivision depth
/// (50 levels) is reached without meeting tolerance.
///
/// Returns [`QuadError::InvalidInput`] if `tol` is not positive.
///
/// # Examples
///
/// ```
/// use numeris::quad::adaptive_simpson;
///
/// // Integrate ln(x) from 1 to 2 (exact: 2*ln(2) - 1)
/// let exact = 2.0 * 2.0_f64.ln() - 1.0;
/// let result = adaptive_simpson(|x: f64| x.ln(), 1.0, 2.0, 1e-12).unwrap();
/// assert!((result - exact).abs() < 1e-12);
/// ```
pub fn adaptive_simpson<T: FloatScalar>(
    f: impl Fn(T) -> T,
    a: T,
    b: T,
    tol: T,
) -> Result<T, QuadError> {
    if !(tol > T::zero()) {
        return Err(QuadError::InvalidInput);
    }

    let two = T::from(2.0).unwrap();
    let four = T::from(4.0).unwrap();
    let six = T::from(6.0).unwrap();
    let fifteen = T::from(15.0).unwrap();

    // Simpson's rule on [a, b] given endpoint and midpoint values
    #[inline]
    fn simpson_val<T: FloatScalar>(h: T, fa: T, fm: T, fb: T) -> T {
        let six = T::from(6.0).unwrap();
        h * (fa + T::from(4.0).unwrap() * fm + fb) / six
    }

    // Stack entry: (left, right, f_left, f_mid, f_right, whole_simpson, current_tol, depth)
    // We store function values to avoid redundant evaluations.
    struct Entry<T> {
        a: T,
        b: T,
        fa: T,
        fm: T,
        fb: T,
        whole: T,
        tol: T,
        depth: usize,
    }

    let fa = f(a);
    let fb = f(b);
    let mid = (a + b) / two;
    let fm = f(mid);
    let whole = simpson_val(b - a, fa, fm, fb);

    // Fixed-size stack (no heap)
    let mut stack: [core::mem::MaybeUninit<Entry<T>>; MAX_DEPTH + 1] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    // Push initial entry
    let mut sp: usize = 1;
    stack[0] = core::mem::MaybeUninit::new(Entry {
        a, b, fa, fm, fb, whole, tol, depth: 0,
    });

    let mut total = T::zero();

    while sp > 0 {
        sp -= 1;
        let entry = unsafe { stack[sp].assume_init_read() };

        let mid = (entry.a + entry.b) / two;
        let h = entry.b - entry.a;

        // Midpoints of left and right halves
        let m1 = (entry.a + mid) / two;
        let m2 = (mid + entry.b) / two;
        let fm1 = f(m1);
        let fm2 = f(m2);

        let left = h * (entry.fa + four * fm1 + entry.fm) / (six * two);
        let right = h * (entry.fm + four * fm2 + entry.fb) / (six * two);
        let refined = left + right;

        // Error estimate: |refined - whole| / 15 (Richardson extrapolation)
        let err = (refined - entry.whole) / fifteen;
        let err_abs = if err < T::zero() { T::zero() - err } else { err };

        if err_abs <= entry.tol || entry.depth >= MAX_DEPTH {
            if entry.depth >= MAX_DEPTH && err_abs > entry.tol {
                return Err(QuadError::MaxDepthExceeded);
            }
            // Accept with Richardson correction
            total = total + refined + err;
        } else {
            // Subdivide: push right half first (so left is processed first)
            let half_tol = entry.tol / two;
            let new_depth = entry.depth + 1;

            if sp + 2 > MAX_DEPTH + 1 {
                return Err(QuadError::MaxDepthExceeded);
            }

            // Right half
            stack[sp] = core::mem::MaybeUninit::new(Entry {
                a: mid,
                b: entry.b,
                fa: entry.fm,
                fm: fm2,
                fb: entry.fb,
                whole: right,
                tol: half_tol,
                depth: new_depth,
            });
            sp += 1;

            // Left half
            stack[sp] = core::mem::MaybeUninit::new(Entry {
                a: entry.a,
                b: mid,
                fa: entry.fa,
                fm: fm1,
                fb: entry.fm,
                whole: left,
                tol: half_tol,
                depth: new_depth,
            });
            sp += 1;
        }
    }

    Ok(total)
}
