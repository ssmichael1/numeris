// Reference implementations:
//   E. Hairer & G. Wanner, "Solving Ordinary Differential Equations II" (1996), §IV.7
//   E. Hairer, rodas.f — http://www.unige.ch/~hairer/prog/stiff/rodas.f
//   Coefficients transcribed from rodas.f subroutine RODAS4

use super::rosenbrock::Rosenbrock;

/// RODAS4 — 6-stage, order 4(3), L-stable Rosenbrock method.
///
/// From Hairer & Wanner, "Solving Ordinary Differential Equations II", §IV.7.
/// Stiffly accurate: the last two stages coincide with the solution, making
/// the error estimate `err = k_6` (the 6th stage solve).
///
/// Well-suited for moderately stiff systems (chemical kinetics, circuit
/// simulation, orbital mechanics with drag).
pub struct RODAS4;

impl Rosenbrock<6> for RODAS4 {
    const GAMMA_DIAG: f64 = 0.25;

    const ALPHA: [f64; 6] = [
        0.0,
        0.386,
        0.21,
        0.63,
        1.0,
        1.0,
    ];

    #[rustfmt::skip]
    const A: [[f64; 6]; 6] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1544000000000000e+01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9466785280815826e+00, 0.2557011698983284e+00, 0.0, 0.0, 0.0, 0.0],
        [0.3314825187068521e+01, 0.2896124015972201e+01, 0.9986419139977817e+00, 0.0, 0.0, 0.0],
        [0.1221224509226641e+01, 0.6019134481288629e+01, 0.1253708332932087e+02, -0.6878860361058950e+00, 0.0, 0.0],
        [0.1221224509226641e+01, 0.6019134481288629e+01, 0.1253708332932087e+02, -0.6878860361058950e+00, 1.0, 0.0],
    ];

    #[rustfmt::skip]
    const C: [[f64; 6]; 6] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5668800000000000e+01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.2430093356833875e+01, -0.2063599157091915e+00, 0.0, 0.0, 0.0, 0.0],
        [-0.1073529058151375e+00, -0.9594562251023355e+01, -0.2047028614809616e+02, 0.0, 0.0, 0.0],
        [ 0.7496443313967647e+01, -0.1024680431464352e+02, -0.3399990352819905e+02,  0.1170890893206160e+02, 0.0, 0.0],
        [ 0.8083246795921522e+01, -0.7981132988064893e+01, -0.3152159432874371e+02,  0.1631930543123136e+02, -0.6058818238834054e+01, 0.0],
    ];

    // Row sums of the full Γ matrix: γ_diag + Σ_j c_ij
    const GAMMA_SUM: [f64; 6] = [
        0.25,
        -0.1043,
        0.1035,
        -0.03620000000000023,
        0.0,
        0.0,
    ];

    // Solution weights (4th order). For stiffly-accurate RODAS4,
    // m[0..4] = a[4,0..4] and m[4] = m[5] = 1.
    const M: [f64; 6] = [
        0.1221224509226641e+01,
        0.6019134481288629e+01,
        0.1253708332932087e+02,
        -0.6878860361058950e+00,
        1.0,
        1.0,
    ];

    // Embedded weights (3rd order). Same as M except m̂[5] = 0,
    // so the error estimate is simply k_6.
    const MHAT: [f64; 6] = [
        0.1221224509226641e+01,
        0.6019134481288629e+01,
        0.1253708332932087e+02,
        -0.6878860361058950e+00,
        1.0,
        0.0,
    ];

    const ORDER: usize = 4;
}
