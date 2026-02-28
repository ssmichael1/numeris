use super::*;

const TOL: f64 = 1e-10;
const LOOSE_TOL: f64 = 1e-6;

fn assert_near(a: f64, b: f64, tol: f64, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{}: {} vs {} (diff {})",
        msg,
        a,
        b,
        (a - b).abs()
    );
}

/// Evaluate the cascade's frequency response magnitude at a given frequency.
/// Uses z = exp(j·2π·f/fs), computed with real arithmetic.
fn freq_response_mag<const N: usize>(
    cascade: &BiquadCascade<f64, N>,
    freq: f64,
    sample_rate: f64,
) -> f64 {
    let omega = 2.0 * core::f64::consts::PI * freq / sample_rate;
    let cos_w = omega.cos();
    let cos_2w = (2.0 * omega).cos();
    let sin_w = omega.sin();
    let sin_2w = (2.0 * omega).sin();

    let mut mag_sq = 1.0;
    for section in &cascade.sections {
        let (b, a) = section.coefficients();
        // Numerator: b0 + b1·e^{-jω} + b2·e^{-j2ω}
        let num_re = b[0] + b[1] * cos_w + b[2] * cos_2w;
        let num_im = -(b[1] * sin_w + b[2] * sin_2w);
        // Denominator: 1 + a1·e^{-jω} + a2·e^{-j2ω}
        let den_re = a[0] + a[1] * cos_w + a[2] * cos_2w;
        let den_im = -(a[1] * sin_w + a[2] * sin_2w);

        let num_mag_sq = num_re * num_re + num_im * num_im;
        let den_mag_sq = den_re * den_re + den_im * den_im;
        mag_sq *= num_mag_sq / den_mag_sq;
    }
    mag_sq.sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// Biquad basics
// ═══════════════════════════════════════════════════════════════════

#[test]
fn biquad_passthrough() {
    let mut bq = Biquad::<f64>::passthrough();
    assert_eq!(bq.tick(1.0), 1.0);
    assert_eq!(bq.tick(0.5), 0.5);
    assert_eq!(bq.tick(-3.0), -3.0);
}

#[test]
fn biquad_normalization() {
    // a[0] = 2 should normalize everything by 2
    let bq = Biquad::new([2.0, 4.0, 6.0], [2.0, 1.0, 0.5]);
    let (b, a) = bq.coefficients();
    assert_near(b[0], 1.0, TOL, "b0");
    assert_near(b[1], 2.0, TOL, "b1");
    assert_near(b[2], 3.0, TOL, "b2");
    assert_near(a[0], 1.0, TOL, "a0");
    assert_near(a[1], 0.5, TOL, "a1");
    assert_near(a[2], 0.25, TOL, "a2");
}

#[test]
fn biquad_reset() {
    let mut bq = Biquad::new([0.1, 0.2, 0.1], [1.0, -0.5, 0.1]);
    bq.tick(1.0);
    bq.tick(0.5);
    bq.reset();
    // After reset, output should match a fresh filter
    let mut bq2 = Biquad::new([0.1, 0.2, 0.1], [1.0, -0.5, 0.1]);
    assert_eq!(bq.tick(1.0), bq2.tick(1.0));
    assert_eq!(bq.tick(0.5), bq2.tick(0.5));
}

#[test]
fn biquad_process_matches_tick() {
    let mut bq1 = Biquad::new([0.1, 0.2, 0.1], [1.0, -0.5, 0.1]);
    let mut bq2 = Biquad::new([0.1, 0.2, 0.1], [1.0, -0.5, 0.1]);

    let input = [1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.0, 0.7];
    let mut output_tick = [0.0; 8];
    let mut output_process = [0.0; 8];

    for (i, &x) in input.iter().enumerate() {
        output_tick[i] = bq1.tick(x);
    }
    bq2.process(&input, &mut output_process);

    for i in 0..8 {
        assert_near(output_tick[i], output_process[i], TOL, "process vs tick");
    }
}

#[test]
fn biquad_process_inplace() {
    let mut bq1 = Biquad::new([0.1, 0.2, 0.1], [1.0, -0.5, 0.1]);
    let mut bq2 = Biquad::new([0.1, 0.2, 0.1], [1.0, -0.5, 0.1]);

    let input = [1.0, 0.5, -0.3, 0.8];
    let mut data = input;
    let mut reference = [0.0; 4];

    bq1.process(&input, &mut reference);
    bq2.process_inplace(&mut data);

    for i in 0..4 {
        assert_near(data[i], reference[i], TOL, "inplace");
    }
}

// ═══════════════════════════════════════════════════════════════════
// BiquadCascade basics
// ═══════════════════════════════════════════════════════════════════

#[test]
fn cascade_process_matches_tick() {
    let mut c1: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    let mut c2: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();

    let input = [1.0, 0.5, -0.3, 0.8, -1.0];
    let mut out_tick = [0.0; 5];
    let mut out_process = [0.0; 5];

    for (i, &x) in input.iter().enumerate() {
        out_tick[i] = c1.tick(x);
    }
    c2.process(&input, &mut out_process);

    for i in 0..5 {
        assert_near(out_tick[i], out_process[i], TOL, "cascade process vs tick");
    }
}

#[test]
fn cascade_order_even() {
    let c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    assert_eq!(c.order(), 4);
}

#[test]
fn cascade_order_odd() {
    let c: BiquadCascade<f64, 3> = butterworth_lowpass(5, 1000.0, 8000.0).unwrap();
    assert_eq!(c.order(), 5);
}

// ═══════════════════════════════════════════════════════════════════
// Butterworth lowpass
// ═══════════════════════════════════════════════════════════════════

#[test]
fn butterworth_lp4_dc_gain() {
    let c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    let dc = freq_response_mag(&c, 0.001, 8000.0);
    assert_near(dc, 1.0, 1e-6, "BW LP4 DC gain");
}

#[test]
fn butterworth_lp4_cutoff_gain() {
    let c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 1000.0, 8000.0);
    // -3 dB point: gain = 1/√2
    assert_near(gain, core::f64::consts::FRAC_1_SQRT_2, 1e-6, "BW LP4 cutoff gain");
}

#[test]
fn butterworth_lp4_stopband() {
    let c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    // Well into the stopband: 3 kHz at order 4 should be heavily attenuated
    let gain = freq_response_mag(&c, 3000.0, 8000.0);
    assert!(gain < 0.01, "BW LP4 stopband: {gain}");
}

#[test]
fn butterworth_lp4_frequency_response() {
    // Verify frequency response matches SciPy: butter(4, 1000, 'low', fs=8000, output='sos')
    // SciPy reference: sosfreqz at various frequencies
    let c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();

    // At DC: gain = 1
    assert_near(freq_response_mag(&c, 0.001, 8000.0), 1.0, 1e-4, "BW LP4 DC");
    // At cutoff: gain = 1/√2 = -3dB
    assert_near(
        freq_response_mag(&c, 1000.0, 8000.0),
        core::f64::consts::FRAC_1_SQRT_2,
        1e-5,
        "BW LP4 fc",
    );
    // Stopband attenuation increases with frequency
    let g2k = freq_response_mag(&c, 2000.0, 8000.0);
    let g3k = freq_response_mag(&c, 3000.0, 8000.0);
    assert!(g2k < 0.25, "BW LP4 at 2kHz: {g2k}");
    assert!(g3k < 0.05, "BW LP4 at 3kHz: {g3k}");
    assert!(g3k < g2k, "monotonic rolloff");
}

#[test]
fn butterworth_lp2_coefficients() {
    // SciPy: butter(2, 1000, 'low', fs=8000, output='sos')
    // section 0: b=[0.0976310729378175, 0.195262145875635, 0.0976310729378175],
    //            a=[1.0, -0.9428090415820631, 0.33333333333333326]
    let c: BiquadCascade<f64, 1> = butterworth_lowpass(2, 1000.0, 8000.0).unwrap();
    let (b, a) = c.sections[0].coefficients();

    assert_near(b[0], 0.0976310729378175, LOOSE_TOL, "BW LP2 b0");
    assert_near(b[1], 0.195262145875635, LOOSE_TOL, "BW LP2 b1");
    assert_near(b[2], 0.0976310729378175, LOOSE_TOL, "BW LP2 b2");
    assert_near(a[1], -0.9428090415820631, LOOSE_TOL, "BW LP2 a1");
    assert_near(a[2], 0.33333333333333326, LOOSE_TOL, "BW LP2 a2");
}

#[test]
fn butterworth_lp1_coefficients() {
    // SciPy: butter(1, 1000, 'low', fs=8000, output='sos')
    // b=[0.292893, 0.292893, 0.0], a=[1.0, -0.414214, 0.0]
    let c: BiquadCascade<f64, 1> = butterworth_lowpass(1, 1000.0, 8000.0).unwrap();
    let (b, a) = c.sections[0].coefficients();

    assert_near(b[0], 0.2928932188134525, LOOSE_TOL, "BW LP1 b0");
    assert_near(b[1], 0.2928932188134525, LOOSE_TOL, "BW LP1 b1");
    assert_near(b[2], 0.0, TOL, "BW LP1 b2");
    assert_near(a[1], -0.4142135623730949, LOOSE_TOL, "BW LP1 a1");
    assert_near(a[2], 0.0, TOL, "BW LP1 a2");
    assert_eq!(c.order(), 1);
}

#[test]
fn butterworth_lp5_dc_gain() {
    let c: BiquadCascade<f64, 3> = butterworth_lowpass(5, 1000.0, 8000.0).unwrap();
    let dc = freq_response_mag(&c, 0.001, 8000.0);
    assert_near(dc, 1.0, 1e-6, "BW LP5 DC gain");
    assert_eq!(c.order(), 5);
}

#[test]
fn butterworth_lp5_cutoff() {
    let c: BiquadCascade<f64, 3> = butterworth_lowpass(5, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 1000.0, 8000.0);
    assert_near(gain, core::f64::consts::FRAC_1_SQRT_2, 1e-5, "BW LP5 cutoff");
}

// ═══════════════════════════════════════════════════════════════════
// Butterworth highpass
// ═══════════════════════════════════════════════════════════════════

#[test]
fn butterworth_hp4_nyquist_gain() {
    let c: BiquadCascade<f64, 2> = butterworth_highpass(4, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 3999.0, 8000.0);
    assert_near(gain, 1.0, 1e-3, "BW HP4 Nyquist gain");
}

#[test]
fn butterworth_hp4_dc_rejection() {
    let c: BiquadCascade<f64, 2> = butterworth_highpass(4, 1000.0, 8000.0).unwrap();
    let dc = freq_response_mag(&c, 0.1, 8000.0);
    assert!(dc < 1e-6, "BW HP4 DC rejection: {dc}");
}

#[test]
fn butterworth_hp4_cutoff_gain() {
    let c: BiquadCascade<f64, 2> = butterworth_highpass(4, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 1000.0, 8000.0);
    assert_near(gain, core::f64::consts::FRAC_1_SQRT_2, 1e-5, "BW HP4 cutoff");
}

#[test]
fn butterworth_hp3_odd() {
    let c: BiquadCascade<f64, 2> = butterworth_highpass(3, 1000.0, 8000.0).unwrap();
    assert_eq!(c.order(), 3);
    let cutoff_gain = freq_response_mag(&c, 1000.0, 8000.0);
    assert_near(cutoff_gain, core::f64::consts::FRAC_1_SQRT_2, 1e-4, "BW HP3 cutoff");
}

#[test]
fn butterworth_hp1() {
    let c: BiquadCascade<f64, 1> = butterworth_highpass(1, 1000.0, 8000.0).unwrap();
    assert_eq!(c.order(), 1);
    let cutoff_gain = freq_response_mag(&c, 1000.0, 8000.0);
    assert_near(cutoff_gain, core::f64::consts::FRAC_1_SQRT_2, 1e-4, "BW HP1 cutoff");
}

// ═══════════════════════════════════════════════════════════════════
// Chebyshev Type I lowpass
// ═══════════════════════════════════════════════════════════════════

#[test]
fn chebyshev1_lp4_cutoff_gain() {
    // At the cutoff, gain should be -ripple_db
    let c: BiquadCascade<f64, 2> = chebyshev1_lowpass(4, 1.0, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 1000.0, 8000.0);
    let gain_db = 20.0 * gain.log10();
    assert_near(gain_db, -1.0, 0.05, "Cheb1 LP4 cutoff gain dB");
}

#[test]
fn chebyshev1_lp4_dc_gain() {
    // Even-order Chebyshev: DC gain = 1/√(1+ε²) = 10^(-Rp/20)
    let c: BiquadCascade<f64, 2> = chebyshev1_lowpass(4, 1.0, 1000.0, 8000.0).unwrap();
    let dc = freq_response_mag(&c, 0.001, 8000.0);
    let expected = 10.0_f64.powf(-1.0 / 20.0); // 0.8913...
    assert_near(dc, expected, 0.01, "Cheb1 LP4 DC gain");
}

#[test]
fn chebyshev1_lp3_dc_gain() {
    // Odd-order Chebyshev: DC gain = 1 (unity)
    let c: BiquadCascade<f64, 2> = chebyshev1_lowpass(3, 0.5, 500.0, 4000.0).unwrap();
    let dc = freq_response_mag(&c, 0.001, 4000.0);
    assert_near(dc, 1.0, 0.01, "Cheb1 LP3 DC gain");
}

#[test]
fn chebyshev1_lp3_cutoff_gain() {
    let c: BiquadCascade<f64, 2> = chebyshev1_lowpass(3, 0.5, 500.0, 4000.0).unwrap();
    let gain = freq_response_mag(&c, 500.0, 4000.0);
    let gain_db = 20.0 * gain.log10();
    assert_near(gain_db, -0.5, 0.05, "Cheb1 LP3 cutoff gain dB");
}

#[test]
fn chebyshev1_lp4_stopband() {
    let c: BiquadCascade<f64, 2> = chebyshev1_lowpass(4, 1.0, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 3000.0, 8000.0);
    assert!(gain < 0.01, "Cheb1 LP4 stopband: {gain}");
}

#[test]
fn chebyshev1_lp_steeper_than_butterworth() {
    // Same order, Chebyshev should have steeper rolloff in the stopband
    let bw: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    let ch: BiquadCascade<f64, 2> = chebyshev1_lowpass(4, 1.0, 1000.0, 8000.0).unwrap();
    let bw_stop = freq_response_mag(&bw, 2500.0, 8000.0);
    let ch_stop = freq_response_mag(&ch, 2500.0, 8000.0);
    assert!(
        ch_stop < bw_stop,
        "Chebyshev stopband ({ch_stop:.6}) should be less than Butterworth ({bw_stop:.6})"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Chebyshev Type I highpass
// ═══════════════════════════════════════════════════════════════════

#[test]
fn chebyshev1_hp4_nyquist_gain() {
    // Even-order HP: Nyquist gain = 1/√(1+ε²)
    let c: BiquadCascade<f64, 2> = chebyshev1_highpass(4, 1.0, 1000.0, 8000.0).unwrap();
    let gain = freq_response_mag(&c, 3999.0, 8000.0);
    let expected = 10.0_f64.powf(-1.0 / 20.0);
    assert_near(gain, expected, 0.01, "Cheb1 HP4 Nyquist gain");
}

#[test]
fn chebyshev1_hp4_dc_rejection() {
    let c: BiquadCascade<f64, 2> = chebyshev1_highpass(4, 1.0, 1000.0, 8000.0).unwrap();
    let dc = freq_response_mag(&c, 0.1, 8000.0);
    assert!(dc < 1e-4, "Cheb1 HP4 DC rejection: {dc}");
}

#[test]
fn chebyshev1_hp3_cutoff_gain() {
    let c: BiquadCascade<f64, 2> = chebyshev1_highpass(3, 0.5, 500.0, 4000.0).unwrap();
    let gain = freq_response_mag(&c, 500.0, 4000.0);
    let gain_db = 20.0 * gain.log10();
    assert_near(gain_db, -0.5, 0.1, "Cheb1 HP3 cutoff gain dB");
}

// ═══════════════════════════════════════════════════════════════════
// Error cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn error_zero_order() {
    let r: Result<BiquadCascade<f64, 1>, _> = butterworth_lowpass(0, 1000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidOrder);
}

#[test]
fn error_wrong_n() {
    // Order 4 needs N=2, not N=3
    let r: Result<BiquadCascade<f64, 3>, _> = butterworth_lowpass(4, 1000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidOrder);
}

#[test]
fn error_wrong_n_odd() {
    // Order 5 needs N=3, not N=2
    let r: Result<BiquadCascade<f64, 2>, _> = butterworth_lowpass(5, 1000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidOrder);
}

#[test]
fn error_cutoff_at_nyquist() {
    let r: Result<BiquadCascade<f64, 1>, _> = butterworth_lowpass(2, 4000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidFrequency);
}

#[test]
fn error_cutoff_above_nyquist() {
    let r: Result<BiquadCascade<f64, 1>, _> = butterworth_lowpass(2, 5000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidFrequency);
}

#[test]
fn error_cutoff_zero() {
    let r: Result<BiquadCascade<f64, 1>, _> = butterworth_lowpass(2, 0.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidFrequency);
}

#[test]
fn error_cutoff_negative() {
    let r: Result<BiquadCascade<f64, 1>, _> = butterworth_lowpass(2, -100.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidFrequency);
}

#[test]
fn error_invalid_ripple_zero() {
    let r: Result<BiquadCascade<f64, 1>, _> = chebyshev1_lowpass(2, 0.0, 1000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidRipple);
}

#[test]
fn error_invalid_ripple_negative() {
    let r: Result<BiquadCascade<f64, 1>, _> = chebyshev1_lowpass(2, -1.0, 1000.0, 8000.0);
    assert_eq!(r.unwrap_err(), ControlError::InvalidRipple);
}

// ═══════════════════════════════════════════════════════════════════
// f32 support
// ═══════════════════════════════════════════════════════════════════

#[test]
fn butterworth_lp_f32() {
    let mut c: BiquadCascade<f32, 2> = butterworth_lowpass(4, 1000.0_f32, 8000.0_f32).unwrap();
    let y = c.tick(1.0_f32);
    assert!(y.is_finite(), "f32 output is finite");

    // Check DC gain (use many samples of DC input)
    c.reset();
    let mut out = 0.0_f32;
    for _ in 0..10000 {
        out = c.tick(1.0);
    }
    assert!((out - 1.0).abs() < 0.01, "f32 DC gain: {out}");
}

#[test]
fn chebyshev1_lp_f32() {
    let mut c: BiquadCascade<f32, 1> =
        chebyshev1_lowpass(2, 1.0_f32, 1000.0_f32, 8000.0_f32).unwrap();
    let y = c.tick(1.0_f32);
    assert!(y.is_finite(), "f32 Cheb1 output is finite");
}

// ═══════════════════════════════════════════════════════════════════
// Impulse response convergence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn butterworth_lp_impulse_decays() {
    let mut c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    // Feed a single impulse
    let first = c.tick(1.0);
    assert!(first > 0.0);
    // After many zeros, output should decay toward zero
    let mut last = first;
    for _ in 0..1000 {
        last = c.tick(0.0);
    }
    assert!(last.abs() < 1e-10, "impulse response decayed: {last}");
}

#[test]
fn butterworth_hp_impulse_decays() {
    let mut c: BiquadCascade<f64, 2> = butterworth_highpass(4, 1000.0, 8000.0).unwrap();
    let first = c.tick(1.0);
    assert!(first.abs() > 0.0);
    let mut last = first;
    for _ in 0..1000 {
        last = c.tick(0.0);
    }
    assert!(last.abs() < 1e-10, "HP impulse decayed: {last}");
}

// ═══════════════════════════════════════════════════════════════════
// Monotonicity / rolloff
// ═══════════════════════════════════════════════════════════════════

#[test]
fn butterworth_lp_monotonic_stopband() {
    // Butterworth is maximally flat → monotonically decreasing in the stopband
    let c: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
    let mut prev = freq_response_mag(&c, 1000.0, 8000.0);
    for f in (1500..3500).step_by(100) {
        let mag = freq_response_mag(&c, f as f64, 8000.0);
        assert!(
            mag <= prev + 1e-12,
            "BW LP4 not monotonic at {f}: {mag} > {prev}"
        );
        prev = mag;
    }
}

#[test]
fn butterworth_hp_monotonic_stopband() {
    let c: BiquadCascade<f64, 2> = butterworth_highpass(4, 1000.0, 8000.0).unwrap();
    // Below cutoff, gain should increase as frequency increases
    let mut prev = freq_response_mag(&c, 10.0, 8000.0);
    for f in (100..1000).step_by(100) {
        let mag = freq_response_mag(&c, f as f64, 8000.0);
        assert!(
            mag >= prev - 1e-12,
            "BW HP4 not monotonic at {f}: {mag} < {prev}"
        );
        prev = mag;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Higher orders
// ═══════════════════════════════════════════════════════════════════

#[test]
fn butterworth_lp8() {
    let c: BiquadCascade<f64, 4> = butterworth_lowpass(8, 1000.0, 8000.0).unwrap();
    assert_eq!(c.order(), 8);
    let dc = freq_response_mag(&c, 0.001, 8000.0);
    assert_near(dc, 1.0, 1e-5, "BW LP8 DC");
    let cutoff = freq_response_mag(&c, 1000.0, 8000.0);
    assert_near(cutoff, core::f64::consts::FRAC_1_SQRT_2, 1e-4, "BW LP8 cutoff");
}

#[test]
fn chebyshev1_lp6() {
    let c: BiquadCascade<f64, 3> = chebyshev1_lowpass(6, 0.5, 1000.0, 8000.0).unwrap();
    assert_eq!(c.order(), 6);
    let cutoff = freq_response_mag(&c, 1000.0, 8000.0);
    let cutoff_db = 20.0 * cutoff.log10();
    assert_near(cutoff_db, -0.5, 0.1, "Cheb1 LP6 cutoff");
}
