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

// ═══════════════════════════════════════════════════════════════
// Lead/Lag compensator tests
// ═══════════════════════════════════════════════════════════════

/// Evaluate a single biquad's frequency response magnitude at a given frequency.
fn biquad_freq_response_mag(bq: &Biquad<f64>, freq: f64, fs: f64) -> f64 {
    let omega = 2.0 * core::f64::consts::PI * freq / fs;
    let (s, c) = omega.sin_cos();
    let (b, a) = bq.coefficients();
    let num_re = b[0] + b[1] * c + b[2] * (2.0 * c * c - 1.0);
    let num_im = -b[1] * s - b[2] * 2.0 * s * c;
    let den_re = a[0] + a[1] * c + a[2] * (2.0 * c * c - 1.0);
    let den_im = -a[1] * s - a[2] * 2.0 * s * c;
    let num_mag = (num_re * num_re + num_im * num_im).sqrt();
    let den_mag = (den_re * den_re + den_im * den_im).sqrt();
    num_mag / den_mag
}

/// Evaluate a single biquad's frequency response phase at a given frequency.
fn biquad_freq_response_phase(bq: &Biquad<f64>, freq: f64, fs: f64) -> f64 {
    let omega = 2.0 * core::f64::consts::PI * freq / fs;
    let (s, c) = omega.sin_cos();
    let (b, a) = bq.coefficients();
    let num_re = b[0] + b[1] * c + b[2] * (2.0 * c * c - 1.0);
    let num_im = -b[1] * s - b[2] * 2.0 * s * c;
    let den_re = a[0] + a[1] * c + a[2] * (2.0 * c * c - 1.0);
    let den_im = -a[1] * s - a[2] * 2.0 * s * c;
    num_im.atan2(num_re) - den_im.atan2(den_re)
}

#[test]
fn lead_compensator_basic() {
    use core::f64::consts::FRAC_PI_4;
    let comp = lead_compensator(FRAC_PI_4, 10.0, 1.0, 1000.0).unwrap();
    // DC gain should be close to 1/(alpha) * alpha = gain = 1
    // Actually for lead: DC = gain (since both zero and pole contribute)
    let dc = biquad_freq_response_mag(&comp, 0.001, 1000.0);
    assert_near(dc, 1.0, 0.01, "lead DC gain ≈ 1");
}

#[test]
fn lead_compensator_adds_phase() {
    use core::f64::consts::FRAC_PI_4;
    let comp = lead_compensator(FRAC_PI_4, 50.0, 1.0, 1000.0).unwrap();
    // Phase at the center frequency should be close to the requested phase lead
    let phase = biquad_freq_response_phase(&comp, 50.0, 1000.0);
    // Bilinear transform warps frequency, so phase won't be exact, but should be positive
    assert!(phase > 0.3, "lead adds positive phase at center freq: {}", phase);
}

#[test]
fn lead_compensator_errors() {
    // Phase out of range
    assert!(lead_compensator(0.0_f64, 10.0, 1.0, 1000.0).is_err());
    assert!(lead_compensator(core::f64::consts::FRAC_PI_2, 10.0, 1.0, 1000.0).is_err());
    assert!(lead_compensator(-0.1_f64, 10.0, 1.0, 1000.0).is_err());
    // Frequency above Nyquist
    assert!(lead_compensator(0.5_f64, 600.0, 1.0, 1000.0).is_err());
    // Zero/negative frequencies
    assert!(lead_compensator(0.5_f64, 0.0, 1.0, 1000.0).is_err());
    assert!(lead_compensator(0.5_f64, 10.0, 1.0, 0.0).is_err());
}

#[test]
fn lag_compensator_dc_boost() {
    let comp = lag_compensator(10.0, 1.0, 1000.0).unwrap();
    // DC gain should be close to dc_boost = 10
    let dc = biquad_freq_response_mag(&comp, 0.001, 1000.0);
    let dc_db = 20.0 * dc.log10();
    assert_near(dc_db, 20.0, 0.5, "lag DC gain ≈ 20 dB");
}

#[test]
fn lag_compensator_hf_unity() {
    let comp = lag_compensator(10.0, 1.0, 1000.0).unwrap();
    // HF gain should be close to 1 (0 dB)
    let hf = biquad_freq_response_mag(&comp, 100.0, 1000.0);
    let hf_db = 20.0 * hf.log10();
    assert!(hf_db.abs() < 1.0, "lag HF gain ≈ 0 dB: {} dB", hf_db);
}

#[test]
fn lag_compensator_errors() {
    // dc_boost must be > 1
    assert!(lag_compensator(1.0_f64, 1.0, 1000.0).is_err());
    assert!(lag_compensator(0.5_f64, 1.0, 1000.0).is_err());
    // Corner above Nyquist
    assert!(lag_compensator(5.0_f64, 600.0, 1000.0).is_err());
}

#[test]
fn lead_compensator_f32() {
    let comp = lead_compensator(0.5_f32, 10.0, 1.0, 1000.0);
    assert!(comp.is_ok(), "lead_compensator works with f32");
}

#[test]
fn lag_compensator_f32() {
    let comp = lag_compensator(5.0_f32, 1.0, 1000.0);
    assert!(comp.is_ok(), "lag_compensator works with f32");
}

// ═══════════════════════════════════════════════════════════════
// PID tuning tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn fopdt_model_construction() {
    let m = FopdtModel::new(2.0_f64, 1.0, 0.3).unwrap();
    assert_eq!(m.gain(), 2.0);
    assert_eq!(m.tau(), 1.0);
    assert_eq!(m.delay(), 0.3);
}

#[test]
fn fopdt_model_errors() {
    // Zero gain
    assert!(FopdtModel::new(0.0_f64, 1.0, 0.3).is_err());
    // Negative tau
    assert!(FopdtModel::new(1.0_f64, -1.0, 0.3).is_err());
    // Negative delay
    assert!(FopdtModel::new(1.0_f64, 1.0, -0.1).is_err());
    // NaN/Inf
    assert!(FopdtModel::new(f64::NAN, 1.0, 0.3).is_err());
    assert!(FopdtModel::new(1.0, f64::INFINITY, 0.3).is_err());
}

#[test]
fn fopdt_zero_delay_allowed() {
    // Zero delay is valid (some tuning rules will panic though)
    let m = FopdtModel::new(1.0_f64, 1.0, 0.0);
    assert!(m.is_ok());
}

#[test]
fn ziegler_nichols_known_values() {
    // K=1, τ=1, L=0.2
    // Kp = 1.2·τ/(K·L) = 1.2·1/(1·0.2) = 6.0
    // Ti = 2L = 0.4
    // Td = L/2 = 0.1
    // Ki = Kp/Ti = 15.0
    // Kd = Kp·Td = 0.6
    let m = FopdtModel::new(1.0_f64, 1.0, 0.2).unwrap();
    let g = m.ziegler_nichols();
    assert_near(g.kp, 6.0, TOL, "ZN kp");
    assert_near(g.ki, 15.0, TOL, "ZN ki");
    assert_near(g.kd, 0.6, TOL, "ZN kd");
}

#[test]
#[should_panic]
fn ziegler_nichols_zero_delay_panics() {
    let m = FopdtModel::new(1.0_f64, 1.0, 0.0).unwrap();
    m.ziegler_nichols();
}

#[test]
fn cohen_coon_positive_gains() {
    let m = FopdtModel::new(1.0_f64, 1.0, 0.5).unwrap();
    let g = m.cohen_coon();
    assert!(g.kp > 0.0, "CC kp positive");
    assert!(g.ki > 0.0, "CC ki positive");
    assert!(g.kd > 0.0, "CC kd positive");
}

#[test]
fn simc_known_values() {
    // K=1, τ=2, L=0.3, tau_c=0.3
    // Kp = τ/(K·(tau_c+L)) = 2/(1·0.6) = 10/3
    // Ti = min(τ, 4·(tau_c+L)) = min(2, 2.4) = 2
    // Td = L/2 = 0.15
    // Ki = Kp/Ti = 10/6 = 5/3
    // Kd = Kp·Td = (10/3)·0.15 = 0.5
    let m = FopdtModel::new(1.0_f64, 2.0, 0.3).unwrap();
    let g = m.simc(0.3);
    assert_near(g.kp, 10.0 / 3.0, TOL, "SIMC kp");
    assert_near(g.ki, 5.0 / 3.0, TOL, "SIMC ki");
    assert_near(g.kd, 0.5, TOL, "SIMC kd");
}

#[test]
#[should_panic]
fn simc_zero_tau_c_panics() {
    let m = FopdtModel::new(1.0_f64, 1.0, 0.2).unwrap();
    m.simc(0.0);
}

#[test]
fn ziegler_nichols_ultimate_known_values() {
    // Ku=10, Tu=0.5
    // Kp = 0.6·10 = 6
    // Ti = 0.5/2 = 0.25
    // Td = 0.5/8 = 0.0625
    // Ki = 6/0.25 = 24
    // Kd = 6·0.0625 = 0.375
    let g = ziegler_nichols_ultimate(10.0_f64, 0.5).unwrap();
    assert_near(g.kp, 6.0, TOL, "ZN-ult kp");
    assert_near(g.ki, 24.0, TOL, "ZN-ult ki");
    assert_near(g.kd, 0.375, TOL, "ZN-ult kd");
}

#[test]
fn ziegler_nichols_ultimate_errors() {
    assert!(ziegler_nichols_ultimate(0.0_f64, 0.5).is_err());
    assert!(ziegler_nichols_ultimate(-1.0_f64, 0.5).is_err());
    assert!(ziegler_nichols_ultimate(10.0_f64, 0.0).is_err());
    assert!(ziegler_nichols_ultimate(10.0_f64, -1.0).is_err());
}

#[test]
fn pid_tuning_f32() {
    let m = FopdtModel::new(1.0_f32, 1.0, 0.2).unwrap();
    let g = m.ziegler_nichols();
    assert!((g.kp - 6.0_f32).abs() < 1e-5, "ZN f32 kp");

    let g2 = ziegler_nichols_ultimate(10.0_f32, 0.5).unwrap();
    assert!((g2.kp - 6.0_f32).abs() < 1e-5, "ZN-ult f32 kp");
}
