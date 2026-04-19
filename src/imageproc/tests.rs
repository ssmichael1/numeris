use super::*;
use crate::DynMatrix;
use crate::Matrix;

// ── kernel generators ──────────────────────────────────────────────────

#[test]
fn gaussian_kernel_sums_to_one() {
    let k = gaussian_kernel_1d::<f64>(1.5, 3.0).unwrap();
    let sum: f64 = k.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
    // Length = 2*ceil(3*1.5)+1 = 11 ... wait ceil(4.5) = 5, so 2*5+1 = 11
    assert_eq!(k.len(), 11);
}

#[test]
fn gaussian_kernel_is_symmetric() {
    let k = gaussian_kernel_1d::<f64>(2.0, 4.0).unwrap();
    let n = k.len();
    for i in 0..n / 2 {
        assert!((k[i] - k[n - 1 - i]).abs() < 1e-15);
    }
}

#[test]
fn gaussian_kernel_rejects_bad_sigma() {
    assert!(gaussian_kernel_1d::<f64>(0.0, 3.0).is_err());
    assert!(gaussian_kernel_1d::<f64>(-1.0, 3.0).is_err());
    assert!(gaussian_kernel_1d::<f64>(f64::NAN, 3.0).is_err());
    assert!(gaussian_kernel_1d::<f64>(1.0, 0.0).is_err());
}

#[test]
fn box_kernel_values() {
    let k = box_kernel_1d::<f64>(5).unwrap();
    assert_eq!(k.len(), 5);
    for v in &k {
        assert!((*v - 0.2).abs() < 1e-15);
    }
}

#[test]
fn box_kernel_rejects_even_or_zero() {
    assert!(box_kernel_1d::<f64>(0).is_err());
    assert!(box_kernel_1d::<f64>(4).is_err());
}

// ── border handling ────────────────────────────────────────────────────

#[test]
fn fetch_border_zero() {
    let s = [1.0_f64, 2.0, 3.0];
    assert_eq!(fetch_border(&s, -1, BorderMode::Zero), 0.0);
    assert_eq!(fetch_border(&s, 3, BorderMode::Zero), 0.0);
    assert_eq!(fetch_border(&s, 1, BorderMode::Zero), 2.0);
}

#[test]
fn fetch_border_replicate() {
    let s = [1.0_f64, 2.0, 3.0];
    assert_eq!(fetch_border(&s, -5, BorderMode::Replicate), 1.0);
    assert_eq!(fetch_border(&s, 10, BorderMode::Replicate), 3.0);
}

#[test]
fn fetch_border_reflect() {
    // reflect without edge duplication: abc | cba | abc  → for [a,b,c]:
    // indices -1,-2,-3,-4  → 1,2,1,0  (period 4)
    // indices  3, 4, 5     → 1,0,1
    let s = [10.0_f64, 20.0, 30.0];
    assert_eq!(fetch_border(&s, -1, BorderMode::Reflect), 20.0);
    assert_eq!(fetch_border(&s, -2, BorderMode::Reflect), 30.0);
    assert_eq!(fetch_border(&s, 3, BorderMode::Reflect), 20.0);
    assert_eq!(fetch_border(&s, 4, BorderMode::Reflect), 10.0);
}

// ── convolution ────────────────────────────────────────────────────────

fn ramp(nrows: usize, ncols: usize) -> DynMatrix<f64> {
    DynMatrix::from_fn(nrows, ncols, |i, j| (i * ncols + j) as f64)
}

#[test]
fn identity_kernel_leaves_image_unchanged() {
    let img = ramp(5, 6);
    let id = Matrix::<f64, 3, 3>::new([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let out = convolve2d(&img, &id, BorderMode::Zero);
    for i in 0..5 {
        for j in 0..6 {
            assert!((out[(i, j)] - img[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn constant_image_convolves_to_kernel_sum() {
    let img = DynMatrix::<f64>::fill(7, 7, 3.0);
    // 3x3 kernel with entries summing to 2.0
    let k = Matrix::<f64, 3, 3>::new([
        [0.1, 0.2, 0.3],
        [0.0, 0.5, 0.0],
        [0.2, 0.5, 0.2],
    ]);
    let out = convolve2d(&img, &k, BorderMode::Replicate);
    // All output pixels should equal 3.0 * sum_of_kernel.
    let ksum: f64 = (0..3)
        .flat_map(|i| (0..3).map(move |j| k[(i, j)]))
        .sum();
    for i in 0..7 {
        for j in 0..7 {
            assert!(
                (out[(i, j)] - 3.0 * ksum).abs() < 1e-12,
                "mismatch at ({i},{j}): {} vs {}",
                out[(i, j)],
                3.0 * ksum
            );
        }
    }
}

#[test]
fn separable_matches_direct() {
    // Outer product of [1,2,3] (y) and [1,1,1] (x) should match the 3x3
    // equivalent dense kernel.
    let ky = [1.0_f64, 2.0, 3.0];
    let kx = [1.0_f64, 1.0, 1.0];
    let dense = Matrix::<f64, 3, 3>::new([
        [ky[0] * kx[0], ky[0] * kx[1], ky[0] * kx[2]],
        [ky[1] * kx[0], ky[1] * kx[1], ky[1] * kx[2]],
        [ky[2] * kx[0], ky[2] * kx[1], ky[2] * kx[2]],
    ]);
    let img = ramp(8, 9);
    let a = convolve2d_separable(&img, &ky, &kx, BorderMode::Replicate);
    let b = convolve2d(&img, &dense, BorderMode::Replicate);
    for i in 0..8 {
        for j in 0..9 {
            assert!(
                (a[(i, j)] - b[(i, j)]).abs() < 1e-10,
                "mismatch at ({i},{j}): sep={} dense={}",
                a[(i, j)],
                b[(i, j)]
            );
        }
    }
}

#[test]
fn delta_impulse_gives_kernel_response() {
    // A one-hot image centered at (3, 3) convolved with a kernel should
    // produce the kernel itself (correlation flips the location but since
    // it's a delta, the footprint is the kernel values around (3,3)).
    let mut img = DynMatrix::<f64>::zeros(7, 7);
    img[(3, 3)] = 1.0;
    let k = Matrix::<f64, 3, 3>::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);
    let out = convolve2d(&img, &k, BorderMode::Zero);
    // out[i, j] = sum over (ki, kj) of k[ki, kj] * img[i+ki-1, j+kj-1]
    //           = k[1-(i-3), 1-(j-3)] when (i+ki-1, j+kj-1) = (3,3)
    //           = k[4-i, 4-j] for |i-3|<=1, |j-3|<=1
    // Actually for correlation with a delta at (3,3):
    // out[i,j] = k[3-i+1, 3-j+1] = k[4-i, 4-j] — only hit when 4-i, 4-j in [0,2].
    // i in [2,4], j in [2,4], map:
    //  (i=2,j=2) → k[2,2]=9
    //  (i=3,j=3) → k[1,1]=5
    //  (i=4,j=4) → k[0,0]=1
    assert!((out[(2, 2)] - 9.0).abs() < 1e-12);
    assert!((out[(3, 3)] - 5.0).abs() < 1e-12);
    assert!((out[(4, 4)] - 1.0).abs() < 1e-12);
    assert!((out[(2, 3)] - 8.0).abs() < 1e-12);
    assert!((out[(3, 2)] - 6.0).abs() < 1e-12);
}

// ── Gaussian blur ──────────────────────────────────────────────────────

#[test]
fn gaussian_blur_preserves_constant() {
    let img = DynMatrix::<f64>::fill(16, 20, 5.0);
    let out = gaussian_blur(&img, 1.5, BorderMode::Replicate);
    for i in 0..16 {
        for j in 0..20 {
            assert!((out[(i, j)] - 5.0).abs() < 1e-10);
        }
    }
}

#[test]
fn gaussian_blur_smooths_impulse() {
    let mut img = DynMatrix::<f64>::zeros(11, 11);
    img[(5, 5)] = 1.0;
    let out = gaussian_blur(&img, 1.0, BorderMode::Zero);
    // Energy is conserved by a normalized Gaussian (up to truncation).
    let mut total = 0.0_f64;
    for i in 0..11 {
        for j in 0..11 {
            total += out[(i, j)];
        }
    }
    assert!((total - 1.0).abs() < 1e-3);
    // Centre should still be the peak.
    for i in 0..11 {
        for j in 0..11 {
            if (i, j) != (5, 5) {
                assert!(out[(i, j)] <= out[(5, 5)] + 1e-12);
            }
        }
    }
}

// ── Sobel ──────────────────────────────────────────────────────────────

#[test]
fn sobel_detects_vertical_step() {
    // Step from 0 to 1 between column 4 and column 5.
    let mut img = DynMatrix::<f64>::zeros(9, 9);
    for i in 0..9 {
        for j in 5..9 {
            img[(i, j)] = 1.0;
        }
    }
    let (gx, gy) = sobel_gradients(&img, BorderMode::Replicate);
    // Sobel_x should peak along the step, Sobel_y should be ~0 everywhere.
    for i in 0..9 {
        assert!(gx[(i, 4)] > 0.0 || gx[(i, 5)] > 0.0);
    }
    for i in 1..8 {
        for j in 1..8 {
            assert!(gy[(i, j)].abs() < 1e-12);
        }
    }
}

// ── Box blur ───────────────────────────────────────────────────────────

#[test]
fn box_blur_on_constant() {
    let img = DynMatrix::<f64>::fill(10, 10, 7.0);
    let out = box_blur(&img, 2, BorderMode::Replicate);
    for i in 0..10 {
        for j in 0..10 {
            assert!((out[(i, j)] - 7.0).abs() < 1e-12);
        }
    }
}

// ── Laplacian / Scharr / LoG / unsharp ─────────────────────────────────

#[test]
fn laplacian_on_constant_is_zero() {
    let img = DynMatrix::<f64>::fill(9, 9, 4.2);
    let out = laplacian(&img, BorderMode::Replicate);
    for i in 1..8 {
        for j in 1..8 {
            assert!(out[(i, j)].abs() < 1e-12, "non-zero at ({i},{j})");
        }
    }
}

#[test]
fn laplacian_detects_impulse() {
    let mut img = DynMatrix::<f64>::zeros(7, 7);
    img[(3, 3)] = 1.0;
    let out = laplacian(&img, BorderMode::Zero);
    // Peak of the 4-neighbour Laplacian on a delta is -4 at the centre.
    assert!((out[(3, 3)] - (-4.0)).abs() < 1e-12);
    // The 4 orthogonal neighbours pick up +1 each.
    assert!((out[(2, 3)] - 1.0).abs() < 1e-12);
    assert!((out[(4, 3)] - 1.0).abs() < 1e-12);
    assert!((out[(3, 2)] - 1.0).abs() < 1e-12);
    assert!((out[(3, 4)] - 1.0).abs() < 1e-12);
}

#[test]
fn scharr_detects_horizontal_step() {
    // Horizontal step from 0 to 1 between row 4 and row 5.
    let mut img = DynMatrix::<f64>::zeros(9, 9);
    for i in 5..9 {
        for j in 0..9 {
            img[(i, j)] = 1.0;
        }
    }
    let (gx, gy) = scharr_gradients(&img, BorderMode::Replicate);
    // Scharr Gy should be strongly positive along the transition.
    for j in 1..8 {
        assert!(gy[(4, j)] > 0.0 || gy[(5, j)] > 0.0, "no response at col {j}");
    }
    // Scharr Gx should be ~0 in the interior (no horizontal variation).
    for i in 1..8 {
        for j in 1..8 {
            assert!(gx[(i, j)].abs() < 1e-12);
        }
    }
}

#[test]
fn gradient_magnitude_matches_sqrt_sum_squares() {
    let gx = DynMatrix::from_rows(2, 3, &[3.0_f64, 0.0, 5.0, 0.0, -4.0, 12.0]);
    let gy = DynMatrix::from_rows(2, 3, &[4.0_f64, 0.0, 12.0, 1.0, 3.0, 5.0]);
    let m = gradient_magnitude(&gx, &gy);
    assert!((m[(0, 0)] - 5.0).abs() < 1e-12);
    assert!((m[(0, 1)] - 0.0).abs() < 1e-12);
    assert!((m[(0, 2)] - 13.0).abs() < 1e-12);
    assert!((m[(1, 0)] - 1.0).abs() < 1e-12);
    assert!((m[(1, 1)] - 5.0).abs() < 1e-12);
    assert!((m[(1, 2)] - 13.0).abs() < 1e-12);
}

#[test]
fn unsharp_mask_on_constant_is_identity() {
    let img = DynMatrix::<f64>::fill(11, 11, 2.5);
    let out = unsharp_mask(&img, 1.0, 1.5, BorderMode::Replicate);
    for i in 0..11 {
        for j in 0..11 {
            assert!((out[(i, j)] - 2.5).abs() < 1e-10);
        }
    }
}

#[test]
fn laplacian_of_gaussian_on_constant_is_zero() {
    let img = DynMatrix::<f64>::fill(13, 13, 6.0);
    let out = laplacian_of_gaussian(&img, 1.2, BorderMode::Replicate);
    for i in 3..10 {
        for j in 3..10 {
            assert!(out[(i, j)].abs() < 1e-10);
        }
    }
}

// ── Integral image ─────────────────────────────────────────────────────

#[test]
fn integral_image_shape_and_zero_padding() {
    let img = DynMatrix::<f64>::fill(3, 4, 2.0);
    let sat = integral_image(&img);
    assert_eq!(sat.nrows(), 4);
    assert_eq!(sat.ncols(), 5);
    // First row and column are zero.
    for j in 0..5 {
        assert_eq!(sat[(0, j)], 0.0);
    }
    for i in 0..4 {
        assert_eq!(sat[(i, 0)], 0.0);
    }
    // Full sum: 3 * 4 * 2 = 24
    assert!((sat[(3, 4)] - 24.0).abs() < 1e-12);
}

#[test]
fn integral_rect_sum_matches_brute_force() {
    // Ramp image.
    let img = DynMatrix::from_fn(5, 6, |i, j| (i * 6 + j + 1) as f64);
    let sat = integral_image(&img);
    // Verify every sub-rectangle matches the naïve sum.
    for r0 in 0..=5 {
        for r1 in r0..=5 {
            for c0 in 0..=6 {
                for c1 in c0..=6 {
                    let mut expected = 0.0_f64;
                    for r in r0..r1 {
                        for c in c0..c1 {
                            expected += img[(r, c)];
                        }
                    }
                    let got = integral_rect_sum(&sat, r0, c0, r1, c1);
                    assert!(
                        (got - expected).abs() < 1e-10,
                        "mismatch at [{r0}..{r1}, {c0}..{c1}]: got {got}, want {expected}",
                    );
                }
            }
        }
    }
}

// ── Resize ─────────────────────────────────────────────────────────────

#[test]
fn resize_preserves_constant() {
    let img = DynMatrix::<f64>::fill(7, 11, 3.25);
    let up = resize_bilinear(&img, 20, 13);
    assert_eq!(up.nrows(), 20);
    assert_eq!(up.ncols(), 13);
    for i in 0..20 {
        for j in 0..13 {
            assert!((up[(i, j)] - 3.25).abs() < 1e-12);
        }
    }
}

#[test]
fn resize_identity_dimensions() {
    // Same dims → output should equal input (bilinear sampling at pixel centres
    // with unit scale is the identity).
    let img = DynMatrix::from_fn(4, 4, |i, j| (i * 10 + j) as f64);
    let out = resize_bilinear(&img, 4, 4);
    for i in 0..4 {
        for j in 0..4 {
            assert!((out[(i, j)] - img[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn resize_upscale_linear_ramp() {
    // A linear horizontal ramp stays linear after bilinear resize.
    let img = DynMatrix::from_fn(2, 4, |_, j| j as f64);
    let up = resize_bilinear(&img, 2, 8);
    // At scale 0.5, input x for output j is (j + 0.5) * 0.5 - 0.5 = (j - 0.5) / 2.
    // Monotonically increasing along columns.
    for i in 0..2 {
        for j in 0..7 {
            assert!(up[(i, j)] <= up[(i, j + 1)] + 1e-12);
        }
    }
}

#[test]
fn resize_empty_output() {
    let img = DynMatrix::<f64>::fill(5, 5, 1.0);
    let out = resize_bilinear(&img, 0, 5);
    assert_eq!(out.nrows(), 0);
    assert_eq!(out.ncols(), 5);
}

// ── Rank / percentile / median ─────────────────────────────────────────

#[test]
fn median_removes_salt_and_pepper() {
    // Uniform 5.0 image with a single salt pixel — median should restore it.
    let mut img = DynMatrix::<f64>::fill(9, 9, 5.0);
    img[(4, 4)] = 999.0;
    let out = median_filter(&img, 1, BorderMode::Replicate);
    for i in 0..9 {
        for j in 0..9 {
            assert!((out[(i, j)] - 5.0).abs() < 1e-12, "spike leaked to ({i},{j})");
        }
    }
}

#[test]
fn median_equals_center_on_constant() {
    let img = DynMatrix::<f64>::fill(7, 7, 3.14);
    let out = median_filter(&img, 2, BorderMode::Replicate);
    for i in 0..7 {
        for j in 0..7 {
            assert!((out[(i, j)] - 3.14).abs() < 1e-12);
        }
    }
}

#[test]
fn rank_min_and_max_match_erosion_dilation() {
    // Bright spot on zero background: the 3×3 minimum (rank 0) should be zero
    // everywhere; the 3×3 maximum (rank 8) should be 1 in a 3×3 block around
    // the spot.
    let mut img = DynMatrix::<f64>::zeros(9, 9);
    img[(4, 4)] = 1.0;
    let mins = rank_filter(&img, 1, 0, BorderMode::Zero);
    let maxs = rank_filter(&img, 1, 8, BorderMode::Zero);
    for i in 0..9 {
        for j in 0..9 {
            assert_eq!(mins[(i, j)], 0.0);
            let in_block = (3..=5).contains(&i) && (3..=5).contains(&j);
            assert_eq!(maxs[(i, j)], if in_block { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn percentile_monotone_in_p() {
    // Monotonic gradient image; percentile outputs must be non-decreasing in p.
    let img = DynMatrix::from_fn(11, 11, |i, j| (i + j) as f64);
    let p25 = percentile_filter(&img, 2, 0.25, BorderMode::Replicate);
    let p50 = percentile_filter(&img, 2, 0.50, BorderMode::Replicate);
    let p75 = percentile_filter(&img, 2, 0.75, BorderMode::Replicate);
    for i in 0..11 {
        for j in 0..11 {
            assert!(p25[(i, j)] <= p50[(i, j)] + 1e-12);
            assert!(p50[(i, j)] <= p75[(i, j)] + 1e-12);
        }
    }
}

#[test]
fn percentile_clamps_out_of_range() {
    let img = DynMatrix::from_fn(5, 5, |i, j| (i * 5 + j) as f64);
    let lo = percentile_filter(&img, 1, -1.0, BorderMode::Replicate);
    let hi = percentile_filter(&img, 1, 2.0, BorderMode::Replicate);
    let min_out = rank_filter(&img, 1, 0, BorderMode::Replicate);
    let max_out = rank_filter(&img, 1, 8, BorderMode::Replicate);
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(lo[(i, j)], min_out[(i, j)]);
            assert_eq!(hi[(i, j)], max_out[(i, j)]);
        }
    }
}

#[test]
fn rank_zero_radius_is_identity() {
    let img = DynMatrix::from_fn(4, 4, |i, j| (i * 4 + j) as f64);
    let out = rank_filter(&img, 0, 0, BorderMode::Replicate);
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(out[(i, j)], img[(i, j)]);
        }
    }
}

// ── Huang median (u16) ─────────────────────────────────────────────────

#[test]
fn huang_matches_quickselect_on_random_u16() {
    // 12-bit-ish data; compare Huang's histogram median against the
    // FloatScalar quickselect median on the same data cast to f64.
    let h = 17;
    let w = 23;
    let img_u16 = DynMatrix::from_fn(h, w, |i, j| {
        let x = (i * 131 + j * 37 + (i ^ j) * 29) as u32;
        (x % 4096) as u16
    });
    let img_f64 = DynMatrix::from_fn(h, w, |i, j| img_u16[(i, j)] as f64);

    for radius in [1_usize, 2, 3] {
        let huang = median_filter_u16(&img_u16, radius, BorderMode::Replicate);
        let slow = median_filter(&img_f64, radius, BorderMode::Replicate);
        for i in 0..h {
            for j in 0..w {
                assert_eq!(
                    huang[(i, j)] as f64,
                    slow[(i, j)],
                    "r={radius}, ({i},{j}) huang={}, qs={}",
                    huang[(i, j)],
                    slow[(i, j)],
                );
            }
        }
    }
}

#[test]
fn huang_removes_salt_u16() {
    let mut img = DynMatrix::<u16>::fill(9, 9, 800);
    img[(4, 4)] = 4095;
    let out = median_filter_u16(&img, 1, BorderMode::Replicate);
    for i in 0..9 {
        for j in 0..9 {
            assert_eq!(out[(i, j)], 800);
        }
    }
}

#[test]
fn huang_constant_image_is_unchanged() {
    let img = DynMatrix::<u16>::fill(12, 10, 1234);
    let out = median_filter_u16(&img, 2, BorderMode::Replicate);
    for i in 0..12 {
        for j in 0..10 {
            assert_eq!(out[(i, j)], 1234);
        }
    }
}

// ── Median pool / pool + upsample ──────────────────────────────────────

#[test]
fn median_pool_shape_and_values() {
    // 4×6 image, block_size=2 → 2×3 output.
    let img = DynMatrix::from_rows(
        4,
        6,
        &[
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
    );
    let out = median_pool(&img, 2);
    assert_eq!(out.nrows(), 2);
    assert_eq!(out.ncols(), 3);
    // 2×2 block of 4 values → quickselect picks index 2 (the upper-middle).
    // Block (0,0): {1,2,7,8} sorted → [1,2,7,8], mid=2 → 7.
    assert_eq!(out[(0, 0)], 7.0);
    // Block (1,2): {17,18,23,24} → mid=2 → 23.
    assert_eq!(out[(1, 2)], 23.0);
}

#[test]
fn median_pool_partial_blocks() {
    // 5×5 with block_size=3 → 2×2 output with partial blocks on the edges.
    let img = DynMatrix::from_fn(5, 5, |i, j| (i * 5 + j) as f64);
    let out = median_pool(&img, 3);
    assert_eq!(out.nrows(), 2);
    assert_eq!(out.ncols(), 2);
    // Partial right-bottom block is 2×2 = {18, 19, 23, 24} → mid=2 → 23.
    assert_eq!(out[(1, 1)], 23.0);
}

#[test]
fn median_pool_upsampled_restores_shape_and_preserves_constant() {
    let img = DynMatrix::<f64>::fill(20, 24, 7.5);
    let bg = median_pool_upsampled(&img, 4);
    assert_eq!(bg.nrows(), 20);
    assert_eq!(bg.ncols(), 24);
    for i in 0..20 {
        for j in 0..24 {
            assert!((bg[(i, j)] - 7.5).abs() < 1e-12);
        }
    }
}

// ── Morphology (Van Herk) ──────────────────────────────────────────────

fn naive_window_extreme<F: Fn(f64, f64) -> f64>(
    src: &DynMatrix<f64>,
    radius: usize,
    border: BorderMode<f64>,
    init: f64,
    combine: F,
) -> DynMatrix<f64> {
    let nrows = src.nrows();
    let ncols = src.ncols();
    let r = radius as isize;
    let mut out = DynMatrix::<f64>::zeros(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            let mut acc = init;
            for dj in -r..=r {
                for di in -r..=r {
                    let val = fetch_border_2d_f64(src, i as isize + di, j as isize + dj, border);
                    acc = combine(acc, val);
                }
            }
            out[(i, j)] = acc;
        }
    }
    out
}

fn fetch_border_2d_f64(
    src: &DynMatrix<f64>,
    i: isize,
    j: isize,
    border: BorderMode<f64>,
) -> f64 {
    let nrows = src.nrows() as isize;
    let ncols = src.ncols() as isize;
    let in_bounds = i >= 0 && i < nrows && j >= 0 && j < ncols;
    if in_bounds {
        return src[(i as usize, j as usize)];
    }
    match border {
        BorderMode::Zero => 0.0,
        BorderMode::Constant(c) => c,
        BorderMode::Replicate => {
            let ii = i.clamp(0, nrows - 1) as usize;
            let jj = j.clamp(0, ncols - 1) as usize;
            src[(ii, jj)]
        }
        BorderMode::Reflect => {
            let period_r = (2 * (nrows - 1)).max(1);
            let mut mi = i.rem_euclid(period_r);
            if mi >= nrows {
                mi = period_r - mi;
            }
            let period_c = (2 * (ncols - 1)).max(1);
            let mut mj = j.rem_euclid(period_c);
            if mj >= ncols {
                mj = period_c - mj;
            }
            src[(mi as usize, mj as usize)]
        }
    }
}

#[test]
fn max_filter_matches_naive() {
    let img = DynMatrix::from_fn(13, 17, |i, j| ((i * 7) ^ (j * 13)) as f64);
    for r in [1_usize, 2, 3, 5] {
        let fast = max_filter(&img, r, BorderMode::Replicate);
        let slow = naive_window_extreme(&img, r, BorderMode::Replicate, f64::NEG_INFINITY, f64::max);
        for i in 0..13 {
            for j in 0..17 {
                assert_eq!(
                    fast[(i, j)],
                    slow[(i, j)],
                    "max r={r} at ({i},{j}): fast={}, slow={}",
                    fast[(i, j)],
                    slow[(i, j)]
                );
            }
        }
    }
}

#[test]
fn min_filter_matches_naive() {
    let img = DynMatrix::from_fn(11, 9, |i, j| ((i * 3 + j) % 19) as f64);
    for r in [1_usize, 2, 4] {
        let fast = min_filter(&img, r, BorderMode::Replicate);
        let slow = naive_window_extreme(&img, r, BorderMode::Replicate, f64::INFINITY, f64::min);
        for i in 0..11 {
            for j in 0..9 {
                assert_eq!(fast[(i, j)], slow[(i, j)], "min r={r} at ({i},{j})");
            }
        }
    }
}

#[test]
fn dilate_erode_on_impulse() {
    let mut img = DynMatrix::<f64>::zeros(11, 11);
    img[(5, 5)] = 1.0;
    let d = dilate(&img, 2, BorderMode::Zero);
    // A 5×5 block around (5,5) should be 1.0, rest 0.
    for i in 0..11 {
        for j in 0..11 {
            let expected = if (3..=7).contains(&i) && (3..=7).contains(&j) { 1.0 } else { 0.0 };
            assert_eq!(d[(i, j)], expected);
        }
    }
    let e = erode(&d, 2, BorderMode::Zero);
    // Erosion by the same kernel should shrink it back — but only near corners,
    // because border rules clip. Just verify the single centre pixel survives.
    assert_eq!(e[(5, 5)], 1.0);
}

// ── median_3x3 / median_5x5 specializations ────────────────────────────

#[test]
fn median_radius_1_matches_rank_filter() {
    let img = DynMatrix::from_fn(13, 19, |i, j| ((i * 17 + j * 23) % 256) as f64);
    let fast = median_filter(&img, 1, BorderMode::Replicate);
    let slow = rank_filter(&img, 1, 4, BorderMode::Replicate);
    for i in 0..13 {
        for j in 0..19 {
            assert_eq!(fast[(i, j)], slow[(i, j)], "3×3 median at ({i},{j})");
        }
    }
}

#[test]
fn median_radius_2_matches_rank_filter() {
    let img = DynMatrix::from_fn(15, 21, |i, j| ((i * 7 + j * 11) % 128) as f64);
    let fast = median_filter(&img, 2, BorderMode::Replicate);
    let slow = rank_filter(&img, 2, 12, BorderMode::Replicate);
    for i in 0..15 {
        for j in 0..21 {
            assert_eq!(fast[(i, j)], slow[(i, j)], "5×5 median at ({i},{j})");
        }
    }
}

#[test]
fn median_radius_1_small_image_fallback() {
    // Images smaller than the interior region must still produce the same
    // result as the generic path (they fall through to rank_filter).
    let img = DynMatrix::<f64>::fill(2, 2, 3.0);
    let out = median_filter(&img, 1, BorderMode::Replicate);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(out[(i, j)], 3.0);
        }
    }
}

// ── Morphology compositions ────────────────────────────────────────────

#[test]
fn opening_removes_small_bright_feature() {
    // A single bright pixel on a dark background — 3×3 opening should remove it.
    let mut img = DynMatrix::<f64>::zeros(9, 9);
    img[(4, 4)] = 1.0;
    let out = opening(&img, 1, BorderMode::Zero);
    for i in 0..9 { for j in 0..9 { assert_eq!(out[(i, j)], 0.0); } }
}

#[test]
fn closing_fills_small_dark_hole() {
    // A single dark pixel in a bright background — 3×3 closing fills it.
    let mut img = DynMatrix::<f64>::fill(9, 9, 1.0);
    img[(4, 4)] = 0.0;
    let out = closing(&img, 1, BorderMode::Replicate);
    assert_eq!(out[(4, 4)], 1.0);
}

#[test]
fn morphology_gradient_highlights_boundary() {
    // Sharp step image — gradient should be nonzero around the step.
    let mut img = DynMatrix::<f64>::zeros(9, 9);
    for i in 0..9 { for j in 5..9 { img[(i, j)] = 1.0; } }
    let g = morphology_gradient(&img, 1, BorderMode::Replicate);
    // Columns adjacent to the step have nonzero response; deep interior is zero.
    assert!(g[(4, 4)] > 0.0);
    assert_eq!(g[(4, 0)], 0.0);
    assert_eq!(g[(4, 8)], 0.0);
}

#[test]
fn top_hat_isolates_bright_spike() {
    let mut img = DynMatrix::<f64>::fill(11, 11, 5.0);
    img[(5, 5)] = 20.0;
    let t = top_hat(&img, 1, BorderMode::Replicate);
    // Background goes to zero; the spike location retains the difference.
    for i in 0..11 { for j in 0..11 {
        if (i, j) == (5, 5) { assert!(t[(i, j)] > 0.0); }
        else { assert_eq!(t[(i, j)], 0.0); }
    }}
}

#[test]
fn black_hat_isolates_dark_spike() {
    let mut img = DynMatrix::<f64>::fill(11, 11, 5.0);
    img[(5, 5)] = 0.0;
    let b = black_hat(&img, 1, BorderMode::Replicate);
    assert!(b[(5, 5)] > 0.0);
}

// ── Geometric ─────────────────────────────────────────────────────────

#[test]
fn flip_horizontal_reverses_cols() {
    let img = DynMatrix::from_rows(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let f = flip_horizontal(&img);
    assert_eq!(f[(0, 0)], 3.0);
    assert_eq!(f[(0, 2)], 1.0);
    assert_eq!(f[(1, 0)], 6.0);
    assert_eq!(f[(1, 2)], 4.0);
}

#[test]
fn flip_vertical_reverses_rows() {
    let img = DynMatrix::from_rows(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let f = flip_vertical(&img);
    assert_eq!(f[(0, 0)], 4.0);
    assert_eq!(f[(1, 2)], 3.0);
}

#[test]
fn rotate_90_180_270_and_back() {
    let img = DynMatrix::from_rows(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r1 = rotate_90(&img);
    let r2 = rotate_180(&img);
    let r3 = rotate_270(&img);
    assert_eq!(r1.nrows(), 3); assert_eq!(r1.ncols(), 2);
    assert_eq!(r2.nrows(), 2); assert_eq!(r2.ncols(), 3);
    assert_eq!(r3.nrows(), 3); assert_eq!(r3.ncols(), 2);

    // Known value checks for rotate_90 clockwise of
    //   1 2 3
    //   4 5 6
    // is
    //   4 1
    //   5 2
    //   6 3
    assert_eq!(r1[(0, 0)], 4.0);
    assert_eq!(r1[(0, 1)], 1.0);
    assert_eq!(r1[(2, 1)], 3.0);

    // rotate_180 is element reversal.
    assert_eq!(r2[(0, 0)], 6.0);
    assert_eq!(r2[(1, 2)], 1.0);

    // rotate_90(rotate_270(x)) == x
    let back = rotate_90(&rotate_270(&img));
    for i in 0..2 { for j in 0..3 { assert_eq!(back[(i, j)], img[(i, j)]); } }
}

#[test]
fn pad_adds_border_pixels() {
    let img = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
    let p = pad(&img, 1, 1, 1, 1, BorderMode::Zero);
    assert_eq!(p.nrows(), 4); assert_eq!(p.ncols(), 4);
    // Corners are border, middle 2×2 is the original.
    assert_eq!(p[(0, 0)], 0.0);
    assert_eq!(p[(1, 1)], 1.0);
    assert_eq!(p[(2, 2)], 4.0);
    assert_eq!(p[(3, 3)], 0.0);
}

#[test]
fn pad_replicate_extends_edges() {
    let img = DynMatrix::from_rows(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
    let p = pad(&img, 1, 1, 1, 1, BorderMode::Replicate);
    // Edges extend the nearest pixel.
    assert_eq!(p[(0, 0)], 1.0);
    assert_eq!(p[(0, 3)], 2.0);
    assert_eq!(p[(3, 0)], 3.0);
    assert_eq!(p[(3, 3)], 4.0);
}

#[test]
fn crop_extracts_subregion() {
    let img = DynMatrix::from_fn(4, 5, |i, j| (i * 10 + j) as f64);
    let c = crop(&img, 1, 2, 2, 3);
    assert_eq!(c.nrows(), 2); assert_eq!(c.ncols(), 3);
    assert_eq!(c[(0, 0)], 12.0); // img[(1, 2)]
    assert_eq!(c[(1, 2)], 24.0); // img[(2, 4)]
}

#[test]
fn resize_nearest_preserves_discrete_labels() {
    // Discrete labels that bilinear would corrupt must stay integral.
    let img = DynMatrix::<f64>::from_fn(2, 2, |i, j| (i * 2 + j) as f64);
    let up = resize_nearest(&img, 4, 4);
    for i in 0..4 {
        for j in 0..4 {
            let v = up[(i, j)];
            assert_eq!(v, v.round(), "non-integer value {v} at ({i},{j})");
        }
    }
}

// ── Local stats ────────────────────────────────────────────────────────

#[test]
fn local_mean_on_constant() {
    let img = DynMatrix::<f64>::fill(10, 10, 7.0);
    let m = local_mean(&img, 2);
    for i in 0..10 { for j in 0..10 { assert!((m[(i, j)] - 7.0).abs() < 1e-12); } }
}

#[test]
fn local_variance_constant_is_zero() {
    let img = DynMatrix::<f64>::fill(8, 8, 3.5);
    let v = local_variance(&img, 1);
    for i in 0..8 { for j in 0..8 { assert!(v[(i, j)] < 1e-12); } }
}

#[test]
fn local_stddev_matches_sqrt_variance() {
    let img = DynMatrix::from_fn(8, 8, |i, j| ((i * 7 + j * 3) % 11) as f64);
    let v = local_variance(&img, 1);
    let s = local_stddev(&img, 1);
    for i in 0..8 { for j in 0..8 {
        assert!((s[(i, j)] - v[(i, j)].sqrt()).abs() < 1e-10);
    }}
}

// ── DoG / pyramid ──────────────────────────────────────────────────────

#[test]
fn dog_on_constant_is_zero() {
    let img = DynMatrix::<f64>::fill(24, 24, 4.0);
    let d = difference_of_gaussians(&img, 1.0, 1.6, BorderMode::Replicate);
    for i in 3..21 { for j in 3..21 { assert!(d[(i, j)].abs() < 1e-10); } }
}

#[test]
fn gaussian_pyramid_halves_sizes() {
    let img = DynMatrix::<f64>::fill(64, 64, 1.0);
    let pyr = gaussian_pyramid(&img, 4, 1.0, BorderMode::Replicate);
    assert_eq!(pyr.len(), 4);
    assert_eq!(pyr[0].nrows(), 64);
    assert_eq!(pyr[1].nrows(), 32);
    assert_eq!(pyr[2].nrows(), 16);
    assert_eq!(pyr[3].nrows(), 8);
}

// ── Thresholding ───────────────────────────────────────────────────────

#[test]
fn threshold_binary() {
    let img = DynMatrix::from_rows(2, 2, &[0.0_f64, 0.5, 1.0, 1.5]);
    let out = threshold(&img, 0.75);
    assert_eq!(out[(0, 0)], 0.0);
    assert_eq!(out[(0, 1)], 0.0);
    assert_eq!(out[(1, 0)], 1.0);
    assert_eq!(out[(1, 1)], 1.0);
}

#[test]
fn threshold_otsu_bimodal() {
    // Bimodal image: one cluster near 1.0, another near 10.0.
    // Otsu should pick a threshold in between.
    let img = DynMatrix::from_fn(16, 16, |i, j| {
        if (i * 16 + j) % 2 == 0 { 1.0_f64 } else { 10.0 }
    });
    let t = threshold_otsu(&img);
    assert!(t > 1.5 && t < 9.5, "Otsu threshold out of expected band: {t}");
}

#[test]
fn adaptive_threshold_responds_to_local_brightness() {
    // Gradient background + isolated bright spot — adaptive threshold
    // should pick up the spot even though a global threshold at half-max
    // would miss it.
    let img = DynMatrix::from_fn(17, 17, |_i, j| j as f64);
    let mut img = img;
    img[(8, 8)] = 100.0;
    let out = adaptive_threshold(&img, 3, 5.0);
    assert_eq!(out[(8, 8)], 1.0);
}

// ── Canny ──────────────────────────────────────────────────────────────

#[test]
fn canny_on_step_produces_edge() {
    // Vertical step between columns 15 and 16 — a single clean edge column
    // should survive Canny's pipeline.
    let mut img = DynMatrix::<f64>::zeros(32, 32);
    for i in 0..32 { for j in 16..32 { img[(i, j)] = 1.0; } }
    let edges = canny(&img, 1.0, 0.05, 0.15, BorderMode::Replicate);
    // Some edge pixels exist near the step in the middle rows.
    let mut edge_count = 0;
    for i in 8..24 { for j in 14..18 { if edges[(i, j)] > 0.5 { edge_count += 1; } } }
    assert!(edge_count > 8, "expected > 8 edge pixels in the centre strip, got {edge_count}");
}

#[test]
fn canny_on_constant_is_empty() {
    let img = DynMatrix::<f64>::fill(16, 16, 5.0);
    let edges = canny(&img, 1.0, 0.05, 0.15, BorderMode::Replicate);
    for i in 0..16 { for j in 0..16 { assert_eq!(edges[(i, j)], 0.0); } }
}

// ── Harris / Shi-Tomasi ────────────────────────────────────────────────

#[test]
fn harris_detects_square_corners() {
    let mut img = DynMatrix::<f64>::fill(32, 32, 0.0);
    for i in 10..22 { for j in 10..22 { img[(i, j)] = 1.0; } }
    let r = harris_corners(&img, 1.0, 0.05, BorderMode::Replicate);
    // Corners should have a stronger response than flat interior or edge interior.
    let corner = r[(10, 10)].max(r[(11, 11)]);
    let flat = r[(16, 16)].abs();
    let edge = r[(10, 16)].abs(); // on the top edge, not at a corner
    assert!(corner > flat, "corner={corner}, flat={flat}");
    assert!(corner > edge, "corner={corner}, edge={edge}");
}

#[test]
fn shi_tomasi_nonnegative() {
    let img = DynMatrix::from_fn(16, 16, |i, j| ((i * 3 + j * 5) % 7) as f64);
    let r = shi_tomasi_corners(&img, 1.0, BorderMode::Replicate);
    for i in 0..16 { for j in 0..16 {
        assert!(r[(i, j)] >= 0.0, "negative response {} at ({i},{j})", r[(i, j)]);
    }}
}

#[test]
fn median_pool_rejects_sparse_bright_sources() {
    // A dark background with a handful of bright pixels — block median
    // should suppress them entirely (they are minority in every block).
    let mut img = DynMatrix::<f64>::fill(16, 16, 2.0);
    // Scatter four bright pixels, none adjacent.
    img[(1, 1)] = 1000.0;
    img[(3, 10)] = 1000.0;
    img[(9, 4)] = 1000.0;
    img[(13, 13)] = 1000.0;
    let pooled = median_pool(&img, 4);
    assert_eq!(pooled.nrows(), 4);
    assert_eq!(pooled.ncols(), 4);
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(pooled[(i, j)], 2.0);
        }
    }
}
