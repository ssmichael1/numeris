# Fast Fourier Transform

Pure-Rust, no-std-first FFT integrated with the crate's `Complex` and SIMD support.

Requires the `fft` Cargo feature:

```toml
numeris = { version = "0.5", features = ["fft"] }
```

## Not FFTW

FFTW's speed comes from runtime planning, autotuned codelets, and a large C codebase
that cannot run under `no_std` / no-alloc — the target `numeris` was built for. The goal
here is portable, zero-C-dependency FFT that **also** runs on embedded, integrated with
the rest of the crate. On desktop, expect roughly 2–4× FFTW throughput; for the audience
that reaches for `numeris` (embedded DSP, no allocator) that is a non-issue, because FFTW
is not an option there.

Sign convention: the forward transform uses $X_k = \sum_n x_n\, e^{-2\pi i kn/N}$; the
inverse is normalized by $1/N$ so that `ifft(fft(x)) == x`.

## Two tiers

| Tier | API | Allocation | Sizes |
|---|---|---|---|
| Fixed-size | `fft` / `ifft` / `fft_inplace` / `ifft_inplace` | none (stack) | power-of-two `N ≤ 4096` |
| Real, fixed | `rfft` / `irfft` | none (stack) | power-of-two `N ≤ 4096` |
| Runtime | `DynFft` | `alloc` | any length |
| Real, runtime | `DynRealFft` | `alloc` | any length |
| 2D | `DynFft2` / `DynRealFft2` | `alloc` | any `rows × cols` |

## Fixed-size complex FFT (no-alloc)

In-place over `[Complex<T>; N]` for power-of-two `N ≤ 4096` (checked at compile time).
`fft` / `ifft` take a precomputed `TwiddleTable` so no `sin`/`cos` runs in the loop —
the path to prefer when repeating a fixed transform size on embedded. `fft_inplace` /
`ifft_inplace` generate stage twiddles inline (no persistent table).

```rust
use numeris::fft::{fft_inplace, ifft_inplace};
use numeris::Complex;

let mut buf = [
    Complex::new(1.0f64, 0.0),
    Complex::new(2.0, 0.0),
    Complex::new(3.0, 0.0),
    Complex::new(4.0, 0.0),
];
fft_inplace(&mut buf);
assert!((buf[0].re - 10.0).abs() < 1e-12); // DC bin = sum of inputs
ifft_inplace(&mut buf);
assert!((buf[0].re - 1.0).abs() < 1e-12);
```

With a reusable table:

```rust
use numeris::fft::{fft, TwiddleTable};
use numeris::Complex;

let tw = TwiddleTable::<f64, 8>::new(); // build once
let mut buf = [Complex::new(0.0, 0.0); 8];
buf[1] = Complex::new(1.0, 0.0);
fft(&mut buf, &tw); // reuse across many transforms
```

`TwiddleTable<f64, 4096>` occupies 64 KiB (full-`N` storage keeps the type on stable
Rust). Memory-constrained callers should use `fft_inplace` or a smaller `N`.

## Runtime-sized FFT — `DynFft` (requires `alloc`)

Build a plan once, reuse it. Power-of-two lengths transform through the SIMD-accelerated
radix core; any other length — including primes — goes through **Bluestein's algorithm**,
which reduces the DFT to power-of-two FFTs. No dedicated radix-3/5 kernels are needed.

```rust
use numeris::fft::DynFft;
use numeris::Complex;

let mut plan = DynFft::<f64>::new(1000); // 1000 = 2^3 · 5^3, mixed length
let mut buf: Vec<Complex<f64>> = (0..1000).map(|i| Complex::new(i as f64, 0.0)).collect();
plan.forward(&mut buf);
plan.inverse(&mut buf);
assert!((buf[1].re - 1.0).abs() < 1e-9);

let mut prime = DynFft::<f64>::new(1009); // prime -> Bluestein
# let mut b: Vec<Complex<f64>> = (0..1009).map(|i| Complex::new(i as f64, 0.0)).collect();
# prime.forward(&mut b);
```

`forward` / `inverse` panic if the buffer length does not equal the plan length.

## Real-input transforms

A length-`N` real signal has a Hermitian spectrum, so only the `N/2 + 1` non-redundant
bins (DC … Nyquist) are computed. The forward transform packs the reals into `N/2`
complex samples for roughly half the work. Output length is passed as a caller-supplied
slice (returning `[_; N/2+1]` would need unstable const generics).

```rust
use numeris::fft::{rfft, irfft};
use numeris::Complex;

let signal = [1.0f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
let mut spec = [Complex::new(0.0, 0.0); 8 / 2 + 1]; // 5 bins
rfft(&signal, &mut spec);

let mut recovered = [0.0f64; 8];
irfft(&spec, &mut recovered);
assert!((recovered[0] - 1.0).abs() < 1e-12);
```

`DynRealFft` is the runtime-sized equivalent (even lengths use the half-size trick; odd
lengths use a full complex FFT).

## Convolution and correlation (requires `alloc`)

FFT-based linear convolution / cross-correlation of real signals. Output length is
`a.len() + b.len() - 1`.

```rust
use numeris::fft::fft_convolve;

let a = [1.0f64, 2.0, 3.0];
let b = [0.5, 1.0];
let c = fft_convolve(&a, &b); // length 4
```

For small kernels a direct convolution is faster; the FFT path wins once both operands
are large.

## 2D FFT (requires `alloc`)

The 2D DFT is *separable*: an `rows × cols` transform is a batch of 1D FFTs along each
axis (every column, then every row — order does not matter), so `DynFft2` is built entirely
on `DynFft` and inherits Bluestein for non-power-of-two dimensions. Data is a column-major
`DynMatrix<Complex<T>>`, so the column pass slices contiguous columns straight out of the
backing buffer while the row pass gathers/scatters through a scratch buffer.

```rust
use numeris::fft::DynFft2;
use numeris::{DynMatrix, Complex};

let mut plan = DynFft2::<f64>::new(4, 4);
let mut img = DynMatrix::from_fn(4, 4, |r, c| Complex::new((r + c) as f64, 0.0));
plan.forward(&mut img);
plan.inverse(&mut img); // normalized by 1/(rows*cols)
assert!((img[(1, 2)].re - 3.0).abs() < 1e-10);
```

`DynRealFft2` transforms a real `rows × cols` image into a `(rows/2 + 1) × cols` complex
half-spectrum (real FFT along the contiguous column axis, full complex FFT along the row
axis) — roughly half the cost and storage, the form image processing wants.

```rust
use numeris::fft::DynRealFft2;
use numeris::{DynMatrix, Complex};

let mut plan = DynRealFft2::<f64>::new(6, 4);
let real = DynMatrix::from_fn(6, 4, |r, c| (r as f64).sin() + c as f64);
let mut spec = DynMatrix::zeros(6 / 2 + 1, 4); // (rows/2+1) × cols
plan.forward(&real, &mut spec);
let mut recon = DynMatrix::zeros(6, 4);
plan.inverse(&spec, &mut recon);
```

## Spectrum centering

`fftshift` / `ifftshift` are no-alloc, in-place, and generic over any element type — pure
rotations. `fftshift` moves the zero-frequency component to the center (NumPy semantics);
`ifftshift` is its exact inverse for odd lengths.

```rust
use numeris::fft::{fftshift, ifftshift};

let mut v = [0, 1, 2, 3, 4];
fftshift(&mut v);
assert_eq!(v, [3, 4, 0, 1, 2]);
ifftshift(&mut v);
assert_eq!(v, [0, 1, 2, 3, 4]);
```

`fftshift2d` / `ifftshift2d` are the 2D analogue for a `DynMatrix` — they swap diagonal
quadrants (a 1D shift along each axis) and allocate nothing.

```rust
use numeris::fft::{fftshift2d, ifftshift2d};
use numeris::DynMatrix;

let mut m = DynMatrix::from_fn(4, 4, |r, c| (r * 4 + c) as f64);
fftshift2d(&mut m);   // DC at (0,0) moves to the center
ifftshift2d(&mut m);  // exact inverse
```

## Performance notes

- The `DynFft` power-of-two path deinterleaves into structure-of-arrays real/imaginary
  buffers and runs radix-2 butterflies through SIMD kernels (NEON / SSE2 / AVX / AVX-512
  via compile-time dispatch, scalar fallback otherwise).
- The no-std fixed tier stays scalar: its audience is embedded (small `N`,
  code-size-sensitive), where deinterleave scratch would undercut the low-memory point.
- Bluestein trades a prime-length DFT for a power-of-two FFT of length `≥ 2N − 1`, so
  prime sizes cost more than nearby power-of-two sizes but remain `O(N log N)`.
