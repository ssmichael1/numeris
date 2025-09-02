# Numeris

General-purpose mathematics and scientific programming library written in pure rust.

The libary has minimal dependencies, and is (*will be*) suitable for use both in general-purpose computing and in high-reliability embedded applications.

Currently in **alpha** development state

## Design Philosophy

- `Numeris` is intended to be a mostly *self-contained* crate, with minimal external dependencies.  This allows for the creation of a highly-integrated package, and ensures robustness when used in embedded applications.

- `Numeris` is performant, but minimizes the use of platform-specific optimizations. Computers are fast and getting faster.  Also, compiler optimizations do a good job of squeezing out available performance.  Avoiding explicit use of these optimizations allows the crate to be platform-agnostic (again, with an eye toward embedded applications), and again minimizes external dependencies. 

- `Numeris` code is simple by design.  Complex idioms can add performance and flexibility, but also are a potential source of errors for developers who cannot follow the design flow.  The code is intended to be understood by those who do not develop rust professionally (such as myself).

## Modules

- `Matrix` : Matrix operations with fixed-size matrices known at compile time
- `DynMatrix` : Matrix operations with dynamically-sized matrices 
- `ODE` : Integration of ordinary differential equations via Runga-Kutta methods
- `Quaternion` : Quaternion representation of attitude and 3-dimensional rotations

## Planned Modules
- `Kalman` : Kalman filter methods (KF, EKF, UKF)
- `DigFilt` : Digital linear filters
- `Image` : 2D Image processing (convolution, transforms)
- `FFT` : Fast-Fourier Transform
- `Optim` : Linear and non-linear optimization (Gauss-Newton, Levenberg-Marquardt, etc..)
- `Random` : Random number generation from multiple possible distributions

## Contributing

I will be developing modules as needed to support my professional work.  However, I am open to making the crate as useful as possible for others. If you are interested in contributing or have suggestions for additional modules, please reach out.

Steven Michael (ssmichael@gmail.com)