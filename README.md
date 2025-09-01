# Numeris

General-purpose mathematics and scientific programming library written in pure rust.

The libary has minimal dependencies, and is (will be) suitable for use in high-reliability embedded applications.

Currently in **alpha** development state

## Modules

- `Matrix` : Matrix operations with fixed-size matrices known at compile time
- `DynMatrix` : Matrix operations with dynamically-sized matrices 
- `ODE` : Integration of ordinary differential equations via Runga-Kutta methods
- `Quaternion` : Quaternion representation of attitude and 3-dimensional rotations

## Planned Modules
- `Kalman` : Kalman filter methods (KF, EKF, UKF)
- `DigFilt` : Digital linear filters
- `Image` : 2D Image processing
- `Optim` : Linear and non-linear optimization (Gauss-Newton, Levenberg-Marquardt, etc..)

## Contributing

This crate is supported publicly, but I will be developing modules as needed to support my professional work. If you are interested in contributing or have suggestions for additional modules, please reach out.

Steven Michael (ssmichael@gmail.com)