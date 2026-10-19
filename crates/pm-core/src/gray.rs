//! Grayscale image representation, conversions, sampling, and ROI extraction.
//!
//! Planned for milestone M2 (README §6). The internal working representation is
//! `GrayF32` in `[0, 1]`; color and 16-bit inputs are converted on load.
