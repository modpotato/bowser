//! Display lists and WebGPU renderer.

pub mod render;

mod borders;

pub mod images;

/// SVG intrinsic dimensions (metadata only; see [`svg::svg_intrinsics`]).
pub mod svg;

mod clip;

mod cpu;

mod text;

mod wgpu_be;

pub use wgpu_be::{WgpuBackend, shutdown_headless_backend};

pub mod display_list;
pub mod filter;
pub mod fonts;
pub mod gradient;
