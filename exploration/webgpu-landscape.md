# WebGPU-in-Rust Landscape Survey (for Project Bowser)
*Researched 2026-09-25. Read-only exploration; no code written.*

## 1. The wgpu crate: maturity and version

- **wgpu 30.0.1** is the latest release (2026-08-22, verified via crates.io API). naga tracks it at 30.0.1.
- **Adoption:** ~36.8M total downloads, ~10.9M in the last 90 days. naga at ~40.7M. This is the dominant Rust GPU API by a wide margin.
- **Battle-tested in browsers:** wgpu is the core of the WebGPU implementation in **Firefox, Servo, and Deno** (per the project's own README). Firefox's MSRV is even hard-tied to wgpu-core's MSRV policy. This is the strongest possible signal: two shipping browser engines already bet their rendering on it.
- **Spec status:** W3C WebGPU reached **Candidate Recommendation in March 2026** (~85% global browser support). wgpu tracks it closely.
- **MSRV:** 1.87 for the `wgpu` crate (deliberately conservative, pinned to Firefox/Servo MSRVs). Breaking releases ship **quarterly**, so budget ~4 upgrade cycles/year.
- **Platforms/backends:** Vulkan (Win/Linux/Android), Metal (macOS/iOS), D3D12 (Windows), GLES/OpenGL (best-effort, via ANGLE where needed), software noop, and on wasm: native WebGPU or WebGL2. All in **pure Rust, no C++ build dependencies** (unlike Google's Dawn, which requires a C++ toolchain).

## 2. What wgpu depends on (the internal stack)

`wgpu` (public API) → `wgpu-core` (validation, state tracking, resource lifetimes) → `wgpu-hal` (per-backend HAL: vulkan/metal/dx12/gles/software) + `naga` (WGSL → SPIR-V/HLSL/MSL/GLSL/DXIL compiler) + `wgpu-types`.

Direct normal dependencies of wgpu 30.0.1 (from crates.io API): arrayvec, bitflags, bytemuck, cfg-if, document-features, hashbrown, log, naga, parking_lot, portable-atomic, profiling, raw-window-handle, smallvec, static_assertions, wgpu-core, wgpu-hal, wgpu-types (+ wasm-only: js-sys, wasm-bindgen, wasm-bindgen-futures, web-sys). These are all tiny, stable utility crates. The real code mass lives inside the wgpu workspace itself (naga, wgpu-core, wgpu-hal).

Key features (verified): `vulkan`, `metal`, `dx12`, `gles`, `webgl`, `webgpu`, `wgsl`, `glsl`, `spirv`, `naga-ir`, `std`, `serde`, `trace`, `angle`, `noop`, `static-dxc`. Default features enable: vulkan, metal, dx12, gles, webgl, webgpu, wgsl, parking_lot, std, web-sys, fragile-send-sync-non-atomic-wasm. Since v25 **every backend is feature-gated and disablable**, so Bowser can select exactly one backend per target OS. no_std support is in progress (hashmaps were removed from public APIs in v25 as preparation).

## 3. Is wgpu viable as Bowser's sole rendering dependency?

**Yes.** Rationale:
- Single pure-Rust dependency replaces raw Vulkan (ash) + Metal (metal crate) + D3D12 (windows crate) + OpenGL (glow) — otherwise Bowser would write and maintain four backends.
- Bowser's mandate is WebGPU; wgpu is the idiomatic Rust implementation of exactly that API, plus a native multi-backend runtime for free.
- Proven in two real browser engines (Firefox, Servo), with an MSRV policy explicitly coordinated with both.
- naga means Bowser writes shaders once in WGSL and gets every backend.

## 4. Alternatives for a browser compositor

| Option | Verdict |
|---|---|
| **Raw Vulkan via `ash`** (0.38, last updated 2024-04) | Linux-only in practice; Windows needs DX12, macOS needs MoltenVK. Means writing 3 backends by hand. Reject. |
| **Metal crate** (0.33) / D3D12 via `windows` crate | Single-platform each; multiplies backend work. Reject. |
| **OpenGL/GLES via `glow`** (0.18) | Cross-platform but legacy ceiling (no bindless/raytracing future), and browser relevance is declining. Only as fallback. |
| **Dawn (C++)** | Would require C++ toolchain and FFI — violates the Rust-first goal. Reject. |
| **wgpu** | One dependency, all platforms, browser-proven, pure Rust. **Choose this.** |

Note: Servo is also the closest architectural reference for Bowser (Rust browser + wgpu).

## 5. What a browser renderer needs beyond the GPU API

A GPU API gives you triangles and textures; a browser compositor additionally needs:

- **2D vector scene rendering (paths, fills, gradients, clips, transforms):** `vello` 0.10.0 — Linebender's GPU compute-centric 2D renderer built on wgpu. Purpose-built for exactly this workload (browser-like 2D scenes). Companion crates: `kurbo` 0.13.1 (geometry), `peniko` 0.6.1 (brushes).
- **Text layout + shaping:** `parley` 0.11.1 (Linebender; rich-text layout, BiDi, line breaking) built on `swash` 0.2.10 (shaping + glyph rasterization, incl. COLR/CBDT color emoji) + `fontique` 0.11.1 (system font discovery/management) + `skrifa` 0.47.0 (font parsing). Alternative: `cosmic-text` 0.19 (heavier, editor-oriented; parley is the leaner, renderer-focused choice). Alternative shaper: `harfrust` 0.13.3 (Rust port of HarfBuzz, full OpenType features) if swash's built-in shaper proves insufficient.
- **Font data pipeline:** skrifa (parse) → fontique (discover/match) → swash (shape/rasterize) → glyph atlas (`etagere` 0.3.0 shelf-packer) → wgpu texture. This is the stack writ/vello apps already use.
- **Image decoding:** `png` 0.18.1 + `zune-jpeg` 0.5.x (116M downloads, minimal footprint) + `image-webp` if WebP is required. Avoid the full `image` crate with all formats (avif/exr/tiff/...) unless Bowser wants them.
- **Windowing / surface:** `winit` 0.31 + `raw-window-handle` (already a wgpu dep) for native windows and swapchain surfaces.
- **Async glue:** wgpu's adapter/device requests are async; `pollster` 1.0.1 (tiny, 33M downloads) blocks on them without pulling in a full async runtime.

## 6. The std-only/minimal-deps tension — recommendation

**The tension is real but manageable, and wgpu is the correct compromise.** Bowser mandates WebGPU; implementing WebGPU-from-scratch on four native APIs in std-only Rust is a multi-year detour before Bowser draws its first pixel. wgpu's "bloat" is almost entirely *inside its own workspace* (naga/wgpu-core/wgpu-hal) — its third-party deps are a short list of tiny utility crates (bitflags, smallvec, arrayvec, hashbrown...), not a sprawling ecosystem grab.

### Recommended pragmatic dependency set for the renderer

```toml
# GPU API — the one big dependency, replacing 4 hand-written backends
wgpu = { version = "30", default-features = false,
         features = ["wgsl", "vulkan"] }   # per-target: metal on macOS, dx12 on Windows, gles as fallback
# 2D vector scene rendering on top of wgpu
vello = "0.10"      # + kurbo, peniko come along
# Text stack
parley = "0.11"
swash = "0.2"
fontique = "0.11"
skrifa = "0.47"
etagere = "0.3"     # glyph atlas packing
# Images (web formats only, no kitchen sink)
png = "0.18"
zune-jpeg = "0.5"
# Windowing + async glue
winit = "0.31"
pollster = "1"
```

### Bloat flags (do not pull these in by default)
- `image` with default features (drags in avif/exr/tiff/gif decoders + heavy codec deps) — use `png` + `zune-jpeg` directly, or `image` with `default-features = false`.
- `cosmic-text` if parley suffices (cosmic-text targets rich text editors; heavier).
- `harfrust` unless swash's shaper hits OpenType-feature limits (start with swash, measure).
- All wgpu backends on all platforms — select one per target via features (vulkan/Linux, metal/macOS, dx12/Windows); keep `gles` only as a fallback and `webgl`/`webgpu` for wasm targets.
- wgpu `serde`/`trace`/`counters` features — debugging/serialization overhead not needed in the hot path.
- `vello_svg`/`velato` (SVG/Lottie) — only if Bowser renders those natively; defer.

### Risks to budget for
- **Quarterly breaking releases** of wgpu/vello/parley — plan upgrade sprints; pin versions and use a lockfile.
- vello and parley are pre-1.0 (0.10 / 0.11) but actively maintained by Linebender with real users; acceptable risk given the alternative is writing a GPU 2D renderer from scratch.
- wgpu default features pull wasm machinery (`fragile-send-sync-non-atomic-wasm`); with `default-features = false` + explicit backends, native builds stay lean.
