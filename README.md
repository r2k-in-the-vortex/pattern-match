# Pattern Match

A classical, teach-by-example computer-vision tool for locating planar objects with
repeatable shape, in the spirit of Cognex PatMax. The implementation is Rust. The
math-heavy core — transforms, homography estimation, and RANSAC guarantees — is
specified in Lean 4 and its theorems are proven before the corresponding Rust code is
written.

**Status: M2 complete.** The image pipeline (grayscale/color/16-bit I/O,
Gaussian pyramid, Scharr gradients, subpixel edges, Shi–Tomasi corners, and
split-and-merge segments) and the feature overlay are implemented in `pm-core`.
The desktop UI loads an image, runs default feature extraction, and displays
the hue-coded overlay. Teaching tools begin at M3. This document is the
implementation plan and the record of decisions made.

---

## 1. Locked decisions

| Area | Decision |
|---|---|
| Motion model | Coarse search over similarity poses `(tx, ty, θ, s)`. Optional 8-DOF homography estimation/refinement from matched features. Every detection reports a full 3×3 homography and a `homography_estimated` flag; when perspective is not supported by the evidence, the matrix embeds the similarity fallback. |
| Matching paradigm | Edge shape model: subpixel edge points with gradient normals, scored by distance-to-edge and orientation agreement, with mask exclusion. Corners and line segments are extracted for visualization, descriptor/segment seeding, and homography correspondences. |
| Coarse search | Hybrid: oriented generalized Hough voting is the primary generator; rotation/scale-invariant corner descriptors and segment-pair invariants seed additional similarity hypotheses. All hypotheses pass the same edge-based verification and refinement. |
| UI | Native desktop `egui`/`eframe` application, Windows-first, pure Rust. No OpenCV or CMake dependency. |
| Lean | Lean 4 + Mathlib, executable specifications plus golden vectors consumed by Rust tests. Core proof scope: geometry G1–G8, DLT/rank/normalization D1–D5, RANSAC R1–R5. Noisy least-squares optimality D6 is a documented follow-on. Scoring and subpixel numerics are Rust-tested only. |
| Training | One sample image per model. The feature mask handles features that vary between instances. |
| Search range | Wide, user-configurable defaults: full 360° rotation, 0.5–2.0× scale, whole-image search, multiple instances. Every range can be tightened. |
| Input | Grayscale is the internal working representation. Accept 8-bit color/grayscale PNG/JPEG/BMP/TIFF, 16-bit monochrome, and webcam/USB capture. Color is converted to luminance; native channel data is retained only for display. |
| Delivery | `egui` GUI + reusable `pm-core` Rust library + headless `pm-cli` + JSON/CSV model and result files. |
| Validation | Synthetic warp/occlusion/clutter benchmark initially. No real-image dataset is available yet; tuning parameters stay exposed. |
| Performance | CPU-only with `rayon`; no hard latency target yet. Stage timings and benchmarks are built in so budgets can be set after profiling. Live camera mode processes frames opportunistically. |

Assumed defaults that can change without architectural impact: Windows-first packaging,
one active model per run, no GPU implementation, no video-file playback in v1, and no
network/C-API service.

---

## 2. Product overview

### 2.1 What it does

1. The user loads a sample image.
2. The user draws a rotated rectangle around the object. This is the **model ROI**.
3. The user places a **zero location** — an origin and direction — that defines the
   object's canonical coordinate frame and therefore what “position and rotation” mean
   in the output.
4. The tool extracts local shape features (subpixel edges, corners, line segments) and
   displays them as thin colored lines/overlays.
5. The user paints exclusion masks over features that are not repeatable between
   instances of the object.
6. The model is saved.
7. On a test image (file or webcam frame) the same feature extraction runs, the tool
   searches for the model, refines the best candidates, estimates a homography where
   the evidence permits, and reports a list of detections with confidence scores.

### 2.2 Goals

- Robust detection of low-texture, high-edge industrial parts such as metal brackets.
- Tolerant to translation, full rotation, moderate scale change, lighting/contrast
  change, partial occlusion, and background clutter.
- Teach-by-example without writing code or tuning a classifier.
- Explainable output: every detection exposes the matched model features, score,
  diagnostics, and the homography used.
- Math-heavy core has a machine-checked specification and proof inventory.
- Reproducible: deterministic given a seed; identical results across runs.

### 2.3 Non-goals for v1

- Learned/deep features or classifiers.
- Full 6-DOF 3D pose or camera calibration.
- Texture/color-based recognition; the match is geometric.
- Multi-image model training, multiple simultaneously active models, video-file
  playback, GPU acceleration, C ABI, REST/TCP integration.
- Formal verification of floating-point rounding, image I/O, SVD numerics, or UI.
- Guaranteed real-time frame rate.

---

## 3. User workflows

### 3.1 Teach

1. **Load**: drag and drop an image onto the canvas, use the file dialog, or freeze a
   webcam frame.
2. **Draw ROI**: click-drag a rotated rectangle. Handles move, resize, and rotate it.
   The ROI defines which image region contributes model features.
3. **Place zero**: click to place the model origin and drag/angle a handle to set the
   model x-axis. The displayed crosshair/arrow is exactly the pose that will be
   reported at run time.
4. **Review features**: edges are drawn as thin lines colored by orientation, corners
   as small markers, and fitted segments. Sliders adjust blur, edge thresholds,
   corner sensitivity, and segment minimum length; the preview updates live.
   Corner descriptors and segment-pair invariants are derived automatically for the
   hybrid search.
5. **Mask**: select the polygon or brush tool, add/erase regions, and toggle the mask
   overlay. Masked features disappear from the model and are excluded from scoring.
6. **Save**: write a `.pmodel` file containing metadata, the model feature sets, the
   mask, and an embedded source image thumbnail. Reopening restores the teach view.

### 3.2 Run

1. Load a test image by drag/drop, file dialog, or webcam.
2. Optionally draw a rotated **search region** in the test image; the default is the
   whole image.
3. Set or accept search parameters: rotation range, scale range, maximum detections,
   score threshold, polarity handling, maximum occlusion, and hypothesis sources
   (Hough, descriptors/segments, or both).
4. Press **Run**. Detection executes on a worker thread with progress and cancel.
5. Inspect the detection list. Selecting a row highlights the transformed ROI,
   zero-frame axes, matched model edges, full homography, score, and diagnostics.
6. Export results as JSON or CSV, and optionally save an annotated overlay PNG.

### 3.3 Feature visual language

- **Edges**: 1-pixel polylines along linked subpixel edge chains; hue encodes normal
  orientation (HSV color wheel). An optional per-point mode draws short tangent strokes
  so every extracted edge point is visibly a thin colored line.
- **Corners**: small colored crosses or dots, distinct from the orientation hue map.
- **Segments**: thin cyan/teal lines fitted to edge chains.
- **Masked features**: hidden, or dimmed when “show masked” is enabled.
- **Detections**: transformed ROI outline, zero origin and axes, and matched model
  edge points in a distinct green/blue overlay. A label shows the confidence.
- A legend maps hue to orientation in degrees.

---

## 4. Coordinate systems and conventions

- **Image coordinates**: `x` right, `y` down; integer coordinates are pixel centers;
  image extent is `[0, W-1] × [0, H-1]`. Bilinear sampling clamps outside the image.
- **Rotation**: `R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]`. With `y` down, positive
  `θ` appears clockwise on screen. Angles are exposed in degrees in `(-180, 180]`.
- **Model frame**: origin at the user's zero location, x-axis at the user's zero
  angle, units equal to taught-image pixels. Scale `1.0` means the taught size.
- **Pose**: `q = t + s R(θ) p`, mapping model point `p` to image point `q`.
  The reported position is where the model zero lands; the reported angle is the
  model x-axis direction.
- **Homography**: a 3×3 invertible matrix `H` mapping homogeneous model points to
  homogeneous image points. The similarity is embedded as

  ```text
  H_sim = [[s cosθ, -s sinθ, tx],
           [s sinθ,  s cosθ, ty],
           [0,       0,       1]]
  ```

  Matrices are normalized for output (divide by `h33` when `|h33|` is non-negligible,
  otherwise by the Frobenius norm) and carry no absolute projective scale.

  For a general `H`, the local affine approximation at the model origin is

  ```text
  w  = h33,   qx = h13/w,   qy = h23/w
  J  = (1/w) · [[h11 - qx·h31, h12 - qx·h32],
                [h21 - qy·h31, h22 - qy·h32]]
  ```

  The reported zero position is `(qx, qy)`; the reported local angle is
  `atan2(J10, J00)` and the local scale is `sqrt(|det J|)`. For a non-similarity
  homography these are local quantities at the zero point, not global constants.
- **ROI**: a rotated rectangle stored as center, width, height, and angle in image
  coordinates. It selects training features; it is not the run-time search region.
- **Search region**: an optional rotated rectangle in the test image that constrains
  candidate translations. Default: the whole image.

The zero frame is applied once at teach time:

```text
p_model = R(-φ0) · (p_image - O)
```

where `O` is the zero origin and `φ0` the zero angle. Changing the zero location
changes the reported pose, never the extracted feature geometry.

---

## 5. System architecture

### 5.1 Workspace layout

```text
pattern-match/
├─ Cargo.toml                # workspace
├─ crates/
│  ├─ pm-core/               # algorithms; no UI dependencies
│  │  ├─ src/
│  │  │  ├─ lib.rs
│  │  │  ├─ gray.rs           # GrayF32 image, conversion, sampling
│  │  │  ├─ pyramid.rs
│  │  │  ├─ gradient.rs
│  │  │  ├─ edge.rs           # Canny, subpixel, chains
│  │  │  ├─ corner.rs         # Shi–Tomasi + subpixel
│  │  │  ├─ segment.rs
│  │  │  ├─ descriptor.rs     # corner shape context, segment-pair invariants
│  │  │  ├─ feature.rs        # feature records, masks, weights
│  │  │  ├─ overlay.rs        # feature overlay rendering
│  │  │  ├─ model.rs          # teaching, model I/O
│  │  │  ├─ geometry.rs       # similarity, homography, DLT, poses
│  │  │  ├─ search.rs         # Hough hypotheses, coarse-to-fine
│  │  │  ├─ score.rs          # scoring, confidence, NMS
│  │  │  ├─ refine.rs         # similarity ICP, RANSAC, LM homography
│  │  │  ├─ capture.rs        # FrameSource trait, image/webcam sources
│  │  │  └─ error.rs
│  │  ├─ tests/
│  │  └─ benches/
│  ├─ pm-ui/                  # eframe application binary
│  └─ pm-cli/                 # headless CLI
├─ lean/
│  ├─ lakefile.toml
│  ├─ lean-toolchain
│  ├─ lake-manifest.json
│  ├─ PatternMatch/
│  │  ├─ Basic.lean
│  │  ├─ Geometry.lean
│  │  ├─ DLT.lean
│  │  ├─ RANSAC.lean
│  │  └─ Golden.lean
│  └─ Main.lean               # emits golden-vector JSON
├─ testdata/
│  ├─ golden/                 # generated by Lean, committed
│  └─ synthetic/              # generated benchmark inputs
├─ models/
├─ scripts/
└─ README.md
```

### 5.2 `pm-core` module responsibilities

| Module | Responsibility |
|---|---|
| `gray` | Internal `GrayF32` image, color/16-bit conversion, bilinear sampling, ROI extraction, thumbnail generation. |
| `pyramid` | Anti-aliased Gaussian pyramid; level scaling and coordinate conversion. |
| `gradient` | Scharr/Sobel gradients, magnitude, orientation, polarity, structure tensor. |
| `edge` | Non-maximum suppression, hysteresis, subpixel localization, edge chaining. |
| `corner` | Shi–Tomasi/Harris response, NMS, subpixel refinement. |
| `segment` | Split-and-merge/RANSAC line fitting over edge chains. |
| `descriptor` | Rotation/scale-invariant corner shape-context descriptors and segment-pair invariants for hypothesis seeding. |
| `feature` | Feature records, weights, mask rasterization and exclusion. |
| `overlay` | Feature overlay rendering: hue-coded edge strokes, corner crosses, segment lines. |
| `model` | Build model at pyramid levels, validate, save/load `.pmodel`. |
| `geometry` | Similarity/homography algebra, DLT, normalization, pose extraction, degeneracy checks. |
| `search` | Oriented Hough voting, peak extraction, hypothesis generation and clustering. |
| `score` | Edge-distance/orientation scoring, coverage, confidence, pose NMS. |
| `refine` | Similarity ICP, correspondence assembly, RANSAC, robust homography LM. |
| `capture` | `FrameSource` trait, image-file source, webcam source (feature-gated). |

### 5.3 Dependencies

| Crate | Purpose |
|---|---|
| `image` | Decode PNG/JPEG/BMP/TIFF, including 16-bit monochrome. |
| `nalgebra` | 2D/3D transforms, SVD/eigen, small dense linear algebra. |
| `rayon` | Row/feature-level parallelism. |
| `serde`, `serde_json` | Metadata and result serialization. |
| `zip` | Single-file `.pmodel` container. |
| `thiserror`, `anyhow`, `tracing` | Errors, context, stage logging/timing. |
| `rfd` | Native file dialogs. |
| `eframe`, `egui` | Desktop UI, drag-and-drop, custom canvas. |
| `clap` | CLI parsing. |
| `nokhwa` (optional feature) | Windows Media Foundation webcam capture. |
| `criterion`, `proptest`, `approx` | Benchmarks, property tests, tolerant comparisons. |

No OpenCV. Image processing primitives are implemented in `pm-core` so subpixel
behavior and thresholds stay under our control.

### 5.4 Threading model

- UI thread owns `egui` and image textures.
- A detection worker owns the model and processes run requests; results return over a
  channel. A cancellation token interrupts long stages between pyramid levels and
  hypothesis batches.
- Webcam capture runs on its own thread and publishes the newest frame; the UI may
  freeze a frame for teaching.
- Live mode drops frames rather than queuing them. Detection runs on the latest frame;
  latency and processed-frame rate are displayed, not guaranteed.
- Determinism: all search loops are ordered; parallelism only splits independent
  reductions. RANSAC uses a seeded PRNG, exposed in parameters for reproducibility.

---

## 6. Image preprocessing

1. **Decode** with `image`. Preserve the original for display.
2. **Convert to grayscale `f32` in `[0,1]`**:
   - 8-bit color: `Y = 0.2126 R + 0.7152 G + 0.0722 B`, using the encoded channel
     values. (No linear-light conversion in v1; this is a repeatable edge detector,
     not a photometric measurement tool.)
   - 8-bit grayscale: divide by 255.
   - 16-bit grayscale: divide by 65535.
3. **Optional blur**: separable Gaussian, default `σ = 1.0` at full resolution.
4. **Pyramid**: levels `L = 0..L_max`, each built by Gaussian blur (`σ = 1.0`) then
   2× decimation. Default `L_max = 3` (image/8), extended only for large inputs.
5. **Gradients**: Scharr 3×3 on each level, giving `gx, gy`, magnitude, orientation,
   and sign. Gradients are computed in `f32`.
6. **Adaptive edge thresholds**: hysteresis high/low thresholds are derived from a
   robust percentile of the ROI/test-image gradient magnitude (defaults below),
   combined with absolute floors. This keeps behavior stable across exposure changes.
7. **Distance transform**: on the test image's edge support, compute an exact
   Euclidean distance transform (Felzenszwalb–Huttenlocher or two-pass chamfer with
   a preference for exact EDT) per pyramid level. Orientation at an edge location is
   sampled from the blurred gradient field.

All buffers are planar and row-parallel. Coordinate conversion between levels is
`q_L = q_0 / 2^L`; model features are stored in level-local coordinates as well as
level-0 coordinates.

---

## 7. Feature extraction

### 7.1 Edges

1. Non-maximum suppression along the gradient direction. Neighboring magnitudes are
   bilinearly interpolated so suppression is not restricted to four quantized angles.
2. Hysteresis thresholding; link surviving points into 8-connected chains.
3. **Subpixel localization.** For the three NMS responses `m₋₁, m₀, m₊₁` along the
   unit normal `n`, fit the parabola and take

   ```text
   δ = (m₋₁ - m₊₁) / (2 · (m₋₁ - 2m₀ + m₊₁))
   ```

   clamped to `[-0.5, 0.5]` when the denominator is safely non-zero and the vertex is
   a maximum. The edge point is `p = p₀ + δ n`. The point carries the unit normal
   `n`, signed gradient polarity, magnitude, and chain id.
4. Discard isolated points and chains shorter than the minimum length.

### 7.2 Corners

- Structure tensor over a Gaussian window (`σ` coupled to the edge blur) with
  response `λ_min` (Shi–Tomasi), plus Harris response available for diagnostics.
- Non-maximum suppression in a local window and an adaptive quality threshold.
- Subpixel corner refinement by a quadratic fit of the response surface, accepted only
  when the fit is concave and the offset is bounded.
- Each corner stores position, response, and dominant orientation.

Corners are primarily visualization and correspondence-seeding features. They do not
drive the coarse search in v1.

### 7.3 Line segments

- Fit segments to edge chains with split-and-merge or sequential RANSAC; reject short
  or high-curvature spans; subpixel endpoints.
- Segments are rendered, feed segment-pair hypothesis seeding, and are available as
  additional homography correspondences (endpoints/intersections).

### 7.4 Feature records

```text
EdgePoint  { x, y, nx, ny, magnitude, sign, chain_id, weight }
Corner     { x, y, response, orientation, weight }
Segment    { x0, y0, x1, y1, length, orientation, weight }
```

Model edge weights default to a normalized magnitude/contrast weight, capped to avoid
one high-contrast edge dominating. Masked or out-of-ROI points never enter a model.

### 7.5 Rendering

- Edge chains are drawn as 1-pixel polylines at display scale. Where chains are
  fragmented, each edge point can be drawn as a 3-pixel tangent stroke. Color is
  `HSV(hue = normal angle, s = 0.85, v = 0.95)`.
- Corners are small crosses in a fixed accent color.
- The orientation legend is always visible when the feature overlay is on.
- The display layer never mutates model data; it is generated from the current feature
  set and mask.

### 7.6 Masking

- Mask primitives are stored in **model coordinates**: polygons, and brush strokes
  with radius, each tagged add/erase.
- The UI rasterizes them into a binary exclusion mask at model resolution; features
  whose centers fall inside are removed from the model. Erasing a brush stroke
  removes previously added mask coverage.
- Undo/redo covers mask edits. An optional “show masked” mode renders excluded
  features dimly for debugging.
- The mask is applied per pyramid level (scaled geometry); total excluded weight is
  reported in the model summary.

---

## 8. Model building and persistence

### 8.1 Teaching transform

The source image is preprocessed and features are extracted. Features inside the ROI
are converted into the model frame; features outside are discarded. The model stores
both level-0 points and per-level downsampled point sets (or enough data to reconstruct
them deterministically from the embedded source image). The zero frame affects only
the stored coordinates and the reported pose convention.

### 8.2 Model summary

- Format/version, model name, creation timestamp.
- Source image metadata, ROI, zero origin/angle, extraction parameters.
- Per-level edge point arrays with normals, polarity, weights.
- Corner and segment arrays, multi-scale corner descriptors, and segment-pair
  invariants used for seeding.
- Mask primitives and rasterized mask hash.
- Default run parameters (rotation/scale range, thresholds, max instances).
- Feature counts and excluded-weight statistics.

### 8.3 `.pmodel` container

A ZIP archive with a versioned manifest and binary payloads:

```text
model.json          # version, metadata, ROI, zero frame, parameters, mask primitives
features.bin        # documented little-endian arrays: levels, edges, corners, segments, descriptors
mask.png            # optional rasterized mask (derived, for fast display)
source.png          # optional embedded source image / thumbnail
```

`model.json` is human-inspectable; `features.bin` avoids JSON bloat. The manifest
includes an ABI/version field and array counts. Loading rejects unknown major versions
and validates all counts and finite values. The embedded source image allows reopening
the teach view and regenerating pyramid data; a “strip source image” option is future
work.

---

## 9. Detection algorithm

### 9.1 Pipeline

```text
test image
  → grayscale + pyramid
  → gradients + edges + EDT per level
  → coarse similarity hypotheses (oriented Hough + corner-descriptor and
     segment-pair seeding)
  → hypothesis clustering / pruning
  → coarse-to-fine similarity verification and refinement
  → correspondence assembly
  → RANSAC homography + robust LM refinement
  → homography validation / similarity fallback
  → multi-instance NMS
  → detections + scores + diagnostics
```

The coarse stage (Hough plus descriptor/segment seeding) is responsible for recall.
The fine stages are responsible for accuracy, homography, and confidence.

### 9.2 Level selection

The coarsest search level `L_c` is the smallest level (most downsampled) at which:

- the model ROI's short side is at least `32` level pixels, and
- the model has at least `120` edge points (sampled/downsampled), if possible.

If no level satisfies this, search starts at level 0. Refinement proceeds from `L_c`
to `L_c - 1`, …, `0`. Coordinates convert as `q_0 = 2^L q_L`; the similarity scale `s`
is defined relative to the taught full-resolution size at every level, so a model
point measured in level-`L` pixels is transformed directly by `s R(θ)` at that level.

### 9.3 Coarse similarity hypotheses: hybrid Hough + descriptor seeding

For each coarse level and each discrete rotation `θ_b` and scale `s_b`, each test edge
point `e` with orientation `ψ_e` votes into a 2-D translation accumulator for every
model edge point `m` whose orientation matches after rotation:

```text
rotation gate:  wrap(ψ_e - ψ_m - θ_b) < Δθ
polarity gate:  sign(e) == sign(m)              # configurable
vote position:  v = e - s_b R(θ_b) m
vote weight:    w_m · w_e · orientation_kernel(ψ_e - ψ_m - θ_b)
accumulator[v] += vote weight
```

The accumulator coordinate is the image position of the model zero (the translation
of the similarity). It is padded by the maximum transformed model radius so objects
partly outside the image can still be proposed; translations outside the optional
search region are ignored.

Details:

- Rotation bins: default `B_θ = 72` over the configured range (5° nominal), with a
  smooth orientation kernel rather than hard binning; orientation histograms for model
  and test points make pair lookup `O(1)` per orientation bin.
- Scale bins: default `B_s = 8` log-spaced over `[s_min, s_max]`
  (`s_b = s_min · (s_max/s_min)^(b/(B_s-1))`).
- Accumulators are reused per `(θ_b, s_b)` to bound memory; peaks are local maxima in a
  `3×3` neighborhood above an adaptive fraction of the strongest peak. The top few
  peaks per bin are emitted, subject to a global hypothesis cap.
- Complexity is `O(N_test · N_model · B_s)` votes plus accumulator scans; orientation
  gating means each model/test point pair contributes to one rotation bin. At coarse
  levels this is the intended budget. Verification at finer levels dominates accuracy,
  not this vote count.

Each Hough hypothesis is `(tx, ty, θ, s, vote_score, level, source)`. Translation is
converted to full-resolution image coordinates for scoring and output.

#### Corner-descriptor seeding

Corners carry rotation/scale-invariant local shape descriptors. A descriptor combines
a shape-context histogram of nearby edge points with a gradient-orientation histogram:

- log-polar bins around the corner (default 5 radius bins × 12 angular bins), with
  angles measured relative to the corner's dominant orientation, plus 8 orientation
  bins weighted by edge magnitude;
- computed at 2–3 descriptor radii (e.g. 12/24/48 px at taught scale) or normalized
  by local edge spacing so matches tolerate scale change;
- L1-normalized and concatenated.

Matching applies a ratio test (best/second-best descriptor distance < 0.8), mutual
nearest neighbors, and a consistency check that the implied relative rotation and
scale agree with corner orientations and pyramid levels. Each surviving match
contributes a similarity seed:

```text
θ = orientation(test corner) - orientation(model corner)
s = descriptor-radius ratio × pyramid-level ratio
t = q_test - s R(θ) p_model
```

Descriptor matching is capped (default 300 model corners and 1500 test corners per
level) and parallelized. Brute force is acceptable at these caps; a kd-tree is added
only if profiling requires it. Descriptor matches are hypotheses, never accepted
detections: they pass through the same edge scoring and refinement as Hough peaks.

#### Segment-pair seeding

Line segments provide a complementary, edge-native seed source for parts with strong
straight features:

- pair invariants invariant to similarity: length ratio, angle between directions,
  midpoint distance normalized by combined length, and midpoint-vector angle relative
  to the segment directions;
- bucket model and test segment pairs by these invariants and match within
  tolerances;
- a matched non-parallel pair determines rotation (direction difference), scale
  (length ratio), and translation (midpoint alignment); near-parallel pairs are
  rejected for scale/rotation determination.

Pair counts are capped (e.g. 500 pairs per image), and matched pairs are deduplicated
before verification.

#### Fusion

Hough and descriptor/segment hypotheses are merged into one candidate list and
clustered in §9.4. The UI/CLI exposes
`hypothesis_sources = hough | descriptors | both` (default `both`). Hough is the
primary generator; descriptor/segment seeds recover objects whose edges are weak,
partially occluded, or poorly quantized by Hough bins. Each detection records which
source produced its winning hypothesis. False descriptor seeds are controlled by the
ratio test, mutual consistency, and the shared verification score.

### 9.4 Hypothesis clustering

Before expensive scoring, hypotheses from Hough, corner descriptors, and segment pairs
are deduplicated in pose space: translation within a fraction of the model radius,
angle within a few degrees, and scale within a small ratio. Representatives keep the
best source-specific seed score. Clustering reduces the number of candidates to
roughly 50–300 per image, which are then scored coarse-to-fine. The winning
hypothesis records its source for diagnostics.

### 9.5 Scoring

Let `P` be model edge points with normalized weights `w_i`, transformed by pose `Pose`
or homography `H` into image points `q_i` and transformed normals `n_i'`. For
similarity, `n_i' = R(θ) n_i`. For a homography, transform the homogeneous line
coefficients (`l' = H⁻ᵀ l`, then take and normalize the first two components), or
equivalently use the inverse-transpose of the local Jacobian at `q_i`; a bare 2-vector
normal is not simply multiplied by `H⁻ᵀ`. For each point inside the image:

```text
d_i     = bilinear(EDT, q_i)                       # distance to nearest test edge, px
g_i     = bilinear(test gradient, q_i)
a_i     = max(0, cos(angle(n_i') - angle(g_i))) · polarity_match · reliability(|g_i|)
quality = w_i · max(0, 1 - d_i / τ_d) · a_i^p
score   = Σ quality / Σ_{visible} w_i
coverage= Σ_{visible} w_i / Σ_all w_i
```

Defaults: `τ_d ≈ 2.5 px` at full resolution (scaled per level), `p = 2`, and
`reliability` ramps from 0 to 1 over a small gradient-magnitude band. Points outside
the image contribute zero and are excluded from the denominator; a minimum `coverage`
(default 0.4, configurable) prevents tiny visible fragments from scoring highly.
Occluded points contribute zero, so the score falls with occlusion but tolerates it
by threshold rather than hard rejection. Polarity can be disabled for parts whose
contrast reverses.

The score is deliberately a normalized fraction, not a chance-corrected statistic;
confidence augmentation and diagnostics are described in §9.9.

### 9.6 Similarity refinement

Starting from a coarse hypothesis, refinement proceeds level by level:

1. **Correspondence search.** For each model point, after applying the current pose,
   search along the transformed normal within `±τ_search` (default 3–4 level pixels)
   for the strongest test-edge response that agrees in orientation and polarity.
   Parabola-interpolate the response along the normal for a subpixel correspondence.
2. **Weighted fit.** Solve the weighted similarity least-squares problem mapping model
   points to correspondences (Umeyama/Procrustes with a similarity constraint).
   Robust weights reject correspondence outliers via a Cauchy/Huber-like weight.
3. Iterate 3–5 times, shrinking `τ_search` at each level.
4. If too few reliable correspondences exist, fall back to a small pattern search over
   `(tx, ty, θ, s)` maximizing the §9.5 score, which is slower but less assumption
   bound.

At full resolution the pose and its §9.5 score are finalized. The best-scoring
candidates proceed to homography estimation.

### 9.7 Homography estimation and refinement

A similarity search is robust but cannot represent perspective. The refinement stage
tries to upgrade each accepted candidate to an 8-DOF homography and explicitly decides
whether the evidence supports it.

1. **Correspondence assembly.**
   - Edge correspondences from the final normal-directed search (model point ↔
     subpixel test edge point), weighted by orientation agreement and fit residual.
   - Corner correspondences: transformed model corners ↔ nearby test corners within a
     gating radius, filtered by orientation/response compatibility.
   - Optional segment endpoint/intersection correspondences.
2. **RANSAC.** Sample 4 correspondences; reject samples with any three model points
   collinear (a hard geometric degeneracy); fit a homography with normalized DLT;
   count inliers by symmetric transfer error or normal-direction residual within a
   pixel threshold (default 2.5 px full resolution). Iteration count follows §10.3’s
   formula, capped, with a fixed seed for reproducibility. The best model is refit on
   all inliers.
3. **Robust LM refinement.** Re-associate correspondences under the current `H` and
   minimize a robust sum of normal-direction residuals `ρ(⟨q_i - c_i, n_i'⟩)` using
   Levenberg–Marquardt on the 8 normalized homography parameters, interleaved with
   2–3 re-associations. This uses edge geometry, not just the sampled correspondences.
4. **Validation.** Reject `H` if:
   - the determinant is near zero, a mapped ROI corner has non-positive homogeneous
     depth, or the numerical condition number exceeds a threshold;
   - fewer than the minimum inliers/correspondences support it;
   - it fails to improve the §9.5 score over the similarity pose by a margin
     (default `Δscore ≥ 0.01`), or the perspective terms are not stable across
     re-association.
5. **Fallback.** If rejected, the reported homography is the similarity embedding
   `H_sim`, with `homography_estimated = false` and reduced confidence. This is
   intentional: on a small, nearly fronto-parallel object an 8-DOF fit is often
   ill-conditioned, and claiming perspective from noise is worse than returning a
   stable similarity.

This stage “attempts to find homographies” as specified, while making the attempt and
its support auditable.

### 9.8 Multi-instance detection and NMS

Hough peaks naturally propose multiple translations for the same rotation/scale, and
multiple objects yield multiple peaks. After verification:

1. Sort detections by score.
2. Suppress a candidate if, relative to a higher-scoring accepted detection, its zero
   position is within a fraction of the model diagonal, its angle is close, and its
   scale is close; or if the transformed ROI overlap exceeds a threshold. Thresholds
   are configurable.
3. Stop at the maximum instance count (default 10) or when the score falls below the
   detection threshold.

Closely overlapping physical objects remain a deliberate user decision: increasing
the overlap allowance permits nested/overlapping matches, while the default favors
one detection per occurrence.

### 9.9 Confidence

Every detection reports:

- `score`: the §9.5 weighted match fraction in `[0,1]`.
- `coverage`: visible model weight fraction.
- `matched_fraction`: model weight with `quality > 0` divided by visible weight.
- `mean_distance`: weighted mean edge distance of matched points (px).
- `seed_source`: `hough`, `corner_descriptor`, or `segment_pair`.
- `inliers`, `correspondences`, `ransac_iterations`.
- `homography_estimated`, homography condition number, and failure reason if any.
- `ambiguity`: margin to the best competing detection that overlaps it.

The initial `confidence` is a documented heuristic:

```text
confidence = clamp01(0.55·score + 0.20·matched_fraction
                     + 0.15·mean_orientation_agreement + 0.10·ambiguity)
             · perspective_factor
```

with `perspective_factor = 1.0` when a validated homography is used and a small
penalty (e.g. 0.85) for a similarity fallback. This is explicitly provisional. Once
synthetic (and later real) data exists, a logistic calibration maps these diagnostics
to an observed correctness probability; the coefficients are stored in configuration
and versioned. Raw diagnostics always remain available so the score is auditable.

### 9.10 Limits and failure behavior

- Weak-texture or badly blurred objects can produce insufficient edge support; the
  detector reports low confidence rather than inventing a pose.
- Strong perspective can move the object outside the similarity basin used by the
  coarse stage. Such cases may be missed; future work adds homography hypothesis
  seeding. The fallback behavior prevents false precision on marginal cases.
- Symmetric or repeated patterns can yield multiple valid peaks; the ambiguity
  diagnostic exposes this and the user chooses thresholds or masks distinguishing
  features.
- Partial objects at the image border are handled by coverage gating and padded
  accumulators; detections below the coverage minimum are rejected.
- If the test image has inverted contrast or a very different edge polarity policy,
  the polarity gate is configurable.

---

## 10. Lean formalization

### 10.1 Purpose and workflow

Lean is the authoritative specification for the math-heavy core. For each covered
module the order is:

1. Define the objects and operations in Lean.
2. State the relevant theorems with their hypotheses.
3. Prove them (no `sorry` in committed code).
4. Emit executable golden vectors from the same definitions.
5. Implement the Rust counterpart and assert conformance in Rust tests.

Lean is **not** used at run time. There is no FFI in the hot path. Rust remains the
implementation; Lean definitions are executable where practical, and golden vectors
catch drift between spec and code.

Toolchain: Lean 4.35.0 via `elan`, with the Mathlib release/commit that supports that
toolchain pinned in `lean-toolchain` and `lake-manifest.json`. If a compatible Mathlib
release is unavailable, the toolchain is adjusted via `elan` to the pinned Mathlib
toolchain; this is a setup concern, not a design change. First build is expected to be
long; CI caches `.lake`.

Non-goals of the formalization: floating-point rounding behavior, the SVD numerical
implementation, image sampling/JPEG decoding, Hough-voting correctness, descriptor
matching, and UI. Those are covered by Rust tests and benchmarks.

Proof scope is staged. The core guarantees in §10.2–§10.4 — G1–G8, D1–D5, and
R1–R5 — are required before the corresponding Rust code is considered complete. D6
(noisy least-squares optimality) is a documented follow-on: it is not stated as a
committed Lean theorem or included in the no-`sorry` gate until its proof is
supported by the available Mathlib API. Rust's SVD/normal-equation behavior is tested
numerically regardless.

### 10.2 Geometry module

Definitions: homogeneous points, 3×3 homography (invertible matrix), similarity
parameters and embedding, action with affine normalization, cross-ratio on a
projective line, local Jacobian at the origin.

| ID | Theorem | Proof approach | Risk |
|---|---|---|---|
| G1 | Homography action is composable: `(H₁H₂)·p = H₁·(H₂·p)`. | Matrix algebra and homogeneous normalization. | Low |
| G2 | Projective scaling invariance: `(cH)·p = H·p` for `c ≠ 0` wherever the action is defined. | Cancel the common non-zero scalar. | Low |
| G3 | Invertibility: `H⁻¹·(H·p) = p`; inverse action is well defined. | `GL(3,ℝ)` group laws. | Low |
| G4 | Similarities form a subgroup of homographies; composition is `(s₁s₂, θ₁+θ₂, t₁+s₁R(θ₁)t₂)`. | Matrix identity and trigonometric addition formulas. | Low |
| G5 | Similarity decomposition: any non-zero block `[[a,-b],[b,a]]` has `s=√(a²+b²)>0`, `cosθ=a/s`, `sinθ=b/s`; the parameter pair is unique up to `2π`. | Positivity of the norm, `cos²+sin²=1`, periodicity lemmas. | Medium |
| G6 | Cross-ratio `CR(x₁,x₂;x₃,x₄)=((x₁-x₃)(x₂-x₄))/((x₁-x₄)(x₂-x₃))` is invariant under `x ↦ (ax+b)/(cx+d)` when `ad-bc≠0` and all denominators are non-zero. | Difference identity `f(x)-f(y)=(ad-bc)(x-y)/((cx+d)(cy+d))`; cancellation. | Medium |
| G7 | Homographies preserve collinearity: `det[p,q,r]=0 ⇒ det[Hp,Hq,Hr]=0`. | Multiplicativity of determinant (`det(H)·det[p,q,r]`). | Low–medium |
| G8 | Local Jacobian of the affine action at the origin, used to extract position/angle/scale from `H`. | Differentiate the rational map; evaluate at the origin; establish non-zero denominator hypotheses. | Medium |

The cross-ratio theorem is included deliberately: it is the invariant that justifies
using projective geometry for planar shape matching, and it exercises the field
reasoning used by the DLT proofs.

Golden vectors for geometry: rational matrices and points, compositions, inverses,
similarity decomposition, and cross-ratio examples, including degenerate cases with
non-zero denominators.

### 10.3 DLT module

For correspondences `(x_i,y_i) ↔ (u_i,v_i)`, each pair contributes two rows:

```text
[-x  -y  -1   0   0   0   ux  uy  u]
[ 0   0   0  -x  -y  -1   vx  vy  v]
```

Stack `A ∈ ℝ^(2n×9)`; a vector `h` in `ker A` is a candidate homography (up to scale).
Rust uses normalized DLT (Hartley) and an SVD; Lean formalizes the algebra and the
properties that justify it.

| ID | Theorem | Proof approach | Risk |
|---|---|---|---|
| D1 | For exact correspondences, `vec(H)` satisfies `A·vec(H)=0`. | Expand both rows; ring simplification. | Low |
| D2 | For four correspondences, `A` is `8×9`, so its kernel is non-trivial. | Rank–nullity. | Low |
| D3 | If four source points are in general position (no three collinear) and four target points are in general position, then `rank A = 8`, `ker A` is one-dimensional, and the non-zero kernel vector is an invertible homography; the solution is unique up to scale. | Projective-basis argument: three source vectors form a basis, the fourth has all-non-zero coefficients, and the target general position makes the induced linear map invertible; kernel uniqueness follows. | High |
| D4 | If three source points are collinear, the four-point configuration is degenerate: the DLT system does not determine a unique homography (`rank A < 8` / kernel dimension at least 2 for consistent data). | Linear dependence of the projective constraints; construct a family or rank bound. Implementation rejects such samples regardless. | Medium–high |
| D5 | Hartley normalization: with invertible normalization transformations `T_model, T_image`, the denormalized solution `H = T_image⁻¹ H_norm T_model` satisfies the original correspondences exactly; solving in normalized coordinates is equivalent up to scale. | Substitute normalized points into the DLT equations; invertibility/cancellation. | Medium |
| D6 (follow-on) | Algebraic least squares: for exact rank-8 data, the kernel vector minimizes `‖Ah‖²` over the unit sphere with value 0 and is unique up to sign; for full-rank noisy data, any unit minimizer satisfies the normal equations `AᵀA h = λh` with `λ = ‖Ah‖²`. | Zero-error case directly; noisy case via compactness of the finite-dimensional unit sphere and the Lagrange multiplier/normal-equation characterization. If Mathlib’s spectral API is insufficient, prove the critical-point characterization and full-rank case, and retain the spectral existence statement as documented future work. | Deferred |

The high-risk item is D3. The proof strategy is a projective-frame/linear-algebra
argument rather than expanding an explicit 8×8 determinant: this keeps the formal
development smaller and matches why four non-collinear correspondences determine a
homography. A research spike is scheduled before committing the full proof; if it
reveals an unexpected Mathlib gap, the fallback is to formalize the kernel/rank
characterization with explicit coordinates and retain D3’s uniqueness statement for
four-point exact data with the same linear-map proof.

D6 is intentionally outside the initial no-`sorry` gate. The core guarantees D1–D5
cover exact DLT, rank/uniqueness, degeneracy, and normalization. If the Mathlib
spectral/calculus API cannot support the noisy-case existence proof within a
reasonable spike, D6 stays a documented follow-on rather than weakening the proven
core; the Rust implementation is numerically tested against synthetic data in the
meantime.

### 10.4 RANSAC module

RANSAC is used for homography fitting from four-point minimal samples. Lean formalizes
sampling probabilities without replacement, the conservative bound used to choose
iteration counts, and monotonicity.

| ID | Theorem | Proof approach | Risk |
|---|---|---|---|
| R1 | For `n` correspondences with `I` inliers, the probability that `k` distinct uniformly sampled correspondences are all inliers is `q = C(I,k)/C(n,k)` (with `C(I,k)=0` if `k>I`). | Combinatorial counting. | Low–medium |
| R2 | `C(I,k)/C(n,k) ≤ (I/n)^k` for `0 < I ≤ n` and `1 ≤ k ≤ n`. | Product form `∏(I-j)/(n-j) ≤ (I/n)^k`. | Medium |
| R3 | For `N` independent iterations, `P(success) = 1 - (1 - q)^N`. | Independence and complementary events. | Low |
| R4 | The success probability is monotone non-decreasing in `N` and inlier count `I`; it is monotone non-increasing in sample size `k`. | Powers with bases in `[0,1]`; combinatorial monotonicity. | Low–medium |
| R5 | For `P∈(0,1)`, `q∈(0,1)`, if `N ≥ ⌈log(1-P)/log(1-q)⌉`, then `1-(1-q)^N ≥ P`; using R2, the same with `q ≤ (I/n)^k`. | Real log inequalities and monotonicity. | Medium |

The Rust implementation computes required iterations from the conservative estimate
(robust when the inlier ratio is uncertain) and caps it; the exact formula is retained
for diagnostics. Golden vectors cover small combinatorial cases and iteration counts.

### 10.5 Golden vectors and CI

`lean/Main.lean` imports the development, computes a versioned set of reference
vectors, and writes JSON:

```text
lake exe golden > testdata/golden/vectors.json
```

Content:

- Geometry: matrices/points as exact rationals, transformed points, composition,
  inverse, decomposition, cross-ratio.
- DLT: exact correspondences from known rational homographies; kernel solution
  normalized to a canonical scale; rank/kernel dimension for degenerate cases;
  normalized-vs-denormalized equivalence.
- RANSAC: exact rational probabilities for small `n, I, k, N`, conservative bounds,
  and iteration counts as decimal values with documented tolerance.

Rust tests deserialize the vectors and compare:

- exact/near-exact rational values converted to `f64` with a relative tolerance;
- projective solutions compared after canonical scaling and sign fixing;
- probabilities with a tight absolute tolerance.

CI checks, in order:

1. `lake build` (no `sorry` in the core modules G1–G8, D1–D5, R1–R5; a script greps
   and fails). D6 is excluded from this gate until promoted.
2. Regenerate golden vectors; fail if the committed file differs.
3. `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace`.
4. A quick synthetic benchmark smoke test and release-mode timings.

A theorem is not considered done until its Rust counterpart is linked in code
comments/tests by theorem ID (`G3`, `D5`, `R5`, …), so the spec-to-implementation map
is navigable.

---

## 11. UI specification

### 11.1 Layout

```text
┌───────────────────────────────────────────────────────────────┐
│ Toolbar: mode | open image | open model | save model | run    │
├───────────────┬───────────────────────────────┬───────────────┤
│ Left panel    │ Canvas (pan/zoom)              │ Right panel   │
│ - image/model │ - image + feature overlay      │ - parameters  │
│ - tools       │ - ROI / zero / mask editing    │ - feature     │
│ - mask list   │ - detection overlay            │   counts      │
│               │                                │ - results     │
├───────────────┴───────────────────────────────┴───────────────┤
│ Status: cursor px · zoom · elapsed ms · model · messages      │
└───────────────────────────────────────────────────────────────┘
```

### 11.2 Teach mode

- Select **ROI**; drag to create, move inside to translate, use corner/edge handles to
  resize, rotation handle to rotate. Numeric fields mirror the geometry. `Enter`
  commits, `Esc` cancels.
- Select **Zero**; click to place the origin, then drag the direction handle to set the
  x-axis. The exact reported pose convention is previewed.
- **Features** section: blur sigma, edge high/low thresholds (absolute + percentile),
  minimum chain length, corner sensitivity/min distance, segment minimum length;
  toggles for edges/corners/segments/mask; orientation legend.
- **Mask** section: polygon tool (click vertices, double-click/Enter closes),
  brush tool (radius slider, add/erase), undo/redo, clear, show-masked toggle.
- **Save model**: name, path, embedded-source-image option, and a model summary with
  edge/corner/segment/descriptor counts.

### 11.3 Run mode

- Load test image or switch to live camera.
- Optional search region tool (rotated rectangle) in the test image.
- Search parameter panel: rotation min/max, scale min/max, max detections, score
  threshold, coverage minimum, polarity on/off, search level (auto/manual),
  hypothesis sources (Hough / descriptors / both), descriptor caps, RANSAC
  confidence/seed.
- Run/cancel. Progress shows the active stage (preprocess, edges, Hough/descriptor
  seeding, verify, refine, homography).
- Results table: id, x, y, angle°, scale, score, confidence, seed source,
  homography-estimated. Selecting a row highlights it in the canvas and opens a
  details pane with the full matrix and diagnostics.
- Export JSON, export CSV, save annotated overlay.

### 11.4 Live camera

- Device enumeration and selection; resolution/frame-rate selection where supported.
- Freeze frame → teach from it, or run continuously.
- Live overlay uses the latest detection; processed-frame rate and end-to-end latency
  are shown. Frames are dropped as needed to keep the UI responsive.
- First version supports webcam/USB via `nokhwa`’s Windows Media Foundation backend
  behind the `camera` feature. If that backend proves incompatible, `capture.rs` is the
  abstraction boundary and a `windows` crate Media Foundation implementation can
  replace it without touching the core.

### 11.5 Interaction conventions

- Pan: middle-drag or space-drag. Zoom: wheel centered on cursor. Fit: `F`.
- Selection: left click; multi-select not required in v1.
- Undo/redo: `Ctrl+Z` / `Ctrl+Y` for mask and teach-geometry edits.
- All long operations are cancellable where safe; the UI never blocks on detection.

---

## 12. CLI, library API, and result formats

### 12.1 Library API (indicative)

```rust
use pm_core::{Model, SearchParams, GrayImage, detect};

let model = Model::load("models/bracket.pmodel")?;
let image = GrayImage::from_path("test.png")?;
let params = SearchParams::default(); // ranges, thresholds, seed
let result: DetectionResult = detect(&model, &image, &params)?;

for det in result.detections {
    println!("{:?} score={:.3} conf={:.3}", det.pose, det.score, det.confidence);
}
```

The public core API deals in `GrayF32`, model/result structs, and plain numeric
parameters. It is independent of `egui` and of file dialogs. UI and CLI are clients of
the same API.

### 12.2 CLI

```text
pm-cli run   --model M.pmodel --image test.png --output results.json [--overlay out.png]
pm-cli batch --model M.pmodel --input images/ --output-dir results/ --recursive
pm-cli model inspect --model M.pmodel
pm-cli bench synthetic --model M.pmodel [--cases 200] [--seed 42]
```

`run` also accepts `--params params.json` for full parameter override and `--timings`
for stage timings. Exit code 0 means the tool ran; detections may be empty. A separate
`--fail-on-none` flag makes empty results exit non-zero for automation.

### 12.3 Results JSON (versioned)

```json
{
  "format": "pattern-match-results",
  "version": 1,
  "model": "bracket-a",
  "image": { "width": 1280, "height": 1024 },
  "parameters": { "...": "..." },
  "timings_ms": { "preprocess": 12.4, "features": 18.1, "search": 90.2, "refine": 22.0 },
  "detections": [
    {
      "id": 0,
      "pose": { "x": 412.35, "y": 508.10, "angle_deg": 12.4, "scale": 0.987 },
      "homography": [[0.985, -0.216, 412.35],
                     [0.217, -0.988, 508.10],
                     [0.000012, -0.000004, 1.0]],
      "homography_estimated": true,
      "score": 0.873,
      "confidence": 0.841,
      "diagnostics": {
        "coverage": 0.94,
        "matched_fraction": 0.91,
        "mean_distance_px": 0.62,
        "mean_orientation_agreement": 0.95,
        "seed_source": "hough",
        "correspondences": 184,
        "inliers": 151,
        "ransac_iterations": 87,
        "homography_condition": 12500.0,
        "ambiguity": 0.44,
        "fallback_reason": null
      }
    }
  ]
}
```

CSV export flattens one row per detection with columns:
`id, seed_source, x, y, angle_deg, scale, h00..h22, homography_estimated, score,
confidence, coverage, matched_fraction, mean_distance_px, inliers, correspondences`.

The model format and results format both carry a semantic version; readers reject
unknown major versions and ignore unknown minor fields.

---

## 13. Testing and validation

### 13.1 Unit and property tests

- Geometry: random invertible homographies, composition/inverse round trips, normal
  transformation under similarity/homography, pose extraction from known matrices.
- DLT: exact recovery of random homographies from 4+ correspondences after
  normalization; degeneracy rejection (collinear triples); sign/scale canonicalization.
- RANSAC: required-iteration formula, exact and conservative probabilities, seeded
  reproducibility, all-outlier and high-outlier cases.
- Edges: synthetic anti-aliased step/ramp/line images with known subpixel position;
  NMS and hysteresis behavior; chain linking.
- Corners: synthetic corners/checkerboards; subpixel error bounds.
- Descriptors and segments: rotation/scale invariance of corner shape context,
  ratio-test behavior, segment-pair invariant matching, and pose-seed accuracy under
  synthetic rotation/scale.
- Masking: feature exclusion and per-level scaling.
- Serialization: round-trip model/result files; version rejection; malformed input.

### 13.2 Golden-vector tests

Rust tests load `testdata/golden/vectors.json`, compare against the Lean-generated
values, and fail with a diagnostic that names the theorem/vector. Regeneration is a
deliberate command, not automatic in tests; CI verifies the committed file is current.

### 13.3 Synthetic benchmark

Since no real dataset is available yet, build a deterministic generator:

- **Warping**: take a taught source image and apply random similarity or homography
  transforms with known parameters; bilinear/anti-aliased sampling.
- **Appearance**: brightness/contrast changes, Gaussian blur, additive noise, JPEG
  compression artifacts.
- **Clutter**: paste procedural textures or other synthetic parts into the background.
- **Occlusion**: random rectangles/circles over the object.
- **Scenes**: zero to many instances, border crops, inverted contrast,
  scale/rotation extremes, and low-contrast/sparse-edge cases that exercise the
  descriptor/segment seeds.
- **Analytic cases**: drawn L-brackets, plates with holes, and grid/checkerboard
  targets for subpixel and corner ground truth independent of the warp source.

Metrics:

- Detection rate at a fixed confidence threshold.
- False detections per image and per scene.
- Position error of the zero location (px), angle error (°), scale error (%).
- Homography error: maximum reprojection error of the model ROI corners.
- Fraction of detections with an accepted homography vs similarity fallback.
- Confidence calibration: reliability curve of confidence vs correctness.

### 13.4 Provisional acceptance targets

These are initial goals, revisable after profiling and real data:

| Scenario | Target |
|---|---|
| Clean synthetic similarity, full 360°, 0.5–2× | ≥ 99% detection; zero-location error ≤ 0.3 px; angle ≤ 0.2°; scale ≤ 0.3% |
| 30% occlusion | ≥ 90% detection at default confidence |
| Sparse/low-contrast edges | Hybrid detection ≥ 85%; Hough-only recall reported alongside for comparison |
| Perspective tilt ≤ 20° | ≥ 90% detection; ROI-corner reprojection ≤ 1.5 px; homography accepted in ≥ 80% of correct detections |
| Clutter scenes | ≤ 0.2 false detections/image at default threshold |
| Border-clipped object | detected if ≥ 50% model weight visible and coverage gate satisfied |
| End-to-end teach/reload/run | identical feature counts and results before/after reload |

The benchmark reports all metrics rather than only pass/fail, so tuning regressions are
visible.

### 13.5 Performance

No hard target is set. `criterion` benches cover preprocessing, edge/corner/segment
extraction, descriptor matching, EDT, Hough, verification, and refinement on
representative image sizes.
The CLI `--timings` flag and a UI stage timer expose the same data. Optimization order
after correctness: coarse-level work reduction, distance-transform reuse, hypothesis
pruning, and parallel reductions. GPU acceleration is explicitly future work.

---

## 14. Milestones

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| M0: Scaffold | Cargo workspace, `pm-core`/`pm-ui`/`pm-cli` skeletons, Lean Lake project, scripts. | `cargo build`, `lake build` skeleton, formatting/clippy clean. |
| M1: Lean geometry + Rust geometry | `Geometry.lean` definitions/proofs G1–G8; Rust similarity/homography/DLT-free geometry; golden vectors. | No `sorry` in `Geometry.lean`; golden tests pass; group/inverse properties tested. |
| M2: Image pipeline and features | Grayscale/color/16-bit I/O, pyramid, gradients, subpixel edges, corners, segments, feature overlay. | Synthetic subpixel/corner accuracy tests pass; overlay renders thin colored features. |
| M3: Teach mode and model I/O | Rotated ROI, zero frame, parameter panel, mask tools, corner descriptors/segment records, `.pmodel` save/load. | Teach → save → reload reproduces features, descriptors, and mask exactly. |
| M4: Lean DLT/RANSAC + Rust estimation | Core proofs D1–D5 and R1–R5; Rust normalized DLT, rank checks, RANSAC, iteration logic; golden tests. D6 follows later. | No `sorry` in core modules; exact recovery/degeneracy tests pass; RANSAC probabilities conform. |
| M5: Hybrid similarity search | Pyramid Hough, corner-descriptor and segment-pair seeding, hypothesis clustering, scoring, ICP/pattern refinement, multi-instance NMS. | Synthetic similarity benchmark meets provisional targets; Hough-only and hybrid recall both reported. |
| M6: Homography refinement | Correspondence assembly, RANSAC H, robust LM, validation/fallback, confidence/diagnostics. | Tilt benchmark meets targets; fallback behavior validated; JSON schema complete. |
| M7: CLI and benchmark harness | `run`/`batch`/`inspect`/`bench`, JSON/CSV, overlay export, deterministic seeds. | CLI reproduces library/UI results; batch outputs valid files. |
| M8: Webcam/live + packaging | Webcam source, freeze-frame teaching, live run, Windows release build. | Webcam frame can be taught and detected live; UI remains responsive; packaged exe launches. |
| M9: Hardening and docs | Parameter tuning, error handling, file validation, performance profiling, user guide. | CI green; acceptance report; known limitations documented. |

Lean modules M1 and M4 are deliberately placed before the corresponding Rust math.
Feature and UI work (M2/M3) can proceed in parallel with proof work because it does not
depend on the formalization. The search stage (M5) depends on geometry, scoring, and
models; homography refinement (M6) depends on DLT/RANSAC. D6 is scheduled as a proof
follow-on after M6; it does not block the detector because the Rust SVD path is tested
numerically.

---

## 15. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| DLT uniqueness proof (D3) is harder than expected in Mathlib. | Blocks the “proven” completion criterion for DLT. | Proof spike first; use the projective-basis argument rather than an 8×8 determinant expansion; keep explicit-coordinate fallback. Do not weaken the theorem silently: if it must be deferred, record it as an explicit open item. |
| Least-squares optimality (D6) depends on spectral theory availability. | DLT numerical story less formally complete. | D6 is explicitly a follow-on; core D1–D5 remain proven and gate completion. Prove zero-error/rank-8 uniqueness now; add noisy-case optimality when Mathlib supports it, with independent Rust numerical tests. |
| Mathlib/toolchain incompatibility or long builds. | Slow setup/CI. | Pin `lean-toolchain` and `lake-manifest.json`; use `elan` to switch to Mathlib’s supported toolchain; cache `.lake`. |
| Hough peaks are too broad or ambiguous on symmetric parts. | Extra false candidates, missed instances. | Orientation/polarity gating, smooth kernels, multi-peak extraction, verification scoring, ambiguity diagnostics, configurable NMS. |
| Descriptor/segment seeds add false correspondences and compute cost. | Extra hypotheses, possible runtime or score inflation. | Ratio test, mutual consistency checks, strict caps, and the shared edge-based verification; Hough remains primary and source toggles allow diagnosis. |
| Perspective fit is ill-conditioned on small/near-planar objects. | Bogus homographies. | Strict conditioning and score-improvement gates; explicit similarity fallback with reduced confidence; report `homography_estimated`. |
| Low-texture parts yield sparse edges. | Low recall. | Adaptive thresholds, contrast weighting, hybrid descriptor/segment seeding, lower resolution options; design accepts confidence degradation. |
| Webcam crate/backend issues on Windows. | No live mode. | `FrameSource` abstraction; fallback to Media Foundation via the `windows` crate; still-image mode remains fully functional. |
| Synthetic-only tuning does not generalize. | Poor real-world accuracy. | Keep all thresholds exposed; benchmark harness supports immediate real-image evaluation; plan a tuning phase when images arrive. |
| Confidence heuristic is not calibrated. | Misleading scores. | Report raw score + diagnostics; mark confidence provisional; add logistic calibration once labeled data exists. |
| No fixed performance target hides a slow design. | Live mode may be unusable initially. | Stage timings and criterion benches from the first search implementation; frame dropping/throttling; documented optimization path. |

---

## 16. Future work

- Multi-sample model training with cross-sample stability weighting.
- Video-file playback and per-frame batch detection.
- Richer or learned local descriptors and segment-graph matching for texture-rich
  parts and stronger perspective.
- Multi-start homography search for large tilt.
- Multiple models active in one run.
- GPU acceleration of Hough/feature stages.
- C API or network/PLC integration once detection semantics and results schema settle.
- Real-image calibration, acceptance data, and confidence calibration.
- Lean formalization of scoring bounds, NCC, and subpixel numerics (currently
  Rust-tested only, by decision).
- 3D pose / calibrated camera support.

---

## Appendix A: Provisional defaults

| Parameter | Default |
|---|---|
| Working image | grayscale `f32`, `[0,1]`; color luma BT.709 on encoded values |
| Gaussian blur at full resolution | `σ = 1.0` |
| Pyramid | levels 0–3 (image/8), auto-extended for large images |
| Coarse level | smallest level with ROI short side ≥ 32 px and ≥ 120 model edge points |
| Edge thresholds | adaptive high = 90th percentile of gradient magnitude (floor 0.08); low = 0.4 × high |
| Minimum edge chain | 4 px |
| Corner response | Shi–Tomasi `λ_min`, quality 0.01 of max, NMS radius 5 px |
| Minimum segment | 10 px |
| Rotation search | `[-180°, 180°]`, 72 bins at coarse level |
| Scale search | `[0.5, 2.0]`, 8 log bins |
| Hough orientation bins | 32 model/test orientation buckets, smooth kernel |
| Hypothesis sources | Hough + corner descriptors + segment pairs (`both`) |
| Corner descriptor | shape context 5×12 log-polar + 8 orientation bins; radii 12/24/48 px; ratio test 0.8 |
| Descriptor caps | 300 model corners, 1500 test corners per level |
| Segment-pair caps | 500 pairs per image; non-parallel pairs only |
| Hypothesis cap | 300 after clustering |
| Score `τ_d` | 2.5 px full resolution, scaled by level |
| Score orientation exponent | 2 |
| Coverage minimum | 0.4 |
| Max occlusion tolerance | implied by score threshold; diagnostics report coverage |
| Detection score threshold | 0.55 (initial) |
| Max detections | 10 |
| NMS | zero distance < 0.35 × model diagonal, angle < 10°, scale ratio within 15%, or transformed-ROI IoU > 0.45 |
| RANSAC sample size | 4 correspondences |
| RANSAC confidence | 0.99; inlier threshold 2.5 px; max 5000 iterations; seed configurable |
| Homography acceptance | score improvement ≥ 0.01; minimum 6 inliers and 12 correspondences; condition threshold documented in code |
| Camera default | device-provided, preferably 1280×720; live processing may throttle |
| Confidence | heuristic §9.9 until calibrated; clamped to `[0,1]` |

All defaults are exposed in the UI/CLI parameter files and are expected to change as
the synthetic benchmark and real images are analyzed.

## Appendix B: Notation

| Symbol | Meaning |
|---|---|
| `p = (x,y)` | model-frame point |
| `q = (u,v)` | image point |
| `t = (tx,ty)` | translation of the model zero in the image |
| `θ` | rotation of the model x-axis |
| `s` | uniform scale relative to the taught size |
| `R(θ)` | 2×2 rotation matrix (positive = clockwise on screen) |
| `H` | 3×3 homography, homogeneous model → image |
| `n`, `n'` | model edge normal, transformed edge normal |
| `d_i` | distance from transformed model point to nearest test edge |
| `ψ` | gradient/normal orientation |
| `τ_d` | scoring distance tolerance |
| `q` (RANSAC) | probability that one minimal sample is all-inlier |
| `k` | RANSAC minimal sample size (4 for homography) |
| `N` | RANSAC iterations |
