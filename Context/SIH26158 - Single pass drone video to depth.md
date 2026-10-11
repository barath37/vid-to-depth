
### The Domain-Fused 3D Reconstruction Pipeline (SIH26158)

Here is the comprehensive, step-by-step architectural pipeline for transforming a single-pass drone video into a metrically accurate 3D model. This design fuses concepts from autonomous robotics, medical imaging, and computer graphics to satisfy near real-time requirements.

### Step 1: Sensor-Fused Intelligent Keyframing (Data Ingestion)

This phase addresses motion blur and optimizes processing by intelligently filtering the input data before it reaches the heavy AI models.

- **Substep 1.1: Video Decoding & Telemetry Synchronization**    
    - Extract sequential frames from the 1080p/4K drone video.
    - Synchronize each frame with its corresponding mandatory GPS coordinates and flight metadata (e.g., altitude, gimbal pitch/yaw).
        
- **Substep 1.2: Motion Blur Rejection**
    
      
    - Calculate the Variance of the Laplacian for each frame.
        
          
        
    - Automatically discard frames that fall below a sharpness threshold to eliminate motion blur and video compression artifacts.
        
          
        
- **Substep 1.3: Spatial Baseline Keyframing**
    
      
    - Instead of processing 60 or 120 frames per second, use the GPS telemetry to extract a frame only after the drone has physically traveled a set distance (e.g., every 2 meters).
        
          
        
    - This guarantees a sufficient parallax baseline (the physical shift in viewpoint) required for accurate geometry calculations.
        
          
        
- **Substep 1.4: Dynamic Object Masking (Optional but Recommended)**
    
      
    - Run a lightweight object detection model (like YOLO-nano) to detect and mask out dynamic objects (vehicles, humans, animals).
        
          
        
    - **Details to Note:** Masking prevents moving objects from introducing ghosting artifacts or breaking the visual tracking algorithms in later steps.
        
          
        

### Step 2: Semantic & Normal-Guided Monocular Depth (The AI Core)

This phase uses deep learning to extract depth from a single angle, explicitly addressing occluded surfaces and scale ambiguities.

  

- **Substep 2.1: Semantic Segmentation**
    
      
    - Classify the pixels in the keyframes into distinct semantic categories, recognizing building facades, rooftops, roads, vegetation, and obstacles[cite: 1, 3].
        
          
        
- **Substep 2.2: Surface Normal Estimation**
    
      
    - Calculate the 3D orientation (surface normals) of the pixels.
        
          
        
    - Apply the geometric assumption that neighboring pixels on structures like walls or rooftops lie on the same local tangent plane in 3D.
        
          
        
- **Substep 2.3: Dense Depth Estimation**
    
      
    - Pass the keyframes through a monocular depth estimator (e.g., ZoeDepth or MiDaS) to generate a dense, pixel-by-pixel depth map.
        
          
        
- **Substep 2.4: Multi-Task Feature Fusion**
    
      
    - Fuse the outputs: Use the semantic masks and surface normals to refine the "wobbly" AI depth map.
        
          
        
    - Force building walls to be perfectly flat, roads to be smooth, and allow vegetation to retain a rough texture.
        
          
        
    - **Details to Note:** The AI relies on learned pictorial cues (like perspective and occlusion) rather than stereo geometry, which allows it to hallucinate and reconstruct occluded surfaces (the back of a building) mathematically[cite: 1, 3].
        
          
        

### Step 3: Scale-Anchored Visual Odometry (Metric Calibration)

This phase locks the AI's relative depth into the real physical world, converting AI guesses into exact metric measurements.

  

- **Substep 3.1: Sparse Feature Tracking (SfM)**
    
      
    - Detect distinct architectural corners and edges in the keyframes.
        
          
        
    - Track these sparse features across multiple frames to establish epipolar geometry.
        
          
        
- **Substep 3.2: Camera Pose Estimation**
    
      
    - Calculate the exact relative rotation ($R$) and translation ($t$) of the camera between every keyframe.
        
          
        
- **Substep 3.3: GPS & Telemetry Scale Fusion**
    
      
    - Monocular depth suffers from scale ambiguity (the system does not natively know if a car is 2 meters or 10 meters long).
        
          
        
    - Map the sparse tracked features to the drone's mandatory GPS coordinates to lock the absolute real-world scale[cite: 1, 3].
        
          
        
    - **Details to Note:** This specific substep is how the pipeline achieves maintaining metric accuracy without extensive Ground Control Points (GCPs). An optimization filter (like a Kalman filter) should be applied here to smooth out GPS inaccuracies and sensor noise.
        
          
        

### Step 4: Depth-Initialized 3D Gaussian Splatting (Reconstruction)

This final phase uses graphics rendering to construct the visual 3D model efficiently enough to meet tactical time constraints.

  

- **Substep 4.1: Pseudo-LiDAR Unprojection**
    
      
    - Use the exact camera poses (from Step 3) to unproject the metric, multi-task depth maps (from Step 2) into a global 3D Euclidean space.
        
          
        
- **Substep 4.2: Gaussian Splat Initialization**
    
      
    - Instead of starting with a random cloud of points, initialize 3D Gaussian splats directly onto the highly accurate, scale-locked Pseudo-LiDAR coordinates.
        
          
        
- **Substep 4.3: Differentiable Rendering Optimization**
    
      
    - Render the 3D Gaussians back into 2D viewpoints and compare them to the original drone video frames.
        
          
        
    - Mathematically tweak the opacity, color, and covariance (shape) of the Gaussians until the rendered lighting perfectly matches the variable illumination and shadows of the real video.
        
          
        
- **Substep 4.4: Export and Delivery**
    
      
    - Bake the optimized scene into the required deliverables: textured 3D meshes or dense point clouds.
        
          
        
    - **Details to Note:** Because the Gaussians are initialized with a highly accurate depth prior, the differentiable rendering converges exponentially faster than traditional photogrammetry, directly satisfying the real-time or near-real-time processing requirements.

### Part II: Pipeline Continuation & System Deployment Strategy

The following phases extend the core 3D reconstruction engine, addressing the translation of raw AI outputs into standardized geospatial formats, hardware deployment for near-real-time execution, and the formal evaluation metrics required by the NTRO problem statement.

### Step 5: Surface-Aligned Mesh Extraction & Topology Optimization
While 3D Gaussian Splatting (Step 4) produces highly realistic novel views rapidly, it is fundamentally a volumetric representation. To satisfy the mandatory deliverable of "Textured 3D meshes," the system must solidify these Gaussians into a polygonal geometry.

- **Substep 5.1: Surface-Aligned Gaussian Regularization (SuGaR)**
  - Apply a regularization loss during the final splatting phase that forces the 3D Gaussians to flatten and align tightly with the underlying geometric surfaces of the buildings and terrain.
- **Substep 5.2: Marching Cubes & Poisson Reconstruction**
  - Extract a dense polygonal mesh from the aligned Gaussians using a marching cubes algorithm or Poisson surface reconstruction, converting density fields into solid physical boundaries (rooftops, facades, roads).
- **Substep 5.3: Mesh Decimation & Optimization**
  - Simplify the dense mesh by reducing polygon count in flat areas (like roads) while preserving high polygon density in complex areas (like vegetation and architectural details) to ensure the file is lightweight and suitable for visualization.
- **Substep 5.4: UV Unwrapping & Texture Baking**
  - Project the variable illumination and color data captured by the Gaussians onto the optimized 3D mesh, baking them into a standardized UV texture map (e.g., `.png` or `.jpg` mapped to an `.obj` or `.gltf` file).

### Step 6: Global Georeferencing & GIS Integration
Step 3 anchored the relative scale of the model, but this phase locks the model into absolute global coordinates, ensuring it is ready for military reconnaissance, urban planning, and damage assessment tools.

- **Substep 6.1: Coordinate Reference System (CRS) Projection**
  - Translate the local 3D Euclidean space into a standardized global coordinate system (e.g., EPSG:4326 for WGS84 or a localized UTM zone).
- **Substep 6.2: Optional RTK/PPK Fusion**
  - If the optional RTK (Real-Time Kinematic) or PPK (Post-Processed Kinematic) corrections are available in the flight metadata, apply them to override standard GPS data. This bumps the spatial accuracy from meter-level variance down to centimeter-level precision.
- **Substep 6.3: Digital Elevation Model (DEM) Extraction**
  - Separate the semantic layers (from Step 2) to strip away vegetation and dynamic objects, generating a bare-earth Digital Terrain Model (DTM) alongside the fully populated Digital Surface Model (DSM).

### Step 7: Edge-to-Cloud Hardware & Execution Architecture
To meet the "real-time or near-real-time processing requirements" in disconnected or tactical environments, the software pipeline must be hardware-aware.

- **Substep 7.1: Edge/Ground Control Station (GCS) Processing**
  - Deploy the inference pipeline via containerized microservices (Docker) onto a ruggedized ground station equipped with dedicated edge AI accelerators (e.g., NVIDIA RTX Ada Generation laptops or Jetson AGX Orin modules).
- **Substep 7.2: Pipeline Pipelining (Asynchronous Execution)**
  - Execute Step 1 (Keyframing) and Step 2 (Depth/Semantics) concurrently as the drone streams video down via a data link. By the time the drone lands, the system only needs to finalize Step 4 (Gaussian Optimization), drastically reducing total mission turnaround time.

---

### Desired Output & Evaluation Criteria

To formally address the problem statement's requirement to define the "Desired Output" and "Evaluation Criteria," the following matrix establishes the benchmarks for system success.

| Category | Metric / Feature | Evaluation Standard & Target |
| :--- | :--- | :--- |
| **Data Deliverables** | 3D File Formats | Outputs must include `.OBJ`, `.GLTF`/`.GLB` (Meshes), and `.LAS`/`.PLY` (Point Clouds). |
| **Data Deliverables** | Georeferencing | Outputs must be globally georeferenced (.TIF for orthomosaics/DEMs) natively importable into QGIS/ArcGIS. |
| **Metric Accuracy** | Relative Scale Error | $< 5\%$ volumetric and distance variance against ground-truth measurements without GCPs. |
| **Metric Accuracy** | Absolute Geolocation | $< 1.5$ meters using standard GPS; $< 5$ centimeters using optional RTK/PPK corrections. |
| **Processing Speed** | Time-to-Model (Near Real-Time) | Total processing time should not exceed 1.5x to 2x the flight time (e.g., a 10-minute flight generates a model in under 20 minutes on edge hardware). |
| **Robustness** | Occlusion & Artifact Handling | The semantic depth estimator must successfully interpolate "blind spots" (e.g., the backsides of buildings facing away from the single flight path) without severe tearing or holes. |
| **Robustness** | Dynamic Object Rejection | $> 95\%$ successful masking of moving vehicles and pedestrians to prevent floating artifacts in the final mesh. |
| **Visual Quality** | Structural Fidelity | High PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) when rendering novel views compared to raw video frames. |

# CONTINUATION — PART III (Addendum to Part I and Part II)
## SIH26158: Single-Pass Drone Video to Accurate 3D Model Generation System

> This is a **continuation**. Part I (Steps 1-4) and Part II (Steps 5-7 + Desired Output table) already exist and are NOT repeated here. Part III adds context, corrections, math, specs, validation, and ideas. Where Part III conflicts with Part I/II, **Part III wins** (see §1).

---

## 0. How to use this document (instructions for the AI reading it)

**Precedence of truth:** (1) official problem statement (PS26158) > (2) Part III > (3) Part I/II.

**Project status:** planning/paper stage only. **No implementation has started.** Solo builder for now (may add teammates later). Demo target: a single machine (Windows laptop with NVIDIA GPU, assumed) or Google Colab, running on a pre-recorded clip. Edge/cloud split is a deployment story, not a demo requirement.

**Interaction rules:**
- Do not write code or generate files unless the user explicitly asks.
- Anything marked *(estimate)* or *(verify)* is unverified. Say so instead of presenting it as fact.
- Do not invent benchmark numbers. Propose how to measure them.
- Ask one clarifying question when a decision depends on missing information.

**Official targets (authoritative):**

| Parameter | Target |
|---|---|
| Reconstruction type | 3D mesh / point cloud |
| Processing time | < 15 min for a 10-min video |
| Spatial accuracy | ≤ 1 m |
| Coverage | Entire visible scene |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualization | Web-based or desktop viewer |

**Official weights:** Accuracy 30% · Completeness 20% · Speed 20% · Innovation 15% · Scalability 10% · UI 5%.

**Mandatory inputs:** drone video (1080p/4K), GPS, flight metadata. **Optional:** IMU, barometric altitude, camera intrinsics, RTK/PPK. Dataset is "provided real time," so the pipeline must tolerate unknown telemetry formats.

---

## 1. Errata and supersessions (Part I/II vs. locked design)

| # | Part I/II says | Status | Use instead |
|---|---|---|---|
| 1 | Depth/semantics (Step 2) runs before SfM (Step 3) | Superseded | SfM → global bundle adjustment → metric scaling first. Depth completion then consumes the drift-corrected sparse points. |
| 2 | Step 2.4: force walls "perfectly flat"; AI "hallucinates" occluded surfaces "mathematically" | Softened | Planar priors are soft, applied only where semantics + normals + multi-view agree. Inferred surfaces are tagged low-confidence (red), never presented as measured. |
| 3 | Step 1.3: keyframe every ~2 m | Superseded | Overlap-driven, budget-capped spacing (§6). A fixed 2 m rule gives ~1,500-2,400 frames on a 10-min flight and breaks the time budget. |
| 4 | Step 2.3: ZoeDepth / MiDaS | Superseded | Lightweight model (e.g., Depth Anything V2-Small), benchmarked on aerial clips (§8). |
| 5 | Step 3.3: map features to GPS; Kalman filter "applied here" | Refined | Fuse GPS + IMU + baro first (EKF, ideally with a smoother), then Sim(3) alignment with observability handling (§4). |
| 6 | Step 4.4 / 5.1-5.2: SuGaR, marching cubes, Poisson on Gaussians | Superseded | Poisson (or a 2.5D hybrid) on the pseudo-LiDAR point cloud. 3DGS is a visual output only. SuGaR-style extraction is an optional research stretch. |
| 7 | Step 5.4: bake texture from Gaussians | Replaced | Project texture from original keyframes with best-view selection (§10). |
| 8 | Part II "Desired Output" table (<5% volumetric error, <1.5 m GPS, <5 cm RTK, >95% masking, PSNR/SSIM, ≤2x flight time) | Non-official | These are self-proposed internal KPIs. Label them "internal targets." The official table is in §0. The official time budget is <15 min for 10 min of video. |
| 9 | Step 6.2: "<5 cm with RTK" | Bonus only | Applies only if RTK/PPK data is supplied. Do not headline it. |
| 10 | Step 7.1: Jetson AGX Orin / RTX Ada edge claims | Unverified | Demo on a single laptop/Colab GPU. Treat edge deployment as a design note. Do not claim 3DGS training on Orin fits the budget without measurement. |
| 11 | Step 7.2: async pipelining means only Step 4 remains at landing | Partly valid | Only local/incremental stages can stream. Final global BA, 3DGS, and Poisson need all frames (§16). |
| 12 | Step 1.4 masking is "optional" | Keep, adjust | Recommended. Use an aerial-trained detector, since COCO-trained YOLO misses tiny top-down vehicles and people. |

---

## 2. Canonical pipeline (locked v3, for reference)

- **S0 Calibration + sensor fusion:** intrinsics/undistortion; EKF (GPS + IMU + baro) → smoothed metric trajectory.
- **S1 Adaptive keyframing (edge-capable):** decode → sharpest-in-window selection → overlap/budget-driven triggers (250-350 frames per 10 min, hard cap ~400) → dynamic-object masking.
- **S2 SfM + global optimization:** feature tracking → poses → triangulation → **global bundle adjustment** → Sim(3) metric alignment to the S0 trajectory.
- **S3 Confidence-aware depth completion:** monocular depth → per-frame scale/shift fit to BA-refined sparse points → temporal consistency from reused sparse tracks → confidence tagging → pseudo-LiDAR unprojection.
- **S4 Dual-tier reconstruction:** Tier 1 orthomosaic + DSM (fast); mesh via Poisson on the point cloud; Tier 2 depth-initialized 3DGS (visual only).
- **S5 Export:** OBJ, PLY, LAS, GeoTIFF, GLB/GLTF, FBX.
- **S6 Viewer:** web viewer + confidence overlay + measurement tools.

---

## 3. Accuracy: definitions, error budget, honest limits

### 3.1 "≤ 1 m" is ambiguous. Define it.
- **Relative (internal) accuracy:** distances, heights, and shapes within the model. This is achievable at sub-meter with good SfM and scale.
- **Absolute (geolocation) accuracy:** where the model sits on Earth. Consumer GNSS typically has meter-level error, and part of it is **correlated bias** (atmosphere, multipath) that averaging and filtering do not remove. *(estimate: 1-3+ m for consumer GNSS; verify per drone model)*
- **Claim we can defend:** relative accuracy ≤ 1 m across the scene; absolute accuracy = GNSS grade unless RTK/PPK is present or the model is registered to reference data (§15, ideas A1-A3).
- **Action:** ask NTRO/mentors which one "≤ 1 m" means, and how it is measured (RMSE_H, RMSE_Z, or 95th percentile).
- **Reporting vocabulary (survey-grade credibility):** RMSE_horizontal, RMSE_vertical, and 95%-confidence values in the style of ASPRS/NSSDA positional-accuracy reporting. Report relative and absolute separately.

### 3.2 Worked error budget (illustrative, not measured)
- **Ground sample distance:** `GSD = H × sensor_width / (f × image_width)`
  Example (1-inch sensor 13.2 mm, f = 8.8 mm, 3840 px, H = 100 m): GSD ≈ 3.9 cm/px.
- **Triangulated depth error:** `σ_z ≈ z² · σ_d / (f_px · B)`
  where `f_px = f/sensor_width × image_width` (≈ 2560 px here), B = baseline, σ_d = matching error in pixels.
  Example: z = 100 m, B = 5 m, σ_d = 0.5 px → σ_z ≈ 0.39 m. Error grows with z² and falls with baseline, which is why keyframe spacing must scale with altitude (§6).
- **Motion blur in pixels:** `blur_px = v × t_exposure / GSD`
  Example: 6 m/s at 1/1000 s → 0.15 px (negligible); at 1/60 s → ~2.6 px (harmful).
- **Implication:** at typical survey altitudes, sub-meter relative accuracy is plausible. Facades are the weak zone (grazing angle, small parallax).

---

## 4. Coordinate systems and georeferencing (the math)

### 4.1 Frames
- **WGS84 geodetic** (lat, lon, alt) → **ECEF** → **local ENU** (East-North-Up, origin at the first frame or scene centroid). All reconstruction happens in local ENU (meters).
- **Camera frame** (x right, y down, z forward, OpenCV convention) vs **world ENU** (Z up). Keep every transform explicit and unit-tested.
- **Output CRS:** projected UTM. India spans zones 42N-46N (EPSG:32642-32646). Example: Coimbatore (~77°E) → zone 43N, EPSG:32643. Also offer EPSG:4326 for interchange.
- **Vertical datum:** GNSS altitude is usually ellipsoidal, maps use orthometric (MSL) height, and drone "relative altitude" is height above takeoff. Geoid undulation (EGM96/EGM2008) can be tens of meters in India. Never mix these silently. Record which one each number uses.

### 4.2 Metric alignment as Sim(3)
Find scale s, rotation R, translation t minimizing
`Σ_i ρ( || p_i^EKF − (s·R·c_i + t) || )`
over SfM camera centers `c_i` and fused-trajectory positions `p_i^EKF` (Umeyama closed form, inside RANSAC or a robust loss such as Huber for GPS outliers).

### 4.3 The observability trap of single-pass flight
A straight, near-collinear flight leaves **rotation about the flight line unconstrained** by camera-center positions alone. The model can "roll" around the path and still fit GPS perfectly, which tilts the whole scene.
Fixes (use several):
1. Gravity from IMU/gimbal attitude fixes roll and pitch of the world frame.
2. Fit the dominant ground plane in the point cloud and align its normal with ENU-up.
3. Use gimbal/compass yaw as a soft constraint (magnetometer yaw is noisy, so weight it low).
4. Prefer flights with lateral offsets or curves if the operator can choose. If not, document the limitation.

### 4.4 Scale
Scale is well constrained over a long path (thousands of meters vs ~1-2 m GNSS noise), so scale error can be tiny. **Position bias** is the residual problem (§3.1).

---

## 5. Telemetry ingestion and fusion details

- **Unknown formats are likely.** Build a format-agnostic parser layer. Common sources *(verify per drone)*: subtitle-style logs (.SRT) carrying per-frame lat/lon/altitude and camera settings, EXIF/XMP tags, CSV exports, MAVLink logs from ArduPilot/PX4.
- **Time synchronization** is a top hidden risk. Video timestamps and GNSS/IMU clocks can be offset by fractions of a second, which at 6 m/s is meters of error. Estimate the offset by cross-correlating visual-odometry speed against GNSS speed, and apply it before fusion.
- **Rates:** GNSS is often 1-10 Hz, IMU 100-400 Hz, video 24-60 fps. Interpolate to frame timestamps (linear for position, SLERP for attitude).
- **Lever arms:** GNSS antenna ≠ camera position, and gimbal attitude ≠ body attitude. Small offsets, but include them if the metadata gives them.
- **Outlier handling:** chi-square gating on GPS innovations (reject multipath jumps near buildings).
- **Filter vs smoother:** the EKF is causal (forward-only). Since the job is offline batch, run an **RTS smoother** (or a small factor-graph, e.g., GTSAM-style) so early poses benefit from later measurements. Keep the EKF as the baseline and the smoother as the upgrade.
- **Vertical channel:** GNSS altitude is typically noisier than horizontal. Barometric altitude is stable over a 10-minute flight but drifts with weather. Fuse it for relative height.
- **Graceful degradation:** without IMU/baro, GPS + visual only, and lower the confidence in outputs.

---

## 6. Keyframing (extended spec)

### 6.1 Overlap-driven spacing (replaces "every 2 m")
- Forward footprint length `L ≈ H_agl × sensor_height / f`. Target forward overlap ~80-85% → baseline `B_target = (1 − overlap) × L`.
- **Consistency check (illustrative):** 10 min at 6 m/s = 3,600 m; with 300 keyframes → 12 m spacing. At H ≈ 100 m and L ≈ 85-100 m, that is ~86-88% overlap. The budget and the geometry agree.
- Height above ground level (AGL) is needed. Relative altitude only approximates it over flat terrain (terrain-following flights break this assumption).

### 6.2 Selection logic
```
N_soft = 25-30 keyframes per minute of footage; N_hard ≈ 400
1. Build cumulative along-track distance from the smoothed trajectory.
2. spacing = max(B_target, total_distance / N_soft_total)
3. For each spacing window, pick the SHARPEST frame (Laplacian variance),
   preferring encoder I-frames (fewer compression artifacts).
4. Insert extra frames when yaw-rate or gimbal-rate exceeds a threshold,
   or when tracked-feature overlap with the last keyframe drops below ~60%.
5. If over N_hard, drop the most redundant keyframes.
```
- **Why sharpest-in-window instead of a fixed sharpness threshold:** thresholds create gaps in low-light or heavily compressed footage. Window selection guarantees coverage.
- **Rolling shutter:** fast motion skews geometry. Prefer slower segments, and note it as an error source.
- **Diagnostics to log:** rejection rate, spacing histogram, overlap per pair.

---

## 7. SfM details

- **Matching strategy:** video is ordered, so use **sequential matching with a window of ±k neighbors** plus a few loop-closure candidates, not exhaustive N² matching.
- **Tools:** COLMAP (self-contained, includes GPS-alignment tooling *(verify flags)*); a faster global SfM such as GLOMAP if installable *(verify)*; hloc-style learned features (SuperPoint + LightGlue) as a robustness upgrade for low-texture aerial scenes.
- **Feed-forward geometry alternatives** (DUSt3R/MASt3R/VGGT family) are worth a small benchmark as a pose/depth initializer or a fallback on short clips. Check their scale handling, drift over many frames, and GPU memory scaling *(verify latest versions)*.
- **Intrinsics:** prefer spec-sheet values, refine in bundle adjustment. Video containers rarely carry full EXIF-style intrinsics. Watch for zoom (varying focal length).
- **Priors:** use telemetry-derived positions as pose priors to speed convergence and reject gross failures.
- **Quality gates:** mean reprojection error, fraction of registered keyframes, triangulation-angle distribution. If registered frames < ~80%, trigger the fallback ladder (§14).

---

## 8. Depth completion details

- **Monocular depth is relative (affine-invariant), not metric.** Per keyframe, solve a robust scale and shift against the BA-refined sparse points:
  `d_metric ≈ s_i · d_rel + b_i` (do the fit in depth or inverse-depth space, whichever the model outputs), using RANSAC/Huber.
- **Cross-frame consistency:** neighboring keyframes' fitted depths must agree in overlaps. Penalize disagreement, and reuse the existing sparse tracks (no separate dense optical-flow pass).
- **Aerial domain gap:** most depth models are trained on ground-level imagery. Nadir and high-oblique aerial views may behave poorly. **Benchmark on aerial clips before committing to a model.** Consider metric-depth-capable models (evaluate on aerial data, *verify availability*).
- **Ensemble disagreement:** run two different depth models, align each to the sparse points, and use their disagreement as a cheap epistemic-uncertainty signal (feeds §9).
- **Sky and far field:** mask sky (no valid depth) and clamp maximum depth, otherwise pseudo-LiDAR produces infinite/garbage points.
- **Dynamic objects:** drop masked pixels before unprojection, and fill resulting holes from neighboring static geometry (e.g., road plane).
- **Semantic segmentation (optional, budget-gated):** only useful for (a) per-class smoothing and (b) LAS classification codes. Use a small model at reduced resolution. Bare-earth (DTM) extraction can be done **without** semantics using LiDAR-style ground filters (morphological or cloth-simulation filters, available in PDAL).

---

## 9. Confidence map specification

**Purpose:** convert the single-pass occlusion limitation into an explicit, honest output.

**Per-pixel score** `C ∈ [0,1]` from normalized signals *(initial design, tune empirically)*:
- **A, anchor proximity:** decays with image distance to nearest verified sparse point.
- **P, parallax:** triangulation angle (saturates at ~5-10°).
- **M, multi-view agreement:** fraction of neighboring keyframes whose depth agrees.
- **E, ensemble agreement:** low disagreement between depth models after alignment.
- **T, temporal stability:** depth stability across adjacent keyframes.

`C = clip(Σ w_k · signal_k, 0, 1)`, with equal weights at first.
**Tags (initial thresholds):** green ≥ 0.7 (measured), yellow 0.4-0.7 (partially supported), red < 0.4 (inferred).

**Honesty rule:** this is a heuristic score, **not a calibrated probability**. Validate by checking that error falls as low-confidence points are removed (a sparsification curve), and by correlating C with error against ground truth on test data.

**Provenance layer (cheap, defensible):** per-point/face **observation count** (how many keyframes saw it) and **max triangulation angle**. Operators can see why something is trusted.

**Propagation:** store confidence per point in LAS (as an extra-bytes dimension), as a vertex attribute in mesh formats that support it, and as a viewer overlay with a legend. Formats without attribute support get a companion file.

---

## 10. Reconstruction details

### 10.1 Point-cloud cleanup (before meshing)
- Voxel downsample (~0.1-0.25 m, tied to GSD and depth noise).
- Statistical/radius outlier removal.
- Compute **normals per depth map** (from depth gradients, oriented toward that camera) rather than one global estimate. This is more reliable for multi-view Poisson.
- Optionally mesh green/yellow points as the "measured" layer and keep red as a separate "inferred" layer.

### 10.2 Meshing
- **Poisson** (depth ~9-11) with **density trimming** (drop vertices in the lowest density quantile, ~1-5%) to remove "balloon" artifacts over gaps and sky.
- **2.5D hybrid (robust, fast):** for nadir-dominant terrain and rooftops, a DSM height-field mesh (grid/Delaunay) is faster and more stable than full 3D Poisson. Use 3D Poisson for facades and structures, then merge. *(design option; decide after seeing real data)*
- **Decimation:** quadric edge collapse, preserving boundaries. Reduce flat areas more than complex areas.

### 10.3 Texturing
- Project from the original keyframes with **best-view selection per face** (view angle, distance, sharpness) and seam leveling (MVS-Texturing/OpenMVS-style). UV atlasing via a tool such as xatlas *(verify)*.
- This removes any dependence on 3DGS for the textured mesh deliverable.

### 10.4 Tier 1 orthomosaic + DSM
- Orthomosaic from nadir-dominant pixels; DSM as a height raster from the point cloud. Set DSM resolution coarser than GSD (~5-10x) because depth noise dominates.
- Optional DTM (bare earth) via ground filtering.

### 10.5 3DGS (visual output only)
- Initialize on the pseudo-LiDAR cloud; train a limited number of iterations (a few thousand vs. the standard 30k) to fit the budget *(estimate; measure)*.
- Hold out ~1 in 8 keyframes for validation (PSNR/SSIM/LPIPS).
- Expect weaker quality on facades. Consider anti-aliased variants when altitude varies.
- Rendering quality is decoupled from measurement accuracy. State this clearly.

### 10.6 Scale-out
- **Tile** the scene (with overlap) for meshing and 3DGS. Process tiles independently, then stitch.

---

## 11. Export specification and gotchas

| Format | Tool (suggested) | Gotchas |
|---|---|---|
| OBJ / PLY | Open3D, trimesh | OBJ has no georeference. Ship a sidecar file with the origin and CRS. |
| GLB / GLTF | trimesh, Open3D | glTF is **Y-up, meters**. Convert from ENU (Z-up). |
| **FBX** | **Blender headless (bpy)** or Assimp *(verify)* | Open3D and trimesh do not, to my knowledge, write FBX. Plan a conversion step from OBJ/GLB. Earlier drafts listing Open3D/trimesh for FBX were wrong. |
| LAS (1.4) | laspy (simple) or PDAL (heavier install) | Coordinates are scaled integers (choose scale, e.g., 0.001 m, and an offset). Embed CRS (WKT VLR). Use a point format with RGB. Set ASPRS classification codes if semantics/ground filter exist. Store confidence as an extra-bytes dimension. |
| GeoTIFF | rasterio, GDAL, pyproj | Orthomosaic (RGB uint8) and DSM (float32) as separate rasters. Define nodata. Prefer Cloud-Optimized GeoTIFF for web use. Embed EPSG. |

**Precision gotcha:** UTM coordinates (~6-7 digits) exceed float32 precision (~6-12 cm steps near northing 1e6), causing visible jitter in viewers. Subtract a **local origin** for mesh/viewer coordinates, keep full-precision doubles in LAS/GeoTIFF, and store the origin offset in a sidecar (e.g., JSON).

**Windows note:** GDAL/PDAL are easiest via conda-forge. gsplat/CUDA builds are often smoother in WSL2 or Colab. COLMAP has Windows binaries.

---

## 12. Viewer and measurement toolkit

- **Baseline:** web viewer (Three.js / model-viewer) for the GLB mesh + confidence overlay toggle + legend.
- **Measurement tools (high demo value, low cost):** point-to-point distance, height, polygon area, slope, and volume above a reference plane (e.g., stockpile). Show an **uncertainty range** derived from local confidence.
- **Coordinate readout:** local ENU and geographic coordinates. Export measurements to CSV.
- **Scalable options (only if time allows):** Potree-style point-cloud viewing for large LAS; OGC 3D Tiles + CesiumJS for georeferenced, streamed meshes.
- UI is 5% of the score: keep it simple, but the confidence toggle and measurement tools are what people remember.

---

## 13. Validation and evaluation protocol

### 13.1 Ground-truth sources (verify licensing and availability)
- **Public datasets with 3D/depth ground truth:** Mid-Air (synthetic drone flights with GPS/IMU/depth), BlendedMVS (includes aerial-style scenes), TartanAir, UseGeo (UAV + LiDAR), DTU / Tanks and Temples (ground-level, for algorithm sanity).
- **Own footage:** measure real-world checkpoints (tape/RTK if available).
- **Known-size objects as free scale checks:** cricket pitch (20.12 m), tennis court (23.77 × 10.97 m), football pitch (~105 × 68 m, varies), shipping containers (6.06 / 12.19 m), standard lane markings.
- **Open geodata cross-checks:** building footprints and roads (OSM), public terrain models (e.g., Copernicus GLO-30) for vertical sanity.

### 13.2 Metrics
- **Geometry:** checkpoint RMSE (H/V), scale error %, Chamfer distance, accuracy/completeness/F-score at a threshold.
- **Poses:** ATE/RPE (e.g., via the `evo` tool).
- **Depth:** AbsRel, δ1.
- **Rendering (3DGS):** PSNR, SSIM, LPIPS on held-out keyframes.
- **Completeness:** % of visible ground area covered; % green/yellow/red.
- **Speed:** per-stage wall-clock and peak VRAM.
- **Masking:** recall on moving objects, and residual ghosting.

### 13.3 Ablations to report (table skeleton)

| Variant | RMSE_H | RMSE_V | Scale err | Time |
|---|---|---|---|---|
| GPS-only scale | | | | |
| + EKF/smoother | | | | |
| + global BA | | | | |
| + depth fusion | | | | |
| + confidence-filtered mesh | | | | |

### 13.4 Rubric → evidence map

| Criterion | Evidence to show |
|---|---|
| Accuracy (30%) | Checkpoint RMSE table; known-size object errors; ablation table |
| Completeness (20%) | Coverage %, hole count, confidence-share chart |
| Speed (20%) | Per-stage timing table vs. 15-min budget; Tier 1 latency |
| Innovation (15%) | Confidence + provenance maps; decoupled meshing; observability-aware alignment; ideas in §15 |
| Scalability (10%) | Keyframes scale with duration; tiling; streaming formats |
| UI (5%) | Viewer with confidence toggle and measurements |

---

## 14. Failure modes and fallback ladder

| Failure mode | Effect | Mitigation |
|---|---|---|
| Textureless surfaces (water, uniform roofs, glass), repetitive patterns (solar arrays) | SfM fails or mismatches | Learned features; depth priors; flag low confidence |
| Rolling shutter, motion blur, GOP compression artifacts | Geometry skew, weak features | Sharpest-in-window, I-frame preference, lower speed |
| Auto-exposure flicker, glare, hard shadows | Photometric inconsistency | Exposure normalization; shadow-robust features |
| Urban-canyon GNSS multipath | Position jumps | Chi-square gating; smoother; baro for vertical |
| Video/telemetry clock offset | Meters of error | Offset estimation (§5) |
| Near-collinear path | Roll ambiguity | Gravity + ground-plane constraints (§4.3) |
| Thin structures (wires, poles), tall vegetation | Missing/noisy geometry | Accept and tag; do not over-claim |
| Tiny top-down vehicles/people | Missed by COCO-trained detector | Aerial-trained detector; geometric-inconsistency filtering |
| Sky pixels | Infinite depth | Sky mask + depth clamp |

**Fallback ladder**
- **L0:** full pipeline.
- **L1:** no IMU/baro → GPS + visual only, lower confidence.
- **L2:** SfM registers too few frames → use telemetry poses as priors + monocular depth; report relative-only accuracy.
- **L3:** everything degrades → deliver Tier 1 orthomosaic/DSM from telemetry poses with explicit warnings.
Every level must say what accuracy it can and cannot claim.

---

## 15. Cross-domain inspiration and innovation candidates

*(Innovation is 15% of the score. Rate each candidate on effort/risk before committing. Baseline pipeline first.)*

**Georegistration without ground control points (targets challenge viii and the absolute-accuracy gap)**
- **A1. Terrain-relative correction:** compare the model's terrain to a public DEM to estimate and remove vertical bias (idea borrowed from terrain-referenced navigation in aerospace). *Effort: low-medium. Risk: DEM resolution and vintage.*
- **A2. Vector-map registration:** align model building footprints and road centerlines to open vector maps (OSM) for horizontal bias correction. *Effort: medium. Needs cached data for offline/tactical use.*
- **A3. Cross-view geo-localization:** match the drone orthomosaic to reference satellite imagery (a GPS-denied-navigation technique) for absolute registration, useful even under GNSS spoofing/denial. *Effort: high. Risk: matching failures, imagery licensing. Treat as stretch.*

**Independent metric cross-checks**
- **B1. Shadow-based height sanity check (remote sensing):** with sun elevation θ (computable from time + location with a solar-position library), object height `h ≈ L_shadow × tan θ`. Use it as a plausibility check on roof/building heights. *Effort: low-medium. Risk: shadow segmentation, sloped ground.*
- **B2. Ensemble disagreement (ML uncertainty):** cheap epistemic signal already used in §9.

**Trust and traceability (defense/intelligence framing)**
- **C1. Provenance layer:** per-face observation count + triangulation angle.
- **C2. Reproducibility manifest:** input hashes, config, software versions, and timing for each run, embedded with the outputs.
- **C3. Operator-facing warnings:** "inferred surface, not for measurement" overlays.

**Robotics / geodesy techniques**
- **D1. Factor-graph smoothing (GNSS + visual + IMU factors)** instead of a forward-only EKF.
- **D2. Pose-graph loop closure** if the path revisits areas.

**GIS / graphics**
- **E1. LoD-style building models:** roof plane fitting on the DSM (RANSAC) → simplified building shells (CityGML-like LOD1/LOD2 thinking).
- **E2. Multi-temporal change detection:** DSM differencing between two flights after registration (disaster damage, construction progress). Demonstrates the listed applications cheaply.
- **E3. Level-of-detail/tiled streaming** borrowed from game engines and web-mapping standards.

---

## 16. Scalability and progressive delivery (honest version)

- **Linear scaling rule:** keyframes ≈ 25-30 per minute of footage. Longer flights → more keyframes → tile processing.
- **What can stream during flight:** decoding, blur filtering, keyframe selection, masking, feature tracking, incremental pose estimation, monocular depth, incremental orthomosaic tiles.
- **What needs all frames:** final global bundle adjustment, final Sim(3) alignment, Poisson meshing, 3DGS training (incremental/local refinement is possible but not final).
- **Progressive delivery plan:** preview orthomosaic within seconds-to-minutes of landing (or during flight) → refined geometry → final mesh + 3DGS.
- **Edge/cloud (deployment context):** run S1 near the drone/GCS, send ~250-350 keyframes (~200-300 MB *(estimate)*) instead of the raw 4-8 GB *(estimate; depends on bitrate)*. Not required for the single-machine demo.

---

## 17. Time budget (planning estimates, NOT benchmarks)

| Stage (≈300 keyframes) | Estimate |
|---|---|
| Decode + blur + masking (edge) | ~1-2 min |
| SfM + global BA | ~1-2 min |
| Depth + sparse-to-dense fusion | ~1-2 min |
| Confidence + unprojection | ~0.5 min |
| Tier 1 orthomosaic/DSM | ~0.5 min |
| 3DGS (visual tier) | ~5-8 min |
| Poisson + texturing | ~1-2 min |
| Export | ~0.5 min |
| **Total** | **~10-15 min, tight** |

**Note:** texturing and tiling add cost that earlier drafts under-counted. 3DGS is the main lever. If time runs short, cap its iterations or deliver it as a background tier after the measurable mesh. **First action once building starts:** profile each stage on real footage and replace these estimates with measurements.

---

## 18. Risk register (top items)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SfM fails on real footage (low texture/parallax) | Med | High | Learned features, telemetry priors, fallback ladder |
| Absolute accuracy misread as ≤ 1 m | High | High | Clarify with NTRO; report relative/absolute separately; A1-A3 |
| 15-min budget missed | Med | High | Budgeted keyframes, lighter models, 3DGS as background tier |
| Depth model fails on aerial imagery | Med | Med | Benchmark early; ensemble; anchor to sparse points |
| Time-sync offset ignored | Med | Med | Offset estimation |
| Roll ambiguity on straight flights | Med | Med | Gravity + ground-plane constraint |
| FBX/format export surprises late | Med | Low | Test the export chain early with a toy mesh |
| Scope creep from §15 ideas | High | Med | Baseline first; pick at most 1-2 ideas |

---

## 19. Open questions (for NTRO/mentors/team)

1. Does "≤ 1 m" mean relative or absolute, and how will it be scored (RMSE, 95%)?
2. What telemetry formats will the real-time dataset use, and at what rate?
3. Will the evaluation set include IMU/baro/intrinsics, or only mandatory inputs?
4. Flight pattern: straight single line, or with lateral offset/curvature? Nadir, oblique, or mixed gimbal angles? Typical altitude and speed?
5. Is the scoring hardware fixed, and is a GPU guaranteed?
6. Is offline access to reference data (DEM/OSM) acceptable for georegistration?
7. Is "processing time" measured on our hardware or on theirs, and does it include export?

---

## 20. Glossary

**SfM** structure from motion · **BA** bundle adjustment · **EKF** extended Kalman filter · **RTS** Rauch-Tung-Striebel smoother · **ENU/ECEF** local East-North-Up / Earth-centered Earth-fixed frames · **CRS/EPSG** coordinate reference system / registry code · **UTM** Universal Transverse Mercator · **Sim(3)** similarity transform (scale + rotation + translation) · **Umeyama** closed-form Sim(3) fit · **GSD** ground sample distance · **AGL** above ground level · **DSM/DTM/DEM** surface / terrain / elevation model · **COG** cloud-optimized GeoTIFF · **LAS** LiDAR point-cloud format · **Pseudo-LiDAR** point cloud unprojected from depth maps · **Poisson** surface reconstruction from oriented points · **3DGS** 3D Gaussian Splatting · **Chamfer / F-score** point-set distance / precision-recall-style geometry metrics · **ATE/RPE** absolute / relative trajectory error · **AbsRel/δ1** depth error metrics · **PSNR/SSIM/LPIPS** image-quality metrics.

---

## 21. Suggested next planning tasks (no code until the user says so)

1. Resolve §19 questions 1 and 4, since they change design decisions.
2. Decide which 1-2 items from §15 to include (recommend one from A, one from C).
3. Draft the rubric → evidence storyline (§13.4) for the presentation.
4. Define the validation protocol (§13) and pick a dataset for early testing.
5. When the user gives the go-ahead: start with a single notebook proving keyframes → SfM → sparse cloud, then add depth fusion.