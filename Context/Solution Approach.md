# SIH26158 — Single-Pass Drone Video to Accurate 3D Model Generation System
## Finalized Technical Architecture & Implementation Specification (v3 — teardown-hardened)

**Problem Statement:** 26158 | **Organization:** National Technical Research Organisation (NTRO) | **Theme:** Robotics and Drones

---

## 0. Target Specification (from official PS — every design decision below is constrained by this)

| Parameter           | Target                                   |
| ------------------- | ---------------------------------------- |
| Reconstruction Type | 3D Mesh / Point Cloud                    |
| Processing Time     | < 15 minutes for a 10-minute video       |
| Spatial Accuracy    | ≤ 1 meter                                |
| Coverage            | Entire visible scene                     |
| Output Formats      | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualization       | Web-based or Desktop Viewer              |

**Evaluation weightage:** Reconstruction Accuracy 30% · Model Completeness 20% · Processing Speed 20% · Innovation 15% · Scalability 10% · UI 5%

**Design implication:** 70% of the score depends on the core pipeline reliably working end-to-end (accuracy + completeness + speed). Innovation (confidence mapping, dual-tier output) is a differentiator, not the main event.

**Deployment architecture note (read before Step 1):** For the hackathon demo, Steps 0-6 run on a single machine (laptop/Colab) against a pre-loaded video file — no live network constraint exists. For real operational deployment (disaster response, field recon), Step 1's keyframing and blur rejection are designed to run **locally on an edge device**, transmitting only the ~250-350 extracted keyframes (~200-300 MB) to a cloud/ground-station GPU for Steps 2-6 — not the raw 4-8 GB source video. This avoids a network-layer bottleneck that would otherwise silently consume the 15-minute budget before processing even begins. State this explicitly as a deployment-context design decision, not a hackathon-demo requirement — it answers the "what about the field?" question without adding demo-day risk.

---

## Step 0: Calibration & Sensor Fusion Setup (The Scale Anchor)

**Purpose:** Establish a drift-resistant, metrically-grounded reference frame before any AI processing begins.

### 0.1 Camera Intrinsic Calibration
- Extract focal length, sensor size, and lens distortion coefficients from drone metadata/spec sheet.
- **Note:** Video containers (MP4) rarely carry full EXIF-style intrinsics. Pull intrinsics from the drone model's published spec sheet, or self-calibrate — refine intrinsics jointly during bundle adjustment (Step 2.4) if the exact drone model is unknown.
- Undistort all frames before any downstream processing.

### 0.2 Extended Kalman Filter (EKF) Setup
- Fuse mandatory GPS coordinates with optional IMU (pitch/roll/yaw) and barometric altitude data.
- **Note:** Consumer GPS alone drifts 2-5m — insufficient for the ≤1m target on its own. Treat GPS as a coarse global correction, not the primary scale source.
- Graceful degradation: if IMU/barometer are unavailable, fall back to GPS + camera-derived scale only, and flag the resulting confidence as lower in the output (feeds Step 3.4).
- Suggested library: `filterpy`.

### 0.3 Trajectory Locking
- Output a smoothed, drift-resistant global trajectory from the EKF — this becomes the absolute metric anchor referenced in Step 2.5.

---

## Step 1: Intelligent Adaptive Keyframing (Data Ingestion — runs on edge/local device)

**Purpose:** Computational gatekeeper. Makes the 15-minute budget achievable and, per the deployment note above, is the boundary between the edge device and the cloud/GPU stage.

### 1.1 Dynamic Video Decoding
- Hardware-accelerated decoding (OpenCV with GPU backend, or FFmpeg with NVDEC).
- Stream-decode frame by frame — don't load the full video into memory.

### 1.2 Motion Blur Rejection
- Variance of Laplacian per candidate frame; discard frames below a sharpness threshold.
- Tune the threshold empirically per dataset; log the rejection rate as a diagnostic.

### 1.3 Budget-Aware Spatial Triggering
- Dynamically calculate frame-extraction spacing based on flight velocity, targeting a **hard-capped ~250-350 keyframes total**, not a fixed "every N meters" rule.
- **Note:** A fixed "every 2m" rule at typical survey speeds (5-8 m/s) over 10 minutes produces 1,500-2,400 candidate keyframes — this alone blows the 15-minute budget before any AI model runs.
- **Scalability note:** State this as ~25-30 keyframes per minute of footage, so it scales linearly to other flight durations rather than looking hardcoded to the demo case.

### 1.4 Angular Triggers
- Force an additional keyframe on sharp gimbal movement or drone rotation, regardless of the spatial trigger.
- **Note:** Can conflict with the hard cap during frequent turns — treat 250-350 as a soft budget with turn-triggered frames as priority insertions; hard-cap the absolute maximum at ~400.

### 1.5 Dynamic Object Masking
- Lightweight detector (YOLO-nano) masks vehicles, humans, animals in each keyframe before they reach feature tracking or depth estimation.

**Output of Step 1 (edge device):** ~250-350 undistorted, masked keyframes (~200-300 MB) — this is what gets transmitted onward, not the raw video.

---

## Step 2: Structure from Motion & Global Optimization (Structure Branch)

**Purpose:** Build a globally-consistent, metrically-scaled geometric skeleton **before** any of it is used as an anchor for AI depth completion — this ordering is the fix for the timeline paradox: depth fusion must consume drift-corrected points, not raw local triangulations.

### 2.1 Sparse Feature Detection & Tracking (SfM)
- Detect distinct architectural corners/edges (ORB, Shi-Tomasi, or SIFT if compute allows) and track across the keyframe sequence.

### 2.2 Camera Pose Estimation
- Calculate relative rotation (R) and translation (t) between keyframes via epipolar geometry.
- **Note:** A single linear flight path gives narrow parallax, especially for surfaces viewed near edge-on (vertical facades from a grazing angle). Pose estimation quality will be visibly better for nadir views than oblique facade views — a structural limitation of single-pass capture, not a bug. This is exactly why Step 3.4's confidence tagging exists.

### 2.3 Sparse 3D Point Triangulation
- Triangulate matched features into a sparse 3D point cloud.

### 2.4 Global Bundle Adjustment (moved up — critical ordering fix)
- Run a global optimization pass over **all** keyframe poses and sparse points simultaneously.
- **Note:** This must complete before any sparse points are used as anchors elsewhere. Injecting un-refined, locally-triangulated points into depth fusion (as earlier drafts did) means anchoring to points that haven't yet been drift-corrected across the full flight path. Reordering costs nothing extra — it's the same computation, just sequenced correctly.
- If camera intrinsics were uncertain (Step 0.1 fallback), refine them jointly here.

### 2.5 EKF-Locked Metric Scaling
- Scale the finalized, globally-optimized pose graph and sparse point cloud using the EKF trajectory from Step 0.3 — not raw GPS.
- **Output of Step 2:** a globally-consistent, metrically-scaled sparse point cloud + camera poses. This is what Step 3 consumes.

---

## Step 3: Confidence-Aware Depth Completion (The AI Core)

**Purpose:** Generate dense per-pixel depth, correctly anchored to the drift-corrected sparse points from Step 2, without over-taxing a consumer/Colab GPU.

### 3.1 Lightweight Monocular Depth Estimation
- Use a lightweight backbone (Depth Anything V2-Small, or MiDaS v2.1 Small) — not full ZoeDepth or large transformer variants.
- **Note:** Licensed by the spec's own ≤1m accuracy target (not sub-cm) — the saved compute goes to 3DGS optimization and meshing instead.

### 3.2 Sparse-to-Dense Depth Fusion
- Inject the **globally bundle-adjusted, metrically-scaled** points from Step 2.5 (not raw Step 2.3 output) into the monocular depth output as anchor constraints.
- The AI model fills in the blanks between verified, drift-corrected measurements rather than guessing the whole scene, or anchoring to still-drifting local triangulations.

### 3.3 Temporal Consistency via Reused Sparse Tracks
- Reuse the sparse feature tracks already computed in Step 2.1 to constrain adjacent-frame depth consistency, **instead of** running a separate dense optical-flow pass (e.g., RAFT/Farneback).
- **Note:** Dense optical flow across ~350 high-resolution frames is a real risk to the 1-2 minute stage budget and is unnecessary — the tracking data already exists. This isn't literally zero added cost (propagating the constraint into the dense map still costs something), but it's near-zero marginal cost compared to a full dense-flow pass — say "near-zero," not "zero," if asked directly.

### 3.4 Epistemic Confidence Tagging
- Per-pixel confidence mask: **Green** = directly observed/triangulated, **Red** = AI-inferred (e.g., occluded backside of a wall).
- **Note:** Highest-leverage design decision in the pipeline — converts the single-pass occlusion limitation into an honest, demo-able feature, and directly supports Model Completeness (20%): an honestly-tagged inferred surface counts as "complete," a silent hole does not.

### 3.5 Pseudo-LiDAR Unprojection
- Using the camera poses from Step 2 and the fused, confidence-tagged depth maps from 3.2-3.4, unproject every pixel into a single global 3D Euclidean point cloud.
- **Output of Step 3:** a dense, metrically-scaled, confidence-tagged 3D point cloud. This — not the Gaussian splats — is what gets meshed in Step 4.3.

---

## Step 4: Dual-Tier 3D Reconstruction

**Purpose:** Satisfy both the "near real-time situational awareness" requirement and the measurable-mesh deliverable, via two deliberately decoupled outputs.

### 4.1 Tier 1 — Fast-Path Mapping (Orthomosaic + DSM)
- Project nadir pixels into a 2D orthomosaic and Digital Surface Model (DSM) immediately after Step 2/3, before the heavier 3DGS stage runs.
- **Note:** Delivers a usable tactical product in seconds, banking early points toward Processing Speed (20%).

### 4.2 Tier 2 — Depth-Initialized 3D Gaussian Splatting (visual output only)
- Initialize Gaussian splats onto the dense point cloud from Step 3.5 instead of random initialization — converges in minutes, not hours.
- **Note (meshing illusion — corrected):** 3DGS is used **only** as a photoreal viewer/visualization output from here on. It is explicitly **not** the source for the measurable mesh — Open3D's Poisson reconstruction has no native support for Gaussian splat density fields (opacity/covariance/spherical-harmonics parameters), and converting 3DGS directly to a clean mesh is an active, unresolved research problem (see SuGaR and related work). Do not attempt a custom splat-to-mesh extractor under hackathon time constraints.
- **Known limitation to state proactively:** splat quality on vertical facades (narrow parallax, grazing angle) will visibly lag nadir-viewed rooftops/terrain — pair this with the confidence map in your presentation; it's the same root cause, stated honestly.

### 4.3 Surface Meshing (corrected source)
- Run Poisson surface reconstruction (Open3D) **directly on the dense pseudo-LiDAR point cloud from Step 3.5** — a standard oriented point cloud with normals, which Open3D handles natively.
- This decouples the measurable-mesh pipeline entirely from the 3DGS optimization pipeline — a meshing failure or slow 3DGS convergence no longer blocks the other.

---

## Step 5: Format Export

### 5.1 Automated Format Conversion
- Mesh/point-cloud formats (OBJ, PLY, .glb/.gltf, .fbx): Open3D or trimesh.
- Georeferenced point cloud (LAS): PDAL.
- Georeferenced raster (GeoTIFF): exported directly from the Tier 1 orthomosaic/DSM (Step 4.1), tagged with the EKF-locked geospatial reference.
- Budget: ~30 seconds — pure engineering, but non-optional (explicit line item in the Desired Output table).

---

## Step 6: Visualization & UI

### 6.1 Lightweight Web Viewer
- `<model-viewer>` or Three.js to display the exported `.glb` mesh.
- **Note:** UI is only 5% of the score — keep this simple, don't over-invest build time here.

### 6.2 Confidence Map Overlay Toggle
- Toggle in the viewer overlaying the Step 3.4 confidence map onto the 3D mesh.
- **Note:** Most demo-friendly feature in the architecture — a one-click way to show judges you've engineered around the system's core limitation rather than hidden it.

---

## Appendix A: Feasibility / Time Budget (for Q&A defense)

**Edge device (local, pre-cloud):**

| Stage | Estimated Time |
|---|---|
| Video decode + blur rejection + YOLO-nano masking (Step 1) | ~1-2 min |
| Transmit ~250-350 keyframes (~200-300 MB) to cloud/GPU *(deployment context only — N/A for local-machine demo)* | ~1-3 min on a constrained field link; negligible on a single local machine |

**Cloud/GPU (Colab T4/A100-class, ~300 keyframes):**

| Stage | Estimated Time |
|---|---|
| SfM (tracking + pose + global bundle adjustment) | ~1-2 min |
| Lightweight monocular depth + sparse-to-dense fusion (batched) | ~1-2 min |
| Temporal consistency (reused sparse tracks — near-zero marginal cost) | negligible |
| Confidence tagging + pseudo-LiDAR unprojection | ~30 sec |
| Tier 1 orthomosaic/DSM | ~30 sec |
| Tier 2: 3DGS optimization (depth-initialized) | ~5-8 min |
| Poisson meshing (on point cloud, not splats) | ~30 sec |
| Format export | ~30 sec |
| **Cloud/GPU total** | **~10-14 min** (within 15-min budget, with margin) |

## Appendix B: Known Limitations to State Proactively

1. **Facade reconstruction quality is inherently lower than rooftop/terrain quality** — narrow parallax on a single linear flight path — mitigated, not eliminated, by depth-initialization and confidence tagging.
2. **The measurable mesh comes from the unprojected dense point cloud, not from the Gaussian splats** — 3DGS is a decoupled, purely visual output; meshing splats directly is an unsolved research problem, not attempted here.
3. **Camera intrinsics may require spec-sheet lookup or self-calibration**, not naive EXIF extraction.
4. **Keyframe budget (250-350) scales at ~25-30 keyframes/minute** — stated explicitly to answer Scalability (10%).
5. **Accuracy target is ≤1m, not survey-grade** — licenses lighter/faster model choices throughout.
6. **Graceful degradation:** if IMU/barometer are unavailable, falls back to GPS+visual scale only, with lower confidence tagged in output.
7. **Edge/cloud data-transfer split is a deployment-architecture decision** for field use, not a requirement of the hackathon demo itself (which runs on a single local machine/Colab against a pre-loaded file).

## Appendix C: Suggested Libraries by Step

| Step | Library/Tool |
|---|---|
| Video decode | OpenCV, FFmpeg (hardware-accelerated) |
| Blur detection | OpenCV (`cv2.Laplacian`) |
| Object masking | YOLO-nano (Ultralytics) |
| EKF sensor fusion | `filterpy`, or custom |
| Sparse feature tracking / SfM / bundle adjustment | OpenCV (ORB/Shi-Tomasi), or COLMAP |
| Monocular depth | Depth Anything V2-Small, MiDaS v2.1 Small |
| 3D Gaussian Splatting (visual output only) | `gsplat`, or original 3DGS reference implementation |
| Surface meshing (on point cloud, not splats) | Open3D (Poisson reconstruction) |
| Point cloud export (LAS) | PDAL |
| Mesh export (OBJ/PLY/glb/fbx) | Open3D, trimesh |
| Web viewer | `<model-viewer>`, Three.js |

---

*v3 — incorporates the timeline-ordering fix (bundle adjustment before depth fusion), the meshing-source correction (point cloud, not splats), the edge/cloud deployment note, and the temporal-consistency compute fix (reused sparse tracks, not dense optical flow).*
