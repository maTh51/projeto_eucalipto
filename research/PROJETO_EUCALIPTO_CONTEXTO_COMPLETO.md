# Comprehensive Project Context and Technical Memory: Eucalyptus Biometric Framework
> **Technical Handover Document & System Context**  
> **Date:** October 15, 2026  
> **Project:** `projeto_eucalipto` / `euc`  
> **Purpose:** Detailed documentation of business context, technical architecture, data structures, algorithms, benchmark findings, and design decisions to guide a clean codebase refactoring in a fresh session.

---

## 1. Project Overview & Research Objectives

The core objective of this project is to automate high-precision forest inventory in clonal Eucalyptus plantations using aerial drone LiDAR point clouds (Airborne Laser Scanning - ALS).

### Key Challenges
Traditional forest inventory relies on manual, highly destructive or heavily localized sample plots ($400\text{ m}^2$ circular plots covering only a fraction of stands spanning tens of hectares). This project addresses several core challenges:
1. **Raw Cloud Processing:** Segmenting individual tree instances (canopies and stems) from dense and sparse airborne LiDAR point clouds.
2. **Semantic Refinement:** Accurately separating stem/wood points from foliage/leaf points.
3. **Missing Spatial Coordinates in Ground Truth:** Field measurements only record relative planting grid positions (`Fila` / `Cova`), not absolute GPS/UTM coordinates of individual trees. The framework solves this through a robust 2D rigid transformation and bipartite matching algorithm.
4. **Biometric Extraction:** Estimating DBH (Diameter at Breast Height at $1.30\text{ m}$), Total Tree Height ($H$), and Commercial Stem Timber Volume ($V$ in $\text{m}^3$).
5. **ALS Volumetric Benchmarking:** Evaluating 5 distinct volumetric modeling methods under single-view aerial occlusion constraints.
6. **Dual Optimization Paradigms:**
   * **Mode 1: Global Best Config:** Identifying a single optimal hyperparameter configuration that minimizes aggregate plot error.
   * **Mode 2: Human-in-the-Loop / Tree-level Oracle:** Selecting the ideal method/parameters per individual tree, establishing the upper bound of expert-guided accuracy.
7. **Stand-Scale Extrapolation & Transferability:** Scaling tree/plot-level volumetric estimates to full stand boundaries ($\text{m}^3/\text{ha}$ and total stand timber $\text{m}^3$), while testing transferability between different flight resolutions ($5\text{ cm}$ vs $10\text{ cm}$).

---

## 2. Data Sources and Repository Layout

### 2.1. Official Ground Truth Field Inventory
* **File:** `/mnt/scratch/dataset/euc/canoa/IFPC-Canoas/Result_IFPC_Canoas.xlsx`
* **Key Sheets:**
  * **`Talhões` (Stand-level summary):**
    * **Talhão 062-063:** Planted area: $41.00\text{ ha}$, $650.00\text{ trees/ha}$, $1081.25\text{ planting holes/ha}$, Mean DBH: $16.0\text{ cm}$, Mean Height: $26.3\text{ m}$, Commercial Volume: $201.29\text{ m}^3/\text{ha}$, Total Timber: $8,252.95\text{ m}^3$.
    * **Talhão 37:** Planted area: $26.46\text{ ha}$, $1034.09\text{ trees/ha}$, $1161.36\text{ planting holes/ha}$, Mean DBH: $13.9\text{ cm}$, Mean Height: $22.4\text{ m}$, Commercial Volume: $191.14\text{ m}^3/\text{ha}$, Total Timber: $5,057.67\text{ m}^3$.
    * **Talhão 34:** Planted area: $24.88\text{ ha}$, $869.44\text{ trees/ha}$, $1069.44\text{ planting holes/ha}$, Commercial Volume: $189.71\text{ m}^3/\text{ha}$, Total Timber: $4,720.01\text{ m}^3$.
  * **`Parcelas` (Sample Plot Summary):** Official circular inventory plots ($400\text{ m}^2$, radius $\approx 11.28\text{ m}$):
    * Talhão 62: Plots `500, 503, 508, 509`.
    * Talhão 37: 11 plots (`594, 602, 611, 612, 621, 622, 623, 634, 635, 636, 648`).
  * **`Covas` (Tree-by-Tree Field Measurements):** Individual tree records indexed by row (`Fila`) and hole (`Cova`), containing measured `Dap (cm)`, `Altura (m)`, and field-cubed ground truth `VTCC Volume (m³)`.
    * Local theoretical coordinate grid: $X_{\text{grid}} = \text{Cova} \times 2.5\text{ m}$, $Y_{\text{grid}} = \text{Fila} \times 3.5\text{ m}$ (nominal $3.5\text{ m} \times 2.5\text{ m}$ spacing).

### 2.2. Airborne Laser Scanning (ALS) Point Clouds
* **Talhão 62 (5 cm resolution / dense flight ~400 pts/m²):**
  * Classified plot crops:
    `/home/matheuspimenta/euc/projeto_eucalipto/results/canoa/parcelas_t62/parcela_{500,503,508,509}/classified_cloud.laz`
* **Talhão 63 (5 cm resolution / continuous canopy block without direct field plots):**
  * Full block segmented cloud:
    `/home/matheuspimenta/euc/projeto_eucalipto/results/canoa/parcelas_t63/ff3d_full/classified_cloud.laz`
* **Talhão 37 (10 cm resolution / sparse flight ~100-150 pts/m²):**
  * Classified plot crops processed via FF3D:
    `/home/matheuspimenta/euc/projeto_eucalipto/results/canoa/parcelas_t37/parcela_{594,602,611,612,621,622,623,634,635,636,648}/classified_cloud.laz`

---

## 3. End-to-End Methodological Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              LIDAR BIOMETRIC PIPELINE                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
     [ Stand 62 — 5 cm ]                                         [ Stand 37 — 10 cm ]
    4 Plots (500,503,508,509)                                  11 Plots (594 to 648)
               │                                                           │
               ├─► 1. Stem Enhancement (KDTree Cylinder-Following) ────────┤
               │      Reassigns unassigned stem points                     │
               │                                                           │
               ├─► 2. Adaptive Spatial Matching with Inventory Grid ───────┤
               │      Grid rotation/translation + biometric consensus      │
               │      (43 trees T62 / 155 trees T37 with Ground Truth)     │
               │                                                           │
               ├─► 3. Multi-Method Biometric Grid Search (Volume & DBH) ───┤
               │      • Voxel: voxel_size ∈ [0.06, 0.07, 0.08, 0.09, 0.10] │
               │      • Axis Profile: slices (15, 20) x percentiles (15-25)│
               │      • Calibrated Cylinder: form_factor ∈ [0.35, ..., 0.45│
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
      [ Mode 1: Global Best ]                                  [ Mode 2: Human in Loop ]
   Single optimal config minimizing                         Best parameter config selected
   aggregate error across plots                             per tree (expert upper bound)
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                        [ Plot Evaluation & Stand Extrapolation ]
                        • Commercial volume per hectare (m³/ha) vs Ground Truth
                        • Total timber volume scaled to stand boundaries
```

### 3.1. FastForest3D (FF3D) Neural Inference
* Executed on GPU (`run_t62_ff3d.py`, `run_t37_ff3d.py`) using the FastForest3D neural network.
* **Output Dimensions:** Writes LAS/LAZ files with custom extra dimensions:
  * `tree_id`: Integer instance ID for each segmented tree.
  * `trunk_leaf_label`: Binary semantic classification ($1 = \text{stem/wood}$, $0 = \text{foliage/leaves}$).

### 3.2. Identified Bottleneck: Unassigned Stem Points (`tree_id <= 0`)
* **Problem:** Neural instance segmentation frequently correctly classifies points as wood (`trunk_leaf_label == 1`), but leaves them unassigned to any tree instance (`tree_id == 0`). This hollows out isolated stems, artificially reducing point density and biasing volume estimates downward.
* **Solution (Cylinder-Following Stem Enhancement):**
  * Queries unassigned wood points (`tree_id <= 0` and `trunk_leaf_label == 1`).
  * Builds a `cKDTree` on all known segmented stem points (`tree_id > 0` and `trunk_leaf_label == 1`).
  * Reassignment constraint: horizontal distance $d_{xy} \le 0.30\text{ m}$ and vertical distance $|\Delta z| \le 0.80\text{ m}$.
  * Reassigns points to the nearest physical stem without generating spurious tree instances or corrupting foliage semantics.

### 3.3. Trunk-Centric Quality Control (QC)
Filters out poorly segmented clusters, ground brush, understory, or severed branches before metric estimation:
1. **Minimum Point Count:** $\ge 40$ points per tree and $\ge 15$ stem points.
2. **Minimum Height:** Height $H \ge 40\%$ of the plot's $P_{90}$ canopy height.
3. **Ground Grounding:** $Z_{\text{base}} - Z_{\text{min\_plot}} \le 2.50\text{ m}$.
4. **Vertical Stem Continuity:** Maximum gap between vertical stem slices $\le 4.0\text{ m}$.
5. **Stem Compactness:** Radial 90th percentile $R_{90}$ of stem points relative to stem center $\le 1.45\text{ m}$ (rejects large lateral branch foliage misclassified as stem).

### 3.4. Adaptive Spatial Matching (Without GPS Coordinates)
* The field inventory lacks absolute GPS coordinates for individual trees.
* **Algorithm (`align_and_match`):**
  * Constructs the planting grid $(X, Y)$ using `Fila` and `Cova`.
  * Evaluates full 2D rotations ($0^\circ$ to $360^\circ$ with $3^\circ$ step) and translations $(\Delta x, \Delta y)$ to match LiDAR QC trees against the grid.
  * **Biometric Consensus Cost Function:** Maximizes the count of close matches ($d < 1.30\text{ m}$), penalizing height discrepancies between LiDAR height and field height (`alt_m`), while prioritizing trees with verified ground-truth volume (`has_gt_volume`).
  * Performs final 1-to-1 assignment via the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`).
  * **Matching Yield:**
    * **Talhão 62:** 43 high-quality trees matched with ground truth across all 4 plots.
    * **Talhão 37:** 155 high-quality trees matched with ground truth across all 11 plots.

---

## 4. Benchmark of the 5 Volumetric Methods & Physical Diagnostics

We benchmarked all 5 methods from [`eucalipto.volume_methods`](file:///home/matheuspimenta/euc/projeto_eucalipto/eucalipto/volume_methods.py) against field-measured trees:

| Method | Algorithm | Performance on Aerial LiDAR (ALS) | Relative Bias (%) |
| :--- | :--- | :--- | :---: |
| **`voxel`** | 3D regular voxel quantization and integration of occupied voxel volumes | **Top Performer**. Highly robust against branch noise, foliage contact, and single-sided scanner occlusion. | **$-1.2\%$ to $-19\%$** |
| **`axis_profile`** | Vertical slicing with radial percentile ($R_{15}, R_{20}$) and trapezoidal numerical integration | Accurate with low percentiles ($R_{15}$–$R_{20}$), but overestimates when low foliage touches the stem. | **$+53\%$ to $+165\%$** |
| **`cylinder` (raw)** | $V = \frac{\pi}{4}\text{DBH}^2 \cdot H$ | Theoretical geometric ceiling. Severe overestimation because it ignores natural stem taper ($f \approx 0.40$). | **$> +1000\%$** |
| **`cylinder_ff`** | $V = f \cdot \frac{\pi}{4}\text{DBH}^2 \cdot H$ ($f \approx 0.38-0.42$) | Robust and dendrometrically sound once calibrated with an empirical Eucalyptus form factor. | **$+5\%$ to $+20\%$** |
| **`frustum`** | Truncated cones fitted via RANSAC 3D circle fitting along slices | **Unstable on ALS**. Single-sided drone flight coverage causes RANSAC to fit massive artificial circles. | **$+2000\%$** |
| **`taper`** | Polynomial taper curve regression along stem slices | **Divergent on ALS**. Runge's phenomenon and noisy slice radii cause extreme polynomial divergence. | **$> +40000\%$** |

> **Key Finding:** Circle-fitting RANSAC methods (`frustum`, `taper`) are designed for Terrestrial Laser Scanning (TLS 360° point clouds). For drone ALS, `voxel` and tuned `axis_profile` (or calibrated form-factor cylinders) are the only viable algorithms.

---

## 5. Framework Results: Global Best vs. Human-in-the-Loop

The consolidated framework script [`run_biometric_framework.py`](file:///home/matheuspimenta/euc/projeto_eucalipto/run_biometric_framework.py) executed across all 15 plots and yielded the following metrics:

### 5.1. Tree-Level Biometric Accuracy

| Calibration Base | Flight Resolution | Matched Sample | Mode 1: Global Best RMSE | Mode 2: Human in the Loop RMSE | Error Reduction (Accuracy Gain) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Talhão 62** | **5 cm (Dense)** | **43 trees** | $0.1717\text{ m}^3$ (`voxel_vs0.07`) | **$0.0647\text{ m}^3$** | **$-62.3\%$** |
| **Talhão 37** | **10 cm (Sparse)** | **155 trees** | $0.1355\text{ m}^3$ (`voxel_vs0.08`) | **$0.0843\text{ m}^3$** | **$-37.8\%$** |

### 5.2. Stand-Scale Volume Extrapolation

| Calibration Base | Target Stand | Official GT ($m^3/\text{ha}$) | Mode 1: Global ($m^3/\text{ha}$) | Global Error (%) | Mode 2: Human ($m^3/\text{ha}$) | Human Error (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **T62 (5 cm)** | **T62/63 (41.0 ha)** | **$201.3$** | $146.5$ | $-27.2\%$ | **$174.3$** | **$-13.4\%$** |
| **T62 (5 cm)** | **T37 (26.46 ha)** | **$191.1$** | $233.1$ | $+21.9\%$ | $277.2$ | $+45.0\%$ |
| **T37 (10 cm)** | **T37 (26.46 ha)** | **$191.1$** | $120.7$ | $-36.8\%$ | **$154.9$** | **$-19.0\%$** |
| **T37 (10 cm)** | **T62/63 (41.0 ha)** | **$201.3$** | $75.9$ | $-62.3\%$ | $97.4$ | $-51.6\%$ |

---

## 6. Repository Layout & File Manifest

### 6.1. Core Production Scripts
* [`run_biometric_framework.py`](file:///home/matheuspimenta/euc/projeto_eucalipto/run_biometric_framework.py): **Master pipeline script**. Handles end-to-end execution: LAS loading, stem enhancement, adaptive matching, grid search, global ranking, human-in-the-loop selection, stand-scale extrapolation, and CSV/JSON export.
* [`visualize_biometric_benchmarks.ipynb`](file:///home/matheuspimenta/euc/projeto_eucalipto/visualize_biometric_benchmarks.ipynb): **Executive presentation notebook**. Features the ASCII pipeline architecture diagram, styled summary tables, 1:1 tree scatter plots, plot-by-plot error analysis, and stand extrapolation bar charts.
* [`compare_all_volume_methods.py`](file:///home/matheuspimenta/euc/projeto_eucalipto/compare_all_volume_methods.py): Benchmarks all 5 classical volume algorithms side-by-side.
* [`extrapolation_report.md`](file:///home/matheuspimenta/euc/projeto_eucalipto/extrapolation_report.md): Summary report with fitted Schumacher-Hall equations and allometric form factors.

### 6.2. Python Package: `eucalipto/`
* `eucalipto/volume_methods.py`: Implementations of volume estimation functions (`estimate_volume`, `estimate_volume_voxel`, `estimate_volume_axis_profile`, `estimate_volume_cylinder`, etc.).
* `eucalipto/dbh_methods.py`: DBH estimation routines (`estimate_dbh` via `ensemble`, `single_ransac`, `ls`).
* `eucalipto/isolation_ff3d.py`: Integration wrapper for FastForest3D model execution.

### 6.3. Output Datasets: `results/benchmarking_framework/`
* `metrics_summary.json`: Master JSON containing all aggregate metrics, method rankings, and extrapolation values.
* `df_trees_t62.csv`: Detailed dataset of 43 matched T62 trees with individual estimates across all methods.
* `df_trees_t37.csv`: Detailed dataset of 155 matched T37 trees with individual estimates across all methods.
* `df_plots_summary.csv`: Summary table across all 15 evaluated field plots.
* `df_extrapolation_summary.csv`: Final stand-scale timber volume extrapolation table.

---

## 7. Guidelines & Architecture Roadmap for Code Refactoring

When refactoring the codebase in a fresh session, follow these architectural priorities:

1. **Modular Package Decoupling:**
   * Break the monolithic script `run_biometric_framework.py` into clean modules inside `eucalipto/`:
     * `eucalipto.matching`: Adaptive 2D spatial grid alignment and bipartite assignment.
     * `eucalipto.enhancement`: KDTree-based stem point recovery (`cKDTree` cylinder-following).
     * `eucalipto.qc`: Trunk quality control filters and validation rules.
     * `eucalipto.extrapolation`: Allometric modeling and per-hectare/stand-scale expansion.
2. **Prune Deprecated Algorithms:**
   * Isolate or remove circle-fitting RANSAC (`pyrsc.Circle`) from the default ALS execution flow, as it generates divide-by-zero warnings and fails to converge on single-sided airborne scans.
3. **Centralized Configuration Management:**
   * Eliminate hardcoded paths and thresholds. Use unified YAML configuration files in `configs/`.
4. **Notebooks as Pure Consumers:**
   * Keep Jupyter notebooks strictly dedicated to rendering precomputed metrics and plots from CSV/JSON artifacts, ensuring maximum speed, portability, and zero state desynchronization.
