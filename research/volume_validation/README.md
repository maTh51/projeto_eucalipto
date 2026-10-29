# Stand-Level Volumetric Validation & Cross-Stand Transfer

> **Directory:** `projeto_eucalipto/research/volume_validation/`  
> **Shared Point Clouds:** `/mnt/scratch/dataset/euc/canoa/recortes/`  
> **Shared Inventory Data:** `/mnt/scratch/dataset/euc/canoa/IFPC-Canoas/`  

---

## 1. Directory Structure

```
projeto_eucalipto/research/volume_validation/
├── README.md                                     <-- Technical documentation & results summary
├── configs/                                      <-- YAML configurations pointing to /mnt/scratch/dataset/euc/canoa
│   ├── config_t37_cv.yaml
│   ├── config_t62_t63.yaml
│   └── config_flight_res.yaml
├── src/                                          <-- Modular validation components
│   ├── __init__.py
│   ├── data_loader.py                            <-- Loads IFPC spreadsheets and LAS/LAZ clouds
│   ├── stem_enhancement.py                       <-- KDTree cylinder-following for unassigned stem points
│   ├── spatial_matcher.py                        <-- 2D rigid alignment + Hungarian matching
│   ├── biometrics.py                             <-- DBH, height, and volumetric estimation grid
│   └── cross_validation.py                       <-- Leave-One-Plot-Out (LOPO) & Monte Carlo saturation
├── experiments/                                  <-- Standalone experiment scripts
│   ├── exp1_t37_learning_curve.py
│   ├── exp2_t62_to_t63_transfer.py
│   ├── exp3_flight_resolution_5vs10cm.py
│   └── exp4_t37_to_t62_10cm_transfer.py
├── results/                                      <-- Exported CSVs, JSONs, and plots
│   ├── exp1_t37/
│   ├── exp2_t62_t63/
│   ├── exp3_resolution_5vs10cm/
│   ├── exp4_cross_stand_10cm/
│   └── figures/
└── notebooks/
    └── volume_validation_analysis.ipynb           <-- Research validation notebook
```

---

## 2. Shared Data Organization (`/mnt/scratch/dataset/euc/canoa/recortes/`)

All point cloud crops (raw `.las` and classified `parcela_{id}/classified_cloud.laz`) are stored in shared storage so any researcher on the team can execute the experiments without relying on user-specific home folders:

- `talhao37_plots_10cm/`: 11 plots of Talhão 37 at $10\text{ cm}$ (`594, 602, 611, 612, 621, 622, 623, 634, 635, 636, 648`).
- `talhao62_plots_5cm/`: 4 plots of Talhão 62 at $5\text{ cm}$ (`500, 503, 508, 509`).
- `talhao62_plots_10cm/`: 4 plots of Talhão 62 at $10\text{ cm}$ (`500, 503, 508, 509`).
- `samples_5cm/`: Sample blocks of Talhão 62 and Talhão 63 at $5\text{ cm}$ (`ff3d_full/classified_cloud.laz`).

---

## 3. Experiments, Results, and Methodological Implications

### Experiment 1: Leave-One-Plot-Out (LOPO) & Sampling Saturation (Talhão 37 — 10 cm)
- **Objective:** Evaluate volumetric error on unseen test plots (eliminating data leakage between calibration and test trees) and observe how held-out plot error varies with the number of calibration plots ($N \in [2, 4, 6, 8, 10]$, 20 Monte Carlo repeats per $N$).
- **Results:**
  - Across 118 matched trees in 11 plots, LOPO mean absolute plot error is **34.38%** in Global Best mode (`voxel_vs0.08`) and **22.40%** in Human-in-the-Loop mode.
  - Held-out error averages **32.4%** ($N=2$), **34.0%** ($N=4$), and **31.4%** ($N=10$).
- **Implications:** Because the global calibration selects a single hyperparameter (optimal voxel size or form factor), increasing the training set beyond $N=4$ plots yields minimal change in the selected global parameter and held-out error.

### Experiment 2: Intra-Clone Extrapolation at Fixed Resolution (Talhão 62 $\to$ Talhão 63 — 5 cm)
- **Objective:** Separate the area records of Talhão 62 ($25.96\text{ ha}$, 4 field plots) and Talhão 63 ($15.04\text{ ha}$, 0 field plots), calibrate the volumetric model on Talhão 62, and evaluate extrapolation on the Talhão 63 sample block under the same $5\text{ cm}$ resolution and clone (`AEC0144`).
- **Results:**
  - Calibrated optimal configuration on T62 (40 trees): `voxel_vs0.07` ($\text{RMSE} = 0.1709\text{ m}^3$).
  - Extrapolated volume on T63 (33 QC trees): **$201.14\text{ m}^3/\text{ha}$** ($3,025.2\text{ m}^3$ total), compared to the inventory reference of **$201.29\text{ m}^3/\text{ha}$** ($3,027.4\text{ m}^3$ total), a relative difference of **$-0.07\%$**.
- **Implications:** Under identical flight resolution ($5\text{ cm}$), clone, and planting grid ($3.50 \times 2.50\text{ m}$), the voxel calibration transfers consistently across stands. Note that the field inventory reference ($201.29\text{ m}^3/\text{ha}$) is reported jointly for the 062-063 management unit.

### Experiment 3: Point Cloud Resolution Sensitivity (5 cm vs. 10 cm on Talhão 62)
- **Objective:** Compare volumetric estimation on the same 4 plots of Talhão 62 (`500, 503, 508, 509`) using $5\text{ cm}$ (~400 pts/m²) and $10\text{ cm}$ (~150 pts/m²) point clouds.
- **Results:**
  - **$5\text{ cm}$ (40 matched trees):** Optimal model `voxel_vs0.07`, tree $\text{RMSE} = 0.1709\text{ m}^3$, mean plot error **28.8%** (Global) and **13.8%** (Human-in-the-Loop).
  - **$10\text{ cm}$ (38 matched trees):** Optimal model `voxel_vs0.08`, tree $\text{RMSE} = 0.1591\text{ m}^3$, mean plot error **31.4%** (Global) and **27.7%** (Human-in-the-Loop).
- **Implications:** Lower point density in the $10\text{ cm}$ cloud shifts the optimal voxel size from $7\text{ cm}$ to $8\text{ cm}$ to compensate for sparser trunk point coverage. Global plot-level error increases by 2.6 percentage points at $10\text{ cm}$, while per-tree Human-in-the-Loop adjustment degrades more noticeably (from 13.8% to 27.7%).

### Experiment 4: Cross-Stand Transfer at Matched Resolution (Talhão 37 $\leftrightarrow$ Talhão 62 — 10 cm)
- **Objective:** Test whether the large extrapolation bias previously observed between Talhão 37 and Talhão 62 was caused by resolution mismatch ($10\text{ cm}$ vs. $5\text{ cm}$) by evaluating cross-stand transfer with both stands at $10\text{ cm}$.
- **Results:**
  - Both Talhão 37 ($10\text{ cm}$) and Talhão 62 ($10\text{ cm}$) independently converge to the same optimal parameter: **`voxel_vs0.08`**.
  - **Mismatched transfer ($\text{T37}_{10\text{cm}} \to \text{T62}_{5\text{cm}}$):** Overestimates stand volume by **$+46.3\%$** ($294.6\text{ m}^3/\text{ha}$ vs. $201.3\text{ m}^3/\text{ha}$).
  - **Matched transfer ($\text{T37}_{10\text{cm}} \to \text{T62}_{10\text{cm}}$):** Predicts **$200.2\text{ m}^3/\text{ha}$** against **$201.3\text{ m}^3/\text{ha}$**, reducing relative error to **$-0.6\%$**.
- **Implications:** Voxel occupancy volume is directly coupled to point cloud density. Transferring voxel parameters across stands requires matching point cloud resolution; when both stands are evaluated at $10\text{ cm}$, resolution-induced bias is eliminated.

---

## 4. Reproducing the Experiments

From the repository root, run:
```bash
python research/volume_validation/experiments/exp1_t37_learning_curve.py
python research/volume_validation/experiments/exp2_t62_to_t63_transfer.py
python research/volume_validation/experiments/exp3_flight_resolution_5vs10cm.py
python research/volume_validation/experiments/exp4_t37_to_t62_10cm_transfer.py
```
