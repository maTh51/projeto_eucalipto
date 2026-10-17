# Research & Biometric Experiments (`research/`)

This directory contains the experimental notebooks, master biometric benchmarking engine, and technical documentation developed on top of the 3 consolidated segmentation/isolation flows (`ff3d_full`, `treeiso_leafwood`, and `rayextract_full`).

---

## Directory Contents

* **`PROJETO_EUCALIPTO_CONTEXTO_COMPLETO.md`** — Comprehensive architectural context, physical diagnostics of the 5 volumetric algorithms on aerial LiDAR (ALS), and benchmark results across Stand 62 ($5\text{ cm}$) and Stand 37 ($10\text{ cm}$).
* **`relatorio_deteccao_multiplas_brotacoes.md`** — Technical report detailing the theory and experimental validation of the 3 multi-stem ("múltiplas brotações" / coppice) detection methods.
* **`run_biometric_framework.py`** — Master benchmarking and stand-scale extrapolation script that generates the evaluation tables in `results/benchmarking_framework/` consumed by the executive benchmark notebook.

### Notebooks (`research/notebooks/`)

* **`grid_search_volume_t62.ipynb`** — Hyperparameter grid search for DBH and volume estimation, `cKDTree` unassigned stem point recovery, Trunk-Centric Quality Control (QC), and 2D adaptive spatial matching on Stand 62.
* **`detect_multi_stems_comparison.ipynb`** — Implementation and side-by-side comparison of 3 multi-stem detection algorithms (2D Slice DBSCAN, Sequential Iterative RANSAC Circle Fitting, and Multi-Height Trajectory Tracking).
* **`visualize_biometric_benchmarks.ipynb`** — Executive visualization dashboard rendering method rankings, 1:1 per-tree volume scatter plots (Global Best vs. Human-in-the-Loop Oracle), plot-level volumetric error ($m^3/\text{ha}$), and stand-scale timber extrapolation.
* **`rayextract_treeinfo_analysis.ipynb`** — Analysis of QSM tree segments and metrics exported by RayExtract (`raycloudtools`).
* **`grid_search_volume.ipynb`** , **`volume_analysis.ipynb`** , **`dados_canoa.ipynb`** , **`Canoas_Integracao_LiDAR.ipynb`** — Exploratory analysis and field inventory integration notebooks.
