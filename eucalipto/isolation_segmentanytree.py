#!/usr/bin/env python3
"""
End-to-end SegmentAnyTree inference pipeline.

Steps:
  1. Convert LAS/LAZ -> PLY            (nibio_inference.pipeline_utm2local_parallel)
  2. Create/edit eval.yaml in output   (data.fold, hydra.run.dir, checkpoint_dir)
  3. Run the model                     (eval.py)
  4. Copy/rename results to final_results/
  5. Merge                             (nibio_inference.merge_pt_ss_is_in_folders)

Run it from the SegmentAnyTree root (or pass --repo-dir).

Example:
  python3 run_pipeline.py \
      -i /mnt/scratch/dataset/euc/canoa/recortes/talhao62_plots_5cm/ \
      -o output/
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd, cwd):
    """Run a command, echoing it first, and stop the pipeline on failure."""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Folder with the input .las/.laz files")
    p.add_argument("-o", "--output", required=True, help="Output folder (ply/, eval.yaml, hydra/, final_results/ go here)")
    p.add_argument("--base-config", default="conf/eval.yaml", help="eval.yaml to copy and edit (default: conf/eval.yaml)")
    p.add_argument("--checkpoint-dir", default=None, help="Folder with the trained model (default: keep the value in the base config)")
    p.add_argument("--repo-dir", default=None, help="SegmentAnyTree root (default: folder containing this script)")
    p.add_argument("--skip-convert", action="store_true", help="Skip step 1 (PLY files already exist in <output>/ply)")
    p.add_argument("--skip-eval", action="store_true", help="Skip steps 2-3 (model already run; reuse <output>/hydra)")
    p.add_argument("--skip-merge", action="store_true", help="Skip the final merge step")
    return p.parse_args()


def pipeline(input_dir, ply_dir, hydra_dir, eval_yaml, final_dir, out_dir, out_abs, repo=None, 
             base_config="conf/eval.yaml", checkpoint_dir=None, skip_convert=False, skip_eval=False, skip_merge=False):
    # ---------------------------------------------------------------- 1. LAS/LAZ -> PLY
    if not skip_convert:
        run([sys.executable, "-m", "nibio_inference.pipeline_utm2local_parallel",
             "-i", input_dir, "-o", ply_dir], cwd=repo)
    else:
        print("Skipping conversion step.")

    # The order of this list defines the result_N index, so it is used for BOTH
    # data.fold and the renaming step.
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        sys.exit(f"No .ply files found in {ply_dir}")
    print(f"\nFound {len(ply_files)} PLY file(s) in {ply_dir}")

    # ---------------------------------------------------------------- 2. eval.yaml
    if not skip_eval:
        base_cfg = Path(base_config)
        if not base_cfg.is_absolute():
            base_cfg = repo / base_cfg
        with open(base_cfg) as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("data", {})["fold"] = [str(p) for p in ply_files]
        cfg.setdefault("hydra", {}).setdefault("run", {})["dir"] = str(hydra_dir)
        if checkpoint_dir:
            cfg["checkpoint_dir"] = str(Path(checkpoint_dir).resolve())

        with open(eval_yaml, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"Wrote {eval_yaml}")

        # ------------------------------------------------------------ 3. run the model
        # Same form as the manual command: --config-name <output>/eval.yaml
        run([sys.executable, "eval.py", "--config-name", out_dir / "eval.yaml"], cwd=repo)
    else:
        print("Skipping model run.")

    # ---------------------------------------------------------------- 4. collect results
    final_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for idx, ply in enumerate(ply_files):
        pairs = [
            (hydra_dir / f"result_{idx}.ply", final_dir / f"instance_segmentation_{ply.name}"),
            (hydra_dir / f"semantic_result_{idx}.ply", final_dir / f"semantic_segmentation_{ply.name}"),
        ]
        for src, dst in pairs:
            if src.exists():
                shutil.copy2(src, dst)
                print(f"cp {src.name} -> {dst.relative_to(out_abs)}")
            else:
                missing.append(src)
    if missing:
        print("\nWARNING: missing model outputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit("Aborting before merge; fix the issue above (or re-run with --skip-convert --skip-eval).")

    # ---------------------------------------------------------------- 5. merge
    if not skip_merge:
        run([sys.executable, "-m", "nibio_inference.merge_pt_ss_is_in_folders",
             "-i", ply_dir, "-s", final_dir, "-o", out_abs, "-v"], cwd=repo)

    print("\nDone.")


def pipeline_default():
    """Run the pipeline with default arguments (for testing)."""

    repo = Path(__file__).resolve().parent / "third_party" / "SegmentAnyTree"
    input_dir = "/mnt/scratch/dataset/euc/canoa/recortes/talhao62_plots_5cm/"  # current data
    base_config = repo / "third_party" / "SegmentAnyTree" / "conf" / "eval.yaml"
    checkpoint_dir = repo / "third_party" / "SegmentAnyTree" / "model_file"

    out_dir = repo / "output"
    out_abs = repo / "output"
    ply_dir = out_abs / "ply"
    hydra_dir = out_abs / "hydra"
    final_dir = out_abs / "final_results"
    eval_yaml = out_abs / "eval.yaml"

    out_abs.mkdir(parents=True, exist_ok=True)
    pipeline(input_dir, ply_dir, hydra_dir, eval_yaml, final_dir, out_dir, out_abs, repo, 
             base_config, checkpoint_dir, skip_convert=False, skip_eval=False, skip_merge=False)


# python3 isolation_segmentanytree.py -i /mnt/scratch/dataset/euc/canoa/LiDAR/T37_10cm/parcelas/ 
# -o /mnt/scratch/keiller/projeto_eucalipto_segmentanytree_outcomes/third_party/SegmentAnyTree/output_t37/ 
# --base-config /mnt/scratch/keiller/projeto_eucalipto_segmentanytree_outcomes/third_party/SegmentAnyTree/conf/eval.yaml 
# --checkpoint-dir /mnt/scratch/keiller/projeto_eucalipto_segmentanytree_outcomes/third_party/SegmentAnyTree/model_file/ 
# --repo-dir /mnt/scratch/keiller/projeto_eucalipto_segmentanytree_outcomes/third_party/SegmentAnyTree/
def main():
    args = parse_args()

    repo = Path(args.repo_dir).resolve() if args.repo_dir else Path(__file__).resolve().parent
    input_dir = Path(args.input).resolve()
    out_dir = Path(args.output)
    out_abs = out_dir.resolve() if out_dir.is_absolute() else (repo / out_dir).resolve()

    ply_dir = out_abs / "ply"
    hydra_dir = out_abs / "hydra"
    final_dir = out_abs / "final_results"
    eval_yaml = out_abs / "eval.yaml"

    out_abs.mkdir(parents=True, exist_ok=True)
    pipeline(input_dir, ply_dir, hydra_dir, eval_yaml, final_dir, out_dir, out_abs, repo, 
             base_config=args.base_config, checkpoint_dir=args.checkpoint_dir, 
             skip_convert=args.skip_convert, skip_eval=args.skip_eval, skip_merge=args.skip_merge)


if __name__ == "__main__":
    main()
