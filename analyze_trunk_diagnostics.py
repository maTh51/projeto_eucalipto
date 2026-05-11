#!/usr/bin/env python3
"""
Analyze trunk extraction diagnostics to understand pipeline performance
and identify trees with potential issues.

Usage:
    python analyze_trunk_diagnostics.py [diagnostics_json_path]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_diagnostics(diag_path: Path) -> Dict:
    """Load diagnostics JSON from pipeline."""
    with open(diag_path) as f:
        return json.load(f)


def analyze_trunk_extraction(diagnostics: Dict) -> None:
    """Print analysis of trunk extraction diagnostics."""
    
    if not diagnostics:
        print("No diagnostics found!")
        return
    
    print("\n" + "="*80)
    print("TRUNK EXTRACTION DIAGNOSTICS ANALYSIS")
    print("="*80)
    
    # Aggregate statistics
    total_trees = len(diagnostics)
    trees_with_warnings = sum(1 for d in diagnostics.values() if d.get("has_warnings", False))
    
    trunk_percentages = [d["trunk_percentage"] for d in diagnostics.values()]
    n_trunk_points = [d["n_final_trunk"] for d in diagnostics.values()]
    
    print(f"\nOVERALL STATISTICS")
    print(f"  Total trees: {total_trees}")
    print(f"  Trees with warnings: {trees_with_warnings} ({100*trees_with_warnings/total_trees:.1f}%)")
    print(f"  Trunk percentage: mean={np.mean(trunk_percentages):.1f}%, "
          f"std={np.std(trunk_percentages):.1f}%, "
          f"range=[{np.min(trunk_percentages):.1f}%, {np.max(trunk_percentages):.1f}%]")
    print(f"  Trunk points: mean={np.mean(n_trunk_points):.0f}, "
          f"median={np.median(n_trunk_points):.0f}")
    
    # Feature statistics
    feat = extract_feature_stats(diagnostics)
    
    print(f"\nFEATURE STATISTICS (per-point)")
    print(f"  Linearity: mean={feat['linearity_mean']:.3f} ± {feat['linearity_std']:.3f}")
    print(f"  Scattering: mean={feat['scattering_mean']:.3f} ± {feat['scattering_std']:.3f}")
    print(f"  Verticality: mean={feat['verticality_mean']:.3f} ± {feat['verticality_std']:.3f}")
    
    # Criterion pass rates
    crit = extract_criterion_pass_rates(diagnostics)
    
    print(f"\nCRITERION PASS RATES (average across trees)")
    print(f"  Distance (<0.4m): {crit['distance_pct']:.1f}%")
    print(f"  Linearity (>0.25): {crit['linearity_pct']:.1f}%")
    print(f"  Scattering (<0.5): {crit['scattering_pct']:.1f}%")
    print(f"  Verticality (>0.85): {crit['verticality_pct']:.1f}%")
    print(f"  Height (<0.90): {crit['height_pct']:.1f}%")
    
    # Outlier removal
    outliers_removed = sum(d["classification"]["axis_diagnostics"]["n_outliers_removed"] 
                          for d in diagnostics.values())
    print(f"\nOUTLIER REMOVAL")
    print(f"  Points removed (IQR-based): {outliers_removed}")
    
    # Problem trees
    problem_trees = [
        (tid, d) for tid, d in diagnostics.items() 
        if d.get("has_warnings", False)
    ]
    
    if problem_trees:
        print(f"\n⚠️  PROBLEM TREES ({len(problem_trees)})")
        problem_trees.sort(key=lambda x: x[1]["trunk_percentage"])
        for tid, d in problem_trees[:10]:  # Show top 10
            print(f"  {tid}: {d['n_final_trunk']} pts ({d['trunk_percentage']:.2f}%)")
    
    # Potential improvements
    print(f"\nPOTENTIAL IMPROVEMENTS")
    
    # Check verticality threshold is working
    vert_pass_rates = []
    for d in diagnostics.values():
        if "verticality_criterion" in d["classification"]:
            rate = d["classification"]["verticality_criterion"]["pct_pass"]
            vert_pass_rates.append(rate)
    
    if np.mean(vert_pass_rates) < 40:
        print(f"  ❌ Verticality threshold too strict: only {np.mean(vert_pass_rates):.1f}% pass")
        print(f"     → Consider reducing verticality_threshold")
    
    if np.mean(trunk_percentages) < 3:
        print(f"  ❌ Very few trunk points extracted ({np.mean(trunk_percentages):.1f}%)")
        print(f"     → Consider relaxing linearity/scattering thresholds")
    
    if trees_with_warnings / total_trees > 0.1:
        print(f"  ❌ High warning rate ({100*trees_with_warnings/total_trees:.1f}%)")
        print(f"     → Check parameter tuning or data quality")
    
    print("\n" + "="*80)


def extract_feature_stats(diagnostics: Dict) -> Dict:
    """Extract and aggregate feature statistics."""
    stats = {
        "linearity_mean": [],
        "linearity_std": [],
        "scattering_mean": [],
        "scattering_std": [],
        "verticality_mean": [],
        "verticality_std": [],
    }
    
    for d in diagnostics.values():
        if "classification" in d and "features_diagnostics" in d["classification"]:
            f = d["classification"]["features_diagnostics"]
            stats["linearity_mean"].append(f.get("linearity_mean", 0))
            stats["linearity_std"].append(f.get("linearity_std", 0))
            stats["scattering_mean"].append(f.get("scattering_mean", 0))
            stats["scattering_std"].append(f.get("scattering_std", 0))
            stats["verticality_mean"].append(f.get("verticality_mean", 0))
            stats["verticality_std"].append(f.get("verticality_std", 0))
    
    return {
        k: np.mean(v) if v else 0 for k, v in stats.items()
    }


def extract_criterion_pass_rates(diagnostics: Dict) -> Dict:
    """Extract criterion pass rates."""
    rates = {
        "distance_pct": [],
        "linearity_pct": [],
        "scattering_pct": [],
        "verticality_pct": [],
        "height_pct": [],
    }
    
    for d in diagnostics.values():
        if "classification" in d:
            c = d["classification"]
            rates["distance_pct"].append(c.get("distance_criterion", {}).get("pct_pass", 0))
            rates["linearity_pct"].append(c.get("linearity_criterion", {}).get("pct_pass", 0))
            rates["scattering_pct"].append(c.get("scattering_criterion", {}).get("pct_pass", 0))
            rates["verticality_pct"].append(c.get("verticality_criterion", {}).get("pct_pass", 0))
            rates["height_pct"].append(c.get("height_criterion", {}).get("pct_pass", 0))
    
    return {k: np.mean(v) if v else 0 for k, v in rates.items()}


def suggest_parameter_tuning(diagnostics: Dict) -> None:
    """Suggest parameter adjustments based on diagnostics."""
    
    print("\n" + "="*80)
    print("SUGGESTED PARAMETER TUNING")
    print("="*80)
    
    feat = extract_feature_stats(diagnostics)
    crit = extract_criterion_pass_rates(diagnostics)
    
    suggestions = []
    
    # Linearity tuning
    if crit["linearity_pct"] < 30:
        suggestions.append(
            f"Linearity threshold too high ({crit['linearity_pct']:.1f}% pass)\n"
            f"  Current: 0.25 → Try: 0.15-0.20"
        )
    
    # Scattering tuning
    if crit["scattering_pct"] < 30:
        suggestions.append(
            f"Scattering threshold too low ({crit['scattering_pct']:.1f}% pass)\n"
            f"  Current: 0.5 → Try: 0.6-0.7"
        )
    
    # Verticality tuning
    if crit["verticality_pct"] < 40:
        suggestions.append(
            f"Verticality threshold too high ({crit['verticality_pct']:.1f}% pass)\n"
            f"  Current: 0.85 → Try: 0.75-0.80"
        )
    
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            print(f"\n{i}. {s}")
    else:
        print("\nParameters appear well-tuned! ✓")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Default path
    default_path = Path("results_treeiso/trunk_diagnostics/trunk_extraction_diagnostics.json")
    
    # From command line
    if len(sys.argv) > 1:
        diag_path = Path(sys.argv[1])
    else:
        diag_path = default_path
    
    if not diag_path.exists():
        print(f"Error: Diagnostics file not found: {diag_path}")
        print(f"Expected path: {default_path}")
        sys.exit(1)
    
    diag = load_diagnostics(diag_path)
    analyze_trunk_extraction(diag)
    suggest_parameter_tuning(diag)
