"""Dependency resolution for submodules and optional external paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping


DEFAULT_DEPENDENCIES: Dict[str, str] = {
    "ff3d_inference": "third_party/FF3D_inference/ff3d_forestsens",
    "forestformer3d": "third_party/ForestFormer3D",
    "treeiso": "third_party/artemis_treeiso",
    "leafwood": "third_party/leaf-wood-segmentation-with-deep-learning",
    "rayextract_manual": "third_party/rayextract-manual",
}


@dataclass(frozen=True)
class ResolvedDependencies:
    repo_root: Path
    paths: Dict[str, Path]


def resolve_dependencies(
    repo_root: str | Path,
    external_paths: Mapping[str, str] | None = None,
) -> ResolvedDependencies:
    """Resolve provider paths using default submodule path + optional overrides."""
    root = Path(repo_root).resolve()
    override = dict(external_paths or {})
    paths: Dict[str, Path] = {}

    for dep_name, default_rel in DEFAULT_DEPENDENCIES.items():
        if dep_name in override and override[dep_name]:
            paths[dep_name] = Path(override[dep_name]).expanduser().resolve()
        else:
            paths[dep_name] = (root / default_rel).resolve()

    return ResolvedDependencies(repo_root=root, paths=paths)


def validate_dependency_paths(
    resolved: ResolvedDependencies,
    required_dependencies: Iterable[str] | None = None,
) -> None:
    """Fail fast if configured dependencies cannot be found.

    When ``required_dependencies`` is provided, validate only those keys.
    """
    if required_dependencies is None:
        deps_to_check = resolved.paths.keys()
    else:
        deps_to_check = required_dependencies

    missing = [
        f"{k}: {resolved.paths[k]}"
        for k in deps_to_check
        if k in resolved.paths and not resolved.paths[k].exists()
    ]
    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(
            "Missing external dependencies. Initialize submodules or configure "
            f"external_paths.\n{missing_str}"
        )

