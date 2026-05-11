"""Pipeline configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping

from .contracts import SCHEMA_VERSION


PipelineMode = Literal[
    "ff3d_full",
    "treeiso_leafwood",
    "treeiso_leafwood_rctqsm",
    "rayextract_full",
    "rct_qsm_metrics",
]

FailurePolicy = Literal["abort", "skip", "fallback_provider", "mark_partial"]


@dataclass
class InputConfig:
    path: str
    format: Literal["auto", "laz", "las", "ply"] = "auto"


@dataclass
class OutputConfig:
    output_dir: str = "results_canonical"
    cloud_format: Literal["laz", "ply"] = "laz"


@dataclass
class RuntimeConfig:
    docker_compose_profile: str = "default"
    use_gpu: bool = True
    retries: int = 0


@dataclass
class FailureConfig:
    on_isolation_error: FailurePolicy = "abort"
    on_segmentation_error: FailurePolicy = "abort"
    on_metrics_error: FailurePolicy = "mark_partial"


@dataclass
class PipelineConfig:
    schema_version: str
    pipeline_mode: PipelineMode
    input: InputConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    external_paths: Dict[str, str] = field(default_factory=dict)
    failure_policy: FailureConfig = field(default_factory=FailureConfig)
    providers: Dict[str, Any] = field(default_factory=dict)


def _read_config_dict(path: str | Path) -> Mapping[str, Any]:
    cfg_path = Path(path).resolve()
    suffix = cfg_path.suffix.lower()

    text = cfg_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported config file extension: {suffix}")


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    raw = dict(_read_config_dict(path))

    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version '{schema_version}'. "
            f"Expected '{SCHEMA_VERSION}'."
        )

    input_cfg = InputConfig(**raw["input"])
    output_cfg = OutputConfig(**raw.get("output", {}))
    runtime_cfg = RuntimeConfig(**raw.get("runtime", {}))
    failure_cfg = FailureConfig(**raw.get("failure_policy", {}))

    cfg = PipelineConfig(
        schema_version=schema_version,
        pipeline_mode=raw["pipeline_mode"],
        input=input_cfg,
        output=output_cfg,
        runtime=runtime_cfg,
        external_paths=dict(raw.get("external_paths", {})),
        failure_policy=failure_cfg,
        providers=dict(raw.get("providers", {})),
    )
    return cfg

