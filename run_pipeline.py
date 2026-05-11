"""Unified CLI entrypoint for canonical configurable pipelines."""

from __future__ import annotations

import argparse
import json

from eucalipto.config_schema import load_pipeline_config
from eucalipto.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical Eucalipto pipeline")
    parser.add_argument("config", help="Path to YAML/JSON pipeline config")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    outputs = run_pipeline(cfg)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

