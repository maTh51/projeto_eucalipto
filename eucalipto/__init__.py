"""Core utilities for Eucalyptus tree point cloud processing.

Primary entrypoint:
    python run_pipeline.py <config.yml|json>
"""

from .contracts import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
