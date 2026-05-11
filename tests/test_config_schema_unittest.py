import json
import tempfile
from pathlib import Path
import unittest

from eucalipto.config_schema import load_pipeline_config


class TestConfigSchema(unittest.TestCase):
    def test_load_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg_path = tmp / "cfg.json"
            (tmp / "input.laz").write_text("", encoding="utf-8")
            cfg_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "pipeline_mode": "ff3d_full",
                        "input": {"path": str(tmp / "input.laz"), "format": "auto"},
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_pipeline_config(cfg_path)
            self.assertEqual(cfg.pipeline_mode, "ff3d_full")
            self.assertEqual(cfg.schema_version, "1.0.0")


if __name__ == "__main__":
    unittest.main()

