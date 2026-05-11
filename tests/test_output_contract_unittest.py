import tempfile
from pathlib import Path
import unittest

from eucalipto.contracts import TreeMetricRow
from eucalipto.output_contract import ensure_output_dir, write_metrics_csv, write_run_manifest


class TestOutputContract(unittest.TestCase):
    def test_write_metrics_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = ensure_output_dir(str(Path(td) / "out"))
            rows = [
                TreeMetricRow(
                    tree_id=1,
                    dbh_cm=10.2,
                    height_m=8.5,
                    volume_m3=0.11,
                    mass_kg=None,
                    metric_provider="test",
                    warnings=[],
                )
            ]
            csv_path = write_metrics_csv(out, rows)
            manifest_path = write_run_manifest(out, {"schema_version": "1.0.0", "pipeline_mode": "ff3d_full"})
            self.assertTrue(csv_path.exists())
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()

