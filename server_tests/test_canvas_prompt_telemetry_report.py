from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path("scripts/canvas_prompt_telemetry_report.py")


class TestCanvasPromptTelemetryReportScript(unittest.TestCase):
    def test_script_generates_csv_and_json_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_file = tmp_path / "mathud_session_test.log"
            csv_out = tmp_path / "out.csv"
            json_out = tmp_path / "out.json"

            telemetry_1 = {
                "mode": "hybrid",
                "prompt_kind": "text",
                "includes_full_state": True,
                "input_estimated_tokens": 100,
                "output_payload_estimated_tokens": 100,
                "reduction_pct": 0.0,
                "normalize_elapsed_ms": 0.2,
            }
            telemetry_2 = {
                "mode": "hybrid",
                "prompt_kind": "text",
                "includes_full_state": False,
                "input_estimated_tokens": 900,
                "output_payload_estimated_tokens": 500,
                "reduction_pct": 44.44,
                "normalize_elapsed_ms": 1.8,
            }
            log_file.write_text(
                "\n".join(
                    [
                        "random line",
                        f"INFO canvas_prompt_telemetry {json.dumps(telemetry_1)}",
                        f"INFO canvas_prompt_telemetry {json.dumps(telemetry_2)}",
                    ]
                ),
                encoding="utf-8",
            )

            cmd = [
                "python",
                str(SCRIPT_PATH),
                "--log-file",
                str(log_file),
                "--csv-out",
                str(csv_out),
                "--json-out",
                str(json_out),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(csv_out.exists())
            self.assertTrue(json_out.exists())
            self.assertIn("rows: 2", result.stdout)

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["totals"]["count"], 2)
            self.assertIn("hybrid|text|includes_full_state=True", payload["groups"])
            self.assertIn("hybrid|text|includes_full_state=False", payload["groups"])

    def test_script_mode_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_file = tmp_path / "mathud_session_test.log"
            json_out = tmp_path / "out.json"

            telemetry_hybrid = {
                "mode": "hybrid",
                "prompt_kind": "text",
                "includes_full_state": False,
                "input_estimated_tokens": 200,
                "output_payload_estimated_tokens": 120,
                "reduction_pct": 40.0,
                "normalize_elapsed_ms": 1.2,
            }
            telemetry_summary = {
                "mode": "summary_only",
                "prompt_kind": "text",
                "includes_full_state": False,
                "input_estimated_tokens": 200,
                "output_payload_estimated_tokens": 80,
                "reduction_pct": 60.0,
                "normalize_elapsed_ms": 1.3,
            }
            log_file.write_text(
                "\n".join(
                    [
                        f"INFO canvas_prompt_telemetry {json.dumps(telemetry_hybrid)}",
                        f"INFO canvas_prompt_telemetry {json.dumps(telemetry_summary)}",
                    ]
                ),
                encoding="utf-8",
            )

            cmd = [
                "python",
                str(SCRIPT_PATH),
                "--log-file",
                str(log_file),
                "--mode",
                "summary_only",
                "--json-out",
                str(json_out),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["totals"]["count"], 1)
            self.assertEqual(payload["rows"][0]["mode"], "summary_only")


if __name__ == "__main__":
    unittest.main()
