"""Stable owner-controlled failure used only for ReproCI dogfooding."""

import json
import os
from pathlib import Path
import sys
import time


started = time.perf_counter()
source = "validation/reproci_dogfood/controlled_failure.py"
report_path = os.environ.get("REPROCI_CAPTURE_REPORT")
if report_path:
    Path(report_path).write_text(
        json.dumps(
            {
                "duration_s": time.perf_counter() - started,
                "failures": [
                    {
                        "nodeid": f"{source}::owner_controlled_exit_23",
                        "traceback": f"{source}: owner-controlled exit 23",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
print("REPROCI_OWNER_CONTROLLED_FAILURE:v0.2.0", flush=True)
print(f"elapsed_before_exit_seconds={time.perf_counter() - started:.6f}", flush=True)
sys.exit(23)
