"""Stable owner-controlled failure used only for ReproCI dogfooding."""

import sys
import time


started = time.perf_counter()
print("REPROCI_OWNER_CONTROLLED_FAILURE:v0.2.0", flush=True)
print(f"elapsed_before_exit_seconds={time.perf_counter() - started:.6f}", flush=True)
sys.exit(23)
