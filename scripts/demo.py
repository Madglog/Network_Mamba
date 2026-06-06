#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data.

Runs the full three-stage pipeline (pre-train -> fine-tune -> anomaly head) on a
small synthetic dataset, then evaluates both operating modes. Useful as a quick
"does everything wire together" check before pointing the pipeline at real data.

    python scripts/demo.py

Requires PyTorch. For the no-torch path (synthetic data + classical baselines)
use:  flowmamba baselines
"""

import sys

from flowmamba.cli import main

if __name__ == "__main__":
    # Small, fast configuration so the whole thing finishes in well under a minute.
    sys.exit(
        main(
            [
                "demo",
                "--n-per-class", "150",
                "--epochs", "3",
                "--d-model", "64",
                "--n-layers", "2",
                "--max-packets", "16",
                "--device", "cpu",
            ]
        )
    )
