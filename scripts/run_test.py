"""Run a single test pipeline from the command line.

Usage:

    python scripts/run_test.py \\
        --test linear-sprint \\
        --video data/sample.mp4 \\
        --gender M \\
        --age 17 \\
        --output outputs/

Prints the resulting JSON report to stdout and writes the annotated video
plus result JSON to the output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test", required=True, help="Test ID, e.g. linear-sprint")
    parser.add_argument("--video", required=True, type=Path, help="Path to the input video")
    parser.add_argument("--gender", required=True, choices=["M", "F", "X"])
    parser.add_argument("--age", required=True, type=int)
    parser.add_argument("--athlete-id", default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    # TODO (Claude Code): Once src/tests/<domain>/<test>.py exists, dispatch
    # to its run() function. The dispatcher should:
    #   1. Resolve test_id → module path via a registry in src/tests/__init__.py
    #   2. Build an AthleteProfile from gender + age (age band derived in scoring)
    #   3. Call test_module.run(video=args.video, athlete=profile)
    #   4. Persist the result JSON and annotated video to args.output
    #   5. Print the result JSON to stdout
    print(
        json.dumps(
            {
                "stub": True,
                "test": args.test,
                "video": str(args.video),
                "gender": args.gender,
                "age": args.age,
                "message": "Wire this up to src/tests/<domain>/<test>.py once Phase 4 begins.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
