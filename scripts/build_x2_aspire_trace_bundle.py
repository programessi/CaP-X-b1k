#!/usr/bin/env python3
"""Build ASPIRE-style trace bundles from saved X2 CaP-X RGB-D runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.integrations.x2.aspire import build_trace_bundles, jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Saved CaP-X run or trial directories.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--visual-artifact-root", default=None, help="Optional X2 visual artifact root for this run.")
    parser.add_argument("--output-root", default="outputs/x2_aspire_traces", help="Trace bundle output root.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable build results.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    visual_root = None
    if args.visual_artifact_root:
        visual_root = Path(args.visual_artifact_root)
        if not visual_root.is_absolute():
            visual_root = repo_root / visual_root

    results = build_trace_bundles(
        repo_root=repo_root,
        paths=[Path(path) for path in args.paths],
        output_root=output_root,
        visual_artifact_root=visual_root,
    )
    rows = [
        {
            "trial_dir": str(result.trial_dir),
            "bundle_dir": str(result.bundle_dir),
            "ok": result.ok,
            "primary_failure": result.primary_failure,
        }
        for result in results
    ]
    if args.json:
        print(json.dumps(jsonable(rows), indent=2, sort_keys=True))
    else:
        print("| trial | ok | primary_failure | bundle |")
        print("|---|---:|---|---|")
        for row in rows:
            print(f"| {row['trial_dir']} | {row['ok']} | {row['primary_failure'] or '-'} | {row['bundle_dir']} |")


if __name__ == "__main__":
    main()
