#!/usr/bin/env python3
"""Audit the current X2 CaP-X tabletop integration state.

This is a local filesystem audit. It checks that the stable files, docs,
snapshot helpers, and oracle evidence are present, and reports whether the
manual direct-API non-oracle evidence has been collected.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_PATHS = [
    "capx/integrations/x2/control.py",
    "capx/envs/tasks/x2/x2_pick_place_red_cube.py",
    "capx/envs/tasks/x2/x2_pick_place_red_cube_two_targets.py",
    "capx/envs/simulators/x2_b1k.py",
    "env_configs/x2/x2_pick_place_red_cube.yaml",
    "env_configs/x2/x2_pick_place_red_cube_two_targets.yaml",
    "env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml",
    "docs/x2-llm-facing-primitives.md",
    "docs/x2-pick-place-current-baseline.md",
    "docs/x2-capx-integration-status.md",
    "docs/x2-version-management.md",
    "scripts/run_x2_two_target_api_stability_and_check.sh",
    "scripts/run_x2_two_target_codex_a_stability_and_check.sh",
    "scripts/check_x2_acceptance.py",
    "scripts/summarize_x2_runs.py",
    "tests/test_x2_llm_api.py",
    "tests/test_x2_run_summary.py",
    "snapshots/x2_two_target_stability_hold_baseline_20260629_1330/README.md",
    "snapshots/x2_two_target_stability_hold_baseline_20260629_1330/scripts/run_x2_two_target_api_stability_and_check.sh",
    "snapshots/x2_two_target_stability_hold_baseline_20260629_1330/scripts/run_x2_two_target_codex_a_stability_and_check.sh",
    "snapshots/x2_two_target_stability_hold_baseline_20260629_1330/scripts/check_x2_acceptance.py",
    "snapshots/x2_two_target_stability_hold_baseline_20260629_1330/docs/x2-capx-integration-status.md",
]

CONTROL_REQUIRED_STRINGS = [
    "class X2PickPlaceApi",
    '"pick_and_place_red_cube"',
    '"pick_and_place_red_cube_to_left_target"',
    '"pick_and_place_red_cube_to_right_target"',
    '"pick_and_place_visual_object"',
    "T_world_tcp",
]

DOC_REQUIRED_STRINGS = [
    "direct-API non-oracle stability run",
    "scripts/run_x2_two_target_api_stability_and_check.sh",
    "scripts/run_x2_two_target_codex_a_stability_and_check.sh",
    "scripts/check_x2_acceptance.py",
]

ORACLE_EVIDENCE_PATHS = [
    "outputs/oracle/stability/oracle/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01",
    "outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01",
]

NON_ORACLE_EVIDENCE_GLOBS = [
    "outputs/stability/two_targets_*_api_stability_*_run*",
    "outputs/stability/two_targets_*_codex_a_stability_*_run*",
    "outputs/stability/*/two_targets_*_api_stability_*_run*",
    "outputs/stability/*/two_targets_*_codex_a_stability_*_run*",
]


def _load_acceptance_module(repo_root: Path):
    script = repo_root / "scripts" / "check_x2_acceptance.py"
    spec = importlib.util.spec_from_file_location("check_x2_acceptance", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _AcceptanceArgs:
    def __init__(
        self,
        *,
        paths: list[str],
        repo_root: str,
        allow_missing_code: bool = False,
    ) -> None:
        self.paths = paths
        self.repo_root = repo_root
        self.require_targets = "right,left"
        self.max_tcp_error_m = 0.02
        self.max_ori_error_rad = 0.10
        self.max_place_error_m = 0.10
        self.allow_missing_video = False
        self.allow_missing_visual = False
        self.allow_missing_code = allow_missing_code


def _path_check(repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for rel_path in REQUIRED_PATHS:
        path = repo_root / rel_path
        rows.append({"name": rel_path, "ok": path.exists(), "detail": "exists" if path.exists() else "missing"})
    return rows


def _content_check(repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    control = (repo_root / "capx/integrations/x2/control.py").read_text(encoding="utf-8", errors="replace")
    for needle in CONTROL_REQUIRED_STRINGS:
        rows.append(
            {
                "name": f"control.py contains {needle}",
                "ok": needle in control,
                "detail": "found" if needle in control else "missing",
            }
        )
    status_doc = (repo_root / "docs/x2-capx-integration-status.md").read_text(encoding="utf-8", errors="replace")
    for needle in DOC_REQUIRED_STRINGS:
        rows.append(
            {
                "name": f"x2-capx-integration-status.md contains {needle}",
                "ok": needle in status_doc,
                "detail": "found" if needle in status_doc else "missing",
            }
        )
    return rows


def _acceptance_check(repo_root: Path, paths: list[str], *, allow_missing_code: bool) -> tuple[bool, str]:
    module = _load_acceptance_module(repo_root)
    args = _AcceptanceArgs(paths=paths, repo_root=str(repo_root), allow_missing_code=allow_missing_code)
    ok, checked = module.check_acceptance(args)
    compact = [
        {
            "trial": item["row"].get("trial_dir"),
            "target": (item["row"].get("result") or {}).get("target"),
            "ok": item["ok"],
            "errors": item["errors"],
        }
        for item in checked
    ]
    return bool(ok), json.dumps(compact, sort_keys=True)


def audit(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    path_rows = _path_check(repo_root)
    content_rows = _content_check(repo_root) if all(row["ok"] for row in path_rows if row["name"].endswith(("control.py", "x2-capx-integration-status.md"))) else []

    oracle_paths = [str(repo_root / rel_path) for rel_path in ORACLE_EVIDENCE_PATHS]
    oracle_ok, oracle_detail = _acceptance_check(repo_root, oracle_paths, allow_missing_code=True)

    direct_paths: list[str] = []
    for pattern in NON_ORACLE_EVIDENCE_GLOBS:
        direct_paths.extend(glob.glob(str(repo_root / pattern)))
    direct_paths = sorted(set(direct_paths))
    if direct_paths:
        direct_ok, direct_detail = _acceptance_check(repo_root, direct_paths, allow_missing_code=False)
        direct_status = "pass" if direct_ok else "fail"
    else:
        direct_ok = False
        direct_status = "pending"
        direct_detail = "no direct API or codex-a stability run folders found"

    required_ok = all(row["ok"] for row in path_rows) and all(row["ok"] for row in content_rows) and oracle_ok
    complete = required_ok and direct_ok
    return {
        "required_files": path_rows,
        "content_contract": content_rows,
        "oracle_evidence": {"ok": oracle_ok, "detail": oracle_detail},
        "direct_api_evidence": {"status": direct_status, "ok": direct_ok, "paths": direct_paths, "detail": direct_detail},
        "required_local_baseline_ok": required_ok,
        "complete": complete,
    }


def _print_markdown(result: dict[str, Any]) -> None:
    print("| section | ok/status | detail |")
    print("|---|---:|---|")
    print(f"| required files | {all(row['ok'] for row in result['required_files'])} | {len(result['required_files'])} paths |")
    print(f"| content contract | {all(row['ok'] for row in result['content_contract'])} | {len(result['content_contract'])} checks |")
    print(f"| oracle evidence | {result['oracle_evidence']['ok']} | {result['oracle_evidence']['detail']} |")
    direct = result["direct_api_evidence"]
    print(f"| direct API evidence | {direct['status']} | {direct['detail']} |")
    print(f"| required local baseline | {result['required_local_baseline_ok']} | files/docs/oracle |")
    print(f"| complete | {result['complete']} | requires direct API evidence pass |")
    print()
    print("X2_INTEGRATION_AUDIT " + ("COMPLETE" if result["complete"] else "INCOMPLETE"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root to audit.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless the direct API evidence is complete.")
    args = parser.parse_args()

    result = audit(Path(args.repo_root))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_markdown(result)
    if args.strict:
        sys.exit(0 if result["complete"] else 1)
    sys.exit(0 if result["required_local_baseline_ok"] else 1)


if __name__ == "__main__":
    main()
