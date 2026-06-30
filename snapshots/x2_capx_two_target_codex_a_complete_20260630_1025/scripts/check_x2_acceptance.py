#!/usr/bin/env python3
"""Check saved X2 CaP-X runs against the current acceptance thresholds.

This reads local CaP-X output folders only. It does not contact an LLM,
simulator, or perception service.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_summary_module():
    script = Path(__file__).resolve().with_name("summarize_x2_runs.py")
    spec = importlib.util.spec_from_file_location("summarize_x2_runs", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_code(row: dict[str, Any]) -> str:
    code_path = row.get("code_path")
    if code_path:
        try:
            return Path(str(code_path)).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
    return str(row.get("code_preview") or "")


def _code_uses_expected_primitive(row: dict[str, Any], target: str) -> tuple[bool, str]:
    code = _read_code(row)
    expected = f"pick_and_place_red_cube_to_{target}_target"
    target_primitives = [
        "pick_and_place_red_cube_to_left_target",
        "pick_and_place_red_cube_to_right_target",
    ]
    forbidden = [
        "plan_visual_grasp_tcp_pose",
        "execute_tcp_grasp_plan",
        "X2ControlApi",
        "move_tcp_joint_ik",
        "open_gripper",
        "close_gripper",
        "env.reset",
    ]
    if expected not in code:
        return False, f"generated code does not call {expected}()"
    primitive_call_count = sum(code.count(f"{name}(") for name in target_primitives)
    if primitive_call_count != 1:
        return False, f"generated code should call exactly one target primitive, found {primitive_call_count}"
    used_forbidden = [name for name in forbidden if name in code]
    if used_forbidden:
        return False, "generated code uses forbidden lower-level APIs: " + ", ".join(used_forbidden)
    return True, "ok"


def _check_row(
    row: dict[str, Any],
    *,
    max_tcp_error_m: float,
    max_ori_error_rad: float,
    max_place_error_m: float,
    require_video: bool,
    require_visual: bool,
    require_code: bool,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    result = row.get("result") or {}
    target = str(result.get("target") or "")

    if not _as_bool_text(result.get("ok")):
        errors.append("result ok is not True")
    if not _as_bool_text(row.get("task_completed")):
        errors.append("task_completed is not 1/True")
    if not _as_bool_text(result.get("object_in_hand_after_close")):
        errors.append("object_in_hand_after_close is not True")

    tcp_error = _as_float(result.get("before_close_tcp_error_m"))
    if tcp_error is None or tcp_error > max_tcp_error_m:
        errors.append(f"before_close_tcp_error_m={tcp_error} exceeds {max_tcp_error_m}")

    ori_error = _as_float(result.get("before_close_ori_error_rad"))
    if ori_error is None or ori_error > max_ori_error_rad:
        errors.append(f"before_close_ori_error_rad={ori_error} exceeds {max_ori_error_rad}")

    place_error = _as_float(result.get("place_error_m"))
    if place_error is None or place_error > max_place_error_m:
        errors.append(f"place_error_m={place_error} exceeds {max_place_error_m}")

    if require_video and not row.get("videos"):
        errors.append("no video_*.mp4 found")
    if require_visual and not row.get("visual_artifact_dirs"):
        errors.append("no matching visual artifact directory with grasp_summary.json found")
    if require_code:
        if not row.get("code_path"):
            errors.append("code.py not found")
        elif target:
            ok, message = _code_uses_expected_primitive(row, target)
            if not ok:
                errors.append(message)

    return not errors, errors


def _collect_rows(paths: list[str], repo_root: Path) -> list[dict[str, Any]]:
    summary = _load_summary_module()
    trial_dirs: list[Path] = []
    for raw_path in paths:
        trial_dirs.extend(summary._find_trial_dirs(Path(raw_path)))
    trial_dirs = sorted(set(path.resolve() for path in trial_dirs))
    return [summary._parse_trial(repo_root, path) for path in trial_dirs]


def check_acceptance(args: argparse.Namespace) -> tuple[bool, list[dict[str, Any]]]:
    repo_root = Path(args.repo_root).resolve()
    rows = _collect_rows(args.paths, repo_root)
    required_targets = [target.strip() for target in args.require_targets.split(",") if target.strip()]
    seen_targets: set[str] = set()
    checked: list[dict[str, Any]] = []

    for row in rows:
        target = str((row.get("result") or {}).get("target") or "")
        if target:
            seen_targets.add(target)
        ok, errors = _check_row(
            row,
            max_tcp_error_m=args.max_tcp_error_m,
            max_ori_error_rad=args.max_ori_error_rad,
            max_place_error_m=args.max_place_error_m,
            require_video=not args.allow_missing_video,
            require_visual=not args.allow_missing_visual,
            require_code=not args.allow_missing_code,
        )
        checked.append({"row": row, "ok": ok, "errors": errors})

    missing_targets = sorted(set(required_targets) - seen_targets)
    if missing_targets:
        checked.append(
            {
                "row": {"trial_dir": "<required-targets>", "result": {"target": ",".join(missing_targets)}},
                "ok": False,
                "errors": ["missing required target run(s): " + ", ".join(missing_targets)],
            }
        )

    if not rows:
        checked.append(
            {
                "row": {"trial_dir": "<paths>", "result": {}},
                "ok": False,
                "errors": ["no trial_* directories found in supplied paths"],
            }
        )

    return all(item["ok"] for item in checked), checked


def _print_report(checked: list[dict[str, Any]]) -> None:
    print("| trial | target | ok | tcp_err_m | ori_err_rad | place_err_m | videos | visual | errors |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for item in checked:
        row = item["row"]
        result = row.get("result") or {}
        errors = item.get("errors") or []
        print(
            "| "
            + " | ".join(
                [
                    str(row.get("trial_dir", "-")),
                    str(result.get("target", "-")),
                    str(bool(item["ok"])),
                    str(result.get("before_close_tcp_error_m", "-")),
                    str(result.get("before_close_ori_error_rad", "-")),
                    str(result.get("place_error_m", "-")),
                    str(len(row.get("videos") or [])),
                    str(len(row.get("visual_artifact_dirs") or [])),
                    "; ".join(errors) if errors else "-",
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Output run directories or trial directories to check.")
    parser.add_argument("--repo-root", default=".", help="Repository root for visual artifact lookup.")
    parser.add_argument("--require-targets", default="right,left", help="Comma-separated targets that must appear.")
    parser.add_argument("--max-tcp-error-m", type=float, default=0.02)
    parser.add_argument("--max-ori-error-rad", type=float, default=0.10)
    parser.add_argument("--max-place-error-m", type=float, default=0.10)
    parser.add_argument("--allow-missing-video", action="store_true")
    parser.add_argument("--allow-missing-visual", action="store_true")
    parser.add_argument("--allow-missing-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    ok, checked = check_acceptance(args)
    if args.json:
        print(json.dumps(checked, indent=2, sort_keys=True))
    else:
        _print_report(checked)
        print()
        print("X2_ACCEPTANCE " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
