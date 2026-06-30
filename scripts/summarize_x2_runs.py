#!/usr/bin/env python3
"""Summarize X2 CaP-X run outputs.

This reads saved CaP-X trial folders only. It does not contact any LLM,
simulator, or perception service.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(
    r"X2_(?:TWO_TARGET_|PICK_PLACE_)?RESULT\s+"
    r"(?P<fields>.*?)(?=\nX2_|\n\s*Stderr:|\Z)",
    re.DOTALL,
)
ATTEMPT_RE = re.compile(r"X2_TWO_TARGET_ATTEMPT\s+(?P<fields>.*)")


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", text):
        fields[key] = value
    return fields


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _find_trial_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return []
    if root.name.startswith("trial_"):
        return [root]
    return sorted(path for path in root.rglob("trial_*") if path.is_dir())


def _trial_parent_run_name(trial_dir: Path) -> str:
    return trial_dir.parent.name


def _find_visual_artifacts(repo_root: Path, run_name: str) -> list[Path]:
    visual_root = repo_root / "outputs" / "x2_visual_artifacts"
    if not visual_root.exists():
        return []
    return sorted(path.parent for path in visual_root.rglob("grasp_summary.json") if run_name in str(path))


def _first_code_lines(code_path: Path, max_lines: int = 12) -> str:
    text = _read_text(code_path)
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines[:max_lines])


def _parse_trial(repo_root: Path, trial_dir: Path) -> dict[str, Any]:
    summary_text = _read_text(trial_dir / "summary.txt")
    code_path = trial_dir / "code.py"
    result_fields: dict[str, str] = {}
    attempts: list[dict[str, str]] = []

    match = RESULT_RE.search(summary_text)
    if match:
        result_fields = _parse_fields(match.group("fields"))
    for attempt_match in ATTEMPT_RE.finditer(summary_text):
        attempts.append(_parse_fields(attempt_match.group("fields")))

    if not result_fields:
        raw_response = _read_text(trial_dir / "raw_response.sh")
        match = RESULT_RE.search(raw_response)
        if match:
            result_fields = _parse_fields(match.group("fields"))
        for attempt_match in ATTEMPT_RE.finditer(raw_response):
            attempts.append(_parse_fields(attempt_match.group("fields")))
    attempts = [attempt for attempt in attempts if "candidate_index" in attempt]

    video_paths = sorted(str(path) for path in trial_dir.glob("video_*.mp4"))
    run_name = _trial_parent_run_name(trial_dir)
    visual_dirs = [str(path) for path in _find_visual_artifacts(repo_root, run_name)]
    reward_match = re.search(r"_reward_([0-9.]+)_taskcompleted_([01])", trial_dir.name)
    reward = reward_match.group(1) if reward_match else None
    task_completed = reward_match.group(2) if reward_match else None

    return {
        "trial_dir": str(trial_dir),
        "run_name": run_name,
        "reward": reward,
        "task_completed": task_completed,
        "result": result_fields,
        "attempts": attempts,
        "code_path": str(code_path) if code_path.exists() else None,
        "code_preview": _first_code_lines(code_path),
        "videos": video_paths,
        "visual_artifact_dirs": visual_dirs,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if text else "-"


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| trial | ok | target | reward | task | tcp_err_m | ori_err_rad | in_hand | place_err_m | videos | visual |")
    print("|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|")
    for row in rows:
        result = row.get("result") or {}
        print(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("trial_dir")),
                    _fmt(result.get("ok")),
                    _fmt(result.get("target")),
                    _fmt(row.get("reward")),
                    _fmt(row.get("task_completed")),
                    _fmt(result.get("before_close_tcp_error_m")),
                    _fmt(result.get("before_close_ori_error_rad")),
                    _fmt(result.get("object_in_hand_after_close")),
                    _fmt(result.get("place_error_m")),
                    str(len(row.get("videos") or [])),
                    str(len(row.get("visual_artifact_dirs") or [])),
                ]
            )
            + " |"
        )

    print()
    for idx, row in enumerate(rows, 1):
        print(f"## Trial {idx}")
        print()
        print(f"- trial_dir: `{row['trial_dir']}`")
        print(f"- code_path: `{row.get('code_path') or '-'}`")
        videos = row.get("videos") or []
        visual_dirs = row.get("visual_artifact_dirs") or []
        if videos:
            print("- videos:")
            for path in videos:
                print(f"  - `{path}`")
        if visual_dirs:
            print("- visual_artifact_dirs:")
            for path in visual_dirs:
                print(f"  - `{path}`")
        attempts = row.get("attempts") or []
        if attempts:
            print("- attempts:")
            for attempt in attempts:
                print(f"  - `{json.dumps(attempt, sort_keys=True)}`")
        code_preview = row.get("code_preview") or ""
        if code_preview:
            print("- code_preview:")
            print("```python")
            print(code_preview)
            print("```")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Output run directories or trial directories to summarize.")
    parser.add_argument("--repo-root", default=".", help="Repository root for visual artifact lookup.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    trial_dirs: list[Path] = []
    for raw_path in args.paths:
        trial_dirs.extend(_find_trial_dirs(Path(raw_path)))
    trial_dirs = sorted(set(path.resolve() for path in trial_dirs))
    rows = [_parse_trial(repo_root, path) for path in trial_dirs]

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_markdown(rows)


if __name__ == "__main__":
    main()
