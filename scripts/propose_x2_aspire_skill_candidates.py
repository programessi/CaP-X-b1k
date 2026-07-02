#!/usr/bin/env python3
"""Propose X2 ASPIRE-lite skill candidates from trace reports.

This is the LLM-facing part of the X2 skill-evolution loop. It reads an
existing ``candidate_search_report.json`` plus the X2 skill library, asks an
OpenAI-compatible model for repair candidates, validates the result against a
small parameter allowlist, and writes a candidate JSON file consumable by
``scripts/run_x2_aspire_rgbd_candidate_search.py --candidate-file``.

The model is deliberately constrained to parameter-level skill candidates. It
cannot rewrite low-level control code or add arbitrary environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ALLOWED_PARAMS: dict[str, dict[str, Any]] = {
    "CAPX_X2_RGBD_CANDIDATE_INDICES": {"type": "csv_int", "min": 0, "max": 8, "max_items": 6},
    "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": {"type": "csv_float", "min": -0.005, "max": 0.03, "max_items": 8},
    "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": {"type": "bool"},
    "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": {"type": "float", "min": 0.04, "max": 0.12},
    "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": {"type": "float", "min": 0.015, "max": 0.06},
    "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": {"type": "float", "min": 0.02, "max": 0.08},
    "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": {"type": "float", "min": 0.04, "max": 0.14},
    "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": {"type": "float", "min": 0.02, "max": 0.06},
    "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": {"type": "float", "min": 0.25, "max": 0.65},
    "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": {"type": "float", "min": 0.018, "max": 0.05},
    "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": {"type": "float", "min": 0.20, "max": 0.45},
    "CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE": {"type": "enum", "values": ["grasp", "post_lift_current"]},
    "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": {"type": "int", "min": 1, "max": 6},
    "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": {"type": "float", "min": 0.004, "max": 0.014},
    "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": {"type": "int", "min": 2, "max": 14},
    "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": {"type": "int", "min": 0, "max": 30},
    "CAPX_X2_RGBD_MAX_JOINT_STEP": {"type": "float", "min": 0.012, "max": 0.028},
    "CAPX_X2_RGBD_INSERT_MAX_JOINT_STEP": {"type": "float", "min": 0.004, "max": 0.012},
    "CAPX_X2_RGBD_TRANSFER_MAX_JOINT_STEP": {"type": "float", "min": 0.012, "max": 0.028},
    "CAPX_X2_RGBD_PLACE_INSERT_MAX_JOINT_STEP": {"type": "float", "min": 0.004, "max": 0.014},
    "CAPX_X2_RGBD_SETTLE_STEPS": {"type": "int", "min": 8, "max": 48},
    "CAPX_X2_RGBD_HOLD_STEPS": {"type": "int", "min": 1, "max": 8},
    "CAPX_X2_RGBD_INSERT_HOLD_STEPS": {"type": "int", "min": 4, "max": 18},
    "CAPX_X2_RGBD_FINE_ALIGN_RETRIES": {"type": "int", "min": 0, "max": 4},
}


BASELINE_PARAMS: dict[str, str] = {
    "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
    "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012,0.016,0.020",
    "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
    "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
    "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.030",
    "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.045",
    "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.085",
    "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.040",
    "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.50",
    "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.035",
    "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.35",
    "CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE": "post_lift_current",
    "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
    "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
    "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
    "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def collect_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    failures: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for split in ("debug", "validation"):
        for item in report.get(f"{split}_results") or []:
            failure = item.get("failure_report") if isinstance(item.get("failure_report"), dict) else {}
            primary = failure.get("primary_failure")
            if primary is not None:
                failures[str(primary)] = failures.get(str(primary), 0) + 1
            if len(examples) < 8:
                examples.append(
                    {
                        "split": split,
                        "candidate_id": item.get("candidate_id"),
                        "seed_id": item.get("seed_id"),
                        "score": item.get("score"),
                        "metrics": item.get("metrics"),
                        "failure_report": failure,
                    }
                )
    return {
        "mode": report.get("mode"),
        "best_candidate_id": report.get("best_candidate_id"),
        "debug_summaries": report.get("debug_summaries"),
        "validation_summaries": report.get("validation_summaries"),
        "failure_counts": failures,
        "examples": examples,
    }


def build_prompt(report: dict[str, Any], skills: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, str]]:
    summary = collect_report_summary(report)
    schema = {
        "candidates": [
            {
                "id": "llm_short_unique_id",
                "description": "why this parameter-level repair should help",
                "params": {"CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE": "post_lift_current"},
                "skills": ["existing_or_new_skill_tag"],
            }
        ]
    }
    content = {
        "task": "Propose X2 RGB-D tabletop pick-place skill candidates.",
        "hard_constraints": [
            "Return JSON only, no markdown.",
            f"Return between 1 and {max_candidates} candidates.",
            "Only use parameters listed in allowed_params.",
            "Do not propose code edits, simulator edits, robot model edits, or new environment variables.",
            "Prefer conservative repairs that can be validated by candidate_search debug/validation seeds.",
            "Include a short unique id beginning with llm_.",
        ],
        "allowed_params": ALLOWED_PARAMS,
        "baseline_params": BASELINE_PARAMS,
        "skill_library": skills,
        "report_summary": summary,
        "output_schema": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "You propose safe parameter-level robot skill candidates for an ASPIRE-lite loop. "
                "You must output strict JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(content, ensure_ascii=True, indent=2)},
    ]


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object.")
    return data


def _parse_csv(value: str, cast: type) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _validate_param_value(key: str, value: Any) -> str:
    spec = ALLOWED_PARAMS[key]
    value_s = str(value)
    kind = spec["type"]
    if kind == "bool":
        if value_s not in {"0", "1", "true", "false", "True", "False"}:
            raise ValueError(f"{key} must be boolean-like, got {value!r}")
        return "1" if value_s in {"1", "true", "True"} else "0"
    if kind == "enum":
        if value_s not in spec["values"]:
            raise ValueError(f"{key} must be one of {spec['values']}, got {value!r}")
        return value_s
    if kind == "int":
        val = int(value_s)
        if not (spec["min"] <= val <= spec["max"]):
            raise ValueError(f"{key}={val} outside [{spec['min']}, {spec['max']}]")
        return str(val)
    if kind == "float":
        val = float(value_s)
        if not (spec["min"] <= val <= spec["max"]):
            raise ValueError(f"{key}={val} outside [{spec['min']}, {spec['max']}]")
        return f"{val:.3f}".rstrip("0").rstrip(".")
    if kind == "csv_int":
        vals = _parse_csv(value_s, int)
        if not vals or len(vals) > spec["max_items"]:
            raise ValueError(f"{key} must contain 1..{spec['max_items']} ints")
        if any(val < spec["min"] or val > spec["max"] for val in vals):
            raise ValueError(f"{key} values outside [{spec['min']}, {spec['max']}]: {vals}")
        return ",".join(str(val) for val in vals)
    if kind == "csv_float":
        vals = _parse_csv(value_s, float)
        if not vals or len(vals) > spec["max_items"]:
            raise ValueError(f"{key} must contain 1..{spec['max_items']} floats")
        if any(val < spec["min"] or val > spec["max"] for val in vals):
            raise ValueError(f"{key} values outside [{spec['min']}, {spec['max']}]: {vals}")
        return ",".join(f"{val:.3f}".rstrip("0").rstrip(".") for val in vals)
    raise ValueError(f"Unhandled param type for {key}: {kind}")


def validate_candidates(payload: dict[str, Any], *, max_candidates: int) -> list[dict[str, Any]]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Payload must contain a non-empty candidates list.")
    if len(raw_candidates) > max_candidates:
        raw_candidates = raw_candidates[:max_candidates]
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate #{index} must be an object.")
        candidate_id = str(candidate.get("id") or "").strip()
        if not re.fullmatch(r"llm_[a-zA-Z0-9_]{3,64}", candidate_id):
            raise ValueError(f"Candidate id must match llm_[a-zA-Z0-9_]+, got {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate id: {candidate_id}")
        raw_params = candidate.get("params")
        if not isinstance(raw_params, dict) or not raw_params:
            raise ValueError(f"{candidate_id} must include non-empty params.")
        params = dict(BASELINE_PARAMS)
        for key, value in raw_params.items():
            if key not in ALLOWED_PARAMS:
                raise ValueError(f"{candidate_id} proposed unsupported parameter {key!r}")
            params[str(key)] = _validate_param_value(str(key), value)
        candidates.append(
            {
                "id": candidate_id,
                "description": str(candidate.get("description") or "LLM-proposed ASPIRE-lite candidate."),
                "params": params,
                "skills": [str(item) for item in candidate.get("skills", [])],
                "source": "llm_skill_proposer",
            }
        )
        seen.add(candidate_id)
    return candidates


def mock_payload_from_report(report: dict[str, Any]) -> dict[str, Any]:
    failures = collect_report_summary(report)["failure_counts"]
    candidates: list[dict[str, Any]] = []
    if failures.get("object_not_in_hand_after_close") or failures.get("preclose_pose_not_reached"):
        candidates.append(
            {
                "id": "llm_deeper_candidate_retry",
                "description": "Try validated candidates with deeper TCP-axis close offsets and slightly relaxed final reach gates.",
                "params": {
                    "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2,3",
                    "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0,0.004,0.008,0.012,0.016,0.02,0.024",
                    "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.038",
                    "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.35",
                },
                "skills": ["try_next_grasp_candidate", "increase_grasp_tcp_axis_offsets"],
            }
        )
    if failures.get("place_pre_pose_not_reached") or failures.get("place_error_too_large") or not candidates:
        candidates.append(
            {
                "id": "llm_post_lift_slow_place",
                "description": "Keep post-lift TCP orientation and slow the vertical place descent to reduce place-stage jumps.",
                "params": {
                    "CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE": "post_lift_current",
                    "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "5",
                    "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.005",
                    "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "10",
                    "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "20",
                },
                "skills": ["use_post_lift_place_orientation", "slow_vertical_place_descent"],
            }
        )
    return {"candidates": candidates}


def call_chat_completions(server_url: str, model: str, messages: list[dict[str, str]], timeout_s: float) -> str:
    response = requests.post(
        server_url,
        json={"model": model, "messages": messages, "temperature": 0.2, "stream": False},
        timeout=float(timeout_s),
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def messages_to_codex_prompt(messages: list[dict[str, str]]) -> str:
    parts = [
        "You are proposing safe parameter-level ASPIRE-lite robot skill candidates.",
        "Return strict JSON only. Do not inspect files or run commands.",
    ]
    for message in messages:
        parts.append(f"### {message['role']}\n{message['content']}")
    return "\n\n".join(parts).strip() + "\n"


def call_codex_cli(
    *,
    messages: list[dict[str, str]],
    codex_bin: str,
    model_provider: str,
    cwd: Path,
    timeout_s: float,
) -> str:
    prompt = messages_to_codex_prompt(messages)
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=True) as out_file:
        cmd = [
            codex_bin,
            "-c",
            f'model_provider="{model_provider}"',
            "-a",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            str(cwd),
            "--output-last-message",
            out_file.name,
            "-",
        ]
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_s),
            check=False,
        )
        output = Path(out_file.name).read_text(encoding="utf-8").strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"codex exec failed rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n\nstderr:\n{completed.stderr[-4000:]}"
        )
    return output or completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Input candidate_search_report.json.")
    parser.add_argument("--skills", default="capx/integrations/x2/aspire_skills.json")
    parser.add_argument("--output-dir", default=None, help="Defaults to <report parent>/llm_proposals/<stamp>.")
    parser.add_argument("--server-url", default=os.getenv("SERVER_URL", "http://127.0.0.1:8120/chat/completions"))
    parser.add_argument("--model", default=os.getenv("ASPIRE_LLM_MODEL", "codex-a"))
    parser.add_argument("--codex-cli", action="store_true", help="Call local Codex CLI directly instead of an HTTP server.")
    parser.add_argument("--codex-bin", default=os.getenv("CODEX_BIN", "codex"))
    parser.add_argument("--model-provider", default=os.getenv("CODEX_MODEL_PROVIDER", "axonhub"))
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local proposal instead of calling an LLM.")
    args = parser.parse_args()

    report_path = Path(args.report).resolve()
    report = read_json(report_path)
    skills = read_json((REPO_ROOT / args.skills).resolve() if not Path(args.skills).is_absolute() else Path(args.skills))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else report_path.parent / "llm_proposals" / stamp
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    messages = build_prompt(report, skills, max_candidates=int(args.max_candidates))
    write_json(output_dir / "prompt_messages.json", messages)
    if args.mock:
        raw_text = json.dumps(mock_payload_from_report(report), indent=2)
    elif args.codex_cli:
        raw_text = call_codex_cli(
            messages=messages,
            codex_bin=str(args.codex_bin),
            model_provider=str(args.model_provider),
            cwd=REPO_ROOT,
            timeout_s=float(args.timeout_seconds),
        )
    else:
        raw_text = call_chat_completions(
            str(args.server_url),
            str(args.model),
            messages,
            timeout_s=float(args.timeout_seconds),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    payload = extract_json_payload(raw_text)
    candidates = validate_candidates(payload, max_candidates=int(args.max_candidates))
    result = {
        "source_report": str(report_path),
        "model": str(args.model),
        "server_url": "mock" if args.mock else ("codex-cli" if args.codex_cli else str(args.server_url)),
        "created_at": stamp,
        "allowed_param_keys": sorted(ALLOWED_PARAMS),
        "candidates": candidates,
    }
    write_json(output_dir / "candidates.json", result)
    print(json.dumps({"output_dir": str(output_dir), "candidate_file": str(output_dir / "candidates.json"), "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
