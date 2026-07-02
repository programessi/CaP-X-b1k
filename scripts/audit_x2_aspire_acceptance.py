#!/usr/bin/env python3
"""Audit an X2 ASPIRE-lite candidate-search report against acceptance gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_trials(summary: dict[str, Any] | None) -> int:
    return int((summary or {}).get("trials") or 0)


def _candidate_successes(summary: dict[str, Any] | None) -> int:
    return int((summary or {}).get("successes") or 0)


def _candidate_metric(summary: dict[str, Any] | None, key: str) -> float | None:
    return _to_float((summary or {}).get(key))


def _split_summaries(report: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    summaries = report.get(f"{split}_summaries")
    return summaries if isinstance(summaries, dict) else {}


def _split_results(report: dict[str, Any], split: str) -> list[dict[str, Any]]:
    results = report.get(f"{split}_results")
    return results if isinstance(results, list) else []


def _best_validation_summary(report: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    best = report.get("best_candidate_id")
    summaries = _split_summaries(report, "validation")
    if isinstance(best, str) and best in summaries:
        return best, summaries[best]
    if len(summaries) == 1:
        candidate_id, summary = next(iter(summaries.items()))
        return str(candidate_id), summary
    return best if isinstance(best, str) else None, None


def _has_non_unknown_failure(report: dict[str, Any]) -> bool:
    for split in ("debug", "validation"):
        for item in _split_results(report, split):
            failure = item.get("failure_report") if isinstance(item.get("failure_report"), dict) else {}
            primary = failure.get("primary_failure")
            if primary not in (None, "unknown"):
                return True
    return False


def _uses_rgbd_non_oracle_obstacles(report: dict[str, Any]) -> bool:
    saw_metrics = False
    for split in ("debug", "validation"):
        for item in _split_results(report, split):
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            if not metrics:
                continue
            saw_metrics = True
            obstacle_source = metrics.get("obstacle_source")
            sim_truth = metrics.get("rgbd_obstacles_sim_truth")
            if obstacle_source is not None and obstacle_source != "rgbd_visual":
                return False
            if sim_truth is not None and sim_truth is not False:
                return False
    return saw_metrics


def audit_report(
    report: dict[str, Any],
    *,
    min_debug_seeds: int = 3,
    min_validation_seeds: int = 3,
    min_validation_successes: int = 3,
    max_avg_tcp_error_m: float = 0.025,
    max_avg_ori_error_rad: float = 0.08,
    max_avg_place_error_m: float = 0.05,
) -> dict[str, Any]:
    """Return structured pass/fail checks for the ASPIRE-lite objective."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, details: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "details": details})

    candidates = report.get("candidates") if isinstance(report.get("candidates"), list) else []
    debug_seeds = report.get("debug_seeds") if isinstance(report.get("debug_seeds"), list) else []
    validation_seeds = report.get("validation_seeds") if isinstance(report.get("validation_seeds"), list) else []
    debug_summaries = _split_summaries(report, "debug")
    best_id, validation_summary = _best_validation_summary(report)

    debug_candidates_with_trials = [
        candidate_id for candidate_id, summary in debug_summaries.items() if _candidate_trials(summary) > 0
    ]
    validation_trials = _candidate_trials(validation_summary)
    validation_successes = _candidate_successes(validation_summary)
    avg_tcp = _candidate_metric(validation_summary, "avg_before_close_tcp_error_m")
    avg_ori = _candidate_metric(validation_summary, "avg_before_close_ori_error_rad")
    avg_place = _candidate_metric(validation_summary, "avg_place_error_m")
    validation_traces = int((validation_summary or {}).get("trace_bundles") or 0)
    validation_videos = int((validation_summary or {}).get("videos") or 0)

    add("execute_mode", report.get("mode") in {"execute", "execute_aggregate"}, {"mode": report.get("mode")})
    add("multiple_candidates", len(candidates) >= 2, {"candidate_count": len(candidates)})
    add("debug_seed_split", len(debug_seeds) >= min_debug_seeds, {"debug_seed_count": len(debug_seeds)})
    add(
        "validation_seed_split",
        len(validation_seeds) >= min_validation_seeds,
        {"validation_seed_count": len(validation_seeds)},
    )
    add(
        "debug_candidate_search_ran",
        len(debug_candidates_with_trials) >= 2,
        {"debug_candidates_with_trials": debug_candidates_with_trials},
    )
    add("best_candidate_selected", bool(best_id), {"best_candidate_id": best_id})
    add(
        "validation_trials_ran",
        validation_trials >= min_validation_seeds,
        {"validation_trials": validation_trials, "required": min_validation_seeds},
    )
    add(
        "validation_success_gate",
        validation_successes >= min_validation_successes,
        {"validation_successes": validation_successes, "required": min_validation_successes},
    )
    add(
        "validation_tcp_error_gate",
        avg_tcp is not None and avg_tcp < max_avg_tcp_error_m,
        {"avg_before_close_tcp_error_m": avg_tcp, "threshold": max_avg_tcp_error_m},
    )
    add(
        "validation_ori_error_gate",
        avg_ori is not None and avg_ori < max_avg_ori_error_rad,
        {"avg_before_close_ori_error_rad": avg_ori, "threshold": max_avg_ori_error_rad},
    )
    add(
        "validation_place_error_gate",
        avg_place is not None and avg_place < max_avg_place_error_m,
        {"avg_place_error_m": avg_place, "threshold": max_avg_place_error_m},
    )
    add(
        "validation_trace_and_video_gate",
        validation_trials > 0 and validation_traces >= validation_trials and validation_videos >= validation_trials,
        {"validation_trials": validation_trials, "trace_bundles": validation_traces, "videos": validation_videos},
    )
    add(
        "non_oracle_rgbd_obstacle_gate",
        _uses_rgbd_non_oracle_obstacles(report),
        {"required_obstacle_source": "rgbd_visual", "required_rgbd_obstacles_sim_truth": False},
    )
    add(
        "failure_taxonomy_evidence",
        _has_non_unknown_failure(report),
        {"required": "at least one non-unknown primary_failure in executed results"},
    )

    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "best_candidate_id": best_id,
    }


def _write_markdown(audit: dict[str, Any]) -> str:
    lines = ["# X2 ASPIRE Acceptance Audit", ""]
    lines.append(f"- ok: `{audit['ok']}`")
    lines.append(f"- best_candidate_id: `{audit.get('best_candidate_id') or '-'}`")
    lines.append("")
    lines.append("| check | ok | details |")
    lines.append("|---|---:|---|")
    for check in audit["checks"]:
        lines.append(
            f"| {check['name']} | {check['ok']} | "
            f"`{json.dumps(check['details'], ensure_ascii=True, sort_keys=True)}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to candidate_search_report.json")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    parser.add_argument("--min-debug-seeds", type=int, default=3)
    parser.add_argument("--min-validation-seeds", type=int, default=3)
    parser.add_argument("--min-validation-successes", type=int, default=3)
    parser.add_argument("--max-avg-tcp-error-m", type=float, default=0.025)
    parser.add_argument("--max-avg-ori-error-rad", type=float, default=0.08)
    parser.add_argument("--max-avg-place-error-m", type=float, default=0.05)
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = audit_report(
        report,
        min_debug_seeds=int(args.min_debug_seeds),
        min_validation_seeds=int(args.min_validation_seeds),
        min_validation_successes=int(args.min_validation_successes),
        max_avg_tcp_error_m=float(args.max_avg_tcp_error_m),
        max_avg_ori_error_rad=float(args.max_avg_ori_error_rad),
        max_avg_place_error_m=float(args.max_avg_place_error_m),
    )
    if args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=True, sort_keys=True))
    else:
        print(_write_markdown(audit), end="")
    raise SystemExit(0 if audit["ok"] else 1)


if __name__ == "__main__":
    main()
