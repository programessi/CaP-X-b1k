#!/usr/bin/env python3
"""Aggregate executed X2 ASPIRE trace bundles into one auditable report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.integrations.x2.aspire import jsonable, read_json


def _score(metrics: dict[str, Any], failure_report: dict[str, Any]) -> float:
    completed = 1.0 if metrics.get("task_completed") or metrics.get("ok") else 0.0
    reward = float(metrics.get("reward") or 0.0)
    tcp_error = float(metrics.get("before_close_tcp_error_m") or 0.20)
    ori_error = float(metrics.get("before_close_ori_error_rad") or 1.0)
    place_error = float(metrics.get("place_error_m") or 0.20)
    failure_penalty = 0.0 if failure_report.get("primary_failure") is None else 50.0
    return 200.0 * completed + 100.0 * reward - 60.0 * tcp_error - 10.0 * ori_error - 60.0 * place_error - failure_penalty


def _mean_metric(results: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for item in results:
        value = (item.get("metrics") or {}).get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _bundle_has_video(bundle_dir: str | None) -> bool:
    if not bundle_dir:
        return False
    return any(Path(bundle_dir).glob("videos/*.mp4"))


def _summarize_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"score": float("-inf"), "successes": 0, "trials": 0}
    successes = sum(1 for item in results if (item.get("metrics") or {}).get("task_completed") or (item.get("metrics") or {}).get("ok"))
    return {
        "score": sum(float(item.get("score", 0.0)) for item in results) / len(results),
        "successes": successes,
        "trials": len(results),
        "primary_failures": [(item.get("failure_report") or {}).get("primary_failure") for item in results],
        "avg_before_close_tcp_error_m": _mean_metric(results, "before_close_tcp_error_m"),
        "avg_before_close_ori_error_rad": _mean_metric(results, "before_close_ori_error_rad"),
        "avg_place_error_m": _mean_metric(results, "place_error_m"),
        "trace_bundles": sum(1 for item in results if item.get("bundle_dir") and Path(str(item["bundle_dir"])).exists()),
        "videos": sum(1 for item in results if _bundle_has_video(item.get("bundle_dir"))),
    }


def _parse_trace_context(bundle_dir: Path) -> tuple[str, str, str] | None:
    parts = list(bundle_dir.parts)
    try:
        trace_idx = len(parts) - 1 - parts[::-1].index("trace")
    except ValueError:
        return None
    if trace_idx < 3:
        return None
    split, candidate_id, seed_id = parts[trace_idx - 3 : trace_idx]
    if split not in {"debug", "validation"}:
        return None
    return split, candidate_id, seed_id


def _result_from_bundle(bundle_dir: Path, *, source: str) -> dict[str, Any] | None:
    context = _parse_trace_context(bundle_dir)
    if context is None:
        return None
    metrics_path = bundle_dir / "metrics.json"
    failure_path = bundle_dir / "failure_report.json"
    if not metrics_path.exists() or not failure_path.exists():
        return None
    split, candidate_id, seed_id = context
    metrics = read_json(metrics_path)
    failure_report = read_json(failure_path)
    return {
        "candidate_id": candidate_id,
        "seed_id": seed_id,
        "split": split,
        "returncode": 0,
        "elapsed_s": None,
        "reused_existing": True,
        "output_dir": None,
        "visual_artifact_dir": None,
        "trace_dir": str(bundle_dir.parent),
        "bundle_dir": str(bundle_dir),
        "metrics": metrics,
        "failure_report": failure_report,
        "trace_build_error": None,
        "score": _score(metrics, failure_report),
        "source": source,
    }


def _seed_rows(results: list[dict[str, Any]], split: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for item in results:
        if item.get("split") != split:
            continue
        seed_id = str(item.get("seed_id"))
        if seed_id in seen:
            continue
        seen.add(seed_id)
        rows.append({"id": seed_id})
    return rows


def _candidate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in results:
        candidate_id = str(item.get("candidate_id"))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        rows.append({"id": candidate_id, "description": "Aggregated from executed trace evidence.", "params": {}, "skills": []})
    return rows


def _add_report_results(results: list[dict[str, Any]], report_path: Path) -> None:
    report = read_json(report_path)
    for split, key in (("debug", "debug_results"), ("validation", "validation_results")):
        for item in report.get(key) or []:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied["split"] = split
            copied["source"] = str(report_path)
            metrics = copied.get("metrics") if isinstance(copied.get("metrics"), dict) else {}
            failure = copied.get("failure_report") if isinstance(copied.get("failure_report"), dict) else {}
            copied["score"] = float(copied.get("score", _score(metrics, failure)) or 0.0)
            results.append(copied)


def aggregate_evidence(paths: list[Path]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    seen_bundles: set[str] = set()

    for root in paths:
        if root.is_file() and root.name == "candidate_search_report.json":
            _add_report_results(results, root)
            for item in results:
                if item.get("bundle_dir"):
                    seen_bundles.add(str(Path(str(item["bundle_dir"])).resolve()))
        elif root.is_dir():
            for report_path in sorted(root.rglob("candidate_search_report.json")):
                _add_report_results(results, report_path)
                for item in results:
                    if item.get("bundle_dir"):
                        seen_bundles.add(str(Path(str(item["bundle_dir"])).resolve()))

    for root in paths:
        search_root = root.parent if root.is_file() else root
        if not search_root.exists():
            continue
        for metrics_path in sorted(search_root.rglob("trace/*/metrics.json")):
            bundle_dir = metrics_path.parent.resolve()
            if str(bundle_dir) in seen_bundles:
                continue
            item = _result_from_bundle(bundle_dir, source=str(search_root))
            if item is None:
                continue
            seen_bundles.add(str(bundle_dir))
            results.append(item)

    debug_results = [item for item in results if item.get("split") == "debug"]
    validation_results = [item for item in results if item.get("split") == "validation"]
    candidate_ids = sorted({str(item.get("candidate_id")) for item in results})
    debug_summaries = {
        candidate_id: _summarize_candidate([item for item in debug_results if item.get("candidate_id") == candidate_id])
        for candidate_id in candidate_ids
    }
    validation_summaries = {
        candidate_id: _summarize_candidate([item for item in validation_results if item.get("candidate_id") == candidate_id])
        for candidate_id in candidate_ids
        if any(item.get("candidate_id") == candidate_id for item in validation_results)
    }
    best_candidate_id = None
    if debug_summaries:
        best_candidate_id = max(debug_summaries, key=lambda candidate_id: debug_summaries[candidate_id]["score"])
    return {
        "mode": "execute_aggregate",
        "candidates": _candidate_rows(results),
        "debug_seeds": _seed_rows(debug_results, "debug"),
        "validation_seeds": _seed_rows(validation_results, "validation"),
        "plan": [],
        "debug_results": debug_results,
        "validation_results": validation_results,
        "debug_summaries": debug_summaries,
        "validation_summaries": validation_summaries,
        "best_candidate_id": best_candidate_id,
    }


def _write_report(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_search_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    lines = ["# X2 ASPIRE Aggregated Evidence", ""]
    lines.append(f"- mode: `{report['mode']}`")
    lines.append(f"- best_candidate: `{report.get('best_candidate_id') or '-'}`")
    lines.append("")
    lines.append("| split | candidate | successes | trials | score | failures |")
    lines.append("|---|---|---:|---:|---:|---|")
    for split in ("debug", "validation"):
        for candidate_id, summary in sorted((report.get(f"{split}_summaries") or {}).items()):
            failures = ", ".join(str(item) for item in summary.get("primary_failures", []))
            lines.append(
                f"| {split} | {candidate_id} | {summary.get('successes', 0)} | "
                f"{summary.get('trials', 0)} | {summary.get('score', 0.0):.3f} | {failures} |"
            )
    (output_root / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Candidate-search roots or candidate_search_report.json files.")
    parser.add_argument("--output-root", required=True, help="Directory for the aggregated report.")
    parser.add_argument("--json", action="store_true", help="Print the aggregated report JSON.")
    args = parser.parse_args()

    report = aggregate_evidence([Path(path).resolve() for path in args.paths])
    output_root = Path(args.output_root).resolve()
    _write_report(output_root, report)
    if args.json:
        print(json.dumps(jsonable(report), indent=2, ensure_ascii=True, sort_keys=True))
    else:
        print(json.dumps({"output_root": str(output_root), "best_candidate_id": report.get("best_candidate_id")}, indent=2))


if __name__ == "__main__":
    main()
