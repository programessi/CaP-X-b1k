#!/usr/bin/env python3
"""Run an LLM-driven X2 ASPIRE-lite skill-evolution loop.

One iteration is:

1. Read a previous ``candidate_search_report.json``.
2. Ask an LLM proposer for parameter-level skill candidates.
3. Run the candidate-search harness with those candidates.
4. Optionally audit the executed report and feed it into the next iteration.

By default this script is plan-only. Pass ``--execute`` to launch the expensive
BEHAVIOR/Isaac simulation runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], *, cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.returncode, completed.stdout


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-report", required=True, help="Initial candidate_search_report.json to learn from.")
    parser.add_argument("--output-root", default=None, help="Defaults to outputs/x2_aspire_llm_skill_evolution/<stamp>.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Run BEHAVIOR/Isaac candidate search. Default only writes plans.")
    parser.add_argument("--mock-llm", action="store_true", help="Use deterministic local proposer instead of calling an LLM.")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--debug-limit", type=int, default=3)
    parser.add_argument("--validation-limit", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=3)
    args = parser.parse_args()

    seed_report = Path(args.seed_report).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs" / "x2_aspire_llm_skill_evolution" / stamp
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    current_report = seed_report
    iterations: list[dict[str, Any]] = []
    for idx in range(max(1, int(args.iterations))):
        iteration_root = output_root / f"iteration_{idx + 1:02d}"
        proposal_dir = iteration_root / "proposal"
        search_dir = iteration_root / "candidate_search"
        proposer_cmd = [
            sys.executable,
            "scripts/propose_x2_aspire_skill_candidates.py",
            "--report",
            str(current_report),
            "--output-dir",
            str(proposal_dir),
            "--max-candidates",
            str(int(args.candidate_limit)),
        ]
        if args.mock_llm:
            proposer_cmd.append("--mock")
        if args.server_url:
            proposer_cmd.extend(["--server-url", str(args.server_url)])
        if args.model:
            proposer_cmd.extend(["--model", str(args.model)])
        rc, out = run_command(proposer_cmd, cwd=REPO_ROOT)
        (iteration_root / "proposer.log").write_text(out, encoding="utf-8")
        if rc != 0:
            iterations.append({"iteration": idx + 1, "stage": "propose", "ok": False, "returncode": rc})
            break
        candidate_file = proposal_dir / "candidates.json"
        search_cmd = [
            sys.executable,
            "scripts/run_x2_aspire_rgbd_candidate_search.py",
            "--candidate-file",
            str(candidate_file),
            "--output-root",
            str(search_dir),
            "--timeout-seconds",
            str(int(args.timeout_seconds)),
            "--debug-limit",
            str(int(args.debug_limit)),
            "--validation-limit",
            str(int(args.validation_limit)),
        ]
        if args.execute:
            search_cmd.append("--execute")
        rc, out = run_command(search_cmd, cwd=REPO_ROOT)
        (iteration_root / "candidate_search.log").write_text(out, encoding="utf-8")
        search_report = search_dir / "candidate_search_report.json"
        audit_path = None
        audit_ok = None
        if args.execute and search_report.exists():
            audit_cmd = [sys.executable, "scripts/audit_x2_aspire_acceptance.py", str(search_report), "--json"]
            audit_rc, audit_out = run_command(audit_cmd, cwd=REPO_ROOT)
            audit_path = iteration_root / "audit.json"
            audit_path.write_text(audit_out, encoding="utf-8")
            audit_ok = audit_rc == 0
        iterations.append(
            {
                "iteration": idx + 1,
                "ok": rc == 0,
                "execute": bool(args.execute),
                "source_report": str(current_report),
                "candidate_file": str(candidate_file),
                "search_report": str(search_report),
                "audit_json": None if audit_path is None else str(audit_path),
                "audit_ok": audit_ok,
            }
        )
        if rc != 0 or not search_report.exists():
            break
        current_report = search_report
        if audit_ok is True:
            break

    summary = {
        "output_root": str(output_root),
        "execute": bool(args.execute),
        "mock_llm": bool(args.mock_llm),
        "iterations": iterations,
    }
    write_json(output_root / "evolution_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
