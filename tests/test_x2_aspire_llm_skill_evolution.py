from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llm_candidate_validation_fills_baseline_and_blocks_unknown_params():
    proposer = _load_script("x2_skill_proposer_test", "scripts/propose_x2_aspire_skill_candidates.py")

    candidates = proposer.validate_candidates(
        {
            "candidates": [
                {
                    "id": "llm_safe_place",
                    "description": "Use post-lift orientation and slower place descent.",
                    "params": {
                        "CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE": "post_lift_current",
                        "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "5",
                    },
                    "skills": ["slow_vertical_place_descent"],
                }
            ]
        },
        max_candidates=3,
    )

    assert candidates[0]["id"] == "llm_safe_place"
    assert candidates[0]["params"]["CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS"] == "5"
    assert candidates[0]["params"]["CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT"] == "1"

    try:
        proposer.validate_candidates(
            {
                "candidates": [
                    {
                        "id": "llm_bad_param",
                        "params": {"CAPX_X2_UNSAFE_NEW_PARAM": "1"},
                    }
                ]
            },
            max_candidates=3,
        )
    except ValueError as exc:
        assert "unsupported parameter" in str(exc)
    else:
        raise AssertionError("unsupported parameter was accepted")


def test_candidate_search_loads_external_candidate_file(tmp_path):
    search = _load_script("x2_candidate_search_test", "scripts/run_x2_aspire_rgbd_candidate_search.py")
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "llm_safe_place",
                        "description": "External candidate.",
                        "params": {
                            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
                            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
                            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
                        },
                        "skills": ["slow_vertical_place_descent"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = search._select_candidates(None, candidate_file=candidate_file)

    assert len(candidates) == 1
    assert candidates[0]["id"] == "llm_safe_place"
    assert candidates[0]["params"]["CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS"] == "4"
