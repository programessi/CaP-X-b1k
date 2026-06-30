# X2 Experiment Script Archive

This directory stores historical X2 smoke tests, diagnostics, and failed or
superseded experiment scripts.

Do not use these files as the current X2 visual grasp path. They are kept for reference only.

## Current Main Path

Use the formal CaP-X task scripts from `scripts/` instead:

```text
scripts/run_x2_two_target_codex_a_stability_and_check.sh
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
scripts/run_x2_two_target_api_stability_and_check.sh
scripts/run_x2_two_target_oracle_stability_smoke.sh
```

The accepted implementation lives in package code, not in these archived
standalone scripts:

```text
capx/integrations/x2/control.py
capx/integrations/x2/vision.py
capx/envs/tasks/x2/
env_configs/x2/
```

## Why These Were Archived

The archived scripts belong to earlier phases:

- wrist/global/chest camera visibility checks
- low-level primitive sweeps
- Jacobian servo and one-shot IK diagnostics
- visual pose to IK trials before the TCP contract was stable
- orientation sweeps and guarded insertion experiments
- early code-exec task trials

The validated path is now:

```text
OWL-ViT + SAM2 + GraspNet raw pose
-> raw GraspNet to X2 TCP adapter
-> proxy-guarded candidate selection
-> PyRoKi precontact
-> joint IK insertion
-> close gripper
```

See:

```text
docs/x2-visual-grasp-to-ik-v11-report.md
```
