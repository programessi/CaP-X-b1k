# Configuration Reference

## CLI flags

Override any YAML config field from the command line:

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path <config.yaml> \
    --model google/gemini-3.1-pro-preview \
    --server-url http://127.0.0.1:8110/chat/completions \
    --temperature 1.0 \
    --total-trials 100 \
    --num-workers 12 \
    --record-video True
```

| Flag                | Default                                  | Description                           |
| ------------------- | ---------------------------------------- | ------------------------------------- |
| `--config-path`     | *(required)*                             | Path to YAML task config              |
| `--model`           | `google/gemini-3.1-pro-preview`          | Model name                            |
| `--server-url`      | `http://127.0.0.1:8110/chat/completions` | LLM endpoint                          |
| `--temperature`     | `1.0`                                    | Sampling temperature                  |
| `--total-trials`    | from YAML                                | Number of evaluation trials           |
| `--num-workers`     | from YAML                                | Parallel worker count                 |
| `--web-ui`          | `False`                                  | Launch interactive web UI             |
| `--use-oracle-code` | `False`                                  | Run human-written reference solutions |

## YAML config format

```yaml
# env_configs/my_task/my_task.yaml
env:
  _target_: capx.envs.tasks.my_robot.my_task.MyTaskCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: my_sim_env
    privileged: false
    apis:
      - FrankaControlApi

record_video: true
output_dir: ./outputs/my_task
trials: 100
num_workers: 12
```

The `_target_` keys enable Hydra-style lazy instantiation via `capx.envs.configs.instantiate()`.

### Perception servers (api_servers)

YAML configs can include an `api_servers` section that **auto-launches** perception servers when the evaluation starts:

```yaml
api_servers:
  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1

  - _target_: capx.serving.launch_contact_graspnet_server.main
    port: 8115
    host: 127.0.0.1

  - _target_: capx.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
    robot: panda_description
    target_link: panda_hand
```

The launcher automatically:
- Skips servers whose port is already in use (e.g. started externally)
- Waits for all servers to be ready before running trials
- Terminates all servers on exit

If you prefer to manage servers separately (e.g. for sharing across multiple eval runs), use `launch_servers.py`:

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

| Profile | Servers | GPU Required |
|---------|---------|-------------|
| `default` | SAM3 (8114) + ContactGraspNet (8115) + PyRoKi (8116) | Yes (~5 GB VRAM) |
| `full` | default + OWL-ViT (8118) + SAM2 (8113) | Yes (~14 GB VRAM) |
| `minimal` | PyRoKi (8116) only | No (CPU-only) |

## Adding new LLM providers

CaP-X queries language models through a local proxy server that exposes an OpenAI-compatible `/chat/completions` endpoint.

### OpenRouter (recommended for getting started)

1. Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Save it to a file in the project root:
   ```bash
   echo "sk-or-v1-your-key-here" > .openrouterkey
   ```
3. Start the proxy (supports automatic key rotation across multiple keys):
   ```bash
   uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
   ```

OpenRouter provides access to Gemini, GPT, Claude, DeepSeek, Qwen, and other models through a single API key.

### Local codex-a GPT configuration

If the GPT API key is already configured in the local `codex-a` command, use
the X2 codex-a scripts. They start `capx/serving/codex_cli_server.py`, expose a
local OpenAI-compatible `/chat/completions` endpoint, and delegate code
generation to the user's local Codex CLI configuration.

```bash
REPEATS=1 \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

If the local executable is named `codex` instead of `codex-a`, override it:

```bash
CODEX_BIN=codex REPEATS=1 scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

On this workstation `codex-a` may be a shell alias for
`codex -c 'model_provider="axonhub"'`. Non-interactive scripts do not see shell
aliases, so the X2 codex-a script defaults to the equivalent executable form:
`CODEX_BIN=codex` and `CODEX_MODEL_PROVIDER=axonhub`.

### Generic OpenAI-compatible direct APIs

`capx/serving/openrouter_server.py` is also a generic OpenAI-compatible proxy:
set `--base-url` to your provider endpoint and pass a local key file. Use this
path when you want to bypass the local Codex CLI and call an explicit
OpenAI-compatible endpoint directly.

```bash
env ALL_PROXY=http://127.0.0.1:7897/ all_proxy=http://127.0.0.1:7897/ \
/home/xingshu/miniforge3/bin/conda run --no-capture-output -n behavior \
python capx/serving/openrouter_server.py \
  --key-file .openai_key \
  --base-url https://api.openai.com/v1/ \
  --host 127.0.0.1 \
  --port 8110
```

The `ALL_PROXY` override is only needed when the shell has a `socks://...`
proxy value and the installed `httpx` version rejects that scheme at startup.
If your environment does not define `ALL_PROXY=socks://...`, omit the override.

X2 two-target direct API smoke:

```bash
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_non_oracle_smoke.sh
```

X2 right/left stability loop:

```bash
REPEATS=1 \
STAMP=<STAMP> \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_smoke.sh
```

Summarize saved local outputs:

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

The summary command does not contact any external API. It reads saved CaP-X
outputs, generated code, metrics, videos, and X2 visual artifacts.

### Option B: vLLM (local models)

```bash
uv run python -m capx.serving.vllm_server --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8080 --tensor-parallel-size 4
```

### Option C: Custom providers

Providers live under `capx/serving/providers/` and implement a simple `generate_code` method. Extend to Gemini/Claude/Bedrock by adding new provider classes.

> **Note:** `.openrouterkey` is git-ignored. Never commit API keys to the repository.
