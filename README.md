


# CaP-X on BEHAVIOR-1K

### Code-as-Policy Agents for Robot Manipulation — Adapted for RTX 5090 + Isaac Sim 5.1

> 本项目基于 [CaP-X](https://github.com/capgym/cap-x)（[论文](https://arxiv.org/abs/2603.22435) | [项目主页](https://capgym.github.io/)），
> 针对 **RTX 5090 + Isaac Sim 5.1** 环境进行了适配，当前覆盖 **BEHAVIOR-1K** 中的 R1Pro 任务，并新增 X2 双臂机器人 tabletop pick-place 基线。

**CaP-X** is an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. This fork focuses on BEHAVIOR-1K tasks with the R1Pro humanoid robot and an X2 tabletop manipulation baseline, adapted for Isaac Sim 5.1 on RTX 5090 GPUs.

## Current Highlights

- **X2 visual pick-place is integrated into CaP-X.** The accepted baseline runs a red-cube tabletop task with OWL-ViT/SAM2-style perception, GraspNet grasp generation, PyRoKi-assisted approach planning, joint-IK execution, gripper control, transfer, and release.
- **The LLM-facing X2 API is intentionally small.** Generated task code calls task-level primitives such as `pick_and_place_red_cube_to_right_target()` instead of directly manipulating IK, cameras, GraspNet, PyRoKi, or simulator setup.
- **The current X2 non-oracle baseline is accepted.** Right and left target runs both passed video recording, visual artifact checks, grasp/placement metrics, and strict local integration audit.
- **This repo uses the sibling BEHAVIOR checkout.** X2 validation uses `/home/xingshu/workspaces/fys/BEHAVIOR-1K`, not `capx/third_party/b1k`.

## X2 Demo

The videos below are from the accepted `codex-a` non-oracle run. The LLM-generated code calls exactly one high-level primitive for each task:

```python
RESULT = pick_and_place_red_cube_to_right_target()
RESULT = pick_and_place_red_cube_to_left_target()
```

| Task | Global View | Robot View |
|------|-------------|------------|
| Right target | <video src="docs/media/x2/x2_right_target_global.mp4" controls width="320"></video><br>[download](docs/media/x2/x2_right_target_global.mp4) | <video src="docs/media/x2/x2_right_target_robot.mp4" controls width="320"></video><br>[download](docs/media/x2/x2_right_target_robot.mp4) |
| Left target | <video src="docs/media/x2/x2_left_target_global.mp4" controls width="320"></video><br>[download](docs/media/x2/x2_left_target_global.mp4) | <video src="docs/media/x2/x2_left_target_robot.mp4" controls width="320"></video><br>[download](docs/media/x2/x2_left_target_robot.mp4) |

Accepted run summary:

| Target | Reward | Task Completed | TCP Error Before Close | Orientation Error | Place Error |
|--------|-------:|---------------:|-----------------------:|------------------:|------------:|
| Right | 1.0 | 1 | 0.0157 m | 0.0548 rad | 0.0141 m |
| Left | 1.0 | 1 | 0.0053 m | 0.0137 rad | 0.0032 m |

Baseline record:

- [Accepted X2 baseline](docs/x2-accepted-baseline-20260630.md)
- [Machine-readable manifest](docs/x2-accepted-baseline-20260630.manifest.json)
- [Snapshot](snapshots/x2_capx_two_target_codex_a_complete_20260630_1025)
- [LLM-facing X2 primitives](docs/x2-llm-facing-primitives.md)

Quick validation:

```bash
python scripts/check_x2_acceptance.py \
  outputs/stability/codex-a/two_targets_*_codex_a_stability_manual_codex_a_20260630_101117_run*

python scripts/audit_x2_capx_integration.py --strict
```

| Component      | What it does                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **CaP-Gym**    | Interactive Gymnasium environments — agents control robots via Python code composing perception & control primitives.       |
| **CaP-Agent0** | Training-free agentic framework: multi-turn visual differencing, auto-synthesized skill libraries, parallel ensembled reasoning. |
| **CaP-RL**     | RL post-training via GRPO — environment rewards fine-tune VLM coding agents. Sim-to-real with minimal gap.                  |

---

## 环境要求

- **GPU:** NVIDIA RTX 5090（或其他 Blackwell 架构 GPU）
- **OS:** Ubuntu 22.04 / 24.04
- **Python:** 3.10（cap-x 本体）/ 3.11（Isaac Sim conda 环境）
- **CUDA:** 12.4+
- **驱动:** 建议 570+（Blackwell 架构支持）

## 安装

### 1. 安装 Isaac Sim 5.1（独立安装）

Isaac Sim 5.1 独立安装在 `/opt/isaac-sim/`，带有 conda 环境 `behavior`（Python 3.11），OmniGibson 和 BDDL 已内置于该环境中。

安装完成后，需要在 behavior 环境中补充 cap-x 适配所需的包（如 cuRobo、PyRoKi 等）。

```bash
# 激活 Isaac Sim 自带的 conda 环境
source /opt/isaac-sim/setup_conda_env.sh

# 补充安装 cap-x 适配包
pip install <capx-required-packages>
```

### 2. 安装 cap-x 本体

cap-x 本体使用 [uv](https://docs.astral.sh/uv/) 管理依赖（Python 3.10）。

```bash
git clone --recurse-submodules <your-repo-url> && cd CaP-X-b1k

# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# 基础安装
uv sync
```

### 3. 无头服务器额外依赖

```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
```

> **注意：** 首次运行时 cuRobo 会 JIT 编译 CUDA kernel（适配 RTX 5090 的 Blackwell 架构），需要 **3-5 分钟**，属于正常现象。

---

## 快速开始

### 1. 感知服务器

感知服务器（SAM3、ContactGraspNet、PyRoKi）由 YAML 配置**自动启动**，大多数情况下无需手动操作。

> **SAM3 认证：** SAM3 权重需要 HuggingFace 授权。在 [SAM3 repo](https://github.com/facebookresearch/sam3) 申请访问权限后，本地执行 `huggingface-cli login`。权重首次下载后会缓存。

如需跨多次评估共享服务器，可预启动：

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

可选 profile：

```bash
--profile full      # 全部感知服务器 (SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2)
--profile minimal   # 仅 PyRoKi (oracle/privileged 评估)
```

### 2. 配置 LLM 代理

评估框架通过本地 OpenAI 兼容 API 代理查询 LLM。

```bash
# OpenRouter（在 openrouter.ai/keys 获取密钥）
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

**本地 Codex CLI 模式：** 如果 GPT API key 已经配置在本机 `codex-a`
命令里，X2 当前推荐直接使用 `codex-a` 脚本，不需要单独找 key 文件：

```bash
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

脚本会启动 `capx/serving/codex_cli_server.py`，把 CaP-X 的
OpenAI-compatible `/chat/completions` 请求转给本机 `codex-a`。如果你的命令
名是 `codex` 而不是 `codex-a`，用：

```bash
CODEX_BIN=codex scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

当前机器上 `codex-a` 可能是 `codex -c 'model_provider="axonhub"'` 的 shell
alias；脚本默认的 `CODEX_BIN=codex` 和 `CODEX_MODEL_PROVIDER=axonhub` 与它等价。

**通用 OpenAI-compatible 直连模式：** 如果你有普通 OpenAI-compatible proxy
和 key 文件，也可以使用直连代理：

```bash
# 先创建本地 key 文件
echo "sk-your-key-here" > .openai_key

# 在 behavior conda 环境中启动代理（无需 source Isaac Sim 脚本）
source /home/xingshu/miniforge3/bin/activate behavior
python capx/serving/openrouter_server.py \
  --key-file .openai_key \
  --base-url https://api.openai.com/v1/ \
  --port 8110
```

详见 [docs/configuration.md](docs/configuration.md) 了解所有 LLM 提供商配置。

### 3. 运行评估

所有 BEHAVIOR 任务需要在 Isaac Sim 的 conda 环境中运行，并设置正确的 `PYTHONPATH` 和 `LD_LIBRARY_PATH`：

```bash
source /opt/isaac-sim/setup_conda_env.sh
```

核心环境变量说明：

| 变量 | 说明 |
|------|------|
| `PYTHONPATH` | 需包含 OmniGibson、bddl3 路径及 cap-x 项目根目录 |
| `LD_LIBRARY_PATH` | 需指向 conda env 的 `lib` 目录 |
| `OMNI_KIT_ACCEPT_EULA=YES` | 必须，接受 Isaac Sim EULA |
| `OMNIGIBSON_HEADLESS=1` | 无头模式（无显示器时必设） |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 建议，避免 CUDA 显存碎片 |
| `HF_HUB_OFFLINE=1` | 可选，离线模式跳过 HuggingFace 检查 |

```bash
# 无头烟测示例（600 秒超时）
source /opt/isaac-sim/setup_conda_env.sh && timeout 600 env \
  CAPX_FAST_EXIT_AFTER_MAIN=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/home/xingshu/miniforge3/envs/behavior/lib/python3.11/site-packages:\
/home/xingshu/workspaces/fys/cap-x/capx/third_party/b1k/OmniGibson:\
/home/xingshu/workspaces/fys/cap-x/capx/third_party/b1k/bddl3:\
/home/xingshu/workspaces/fys/cap-x:$PYTHONPATH \
  LD_LIBRARY_PATH=/home/xingshu/miniforge3/envs/behavior/lib:$LD_LIBRARY_PATH \
  NUMBA_CACHE_DIR=/tmp/numba-cache \
  MPLCONFIGDIR=/tmp/matplotlib-cache \
  OMNI_KIT_ACCEPT_EULA=YES \
  OMNIGIBSON_HEADLESS=1 \
  python capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio_sam2_smoke.yaml \
    --model gpt-5 \
    --output-dir ./outputs/r1pro_pick_up_radio_sam2_smoke
```

其他常用任务示例：

```bash
source /opt/isaac-sim/setup_conda_env.sh

# R1Pro 拾取收音机
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
python capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# R1Pro Oracle（特权信息，用于基准测试）
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
python capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml \
    --model "google/gemini-3.1-pro-preview"

# B1K 通用任务（替换为具体活动名）
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
python capx/envs/launch.py \
    --config-path env_configs/r1pro/b1k_hiding_Easter_eggs.yaml \
    --model "google/gemini-3.1-pro-preview"
```

> **RTX 5090 注意事项：**
> - Isaac Sim 使用 `OMNIGIBSON_GPU_ID`（非 `CUDA_VISIBLE_DEVICES`）选择 GPU
> - 多 GPU 系统建议：评估用 `OMNIGIBSON_GPU_ID=0`，感知服务器用 `CUDA_VISIBLE_DEVICES=1`
> - 务必设置 `OMNI_KIT_ACCEPT_EULA=YES` 和 `OMNIGIBSON_HEADLESS=1`（无头模式）
> - 建议设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 避免 5090 显存碎片问题

### 4. X2 Red-Cube Pick-Place Baseline

当前 X2 推荐入口是一个 CaP-X 非 oracle 任务：LLM 生成代码调用
`pick_and_place_red_cube()`，该 primitive 内部串联视觉检测/分割/抓取位姿
生成和 X2 动作执行。

一键烟测：

```bash
scripts/run_x2_pick_place_red_cube_non_oracle_smoke.sh
```

默认使用：

```text
BEHAVIOR_ROOT=/home/xingshu/workspaces/fys/BEHAVIOR-1K
CONDA_ENV=behavior
```

任务配置：

```text
env_configs/x2/x2_pick_place_red_cube.yaml
```

当前成功基线和视频证据见：

```text
docs/x2-accepted-baseline-20260630.md
snapshots/x2_pick_place_red_cube_capx_baseline_20260626_1530
docs/x2-pick-place-current-baseline.md
docs/x2-capx-integration-status.md
```

两目标 codex-a 非 oracle 烟测：

```bash
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

两目标 codex-a 稳定性循环，并在跑完后汇总和验收：

```bash
REPEATS=1 \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

两目标 generic direct API 非 oracle 烟测：

```bash
scripts/run_x2_two_target_api_non_oracle_smoke.sh
```

左目标变体使用同一个脚本，通过 `CONFIG_PATH` 覆盖任务配置：

```bash
CONFIG_PATH=env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml \
scripts/run_x2_two_target_api_non_oracle_smoke.sh
```

两目标 generic direct API 稳定性循环：

```bash
REPEATS=1 \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_smoke.sh
```

也可以直接使用“跑完并验收”的封装脚本：

```bash
REPEATS=1 \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_and_check.sh
```

该路径不启动 `codex_cli_server.py`，只使用 OpenAI-compatible
`SERVER_URL`。如果使用本地 OpenAI-compatible 代理，且当前 shell 中存在
`ALL_PROXY=socks://...`，需要把它改为 http scheme：

```bash
env ALL_PROXY=http://127.0.0.1:7897/ all_proxy=http://127.0.0.1:7897/ \
/home/xingshu/miniforge3/bin/conda run --no-capture-output -n behavior \
python capx/serving/openrouter_server.py \
  --key-file .openai_key \
  --base-url https://api.openai.com/v1/ \
  --host 127.0.0.1 \
  --port 8110
```

运行后可汇总结果：

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

也可以直接跑当前验收门槛：

```bash
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

完整本地集成审计：

```bash
python scripts/audit_x2_capx_integration.py
```

X2 脚本入口索引见 [scripts/README.md](scripts/README.md)。当前如果 GPT key
已经在 `codex-a` 中配置，推荐使用 `codex-a` 路径；generic direct API
脚本保留给显式启动 OpenAI-compatible proxy 的场景。

当前 accepted baseline 已通过 `codex-a` 非 oracle right/left 两目标闭环、
视频保存、视觉 artifact 保存、验收脚本和 strict audit：

```text
docs/x2-accepted-baseline-20260630.md
snapshots/x2_capx_two_target_codex_a_complete_20260630_1025
```

历史两目标稳定快照见：

```text
snapshots/x2_two_target_codex_a_baseline_20260629_0935
snapshots/x2_two_target_left_codex_a_baseline_20260629_1012
snapshots/x2_two_target_stability_hold_baseline_20260629_1330
```

---

## 可用任务

`env_configs/r1pro/` 目录下包含 R1Pro BEHAVIOR-1K 任务配置；`env_configs/x2/` 包含当前 X2 tabletop 任务配置：

| 类型 | 数量 | 示例 |
|------|------|------|
| **R1Pro 专项** | 6 个 | `r1pro_pick_up_radio.yaml`, `r1pro_pick_up_trash.yaml` 及其 oracle/multiturn 变体 |
| **B1K 通用** | 48 个 | `b1k_assembling_gift_baskets.yaml`, `b1k_chop_an_onion.yaml` 等 |
| **X2 tabletop** | 3 个基线/扩展任务 | `x2_pick_place_red_cube.yaml`, `x2_pick_place_red_cube_two_targets.yaml`, `x2_pick_place_red_cube_two_targets_left.yaml` |

R1Pro / B1K 详见 [docs/behavior-tasks.md](docs/behavior-tasks.md)。X2 当前基线详见 [docs/x2-pick-place-current-baseline.md](docs/x2-pick-place-current-baseline.md)。

---

## 文档

| 文档 | 内容 |
|------|------|
| [BEHAVIOR 任务](docs/behavior-tasks.md) | 环境变量、R1Pro 任务详情、基准性能 |
| [配置说明](docs/configuration.md) | YAML 格式、CLI 参数、LLM 提供商配置 |
| [添加新环境](docs/adding-environments.md) | 创建模拟器、任务环境、YAML 配置 |
| [添加新 API](docs/adding-apis.md) | 实现并注册机器人控制 API |
| [X2 Accepted Baseline](docs/x2-accepted-baseline-20260630.md) | X2 两目标 codex-a 非 oracle 完整通过基线、证据和复现命令 |
| [X2 集成状态](docs/x2-capx-integration-status.md) | X2 导入 CaP-X 的完成项、验收证据和后续扩展方向 |
| [X2 当前基线](docs/x2-pick-place-current-baseline.md) | X2 red-cube pick-place 调用链、指标、复现命令 |
| [X2 LLM 原语](docs/x2-llm-facing-primitives.md) | X2 给 LLM 暴露的任务级/视觉/动作原语 |
| [X2 版本管理](docs/x2-version-management.md) | X2 成功路径、snapshot 和历史输出管理 |
| [真机 Franka Panda](docs/real-franka.md) | 真机部署、QuickStart |
| [RL 训练](docs/rl-training.md) | CaP-RL + GRPO/VeRL、sim-to-real 迁移 |

---

## Citation

```bibtex
@article{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and Huang, Huang and Xiao, Wenli and Wang, Guanzhi and Li, Fei-Fei and Shi, Guanya and Wu, Jiajun and Sastry, Shankar and Zhu, Yuke and Goldberg, Ken and Fan, Jim},
  journal   = {arXiv preprint arXiv:2603.22435},
  year      = {2025},
  url       = {https://arxiv.org/abs/2603.22435}
}
```

## License

This project is released under the [MIT License](LICENSE).
