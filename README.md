


# CaP-X on BEHAVIOR-1K

### Code-as-Policy Agents for Robot Manipulation — Adapted for RTX 5090 + Isaac Sim 5.1

> 本项目基于 [CaP-X](https://github.com/capgym/cap-x)（[论文](https://arxiv.org/abs/2603.22435) | [项目主页](https://capgym.github.io/)），
> 针对 **RTX 5090 + Isaac Sim 5.1** 环境进行了适配，专注于 **BEHAVIOR-1K** 基准测试中的 R1Pro 人形机器人操作任务。

**CaP-X** is an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. This fork focuses on BEHAVIOR-1K tasks with the R1Pro humanoid robot, adapted for Isaac Sim 5.1 on RTX 5090 GPUs.

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
    --model MiniMax-M2.7 \
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

---

## 可用任务

`env_configs/r1pro/` 目录下包含所有 BEHAVIOR-1K 任务配置：

| 类型 | 数量 | 示例 |
|------|------|------|
| **R1Pro 专项** | 6 个 | `r1pro_pick_up_radio.yaml`, `r1pro_pick_up_trash.yaml` 及其 oracle/multiturn 变体 |
| **B1K 通用** | 48 个 | `b1k_assembling_gift_baskets.yaml`, `b1k_chop_an_onion.yaml` 等 |

详见 [docs/behavior-tasks.md](docs/behavior-tasks.md)。

---

## 文档

| 文档 | 内容 |
|------|------|
| [BEHAVIOR 任务](docs/behavior-tasks.md) | 环境变量、R1Pro 任务详情、基准性能 |
| [配置说明](docs/configuration.md) | YAML 格式、CLI 参数、LLM 提供商配置 |
| [添加新环境](docs/adding-environments.md) | 创建模拟器、任务环境、YAML 配置 |
| [添加新 API](docs/adding-apis.md) | 实现并注册机器人控制 API |
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
