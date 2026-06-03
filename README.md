


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
- **Python:** 3.10（Isaac Sim wheels 仅支持 cp310）
- **CUDA:** 12.4+
- **驱动:** 建议 570+（Blackwell 架构支持）

## 安装

使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```bash
git clone --recurse-submodules <your-repo-url> && cd CaP-X-b1k

# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# 基础安装
uv sync
```

### BEHAVIOR + Isaac Sim 5.1 安装

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset --accept-dataset-tos
cd ../../..
```

此命令会安装 OmniGibson、Isaac Sim 5.1、BDDL、cuRobo，并下载机器人资产、BEHAVIOR-1K 场景/物体资产及 2025 challenge 任务实例。

> **注意：** 首次运行时 cuRobo 会 JIT 编译 CUDA kernel（适配 RTX 5090 的 Blackwell 架构），需要 **3-5 分钟**，属于正常现象。

### 安装后修复

```bash
# 激活 b1k 虚拟环境
source capx/third_party/b1k/.venv/bin/activate

# 修复 cuRobo CUDA JIT 头文件
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/

# 修复 Vulkan ICD 冲突（多 GPU 系统可能 segfault）
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

### 无头服务器额外依赖

```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
```

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

所有 BEHAVIOR 任务均需激活 b1k 虚拟环境：

```bash
source capx/third_party/b1k/.venv/bin/activate
```

```bash
# R1Pro 拾取收音机（20 trials）
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# R1Pro 多轮 + 视觉差分
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview"

# R1Pro Oracle（特权信息，用于基准测试）
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml \
    --model "google/gemini-3.1-pro-preview"

# B1K 通用任务（替换为具体活动名）
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/b1k_hiding_Easter_eggs.yaml \
    --model "google/gemini-3.1-pro-preview"
```

> **RTX 5090 注意事项：**
> - Isaac Sim 使用 `OMNIGIBSON_GPU_ID`（非 `CUDA_VISIBLE_DEVICES`）选择 GPU
> - 多 GPU 系统建议：评估用 `OMNIGIBSON_GPU_ID=0`，感知服务器用 `CUDA_VISIBLE_DEVICES=1`
> - 务必设置 `OMNI_KIT_ACCEPT_EULA=YES` 和 `OMNIGIBSON_HEADLESS=1`（无头模式）

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
