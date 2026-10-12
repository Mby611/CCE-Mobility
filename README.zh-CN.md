# CCE-Mobility: Counterfactual World-Agent Co-Evolution for Mobility Agents

本仓库是论文 **“CCE-Mobility: Counterfactual World-Agent Co-Evolution for Mobility Agents”** 的代码实现。

[English](README.md)

## Introduction

当出行 Agent 已形成决策后，支持该决策的资源或环境条件可能发生变化。CCE-Mobility 从事实轨迹中构造局部、可执行的反事实干预，同时保持用户目标和已观察历史不变，让 Agent 学习基于证据的恢复技能。

候选技能在同一冻结挑战和上下文上进行成对评价与整理。更新后的 Agent 在**同一批挑战**上复评，仅将更新前后都失败的 **F→F** 样本交给 World memory；S→F 是 Agent regression，不路由给 WM。Agent 根据当前可观察状态自行检索，最多使用两个技能，不更新基座模型权重。

<p align="center">
  <img src="assets/method.png" alt="共进化方法图" width="1000">
</p>

## Setup

推荐 Python 3.12。核心依赖为 Pydantic；数据准备、TP 官方评测使用可选依赖。MB 与 SkillOpt 的上游环境分别安装，避免不兼容依赖混装。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data,travelplanner,mobilitybench]'
cp src/cce_mobility/local_secrets.example.py src/cce_mobility/local_secrets.py
mkdir -p configs/local
cp configs/models/openai_compatible.example.json configs/local/models.json
```

### API Key Setup

将自己的 Key 填入被 Git 忽略的 `src/cce_mobility/local_secrets.py`，不要提交。接口类型、URL、模型名称与推理参数填写在 `configs/local/models.json`。Agent/WM 默认共享基座，Judge 独立配置；历史模型不可用时明确报错，由读者自行选择替代模型，不静默替换。

默认密钥文件只有**两个字段**：`CCE_MODEL_API_KEY` 供 Agent/WM 使用，`CCE_JUDGE_API_KEY` 供 Judge 使用。填入各角色所配置服务商签发的密钥即可，不需要额外申请所谓的 CCE 账号。例如 Qwen Agent/WM 加 DeepSeek Judge，前者填 Qwen Key，后者填 DeepSeek Key；所有角色使用同一账号时，两项可以相同。配置中的 `api_key_env` 按名称读取本地文件或环境变量，并不要求导出环境变量。

官方申请入口：[OpenAI](https://developers.openai.com/api/docs/quickstart)、[Qwen](https://help.aliyun.com/zh/model-studio/get-api-key)、[DeepSeek](https://api-docs.deepseek.com/en/)、[GLM](https://docs.bigmodel.cn/cn/guide/start/quick-start)、[Claude](https://platform.claude.com/docs/en/manage-claude/authentication)、[Gemini](https://ai.google.dev/gemini-api/docs/api-key)。接口不是统一协议，详见[模型服务指南](docs/setup/providers.md)。

### Data Setup

官方来源：[TravelPlanner GitHub](https://github.com/OSU-NLP-Group/TravelPlanner) 与 [HF 数据](https://huggingface.co/datasets/osunlp/TravelPlanner)，[MobilityBench GitHub](https://github.com/AMAP-ML/MobilityBench) 与 [HF 数据和 sandbox](https://huggingface.co/datasets/GD-ML/MobilityBench)。不提供重新打包的数据副本。

`prepare-data` 固定下载 revision，验证格式与划分，输出哈希和 split manifest；已有官方本地数据可通过 `--source` 读取。下载数据、上游代码和运行产物分别放入不提交的 `data/`、`third_party/`、`outputs/`。TP 官方评测还需要官方 database；MB 完整 continuation 需要官方 sandbox 与 Planner–Worker–Reporter 环境。详见 [TP 指南](docs/reproducibility/travelplanner.md) 与 [MB 指南](docs/reproducibility/mobilitybench.md)。

## Quick Start

先修改模型配置与 Key，再按顺序运行。最后的官方评测前，需要将官方 TP database 解压到 `third_party/TravelPlanner/database/`。

```bash
git clone https://github.com/OSU-NLP-Group/TravelPlanner third_party/TravelPlanner
cce prepare-data --benchmark travelplanner --output data/travelplanner
cce prepare-config --benchmark travelplanner \
  --models configs/local/models.json --data-dir data/travelplanner \
  --repo-path third_party/TravelPlanner \
  --output configs/generated/travelplanner.json \
  --output-root outputs/travelplanner --name travelplanner_cce
cce check-models configs/generated/travelplanner.json
cce train configs/generated/travelplanner.json
cce validate configs/generated/travelplanner.json
cce test configs/generated/travelplanner.json
cce evaluate-official configs/generated/travelplanner.json
```

同一生成配置用于训练、验证和测试。默认 TP train45、val180、test1000、seed42，以及三个严格成对 cycle。通过 `--train-tasks`、`--cycles`、`--max-phase-updates`、`--validation-tasks`、`--test-tasks`、`--seed` 自定义新实验。一个 cycle 包含 Agent 更新、同题最终复评与残余失败驱动的 World 更新；历史的 42 phase update 上限不是 42 个完整 cycle。[预算说明](docs/reproducibility/budgets.md)区分历史自适应预算与严格成对机制预算。

验证输出 `selected_checkpoint.json`，包含选点规则、验证指标和记忆哈希。测试只能读取验证选点，不再更新方法，输出 `test_manifest.json`。官方 factual 指标保存在 `official/`，与 CCE 动态恢复指标分开报告。MB 使用相同 CLI，独立运行环境及完整命令见 [MB 指南](docs/reproducibility/mobilitybench.md)。

无密钥最小检查仅使用合成数据，不代表模型能力：

```bash
cce run-config configs/smoke/travelplanner_offline.json
cce run-config configs/smoke/mobilitybench_mock.json
python -m pytest -q
```

[三任务、一 cycle 的付费验收](docs/reproducibility/paid-smoke.md)用于真实服务检查。完整流程可运行不等于精确重复论文数值；新生成挑战或不同模型快照不能冒充论文的原始冻结产物。

## Results

以下数值保持论文原值，表格与英文版一致。主表所有数值为百分比；记忆整理和 SkillOpt 增量小表使用原始 0–4 Recovery 与比例。机制表区分验证面板和独立测试面板，不混用口径。

### TravelPlanner

Main-paper Table 1: 1,000 factual tasks and 1,503 configured counterfactual cases. All columns below are percentages. Recovery is the raw 0-4 mean divided by 4; Poor Recovery uses the strict raw-score threshold < 2. Official TP columns evaluate factual planning, not dynamic recovery.

| Backbone | Method | Recovery | Poor Recovery | CF Pass | TP CS Micro | TP Hard Micro | TP Final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | No Evolution | 61.40 | 31.94 | 37.59 | 73.40 | 7.77 | 1.20 |
| Qwen3.6-35B-A3B | Reflexion | 53.86 | 45.64 | 34.46 | 73.28 | 7.34 | 1.20 |
| Qwen3.6-35B-A3B | Voyager | 55.31 | 43.05 | 35.33 | 73.31 | 7.99 | 0.70 |
| Qwen3.6-35B-A3B | SkillOpt | 72.01 | 21.69 | 37.13 | 73.46 | 5.24 | 0.60 |
| Qwen3.6-35B-A3B | **CCE-Mobility** | 73.12 | 20.90 | 37.79 | 74.48 | 7.86 | 1.00 |
| Qwen3.8-Max | No Evolution | 48.67 | 46.91 | 37.19 | 82.21 | 34.37 | 6.50 |
| Qwen3.8-Max | Reflexion | 50.54 | 44.71 | 38.99 | 81.33 | 32.18 | 5.60 |
| Qwen3.8-Max | Voyager | 62.11 | 30.14 | 36.66 | 81.56 | 31.88 | 5.50 |
| Qwen3.8-Max | SkillOpt | 55.79 | 38.86 | 36.33 | 81.75 | 33.45 | 5.90 |
| Qwen3.8-Max | **CCE-Mobility** | 68.33 | 23.00 | 36.46 | 83.65 | 34.23 | 5.80 |
| GPT-5.6-sol | No Evolution | 65.19 | 27.21 | 2.33 | 60.84 | 0.44 | 0.10 |
| GPT-5.6-sol | **CCE-Mobility** | 70.12 | 22.22 | 2.20 | 70.15 | 20.83 | 2.30 |
| Claude Opus 4.8 | No Evolution | 53.68 | 41.78 | 1.73 | 60.61 | 9.35 | 1.30 |
| Claude Opus 4.8 | **CCE-Mobility** | 56.41 | 34.80 | 1.80 | 71.90 | 8.86 | 1.40 |
| GLM-5.2 | No Evolution | 35.46 | 58.75 | 26.01 | 72.84 | 8.65 | 0.60 |
| GLM-5.2 | **CCE-Mobility** | 60.59 | 32.47 | 28.68 | 72.88 | 9.56 | 1.00 |

### MobilityBench

Main-paper Table 2: the historical heldout500 panel, with 500 factual inputs and 500 counterfactual cases per method. All columns are percentages. This is a fixed development-panel comparison, not an untouched development-independent test set.

| Backbone | Method | Recovery | Poor Recovery | CF Pass | MB FPR | MB TS-R | MB DEC-R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | No Evolution | 74.16 | 5.60 | 98.80 | 56.00 | 60.58 | 76.99 |
| Qwen3.6-35B-A3B | Reflexion | 75.05 | 3.40 | 96.80 | 58.00 | 59.63 | 75.34 |
| Qwen3.6-35B-A3B | Voyager | 74.61 | 8.80 | 99.00 | 56.20 | 60.04 | 76.58 |
| Qwen3.6-35B-A3B | SkillOpt | 84.84 | 2.20 | 99.20 | 57.00 | 61.12 | 75.34 |
| Qwen3.6-35B-A3B | **CCE-Mobility** | 96.75 | 1.40 | 100.00 | 59.60 | 61.01 | 76.16 |
| DeepSeek-V4-Flash-0731 | No Evolution | 68.09 | 12.20 | 99.60 | 45.20 | 45.14 | 71.70 |
| DeepSeek-V4-Flash-0731 | Reflexion | 75.25 | 5.20 | 98.60 | 48.00 | 46.99 | 70.74 |
| DeepSeek-V4-Flash-0731 | Voyager | 69.96 | 4.40 | 98.60 | 45.80 | 46.39 | 69.97 |
| DeepSeek-V4-Flash-0731 | SkillOpt | 77.88 | 1.60 | 99.60 | 49.20 | 47.37 | 70.52 |
| DeepSeek-V4-Flash-0731 | **CCE-Mobility** | 75.66 | 10.20 | 99.60 | 51.20 | 45.72 | 72.59 |

### Co-Evolution Ablation

Main-paper Table 3: TP validation uses 247 challenges from 149 tasks; MB uses a separate fresh-disjoint 500-case continuation panel. Intervals are paired task-cluster bootstrap intervals, not training-seed uncertainty. The MB Full-versus-Agent-only interval crosses zero.

| Panel | Comparator | Comparator Poor (%) | Full CCE Poor (%) | Improved | Regressed | Delta Poor (pp) | 95% CI (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TravelPlanner validation | No Evolution | 21.46 | 14.57 | 20 | 3 | -6.88 | [-10.61, -3.24] |
| TravelPlanner validation | Agent-only | 18.62 | 14.57 | 14 | 4 | -4.05 | [-7.35, -0.83] |
| MobilityBench fresh test | No Evolution | 64.00 | 41.60 | 130 | 18 | -22.40 | [-26.80, -18.20] |
| MobilityBench fresh test | Agent-only | 42.60 | 41.60 | 64 | 59 | -1.00 | [-5.40, 3.40] |

### Memory Curation and SkillOpt Complementarity

These compact supplementary diagnostics use raw Recovery on the 0-4 scale and rates as fractions. The curation panel is validation, not held-out test; complementarity uses Qwen3.6 and the matched TP test1000 panel. They show trade-offs, not uniform or statistically established superiority.

| Variant | Recovery | Low | CF-Final | Unavailable | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired curation | 2.9502 | 0.1822 | 0.0769 | 0.0000 | 2,241,210 |
| Without curation | 2.9581 | 0.1862 | 0.0850 | 0.0000 | 2,355,019 |

| Method | Recovery | Low | Unavailable | CF-Final | TP Hard Micro | TP Final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SkillOpt | 2.8803 | 0.2169 | 0.0279 | 0.3713 | 5.2402 | 0.6000 |
| SkillOpt + CCE-Mobility | 2.9409 | 0.1989 | 0.0070 | 0.3673 | 5.5895 | 0.8000 |

See [metric definitions](docs/metrics.md) and [evaluation protocol](docs/protocol.md) for denominators, failure predicates, and factual/counterfactual boundaries. Values are transcribed from the supplied paper, not recomputed from the example runs.

## Baselines

[SkillOpt](https://github.com/microsoft/SkillOpt) 使用上游 trainer；本仓库将 [Reflexion](https://github.com/noahshinn/reflexion) 和 [Voyager](https://github.com/MineDojo/Voyager) 适配到出行规划与恢复任务，保留在线记忆更新，不属于原始上游环境的复现。步骤见[基线指南](docs/baselines.md)。

消融通过 `--variant agent-only` 或 `--variant without-memory-curation` 配置。同题最终复评和 F→F 筛选在去掉整理时仍然保留。

## Citation / License

引用信息与英文版一致，题目：CCE-Mobility: Counterfactual World-Agent Co-Evolution for Mobility Agents，2027。

原创代码采用 [MIT License](LICENSE)。第三方代码、数据及其许可独立保留，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

方法实现和工作流统一放在 `src/cce_mobility/`；`configs/` 存放实验配置，`scripts/` 仅保留 MobilityBench 官方程序的 Shell 启动脚本，`tools/` 存放仓库检查和发布审计工具。主流程使用 `cce` 命令，高级工作流使用 `python -m cce_mobility.workflows.<group>.<module>`，不再提供重复的外层 Python 转发脚本。目录结构与开发说明见[开发指南](docs/development.md)。
