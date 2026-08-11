---
title: "report"
tags:
  - "type/reproduction"
  - "methods/lead"
result: "partial"
repo_url: "https://github.com/mlrm-LEAD/mlrm-LEAD.git"
published: 2026-08-11
updated: 2026-08-11
---

# LEAD（arXiv 2603.13366）复现审计报告

| 字段 | 值 |
|---|---|
| Run ID | `20260804-093724-a1a4fedd` |
| 工作区 | `[本地路径]lbj/mlrm-LEAD`（git commit `dd2651d2ce04abba0e77af803fad42177a51d7b1`，main） |
| 仓库 | http[本地路径]/github.com/mlrm-LEAD/mlrm-LEAD.git |
| 论文 | arXiv 2603.13366v1（`input/2603.13366.pdf`，12 页） |
| 开始 / 结束 | 2026-08-04T01:37:25Z / 2026-08-04T05:51Z（reporting 阶段） |
| 问题归档 | 见 `report/RUN_ISSUES.md`（含系统 issue 与本 run 复现级问题 B1-B9） |
| 本报告结论 | 机制级复现成功；数值级复现失败（0/12 精确）；方向级部分复现（3/6 基准，VStar/RealWorldQA 稳定正效应） |

---

## 1. 概述

### 1.1 论文是什么

论文《Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding》（arXiv 2603.13366，cs.CV，2026-03-09，Monash/Georgia Tech/Cornell/Northeastern/Khalifa/UMN 合作）提出 **LEAD（Latent Entropy-Aware Decoding）**——一种**训练无关（training-free）的即插即用解码策略**，用于缓解多模态大推理模型（MLRM）的幻觉：

- 观察：转折词（because/however/wait 等）与幻觉高度共现，且伴随 token 级高熵状态（Fig 1/2）。
- 方法：以 token 概率分布的熵 `H_t`（Eq 4）为不确定性度量，在高熵态使用**概率加权连续嵌入**（latent 推理，Eq 3），低熵态回到离散 token 嵌入（discrete 推理，Eq 5）；通过非对称持久窗口（Eq 6-8）与开关计数上限 `C_max=5` 控制模式切换，并在每个高熵阶段首个 latent 步注入**视觉锚点**（Eq 9，`λ=0.4` 最优）。
- 主张：5 个 7B 模型（R1-Onevision-7B / Vision-R1-7B / VL-Rethinker-7B / VL-Cogito-7B / OpenVLThinker-7B）× 17 个基准（Table 2/3 + Fig 5-10），LEAD 在所有基准均为正提升（如 R1-Onevision 通用推理平均 +3.6%、MMHalu +4.7%、Bingo +3.8%、数学平均 +2.0%、科学平均 +3.2%）。

论文关键缺陷（对复现的影响）：**正文未给出任何模型权重/数据/代码链接**，未报告采样超参与 `max_new_tokens`，未报告 GPU 与样本量，附录 A-D（模型规模、基线细节、完整 Pass@k）不在 PDF 中，Algorithm 1 的 `eps` 未给数值。详见 `analysis/paper_manifest.json`。

### 1.2 仓库是什么

`mlrm-LEAD` 是 LEAD 的官方配套仓库（README 即本论文项目页内容）：`main.py`（21 个 CLI 参数，默认模型 `Qwen/Qwen2.5-VL-7B-Instruct`、默认数据集 `data/physunibench.jsonl`）、`lead/generation_utils.py`（**手写解码循环** `generate_lead`/`generate_cot`，非 `model.generate()`，10 步流程 A-J）、`lead/evaluator.py`（评估）、8 个本地 JSONL 基准、41 个 CPU 单测、13 项已知坑。详见 `analysis/repo_manifest.json`。

**仓库与论文的错位**（审计结论，证据见 `analysis/repo_manifest.json` 的 `paper_vs_code_notes`）：论文 17 基准中 11 个在本仓库不存在（MMEval-Pro/VMCBench/MMHalu/Bingo/POPE/MathVerse/Geometry3K/MMK12-*）；论文 5 个模型在代码中零接线（README 声称的默认模型 Fancy-MLLM/R1-Onevision-7B-RL 与代码实际默认 Qwen 不符）；论文 3 条基线（VCD/MemVR/SID）零实现；论文主基准 PhysUniBench 在论文正文 0 次出现。

### 1.3 本次复现范围

在 2×A100-40GB 上执行八步审计管线（paper-audit → repo-audit → coverage → assets → environment → execution → verification → reporting）：

- **可运行范围**：仓库 6 个与论文同名的基准（vstar/mmvp/realworldqa/math_vision/math_vista/visulogic）× 2 个可用模型（Qwen2.5-VL-7B-Instruct 现成代理 + 本次下载的 Fancy-MLLM/R1-Onevision-7B-RL 作为论文 R1-Onevision-7B 的 proxy）× {lead, cot, cot_greedy}，共 **28 个组合全部成功**（limit=100, seed=42, max_new_tokens=512, T=0.6, top_p=0.95, top_k=20）。
- **对齐范围**：论文 Table 2/3 中 R1-Onevision-7B 行 × 6 个有输出的基准 × 2 方法 = **12 个可对齐单元格**。
- **明确不覆盖**：Table 1（λ 消融）、Table 2/3 其余 4 模型行、VCD/MemVR/SID 基线行、11/17 基准、Fig 5-10 消融与 Pass@k、MMHalu/Bingo/POPE 打分指标——全部被 B1-B9 阻塞（见 §6）。

---

## 2. 环境与资产

### 2.1 控制 / 项目环境分离

| 环境 | 名称 | 前缀 | Python | 角色 |
|---|---|---|---|---|
| 控制（orchestration） | `paper-repro-control` | `[本地路径][本地路径]3/envs/paper-repro-control` | 3.11.15 | 审计管线、PDF 提取、资产下载编排 |
| 执行（项目） | `mlrm` | `[本地路径][本地路径]3/envs/mlrm` | 3.10.20 | 全部推理/评测/测试（repro_environment 配置，2026-08-04T02:15:21Z） |

`mlrm` 关键版本（`environment/verify.txt`、`environment/environment.yml`，全量锁文件见 `environment/conda-explicit.txt` / `pip-freeze.txt` / `environment_full.yml`）：

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.11.0+cu128 | cuda_available=True，device_count=2，cudnn 91900，matmul 实测 OK |
| torchvision | 0.26.0+cu128 | |
| transformers | 5.14.1 | 与手写解码循环兼容（cache_position 被 **kwargs 吸收） |
| qwen-vl-utils | 0.0.14 | |
| accelerate | 1.14.0 | |
| huggingface-hub | 1.26.0 | 含 hf CLI |
| tokenizers / numpy / pillow / pytest | 0.22.2 / 2.2.6 / 12.2.0 / 9.1.1 | |
| 测试 | **41/41 通过**（pytest 3.83s，`bash script/run_tests.sh`） | |

### 2.2 GPU

- 硬件：**2× NVIDIA A100-PCIE-40GB**（Ampere，驱动 580.126.20，系统 CUDA 13.0，`environment/nvidia-smi.txt`）。
- 本次使用：L2 两卡并行——GPU0 跑模型 A 14 组（会话 8471s），GPU1 跑模型 B 14 组（会话 11234s）；单卡峰值显存 **31455 MiB**；总推理耗时 19695s = **5.47 GPU 时**（`execution/experiment_summary.json` gpu_usage；GPU 采样曲线见 `results/gpu_A.csv` / `results/gpu_B.csv`）。
- 磁盘：run 开始时 28GB 可用；下载 R1-RL + 6 数据集后余 8.8GB（A4）。

### 2.3 模型资产（`assets/assets.lock.json`）

| 资产 id | 来源 | 状态 | 大小 | revision | SHA256（分片，实测） | 许可 | 备注 |
|---|---|---|---|---|---|---|---|
| model-r1-onevision-7b-rl | huggingface.co/Fancy-MLLM/R1-Onevision-7B-RL（经 [本地路径].com） | **downloaded** | 16.6GB（15 文件） | 11a494646839cfde2f19f39e76935bfc38ad0f7c | `2c4e3adc…|35e31a85…|223baafd…|8b83c1f1…`（4 safetensors shard） | apache-2.0 | 549s @30.2MB/s；config 确认 qwen2_5_vl 架构；**身份存疑**（见 §7） |
| model-qwen25-vl-7b-instruct | huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | **present**（旧缓存完整） | 16.6GB（14 文件） | cc594898137f460bfe9f0759e9844b3ce807cfb5 | `e97b877e…|ef47f634…|111223d1…|a9a300a4…|0c859795…`（5 shard） | apache-2.0 | 14/14 符号链接无悬挂 |
| model-vision-r1-7b | Osilly/Vision-R1-7B | **deferred** | 需 16.6GB | — | — | unknown | 磁盘不足（A4/A5） |
| model-vl-rethinker-7b / vl-cogito-7b / openvlthinker-7b | 无确认来源（cogito source=unknown） | **deferred** | 各需 16.6GB | — | — | unknown | 论文未给链接（B6） |

### 2.4 数据资产（`assets/assets.lock.json`）

本次经 [本地路径] 新下载 6 个论文相关 HF 数据集（共 3.91GB，188s @20.8MB/s；首次命令缺 `--repo-type dataset` 全失败，修正后一次成功——见 downloads 记录与 A1/A7）：

| 数据集 | 大小 | revision | SHA256 | 对应论文基准 |
|---|---|---|---|---|
| MM-Diagnose/MMEvalPro | 0.25GB | 699a7d67… | fb214972… | MMEval-Pro |
| suyc21/VMCBench | 2.72GB | ac748b6f… | 7649e10d… | VMCBench |
| Shengcao1006/MMHal-Bench | 0.35GB | f5f49a93… | d5622962… | MMHalu |
| AI4Math/MathVerse | 0.31GB | 3bc86196… | 80528135… | MathVerse |
| hiyouga/geometry3k | 0.06GB | fd21e533… | 61952d49… | Geometry3K |
| Cierra0506/MM-K12 | 0.23GB | 4164afca… | 443842cf… | MMK12-Math/Phys/Chem/Bio |

> 注：这 6 个数据集**已下载但尚未转换为仓库 JSONL schema**（仓库无转换工具，B2），本次 L2 未使用；Bingo/POPE 因无 LICENSE 许可阻塞未下载。

### 2.5 本地基准数据（`assets/assets.lock.json` local_data）

仓库 8 个 tracked JSONL 的图片路径全部是作者机器路径（`[本地路径][本地路径]-tmp/开源/已解压/LEAD/…`，失效，A3）；可用版本为未跟踪的 `data/*_local.jsonl`（图片重定位到 `[本地路径]bench_images/<bench>/`）：

| 文件 | 样本数 | 图片 OK | 状态 |
|---|---|---|---|
| demo_local.jsonl | 1 | 1/1 | present（example/demo.jpeg 在仓库内） |
| vstar_local.jsonl | 191 | 191/191 | present |
| mmvp_local.jsonl | 300 | 300/300 | present |
| math_vista_local.jsonl | 1000 | 1000/1000 | present |
| math_vision_local.jsonl | 3040 | 3040/3040 | present |
| visulogic_local.jsonl | 1000 | 1000/1000 | present |
| realworldqa_local.jsonl | 765 | **697/765** | **partial：缺 68 图（8.9%）** |
| physunibench_local.jsonl | 3304 | 3304/3304 | present（与论文无对应关系，不进矩阵） |

---

## 3. 八步流程摘要

| # | 阶段 | 做了什么 | 产物（本 run 内路径） |
|---|---|---|---|
| 1 | paper-audit | pdftotext/pdfimages 提取 12 页 PDF 文本与 28 张位图；转录 Table 1-3 全部数值、Eq 1-9、Algorithm 1、超参（Cmax=5、λ=0.4、窗口消融 64/128/256/inf、动态阈值）、Fig 1-10 说明与 6 条核心主张 | `analysis/paper_manifest.json`、`analysis/paper_text_layout.txt`、`analysis/paper_text_raw.txt` |
| 2 | repo-audit | 审计 `main.py` 21 个 CLI 参数、`lead/` 8 个模块、`generate_lead` 10 步流程（A-J）、8 个 JSONL、4 个脚本、41 测试、13 项 known_issues、3 处工作区改动（embedding `.to(device)`、四级匹配器、bf16 单设备加载） | `analysis/repo_manifest.json` |
| 3 | coverage | 构造 **102 项**论文×仓库覆盖矩阵：机制 7 项 partial、Table 1-3 行/列、Fig 1-10、6 条主张、5 指标、3 基线、5 模型、数据资产；**exact=0**；定义 B1-B9 与 P0-P4 路径 | `analysis/reproduction_matrix.json` |
| 4 | assets | 实测缓存（Qwen 16G 完整、R1-RL/Vision-R1 为 4KB 空壳、6 数据集只剩 refs）、网络（[本地路径] 200 / huggingface.co 不通）、磁盘（27.9GB）；下载 R1-Onevision-7B-RL 16.6GB（549s）+ 6 数据集 3.91GB（188s），SHA256 实测；登记 7 项问题 A1-A7 | `assets/assets.lock.json`、`assets/artifacts.jsonl` |
| 5 | environment | 项目 conda `mlrm` 安装 torch 2.11.0+cu128 / transformers 5.14.1 等（含一次失败的 torch 探测命令，见 RUN_ISSUES.md），import/CUDA 冒烟、41/41 测试通过、导出 4 份锁文件 | `environment/verify.txt`、`environment.yml`、`environment_full.yml`、`conda-explicit.txt`、`pip-freeze.txt`、`nvidia-smi.txt`、`git-status.txt` |
| 6 | execution | L0：41/41 测试 + `main.py --help`；L1：Qwen 单样本 demo/vstar 管线验证；L2：**28/28 组合**（2×A100 并行，GPU0=模型A 8471s，GPU1=模型B 11234s，limit=100/seed=42/max_new_tokens=512）；L3：全量复现跳过（理由见 §6.2） | `execution/experiment_summary.json`、`execution/commands.jsonl`、`execution/gpu.csv`、`execution/logs/*.log`（28 份 run 日志 + 阶段日志）、`results/l2_*_eval.json`（28 份） |
| 7 | verification | 以 R1-Onevision-7B 论文行（Table 2/3）对齐 B 模型输出，生成 **12 单元格**（exact/±3pp/方向一致/方向冲突/不可比 五分类）、LEAD 效应表、截断率实测（分词器对 model_answer 计数）、Top-3 归因与 caveats | `results/verification_report.json`、`results/l2_summary.jsonl` |
| 8 | reporting | 汇总全部产物，写本报告与 summary.json，归档问题 | `report/report.md`、`report/summary.json`、`report/RUN_ISSUES.md`（run 根目录 `report.md`/`summary.json`/`RUN_ISSUES.md` 为指向本目录的符号链接） |

---

## 4. 复现结果

### 4.1 28 组合全表

统一配置：`limit=100, seed=42, max_new_tokens=512, temperature=0.6, top_p=0.95, top_k=20, alpha=0.6, max_switch_count=5`。
模型 A = `Qwen/Qwen2.5-VL-7B-Instruct`（代理，非论文模型）；模型 B = `Fancy-MLLM/R1-Onevision-7B-RL`（论文 R1-Onevision-7B 的 proxy）。
数据：`data/*_local.jsonl`。accuracy 口径：本地四级匹配器，failed_extraction 计入分母（B9）。日志：`execution/logs/run_{A,B}_{bench}_{method}.log`。

| id | 模型 | 基准 | 方法 | 准确率 | correct | failed_extraction | 耗时(s) |
|---|---|---|---|---|---|---|---|
| l2_A_vstar_lead | A | vstar | lead | **0.82** | 82 | 0 | 259 |
| l2_A_vstar_cot | A | vstar | cot | 0.79 | 79 | 0 | 240 |
| l2_A_vstar_cot_greedy | A | vstar | cot_greedy | 0.84 | 84 | 0 | 258 |
| l2_A_mmvp_lead | A | mmvp | lead | 0.69 | 69 | 0 | 408 |
| l2_A_mmvp_cot | A | mmvp | cot | **0.73** | 73 | 0 | 374 |
| l2_A_mmvp_cot_greedy | A | mmvp | cot_greedy | 0.67 | 67 | 0 | 386 |
| l2_A_math_vista_lead | A | math_vista | lead | 0.57 | 57 | 0 | 734 |
| l2_A_math_vista_cot | A | math_vista | cot | 0.57 | 57 | 0 | 678 |
| l2_A_math_vision_lead | A | math_vision | lead | 0.17 | 17 | 0 | 1416 |
| l2_A_math_vision_cot | A | math_vision | cot | **0.18** | 18 | 0 | 1341 |
| l2_A_visulogic_lead | A | visulogic | lead | **0.59** | 59 | 0 | 1062 |
| l2_A_visulogic_cot | A | visulogic | cot | 0.55 | 55 | 0 | 961 |
| l2_A_realworldqa_lead | A | realworldqa | lead | 0.38 | 38 | **17** | 177 |
| l2_A_realworldqa_cot | A | realworldqa | cot | 0.38 | 38 | **17** | 168 |
| l2_B_vstar_lead | B | vstar | lead | **0.76** | 76 | 0 | 761 |
| l2_B_vstar_cot | B | vstar | cot | 0.69 | 69 | 0 | 701 |
| l2_B_vstar_cot_greedy | B | vstar | cot_greedy | 0.81 | 81 | 0 | 445 |
| l2_B_mmvp_lead | B | mmvp | lead | 0.60 | 60 | 0 | 472 |
| l2_B_mmvp_cot | B | mmvp | cot | **0.62** | 62 | 0 | 427 |
| l2_B_mmvp_cot_greedy | B | mmvp | cot_greedy | 0.71 | 71 | 0 | 348 |
| l2_B_math_vista_lead | B | math_vista | lead | **0.51** | 51 | 0 | 929 |
| l2_B_math_vista_cot | B | math_vista | cot | 0.50 | 50 | 0 | 879 |
| l2_B_math_vision_lead | B | math_vision | lead | 0.17 | 17 | 0 | 1293 |
| l2_B_math_vision_cot | B | math_vision | cot | 0.17 | 17 | 0 | 1208 |
| l2_B_visulogic_lead | B | visulogic | lead | 0.41 | 41 | 0 | 1362 |
| l2_B_visulogic_cot | B | visulogic | cot | **0.43** | 43 | 0 | 1281 |
| l2_B_realworldqa_lead | B | realworldqa | lead | **0.37** | 37 | **17** | 590 |
| l2_B_realworldqa_cot | B | realworldqa | cot | 0.29 | 29 | **17** | 537 |

**28/28 全部成功、0 失败重试**；realworldqa 两模型各 17 条 failed_extraction（前 100 条中含 17 条缺图样本，与已知 68/765 缺图一致）。粗体为该模型×基准下方法最优。

### 4.2 LEAD vs cot / cot_greedy 对比

| 基准 | A: cot | A: lead | ΔA | A: cot_greedy | B: cot | B: lead | ΔB | B: cot_greedy |
|---|---|---|---|---|---|---|---|---|
| vstar | 79 | 82 | **+3.0** | 84 | 69 | 76 | **+7.0** | 81 |
| realworldqa | 38 | 38 | 0.0 | — | 29 | 37 | **+8.0** | — |
| math_vista | 57 | 57 | 0.0 | — | 50 | 51 | **+1.0** | — |
| math_vision | 18 | 17 | -1.0 | — | 17 | 17 | 0.0 | — |
| mmvp | 73 | 69 | -4.0 | 67 | 62 | 60 | -2.0 | 71 |
| visulogic | 55 | 59 | **+4.0** | — | 43 | 41 | -2.0 | — |

- **B（最接近论文的代理）**：LEAD vs cot = **3 胜 1 平 2 负**。胜：VStar +7.0、RealWorldQA +8.0、MathVista +1.0；平：MathVision 0.0；负：MMVP -2.0、VisuLogic -2.0。
- **A（Qwen 代理）**：2 胜 2 平 2 负。胜：VStar +3.0、VisuLogic +4.0；平：RWQA/MathVista 0.0；负：MMVP -4.0、MathVision -1.0。
- **cot_greedy 参考**：采样影响显著——B 模型 MMVP 71 vs cot 62、VStar 81 vs cot 69；论文仅说明"示例用 greedy"，主实验采样配置未知（B6）。
- 论文主张"LEAD 全基准正提升"：**两个模型上均未获支持**（B 3/6、A 2/6 基准正效应）。

### 4.3 Qwen（A）vs R1-RL（B）对比

- 绝对水平：A 在 vstar（cot 79 vs 69）、mmvp（73 vs 62）、math_vista（57 vs 50）、visulogic（55 vs 43）均高于 B；B 仅在 realworldqa lead（37 vs 38 接近）不落后。B 作为 RL 后训练模型并未系统性更强——进一步支持"B 与论文 R1-Onevision-7B 不是同一模型"的推断（§7）。
- LEAD 效应：两模型在 **VStar 均为正**（A +3.0 / B +7.0），是跨模型最稳定的正效应；MMVP 两模型均为负；MathVision 均≈0。
- 旧 run 曾记录 Qwen 代理"LEAD<CoT 于 6/7 基准"，本轮 Qwen 为 2 胜 2 平 2 负（VStar/VisuLogic 转正），与 N=50 抽样波动一致，**LEAD 在 Qwen 上的负效应不稳健**。

---

## 5. 论文对齐（12 单元格）

对齐口径：论文 Table 2/3 中 R1-Onevision-7B 行（Base 与 +LEAD）× 6 个有本地数据与实验输出的基准；复现侧 B 模型 `cot ≈ Base`、`lead ≈ +LEAD`。Δ = 复现 − 论文（百分点）。Base 单元格方向 = 复现是否达到/超过论文基线（Δ≥0）；LEAD 单元格方向 = LEAD 效应（lead−cot）符号是否与论文一致（论文 12 个 LEAD 效应全为正）。分类优先级：not_comparable > exact > within_3pp > direction_match/conflict。完整归因与 caveats 见 `results/verification_report.json`。

### 5.1 逐项表

| # | 基准 | 方法 | 论文值 (R1-Onevision-7B) | 复现值 (R1-RL, N=100) | Δ(pp) | 方向 | 分类 | 归因（Top） |
|---|---|---|---|---|---|---|---|---|
| 1 | VStar | Base | 66.5 | 69.0 | +2.5 | ✓ | **within_3pp** | 抽样噪声（100/191，σ≈4.7pp）+ 模型身份 + 匹配口径 |
| 2 | VStar | LEAD | 71.2 | 76.0 | +4.8 | ✓ | direction_match | LEAD 效应方向一致且幅度更大（+7.0 vs +4.7） |
| 3 | RealWorldQA | Base | 62.5 | 29.0 | −33.5 | ✗ | not_comparable | 前 100 条 17% 缺图（failed 计入分母）；有效样本 29/83=34.9% 仍远低于论文 |
| 4 | RealWorldQA | LEAD | 66.4 | 37.0 | −29.4 | ✓* | not_comparable | 同上；LEAD 效应方向一致且更大（+8.0 vs +3.9），方向证据仍有效 |
| 5 | MMVP | Base | 43.0 | 62.0 | +19.0 | ✓ | direction_match | +19pp 远超噪声→模型身份差异或本地子集偏易 |
| 6 | MMVP | LEAD | 45.0 | 60.0 | +15.0 | ✗ | **direction_conflict** | LEAD 效应反向（论文 +2.0 vs 复现 −2.0）；采样交互 + 实现差异 |
| 7 | MathVision | Base | 29.9 | 17.0 | −12.9 | ✗ | not_comparable | 512 token 截断（触顶 25-26%，答案尾部丢失）；匹配口径非官方 |
| 8 | MathVision | LEAD | 32.4 | 17.0 | −15.4 | ✗ | not_comparable | 同 7；LEAD 效应 0.0 vs +2.5 被截断掩盖，无法评估 |
| 9 | MathVista | Base | 64.1 | 50.0 | −14.1 | ✗ | **direction_conflict** | 子集/身份/匹配口径系统性差异（旧 run Qwen N=50=52.0 同量级佐证） |
| 10 | MathVista | LEAD | 66.4 | 51.0 | −15.4 | ✓ | direction_match | LEAD 效应方向一致（+1.0 vs +2.3）但幅度不足一半 |
| 11 | VisuLogic | Base | 24.9 | 43.0 | +18.1 | ✓ | direction_match | +18.1pp 超噪声；截断反而压低复现值，真实差距更大 |
| 12 | VisuLogic | LEAD | 26.1 | 41.0 | +14.9 | ✗ | **direction_conflict** | LEAD 效应反向（论文 +1.2 vs 复现 −2.0）；截断率相近非直接原因 |

\* 单元格 4 的"方向"指 LEAD 效应方向（与论文一致），但绝对水平因缺图口径不可比，故分类为 not_comparable。

### 5.2 分类统计

| 分类 | 数量 | 单元格 |
|---|---|---|
| exact（精确复现） | **0** | — |
| within_3pp（±3pp 内） | **1** | VStar Base（+2.5） |
| direction_match（方向一致） | **4** | VStar LEAD、MMVP Base、MathVista LEAD、VisuLogic Base |
| direction_conflict（方向冲突） | **3** | MMVP LEAD、MathVista Base、VisuLogic LEAD |
| not_comparable（不可比） | **4** | RWQA Base/LEAD、MathVision Base/LEAD |
| 合计 | **12** | |

LEAD 效应方向一致性：论文 12 个正效应中，B 模型复现 **3/6** 正（VStar、RWQA、MathVista；其中 VStar +7.0 vs +4.7、RWQA +8.0 vs +3.9 幅度超过论文），A 模型 2/6 正。

### 5.3 结论表述（严格采用 verification 报告口径）

> **机制级复现成功**：LEAD 的熵感知模式切换（Eq 4-8）、概率加权连续嵌入（Eq 3）、视觉锚点注入（Eq 9 的机制形态）、Cmax=5 与动态熵阈值在 `generate_lead` 中均有真实实现（`analysis/reproduction_matrix.json` 机制 7 项全 partial/high-confidence），并已在 7B 视觉推理模型上端到端运行（L1/L2，`[LEAD] Step` 切换日志见 `execution/logs/run_*_lead.log`）。
>
> **数值级复现失败**：12 个可对齐单元格中 0 个精确复现、仅 1 个落在 ±3pp 内；8 个单元格偏差 >12pp，超出 100 样本抽样噪声（σ≈3-5pp）可解释范围。系统性偏离（RWQA −33.5、MathVista −14.1、MathVision −12.9 显著低于论文；MMVP +19.0、VisuLogic +18.1 显著高于论文）指向模型身份差异（R1-RL ≠ 论文 R1-Onevision-7B，论文未给权重链接）与本地数据/匹配口径差异，而非单一可归因因素。
>
> **方向级部分复现**：LEAD 的正效应在 6 个基准中 3 个被复现（VStar/RealWorldQA 两个通用核心基准方向一致且幅度更大），MMVP 与 VisuLogic 出现效应反向，MathVision 无效应。论文"全基准提升"的强主张在本复现证据下未获支持。

（禁止性表述说明：本报告不使用"论文数值得到验证/证伪"式结论；上述表述限定于机制/数值/方向三个层级。）

---

## 6. 阻塞与未复现项

### 6.1 B1-B9 最终状态（定义见 `analysis/reproduction_matrix.json` blockers）

| 阻塞 | 严重度 | 内容 | 最终状态 |
|---|---|---|---|
| B1 | critical | 论文 5 模型不可用 | **部分解除**：R1-Onevision-7B-RL 已下载（16.6GB，SHA256 实测）并跑通 14 组实验；**身份存疑未解除**（RL 后训练模型 ≠ 论文 R1-Onevision-7B）；Vision-R1（磁盘/许可）、VL-Rethinker/VL-Cogito/OpenVLThinker（无来源）仍不可用 |
| B2 | critical | 11/17 论文数据集缺失 | **部分解除**：6 个 HF 数据集已下载（3.91GB，SHA256 实测）但**未转换**为仓库 JSONL；Bingo/POPE 许可阻塞未解除 |
| B3 | high | 消融参数硬编码（λ/a/b1/b2 模块常量、window_size=256、单 token 锚点） | **未解除**（无代码改动；window_size 256 vs 论文消融最优 128，λ 不可配置） |
| B4 | high | VCD/MemVR/SID 基线零实现 | **未解除**（Table 2 三行对比无法产出） |
| B5 | high | MMHalu/Bingo 打分器、POPE R/P/A 无实现 | **未解除**（依赖 GPT-4 类 API 与官方脚本） |
| B6 | medium | 论文侧缺失（模型链接/采样超参/eps/附录 A-D） | **未解除**（无法从论文获得） |
| B7 | medium | realworldqa 缺图 68/765 | **未解除**（已量化为 17/100 failed_extraction） |
| B8 | medium | 全量复现资源（数百 GPU 时、磁盘、HF 直连） | **部分缓解**：2×A100 并行完成 L2；全量仍不可行；下载已走 [本地路径] |
| B9 | medium | 评估口径非官方（四级匹配器为本地发明、failed 计入分母） | **未解除**（所有 accuracy 均为本地口径） |

### 6.2 L3（全量复现）跳过理由

`execution/experiment_summary.json` full_reproduction：状态 skipped，原因 **B1/B2/B4/B5**——论文模型 3/5 缺失、11/17 数据集未就绪、3 条基线零实现、幻觉打分器缺失；且全量 physunibench 3304×25600 tokens×多方法 = 数百 GPU 小时而无论文对照列。L2 以 100 样本子集提供方向性证据。

### 6.3 未覆盖项

- **11/17 论文基准无输出**：MMEval-Pro、VMCBench、MMHalu、Bingo、POPE（R/P/A）、MathVerse、Geometry3K、MMK12-Math/Phys/Chem/Bio（B2/B5）。
- **3 条基线无代码**：VCD、MemVR、SID（B4）；`--method` 仅有 lead/cot/cot_greedy。
- **Table 1 全部 8 行**（λ 消融：λ 不可配置 + 模型/数据/打分器缺，B3/B1/B2/B5）、**Table 2/3 其余 4 模型行**（B1/B2）。
- **Fig 1-3、7-10**：过渡词共现统计、熵曲线、掩码消融、注意力可视化、GPT-5 评分、Pass@k——全部无工具实现（not_implemented/blocked，见矩阵）。
- **论文 6 条核心数值主张**（+3.6%/+4.7%/+3.8%/+2.0%/+3.2%/动态阈值增益）：全部不可精确产出（blocked）。

---

## 7. 已知坑与复现注意事项

1. **图片路径失效**（最高优先级）：tracked `data/*.jsonl` 的 `image` 均为作者机器绝对路径（`[本地路径][本地路径]-tmp/开源/已解压/LEAD/…`），`load_dataset` 对绝对路径原样透传（`lead/data.py:51`），直接用 tracked 文件必得 ~0% 准确率。**必须用 `data/*_local.jsonl`**（`[本地路径]bench_images/`）。`script/normalize_image_paths.py` 会原地改写 tracked 文件，勿运行。
2. **max_new_tokens=512 截断**：本轮设定 512（论文未公布上限；仓库 run.sh 默认 25600）。分词器实测触顶比例：B math_vision 25-26%、B visulogic 31-36%、B math_vista 8-10%、B vstar/mmvp 0%、B rwqa 1%。截断直接压低 MathVision/VisuLogic 绝对水平并掩盖 LEAD 效应评估。
3. **评估口径（B9）**：本地四级匹配器（字母/字母+文本/数值 isclose/自由文本尾部 300 字符）为工作区发明，非官方脚本；`failed_extraction` 计入分母。HEAD 版 evaluator 只有字母匹配（无法评 `C. $72^{\circ}$` 类答案）——**必须用工作区版**。任何对外数值对比需声明口径。
4. **R1-RL 身份存疑**：`config.json` 的 `_name_or_path` 显示该权重是 LLaMA-Factory 基于 Qwen2.5-VL-7B-Instruct 的 RL 后训练产物（cot_0227/full）。作者仓库 README 指向此模型，但论文正文无链接、无法核对是否为论文所用 checkpoint（B6）。所有 B 模型数值只能作为 **proxy 证据**。
5. **采样敏感性**：T=0.6 采样下 cot_greedy 与 cot 差距显著（B mmvp 71 vs 62、vstar 81 vs 69）；论文仅说明示例用 greedy，主实验配置未知——建议任何复现固定 seed 并报告采样超参。
6. **实现差异（B3）**：`thinking_token_id` 硬覆盖为 `<|image_pad|>`（`generation_utils.py:240-243`，`<think>` 解析为死代码）；锚点为单 token（论文为 3-token 均值）、权重 `a*(1-alpha)` 随步数衰减（a=1.0 模块常量，λ 不可配置）；数学 token 强制 normal 使 soft 模式仅覆盖散文 token；window_size=256（论文最优 128）；终止语义为 `[C,2C]` 收敛注入 + 32 token 预算（论文为"超限即停"）。
7. **HEAD 代码不可直接跑**：HEAD `main.py` 用 `device_map='auto'`（fp32，多卡分片会与手写解码循环的 embedding matmul / `batch_select_indices` 错位），HEAD `generation_utils.py` 的 embedding 矩阵留在 CPU——需工作区三处补丁（`.to(device)`、bf16 单设备加载、四级匹配器）。
8. **网络**：huggingface.co 直连超时，所有 HF 操作必须 `HF_ENDPOINT=http[本地路径]/[本地路径].com` + `HF_HUB_DISABLE_XET=1`（A1）；下载注意 `--repo-type dataset`。
9. **其他仓库坑**（详见 repo_manifest known_issues）：demo answer 为空（acc 恒 0，仅验管线）；results.jsonl 不持久化 extracted_answer/is_correct；generate_lead 默认 max_new_tokens=32768 与 CLI 25600 不一致；`lead/__init__.py` 触发 transformers/qwen-vl-utils 导入（测试收集 gotcha）；batch>1 声称支持但实际单样本。

---

## 8. 可复现性评估与后续建议

### 8.1 评估

- **机制/流程级**：可复现（L0/L1/L2 全通过，7 项机制 partial/high-confidence）。
- **数值级**：当前不可复现（exact=0；12 单元格仅 1 个 ±3pp 内）；且按 B1-B9 现状，**在给定仓库+数据+模型条件下不存在任何可精确产出的论文数值**（矩阵 `summary.exact=0`）。
- **方向级**：部分复现（B 模型 3/6 正效应，VStar/RealWorldQA 稳定）。
- 论文侧可复现性短板：无权重/数据/脚本链接、无采样与截断超参、无样本量、附录缺失——**即使资产齐备，第三方也无法重建完全同口径的评测**。

### 8.2 后续路径（P0-P4，来源 `reproduction_matrix.recommended_path`）

| 路径 | 内容 | 现状 / 下一步 |
|---|---|---|
| P0 机制验证 | 41 测试 + demo 单样本 | **已完成**（L0/L1） |
| P1 代理实验 | R1-RL × 6 本地基准 × {lead,cot,cot_greedy} × N≥100 × 多 seed | **本轮 N=100 已完成**；下一步：多 seed 复核 VStar/RWQA 正效应、MMVP/VisuLogic 反效应；补 RWQA 缺图样本（向作者索取或过滤） |
| P2 数据接入 | 6 个已下载 HF 数据集 → JSONL+图片转换 → 接入 MathVerse/Geometry3K/MMK12 | 数据集已就位（B2 半解除）；需写转换脚本（仓库无），再跑 R1-RL 对应列 proxy |
| P3 参数对齐 | window_size→128、3-token 锚点均值、λ 参数化（代码改动，需 patch 记录） | 未开始（B3）；改动后重跑 P1 子集验证机制语义 |
| P4 打分线 | VCD 实现 + MMHalu/Bingo 打分 + POPE 官方脚本 | 未开始（B4/B5，依赖外部 API 与基线代码） |
| 不建议 | 全量 5 模型 × 17 基准数值复现 | B1/B2/B4/B5 未解除前无意义 |

---

## 9. 引用与证据索引

| 证据 | 路径（run 根目录相对） |
|---|---|
| 论文清单（数值转录、超参、主张） | `analysis/paper_manifest.json` |
| 论文原始文本 | `analysis/paper_text_layout.txt`、`analysis/paper_text_raw.txt` |
| 仓库清单（CLI、解码流程、已知坑） | `analysis/repo_manifest.json` |
| 覆盖矩阵（102 项、B1-B9、P0-P4） | `analysis/reproduction_matrix.json` |
| 资产锁（SHA256、revision、下载记录、A1-A7） | `assets/assets.lock.json`、`assets/artifacts.jsonl` |
| 环境锁与验证 | `environment/verify.txt`、`environment/environment.yml`、`environment/conda-explicit.txt`、`environment/pip-freeze.txt`、`environment/environment_full.yml`、`environment/nvidia-smi.txt`、`environment/git-status.txt` |
| 命令流水（含时间戳/退出码） | `execution/commands.jsonl` |
| 实验汇总（28 runs、GPU 统计、L3 理由） | `execution/experiment_summary.json`、`execution/gpu.csv`、`results/l2_summary.jsonl` |
| 28 份评测报告 | `results/l2_*_eval.json`、`results/l1_vstar_eval.json` |
| 逐 run 日志 | `execution/logs/run_{A,B}_{bench}_{method}.log`、`execution/logs/*.log` |
| GPU 采样 | `results/gpu_A.csv`、`results/gpu_B.csv`、`execution/gpu.csv` |
| 验证报告（12 单元格、归因、caveats） | `results/verification_report.json` |
| 问题归档（系统 issue + B1-B9 摘要 + 实验归因） | `report/RUN_ISSUES.md`（run 根符号链接） |

---

*报告生成：report-writer（2026-08-04）；所有结论均可回溯到上述 run 内文件；与本 run 无关的外部文档（README_REPRO.md、docs/full_eval_record.md）仅在归因时作为参考佐证引用。*
