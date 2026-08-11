---
title: "Revealing and Enhancing Core Visual Regions: Harnessing Internal Attention Dynamics for Hallucination Mitigation in LVLMs"
tags:
  - type/paper
  - methods/attention-intervention
  - methods/hallucination-mitigation
published: 2026-08-11
updated: 2026-08-11
---

# Revealing and Enhancing Core Visual Regions: Harnessing Internal Attention Dynamics for Hallucination Mitigation in LVLMs

- **作者**：Guangtao Lyu, Qi Liu, Chenghao Xu, Jiexi Yan, Muli Yang, Xueting Li, Fen Fang, Cheng Deng
- **年份**：2026
- **DOI**：10.48550/arXiv.2602.15556
- **类型**：训练-free 注意力干预（Preprint）

## 解决的问题

大型视觉语言模型（LVLM）易产生与图像事实不符的幻觉输出。现有训练-free 方法（对比解码、辅助专家模型、静态内部信号增强）存在计算开销大、易受 attention sink 影响等问题。

## 核心方法

- **Positive Attention Dynamics (PAD)**：利用 LVLM 内部正向注意力动态自然地揭示语义核心视觉区域；
- **Positive Attention Dynamics Enhancement (PADE)**：基于 PAD 图识别核心视觉区域；
- **Median Absolute Deviation (MAD) Scaling**：按注意力头自适应控制干预强度；
- **System-Token Compensation (STC)**：维持对复杂用户指令的注意力与长输出一致性。

## 相关页面

- **[[projects/lvlm-hallucination|LVLM 幻觉缓解研究项目]]**
- 复现记录：暂无公开复现

## 关键结论

- 在多个 LVLM 与 benchmark 上提升视觉 grounding 并减少幻觉（论文声称，训练-free 干预有效）。
