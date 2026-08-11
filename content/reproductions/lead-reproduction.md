---
title: "LEAD 复现记录：Latent Entropy-Aware Decoding"
tags:
  - type/reproduction
  - methods/latent-entropy-decoding
  - methods/hallucination-mitigation
published: 2026-08-11
updated: 2026-08-11
---

# LEAD 复现记录：Latent Entropy-Aware Decoding

## 对应论文

- 论文：*Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding*（arXiv 2603.13366）
- 仓库：https://github.com/mlrm-LEAD/mlrm-LEAD

## 复现结果

- **结果**：partial（部分成功）
- **状态**：demo 运行成功，完整评测部分完成

## 使用的组件

- latent entropy 感知解码策略
- 概率分布语义表征

## 实验经验

- demo 路径可端到端跑通；
- 完整评测受限于资源与模型加载，部分指标未完成（详见后续更新）。

## 相关页面

- **[[projects/lvlm-hallucination|LVLM 幻觉缓解研究项目]]**
- **[[papers/pade|PADE 论文笔记]]**（同方向的注意力干预方法）
