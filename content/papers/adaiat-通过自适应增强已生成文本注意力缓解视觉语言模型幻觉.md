---
title: "AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM"
tags:
  - "type/paper"
  - "methods/iat"
  - "methods/adaiat"
description: "AdaIAT：通过自适应增强已生成文本注意力缓解视觉语言模型幻觉 摘要 大型视觉语言模型在图像描述、视觉问答等任务中，常常生成图像中不存在的物体、属性或关系，即视觉幻觉。已有注意力干预方法主要通过增强模型对图像 token 的注意力来降低幻觉，但这种做法会相对削弱模型对已生成文本的关注，使模型遗忘前文并产生重复、单调的描述。 本文围绕 AdaIAT 方法展开。该方法首先发现：模型生成真实物体时，对已生成文本的注意力明显高于生成幻觉物体时的注意力。基于这一现象，论文提出 IAT，通过增强模型对已生成文本 token 的注意力，利用其中已经被语言化、压缩和指"
authors:
  - "Zhong, Li'an"
  - "He, Ziqiang"
  - "Zheng, Jibin"
  - "Li, Jin"
  - "Wang, Z. Jane"
  - "Kang, Xiangui"
year: "2026"
published: 2026-08-11
updated: 2026-08-14
modified: 2026-08-14
---

# AdaIAT：通过自适应增强已生成文本注意力缓解视觉语言模型幻觉

## 摘要

大型视觉语言模型在图像描述、视觉问答等任务中，常常生成图像中不存在的物体、属性或关系，即视觉幻觉。已有注意力干预方法主要通过增强模型对图像 token 的注意力来降低幻觉，但这种做法会相对削弱模型对已生成文本的关注，使模型遗忘前文并产生重复、单调的描述。

本文围绕 AdaIAT 方法展开。该方法首先发现：模型生成真实物体时，对已生成文本的注意力明显高于生成幻觉物体时的注意力。基于这一现象，论文提出 IAT，通过增强模型对已生成文本 token 的注意力，利用其中已经被语言化、压缩和指令筛选过的视觉信息；随后进一步提出 AdaIAT，使用层级阈值决定是否干预，并根据不同注意力头的真实—幻觉差异自适应确定干预强度。该方法在不更新模型参数的情况下，实现了幻觉率、预测能力和文本多样性之间更好的平衡。

---

# 一、从“降低幻觉”到“维持模型能力：论文要解决的核心问题

## 1.1 大型视觉语言模型中的视觉幻觉

大型视觉语言模型将图像编码器、模态连接器和大型语言模型结合起来，能够根据图像生成描述或回答问题。但由于视觉特征与语言特征之间存在模态差异，模型在生成过程中可能过度依赖语言先验，而没有充分依据输入图像。

视觉幻觉主要表现为：

- 描述图像中不存在的物体；
- 为真实物体生成错误属性；
- 错误描述物体数量、位置或关系；
- 根据语言常识补充图像中并不存在的细节。

例如，图像中只有一辆摩托车，模型却生成"背景中还有两辆汽车"。这种错误会显著降低视觉语言模型在自动驾驶、医疗图像、机器人交互等场景中的可信度。

## 1.2 现有注意力增强方法的局限

PAI、HGAI 等方法认为，视觉幻觉产生的重要原因是模型对图像 token 的关注不足，因此在推理阶段直接增强图像 token 的注意力。

这一思路可以表示为：

$$
\text{幻觉风险上升}
\quad\Longrightarrow\quad
\text{增强图像 token 注意力}
$$

这种方法能够降低幻觉率，但注意力总量是有限的。由于 Softmax 后的注意力满足：

$$
\sum_i A(i)=1,
$$

增强图像 token 的注意力必然会相对压低其他 token 的注意力，尤其是已生成文本 $T_p$ 的注意力。

当模型无法充分关注已经生成的内容时，就容易"忘记自己说过什么"，反复描述图像中最显著的对象。例如：

> The motorcycle is parked on the street.
> The motorcycle is an old-fashioned model.
> The motorcycle is parked next to a building.

因此，已有方法虽然减少了幻觉，却可能造成：

$$
\text{幻觉减少}
\quad+\quad
\text{重复增加}
\quad+\quad
\text{语言多样性下降}.
$$

## 1.3 论文真正解决的是一个多目标平衡问题

这篇论文并不只是追求尽可能低的幻觉率，而是试图同时实现三个目标：

$$
\boxed{
\text{降低幻觉}
+
\text{保持语言连贯性和多样性}
+
\text{保留模型原有预测能力}
}
$$

因此，它解决的本质问题可以写成：

$$
\min \ \text{Hallucination},
$$

同时约束：

$$
\text{Language Quality}\not\downarrow,
\qquad
\text{Prediction Capability}\not\downarrow.
$$

换句话说，论文希望在幻觉率、文本质量和预测准确性之间取得更优的折中，而不是通过过强的视觉干预换取单一指标提升。

---

# 二、从注意力模式中寻找突破口：已生成文本为何能够帮助抑制幻觉

## 2.1 模型输入的四个组成部分

在自回归生成第 $n+1$ 个 token 时，LVLM 的输入可以写为：

$$
\mathcal I=[S,V,U,T_p],
$$

其中：

$$
S=\{s_1,\ldots,s_a\}
$$

表示系统提示词；

$$
V=\{v_1,\ldots,v_m\}
$$

表示图像编码器产生的图像 token；

$$
U=\{u_1,\ldots,u_b\}
$$

表示用户指令；

$$
T_p=\{t_1,\ldots,t_n\}
$$

表示模型此前已经生成的文本 token。

总输入长度为：

$$
\operatorname{len}=a+m+b+n.
$$

这里需要区分两个容易混淆的符号：

- $V$ 可以表示输入中的图像 token 集合；
- 标准注意力公式中的 $V_{\text{value}}$ 表示 Value 矩阵。

二者含义不同。

## 2.2 标准自注意力计算

对于第 $l$ 层、第 $h$ 个注意力头，当前 token 对所有历史 token 的注意力 logit 为：

$$
\tilde{\mathbf A}^{(l,h)}
=
\frac{
\mathbf Q_{t_n}^{(l,h)}
\left(\mathbf K^{(l,h)}\right)^\top
}{
\sqrt{d_k}
}.
\tag{1}
$$

其中：

- $\mathbf Q_{t_n}^{(l,h)}$ 是当前 token 对应的 Query；
- $\mathbf K^{(l,h)}$ 是全部历史 token 的 Key；
- $d_k$ 是 Key 的维度；
- $\tilde{\mathbf A}$ 是 Softmax 前的注意力分数。

经过 Softmax 后得到注意力概率：

$$
\mathbf A^{(l,h)}
=
\operatorname{softmax}
\left(
\tilde{\mathbf A}^{(l,h)}
\right).
\tag{2}
$$

注意力头的输出为：

$$
\mathbf O^{(l,h)}
=
\mathbf A^{(l,h)} \mathbf V_{\text{value}}^{(l,h)}
=
\sum_i A^{(l,h)}(i)\,
v_{\text{value},i}^{(l,h)}.
\tag{3}
$$

从功能上看：

- $A(i)$ 决定"从第 $i$ 个 token 读取多少信息"；
- $v_{\text{value},i}$ 决定"第 $i$ 个 token 提供什么信息"；
- $A V_{\text{value}}$ 是全部 token 信息的加权混合。

## 2.3 为什么修改注意力 $A$，而不是修改 $A V_{\text{value}}$

AdaIAT 的目标是提高模型从历史文本 $T_p$ 中读取信息的比例，这是一个"从哪里读取信息"的问题，因此直接修改注意力权重 $A$ 最自然。

在乘以 Value 之前，注意力中的每个位置仍然与一个具体 token 对应：

$$
A(i)\leftrightarrow \mathcal I(i).
$$

因此可以精确选择：

$$
\mathcal I(i)\in T_p
$$

的所有位置进行增强。

但在计算：

$$
O=A V_{\text{value}}
=\sum_i A(i)v_i
$$

之后，token 维度已经被压缩，图像 token、提示词和历史文本的信息都混合进同一个向量，通常无法再精确区分哪部分来自 $T_p$。

直接修改 $A$ 还有三个好处：

1. **只改变信息路由，不改变 token 表示本身**；
2. **可以继续保证注意力非负且总和为 1**；
3. **便于在层、头和 token 三个粒度上进行分析与控制**。

因此，注意力干预更接近：

$$
\text{重新分配模型的信息读取预算},
$$

而不是直接修改模型已经形成的隐藏语义表示。

## 2.4 已生成文本不仅是语言上下文，也是"语言化视觉记忆"

论文最重要的观察是：

$$
T_p
\neq
\text{纯语言历史}.
$$

模型生成前文时已经同时接收了图像和用户指令，因此已生成文本通常包含：

$$
T_p
\approx
\text{指令相关视觉信息}
+
\text{语言上下文知识}.
$$

与原始图像 token 相比，$T_p$ 具有两个优势。

### 第一，信息经过指令筛选

原始图像 token 包含大量信息，其中一部分与当前问题或图像描述任务无关。

而 $T_p$ 是模型根据用户指令生成的，因此已经对原始视觉信息进行了选择、组织和压缩。

### 第二，信息已处于语言特征空间

图像 token 来自视觉编码器，即使经过连接器对齐，仍可能与 LLM 的语言表示空间存在模态差异。

而 $T_p$ 本身由 LLM 生成，天然属于语言空间。因此可将其理解为：

$$
\boxed{
\text{处于语言空间中的、指令相关的、压缩视觉表示}
}
$$

这也是为什么增强 $T_p$ 可能同时提高事实一致性和语言连贯性。

---

# 三、由统计观察建立方法依据：真实物体与幻觉物体的注意力差异

## 3.1 历史文本平均注意力的定义

论文定义第 $l$ 层、第 $h$ 个注意力头对已生成文本 $T_p$ 的平均注意力为：

$$
A_{T_p}^{(l,h)}
=
\frac{1}{n}
\sum_{[本地路径]mathcal I(i)\in T_p}
A^{(l,h)}(i).
\tag{4}
$$

它表示：

> 当前模型预测下一个 token 时，平均给每个历史文本 token 分配了多少注意力。

随后跨注意力头聚合：

$$
\bar A_{T_p}^{(l)}
=
\sum_{h=1}^{H}
A_{T_p}^{(l,h)}.
\tag{5}
$$

论文正文有时称其为头平均，但公式实际使用的是求和。严格的头平均应为：

$$
\frac{1}{H}
\sum_{h=1}^{H}
A_{T_p}^{(l,h)}.
$$

不过，只要离线阈值计算和在线检测使用相同定义，是否除以 $H$ 不会改变最终比较结果。

## 3.2 真实物体与幻觉物体注意力如何统计

论文从 COCO 2014 中随机选择 10,000 张图像，并让 LLaVA-1.5-7B 根据指令：

> Please describe the image in detail.

生成图像描述。

随后从描述中识别模型提及的物体，并与 COCO 的真实标注比较。

例如模型生成：

> A dog is sitting beside a bicycle and a car.

若图像标注中存在 dog 和 bicycle，但不存在 car，则：

$$
dog,bicycle\in D_r,
$$

$$
car\in D_h,
$$

其中 $D_r$ 表示真实物体生成事件集合，$D_h$ 表示幻觉物体生成事件集合。

论文共收集：

- 22,015 个真实物体 token；
- 9,473 个幻觉物体 token。

对于每一个物体生成事件 $e$，记录模型生成该物体 token 时所有层、所有头的注意力图，并计算：

$$
a_e^{(l,h)}
=
\frac{1}{n_e}
\sum_{i\in T_{p,e}}
A_e^{(l,h)}(i).
\tag{6}
$$

其中 $n_e$ 是该物体生成前已经生成的文本长度。

然后分别对真实物体和幻觉物体集合求平均：

$$
A_{T_p}^{r,(l,h)}
=
\frac{1}{|D_r|}
\sum_{e\in D_r}
a_e^{(l,h)},
\tag{7}
$$

$$
A_{T_p}^{h,(l,h)}
=
\frac{1}{|D_h|}
\sum_{e\in D_h}
a_e^{(l,h)}.
\tag{8}
$$

最终得到两个矩阵：

$$
A_{T_p}^{r},A_{T_p}^{h}
\in\mathbb R^{L\times H}.
$$

每个矩阵元素表示某一层、某一个头在生成真实或幻觉物体时，对历史生成文本的平均注意力。

## 3.3 关键统计发现

实验发现：

$$
\bar A_{T_p}^{r}
>
\bar A_{T_p}^{h}.
$$

真实物体与幻觉物体在历史文本注意力上的差异通常达到：

$$
\frac{\bar A_{T_p}^{r}}
{\bar A_{T_p}^{h}}
\approx1.5\sim2.5.
$$

而二者在图像 token 注意力上的差异通常只有：

$$
\frac{\bar A_V^{r}}
{\bar A_V^{h}}
\approx1.0\sim1.5.
$$

这说明：

$$
\boxed{
\text{真实预测与幻觉预测之间，对历史文本的注意力差异更显著}
}
$$

因此，幻觉发生时的问题可能不仅是模型"看图不够"，还包括模型没有充分利用前文已经形成的视觉—语言上下文。

需要注意的是，这一结果说明的是统计相关性：

$$
A_{T_p}\text{ 偏低}
\Rightarrow
\text{幻觉风险较高},
$$

并不能直接证明：

$$
A_{T_p}\text{ 偏低}
\Rightarrow
\text{下一个 token 一定是幻觉}.
$$

所以论文后续阈值检测更准确地说是在检测"幻觉风险代理信号"，而不是直接识别幻觉。

---

# 四、从统一增强到自适应控制：IAT 与 AdaIAT 的方法演进

## 4.1 第一阶段：IAT 统一增强已生成文本注意力

论文首先提出 Increased Attention to Generated Text，即 IAT。

IAT 对中间层中属于 $T_p$ 的注意力 logit 进行修改：

$$
\tilde A^{(l,h)}(i)
\leftarrow
\tilde A^{(l,h)}(i)
+
\alpha
\left|
\tilde A^{(l,h)}(i)
\right|,
\qquad
\mathcal I(i)\in T_p.
\tag{9}
$$

其中：

- $\alpha$ 为统一放大系数；
- 论文主要在中间层 $5\sim18$ 进行干预；
- 干预发生在 Softmax 之前。

设某个历史 token 的原始 logit 为 $z$。

当 $z\geq0$ 时：

$$
z'=z+\alpha|z|
=(1+\alpha)z.
$$

当 $z<0$ 时：

$$
z'=z+\alpha|z|
=(1-\alpha)z.
$$

当 $0<\alpha<1$ 时，负值会变得不那么负。因此，无论原始 logit 正负，历史文本 token 在 Softmax 后的相对注意力通常都会提高。

IAT 的完整流程为：

$$
QK^\top
\rightarrow
\boxed{\text{修改 }\tilde A}
\rightarrow
\operatorname{softmax}
\rightarrow
A'
\rightarrow
A'V_{\text{value}}.
$$

与增强图像 token 的 PAI 相比，IAT 增强的是：

$$
\mathcal I(i)\in T_p,
$$

而不是：

$$
\mathcal I(i)\in V.
$$

因此，它在利用视觉相关信息的同时，也增强了模型对前文的记忆，从而减少重复描述。

## 4.2 IAT 的不足：持续干预与统一强度

虽然 IAT 有效，但仍有两个问题。

### 问题一：所有时间步都进行干预

模型大多数预测本来就是正确的。若无论是否出现异常都增强 $T_p$，可能导致模型过度依赖前文，反而降低对新图像信息或用户指令的关注。

### 问题二：所有注意力头使用相同强度

不同注意力头承担不同功能。某些头在真实与幻觉生成之间差异明显，另一些头差异很小。

统一使用 $\alpha$ 会导致：

- 真正关键的头干预不足；
- 不相关的头被过度修改；
- 模型原有注意力模式受到不必要破坏。

因此，论文进一步提出 AdaIAT，使模型能够回答两个问题：

$$
\text{什么时候干预？}
$$

以及：

$$
\text{每个头应该干预多少？}
$$

---

# 五、AdaIAT 的第一层控制：用层级阈值决定何时干预

## 5.1 层级阈值的构造

根据离线统计，论文首先跨头聚合真实和幻觉状态下的历史文本注意力：

$$
\bar A_{T_p}^{r,(l)}
=
\sum_{h=1}^{H}
A_{T_p}^{r,(l,h)},
$$

$$
\bar A_{T_p}^{h,(l)}
=
\sum_{h=1}^{H}
A_{T_p}^{h,(l,h)}.
$$

随后定义第 $l$ 层的阈值：

$$
\mathcal T^{(l)}
=
\bar A_{T_p}^{h,(l)} + \beta \left( \bar A_{T_p}^{r,(l)}
-
\bar A_{T_p}^{h,(l)}
\right).
\tag{10}
$$

等价地：

$$
\mathcal T^{(l)}
=
(1-\beta)\bar A_{T_p}^{h,(l)}
+
\beta\bar A_{T_p}^{r,(l)}.
\tag{11}
$$

因此，当：

$$
0\leq\beta\leq1
$$

时，阈值位于真实注意力模式和幻觉注意力模式之间。

例如 $\beta=0.5$ 时：

$$
\mathcal T^{(l)}
=
\frac{
\bar A_{T_p}^{r,(l)}
+
\bar A_{T_p}^{h,(l)}
}{2}.
$$

即取二者的中点。

## 5.2 在线异常检测

在当前生成时间步 $t$，模型先正常计算注意力：

$$
A_t^{(l,h)}
=
\operatorname{softmax}
\left(
\frac{
Q_t^{(l,h)}K_t^{(l,h)\top}
}{
\sqrt{d_k}
}
\right).
\tag{12}
$$

随后计算该层当前对历史文本的聚合注意力：

$$
\bar A_{T_p,t}^{(l)}
=
\sum_{h=1}^{H}
\frac{1}{n_t}
\sum_{i\in T_p}
A_t^{(l,h)}(i).
\tag{13}
$$

若：

$$
\bar A_{T_p,t}^{(l)}
\geq
\mathcal T^{(l)},
$$

说明该层对历史文本关注充分，模型保持正常预测，不进行干预。

若：

$$
\bar A_{T_p,t}^{(l)}
<
\mathcal T^{(l)},
$$

则认为当前注意力模式更接近幻觉状态，触发 AdaIAT。

可以定义层级开关：

$$
g_t^{(l)}
=
\begin{cases}
1,
&
\bar A_{T_p,t}^{(l)}
<
\mathcal T^{(l)},\\[4pt]
0,
&
\bar A_{T_p,t}^{(l)}
\geq
\mathcal T^{(l)}.
\end{cases}
\tag{14}
$$

其中：

- $g_t^{(l)}=1$：该层启动干预；
- $g_t^{(l)}=0$：该层保持原始注意力。

## 5.3 $\beta$ 的实际作用

$\beta$ 不只是控制阈值位于真实和幻觉均值之间的位置，它还控制干预频率。

在通常情况下：

$$
\bar A_{T_p}^{r,(l)}
>
\bar A_{T_p}^{h,(l)}.
$$

因此：

$$
\beta\uparrow
\quad\Longrightarrow\quad
\mathcal T^{(l)}\uparrow.
$$

而触发条件是：

$$
\bar A_{T_p,t}^{(l)}
<
\mathcal T^{(l)}.
$$

所以：

$$
\beta\uparrow
\quad\Longrightarrow\quad
\text{更容易触发干预}.
$$

论文的消融实验表明：

- $\beta$ 太小：触发不足，幻觉改善有限；
- $\beta$ 适中：幻觉、F1 和 D1 之间平衡最佳；
- $\beta$ 太大：正常预测也频繁受到修改，性能开始下降。

因此，$\beta$ 类似于异常检测器中的敏感度参数。

---

# 六、AdaIAT 的第二层控制：用头级矩阵决定干预强度

## 6.1 真实—幻觉注意力比值

论文根据离线统计定义头级放大矩阵：

$$
\mathcal M
=
\frac{
A_{T_p}^{r}
}{
A_{T_p}^{h}
}.
\tag{15}
$$

其中：

$$
\mathcal M\in\mathbb R^{L\times H},
$$

且：

$$
\mathcal M^{(l,h)}
=
\frac{
A_{T_p}^{r,(l,h)}
}{
A_{T_p}^{h,(l,h)}
}.
\tag{16}
$$

该比值表示：

> 在第 $l$ 层、第 $h$ 个注意力头中，真实物体生成时对历史文本的注意力，是幻觉物体生成时的多少倍。

若：

$$
\mathcal M^{(l,h)}\gg1,
$$

说明该头在幻觉状态下存在明显的历史文本注意力缺失，应受到更强干预。

若：

$$
\mathcal M^{(l,h)}\approx1,
$$

说明该头在真实和幻觉状态下差异不大，应尽量保持原始模式。

## 6.2 自适应注意力更新

当第 $l$ 层触发干预后，对于属于历史文本区域的位置：

$$
\mathcal I(i)\in T_p,
$$

AdaIAT 执行：

$$
\hat A_t^{(l,h)}(i)
=
A_t^{(l,h)}(i)
+
\alpha
\mathcal M^{(l,h)}
A_t^{(l,h)}(i).
\tag{17}
$$

等价于：

$$
\hat A_t^{(l,h)}(i)
=
\left(
1+\alpha\mathcal M^{(l,h)}
\right)
A_t^{(l,h)}(i).
\tag{18}
$$

对于非历史文本区域：

$$
\hat A_t^{(l,h)}(i)
=
A_t^{(l,h)}(i),
\qquad
\mathcal I(i)\notin T_p.
\tag{19}
$$

随后对全部注意力位置重新归一化：

$$
A_t'^{(l,h)}(k)
=
\frac{
\hat A_t^{(l,h)}(k)
}{
\sum_j\hat A_t^{(l,h)}(j)
}.
\tag{20}
$$

最后计算注意力输出：

$$
O_t^{(l,h)}
=
A_t'^{(l,h)}
V_{\text{value}}^{(l,h)}.
\tag{21}
$$

## 6.3 统一表示 AdaIAT 的两级控制

将层级开关和头级强度合并，可以写成：

$$
\hat A_t^{(l,h)}(i)
=
\left[
1+
g_t^{(l)}
\alpha
\mathcal M^{(l,h)}
\mathbf 1\!\left(\mathcal I(i)\in T_p\right)
\right]
A_t^{(l,h)}(i).
\tag{22}
$$

其中：

- $g_t^{(l)}$ 决定该层是否干预；
- $\alpha$ 决定全局干预强度；
- $\mathcal M^{(l,h)}$ 决定各注意力头的相对强度；
- $\mathbf 1(\mathcal I(i)\in T_p)$ 决定只修改历史文本区域。

因此 AdaIAT 是一个两级控制器：

$$
\boxed{
\text{层级异常检测}
+
\text{头级强度调节}
}
$$

## 6.4 放大后历史文本总注意力如何变化

设某个注意力头原本分配给历史文本的总注意力为：

$$
p_T
=
\sum_{i\in T_p}A(i).
$$

设该头的放大倍数为：

$$
c=1+\alpha M.
$$

放大并归一化后，历史文本总注意力变为：

$$
p_T'
=
\frac{
cp_T
}{
cp_T+(1-p_T)
}.
\tag{23}
$$

代入 $c=1+\alpha M$：

$$
\boxed{ p_T'
=
\frac{
(1+\alpha M)p_T
}{
1+\alpha Mp_T
}
}
\tag{24}
$$

当：

$$
\alpha>0,\quad M>0,\quad0<p_T<1
$$

时，有：

$$
p_T'>p_T.
$$

但由于重新归一化：

$$
p_T'<1.
$$

这说明 AdaIAT 不是无约束放大注意力，而是把有限的注意力预算从其他区域重新分配给历史文本。

相应地，图像 token、系统提示词和用户指令获得的相对注意力都会下降。因此，自适应触发非常重要，否则持续增强 $T_p$ 也可能使模型忽视新视觉信息。

---

# 七、方法在完整注意力流程中的位置

## 7.1 标准注意力流程

$$
Q,K,V_{\text{value}}
$$

$$
\Downarrow
$$

$$
\tilde A
=
\frac{QK^\top}{\sqrt{d_k}}
$$

$$
\Downarrow
$$

$$
A=\operatorname{softmax}(\tilde A)
$$

$$
\Downarrow
$$

$$
O=AV_{\text{value}}
$$

$$
\Downarrow
$$

$$
\text{多头拼接、输出投影与残差连接}.
$$

## 7.2 IAT 的位置

IAT 位于：

$$
QK^\top
\quad\text{之后，Softmax 之前}.
$$

即：

$$
\tilde A
\overset{\mathrm{IAT}}{\longrightarrow}
\tilde A'
\overset{\operatorname{softmax}}{\longrightarrow}
A'.
$$

## 7.3 AdaIAT 的位置

AdaIAT 位于：

$$
\text{Softmax 之后，乘 Value 之前}.
$$

完整流程为：

$$
QK^\top
\rightarrow
\operatorname{softmax}
\rightarrow
A
$$

$$
\rightarrow
\text{计算 }A_{T_p}
\rightarrow
\text{阈值检测}
$$

$$
\rightarrow
\text{必要时修改并归一化 }A
\rightarrow
A'
$$

$$
\rightarrow
A'V_{\text{value}}.
$$

AdaIAT 不只是给 IAT 增加了一个阈值，它还将干预位置从 Softmax 前的 logit 空间转移到了 Softmax 后的概率空间。

原因是：

$$
M^{(l,h)}
=
\frac{
A_{T_p}^{r,(l,h)}
}{
A_{T_p}^{h,(l,h)}
}
$$

本身就是在概率空间统计得到的。在概率空间应用 $M$，其"真实状态是幻觉状态多少倍"的含义更直接。

若将 $M$ 直接作用于 logit，则由于 Softmax 非线性：

$$
\operatorname{softmax}(cZ)
\neq
c\operatorname{softmax}(Z),
$$

$M$ 将不再具有清晰的比例解释。

---

![image-20260728153130397](static/img/56e9205596b3.png)

# 八、为什么主要干预中间层

论文比较了多个层区间：

- 浅层：0–5；
- 中间层：5–18；
- 深层：18–31；
- 浅层加中间层：0–18；
- 中间层加深层：5–31。

实验表明，中间层 5–18 能获得更好的综合表现。

一种合理解释是：

- 浅层更接近输入和基础特征表示，干预可能破坏底层编码；
- 深层更接近最终语言决策，此时预测倾向已经形成，干预容易直接损害输出；
- 中间层承担跨模态融合、实体语义形成和上下文整合，更适合调整视觉—文本信息路由。

当 IAT 干预过多层时，甚至可能出现严重退化。例如对 5–31 层进行干预时，Distinct-1 降至约 0.036，说明模型输出几乎陷入重复。

因此，5–18 层是由消融实验得到的经验性平衡，并非理论上适用于所有模型的固定区间。

---

# 九、论文的主要创新点

## 9.1 创新一：将已生成文本重新定义为视觉信息来源

已有方法主要遵循：

$$
\text{减少幻觉}
\Rightarrow
\text{增强图像 token}.
$$

本文提出：

$$
\text{减少幻觉}
\Rightarrow
\text{也可以增强已生成文本}.
$$

其核心思想是：

$$
\boxed{ T_p
=
\text{语言空间中的指令相关视觉记忆}
}
$$

这改变了视觉幻觉干预的信息来源：模型不必始终直接回到原始图像 token，也可以利用自己此前已经组织好的视觉描述。

## 9.2 创新二：同时缓解幻觉与重复生成

图像注意力增强方法容易相对压低历史文本注意力，使模型遗忘前文。

IAT 增强 $T_p$，同时提供：

- 已确认的视觉实体；
- 已生成的属性和关系；
- 句法和篇章信息；
- 避免重复所需的上下文记忆。

因此，它并非简单提高语言先验，而是利用"经过视觉条件化的语言上下文"。

实验中，PAI 和 HGAI 虽然降低了幻觉，但 LLaVA-1.5-7B 上的 Distinct-1 从约 0.60 降至约 0.50；IAT 和 AdaIAT 则维持在约 0.60。

## 9.3 创新三：层级触发与头级放大的两粒度机制

AdaIAT 使用：

$$
\mathcal T^{(l)}
$$

决定第 $l$ 层是否需要干预；

使用：

$$
\mathcal M^{(l,h)}
$$

决定第 $h$ 个头的干预强度。

因此，其控制结构是：

$$
\text{层级是否异常}
\rightarrow
\text{头级如何修正}.
$$

相比所有层、所有头统一增强，这种方式更少破坏模型原有预测模式。

## 9.4 创新四：训练免费、可插拔

AdaIAT 不更新模型参数，只在推理阶段读取并修改注意力，因此可以与不同解码方法结合。

论文在以下模型中进行了验证：

- LLaVA-1.5-7B；
- LLaVA-1.5-13B；
- Janus-Pro-7B；
- Qwen2.5-VL-7B。

同时支持 Greedy 和 Sampling 解码。

不过，更准确地说，它是：

$$
\boxed{
\text{无需参数训练，但需要离线统计校准}
}
$$

因为 $\mathcal T$ 和 $\mathcal M$ 仍需通过带有真实/幻觉标记的样本预先估计。

## 9.5 创新五：优化目标由单一幻觉率扩展为综合权衡

以 LLaVA-1.5-7B 为例：

| 方法   | $C_S\downarrow$ | $C_I\downarrow$ | F1 $(\uparrow)$ | D1 $(\uparrow)$ |
| ------ | --------------- | --------------- | ------------- | ------------- |
| Greedy | 49.0            | 13.3            | 77.9          | 0.60          |
| PAI    | 31.8            | 7.8             | 77.7          | 0.50          |
| HGAI   | 31.4            | 6.9             | 78.3          | 0.50          |
| IAT    | 29.8            | 9.0             | 76.8          | 0.61          |
| AdaIAT | 31.4            | 8.3             | 79.4          | 0.60          |

其中：

$$
C_S
=
\frac{
|\{\text{包含幻觉物体的描述}\}|
}{
|\{\text{全部描述}\}|
}
\tag{25}
$$

衡量句子级幻觉率；

$$
C_I
=
\frac{
|\{\text{幻觉物体}\}|
}{
|\{\text{全部提及物体}\}|
}
\tag{26}
$$

衡量实例级幻觉率；

$$
D_1
=
\frac{
|\{\text{不同 unigram}\}|
}{
|\{\text{全部 unigram}\}|
}
\tag{27}
$$

衡量词汇多样性。

AdaIAT 不一定在每个单项指标上绝对最优，但它获得了更好的整体平衡：

$$
\boxed{
\text{较低幻觉率}
+
\text{较高 F1}
+
\text{保留文本多样性}
}
$$

---

# 十、方法中需要特别理解的若干细节

## 10.1 阈值检测的不是幻觉本身

判断：

$$
\bar A_{T_p,t}^{(l)}
<
\mathcal T^{(l)}
$$

只能说明历史文本注意力低于典型真实预测模式。

它检测的是：

$$
\text{低历史注意力异常},
$$

而不是：

$$
\text{下一个 token 必然为幻觉}.
$$

所以 AdaIAT 属于风险触发式校正，而不是直接的幻觉分类器。

## 10.2 离线统计基于物体 token，在线却可能干预所有时间步

$A_{T_p}^{r}$ 和 $A_{T_p}^{h}$ 是在真实和幻觉物体 token 生成时统计的。

但在线推理时，模型并不知道下一个 token 会是：

- 名词；
- 属性词；
- 动词；
- 介词；
- 功能词。

因此阈值一般在每个生成时间步执行。

这隐含假设：

$$
\text{物体 token 上发现的注意力模式}
$$

能够迁移到：

$$
\text{一般自回归生成时间步}.
$$

论文通过整体指标验证了有效性，但没有细分不同词类的触发效果。

## 10.3 $M$ 主要用于确定头的重要性，不保证精确恢复真实模式

虽然：

$$
M^{(l,h)}
=
\frac{
A_{T_p}^{r,(l,h)}
}{
A_{T_p}^{h,(l,h)}
},
$$

但更新后不一定有：

$$
A_{T_p}'^{(l,h)}
=
A_{T_p}^{r,(l,h)}.
$$

因为最终注意力还取决于：

- 当前样本的原始 $A$；
- 其他 token 的注意力；
- 生成长度；
- 归一化结果。

因此 $M$ 的主要作用是：

$$
\boxed{
\text{确定哪些头更需要增强}
}
$$

而不是将当前注意力精确映射到真实物体均值。

## 10.4 IAT 与 AdaIAT 中的 $\alpha$ 不可直接比较

IAT 中：

$$
\tilde A'
=
\tilde A+\alpha|\tilde A|
$$

作用于 Softmax 前 logit。

AdaIAT 中：

$$
A'
\propto
(1+\alpha M)A
$$

作用于 Softmax 后概率。

由于 Softmax 具有指数非线性，两种 $\alpha$ 所处尺度不同。因此 IAT 中的 $\alpha=0.8$ 与 AdaIAT 中的 $\alpha=6$ 不能直接比较为干预强度相差 7.5 倍。

---

# 十一、论文的主要局限

## 11.1 可能强化已经生成的错误

论文默认：

$$
T_p
$$

主要包含可靠的压缩视觉信息。

但如果模型早期已经生成幻觉，例如：

> There is a red car beside the building.

那么后续增强对 $T_p$ 的注意力可能使模型继续生成：

> The car has black wheels and is parked near the road.

形成：

$$
\text{早期幻觉}
\rightarrow
\text{写入 }T_p
\rightarrow
\text{注意力增强}
\rightarrow
\text{错误扩展}.
$$

因此，AdaIAT 更擅长防止模型偏离已经建立的正确上下文，但不一定能纠正已经写入历史文本的错误。

## 11.2 所有历史 token 被统一增强

论文在同一个注意力头中，对全部：

$$
i\in T_p
$$

使用相同系数。

它没有区分：

- 实体词与功能词；
- 正确 token 与可疑 token；
- 近期上下文与远期上下文；
- 与当前预测相关或无关的 token。

因此可能增强大量没有帮助的信息。

## 11.3 阈值和放大矩阵是静态统计量

$$
\mathcal T^{(l)},\quad
\mathcal M^{(l,h)}
$$

由离线数据估计，并在测试时固定。

它们可能随以下因素变化：

- 图像领域；
- 用户指令类型；
- 输出语言；
- 模型架构；
- 解码策略；
- 描述长度；
- 幻觉类型。

因此，从 COCO 自然图像得到的统计模式不一定适用于医学、遥感、文档或长篇视觉推理。

## 11.4 长度对每 token 平均注意力有影响

论文使用：

$$
A_{T_p}^{(l,h)}
=
\frac{1}{n}
\sum_{i\in T_p}A^{(l,h)}(i).
$$

设历史文本总注意力为：

$$
P_{T_p}
=
\sum_{i\in T_p}A(i),
$$

则：

$$
A_{T_p}
=
\frac{P_{T_p}}{n}.
$$

即使总注意力 $P_{T_p}$ 不变，随着生成长度 $n$ 增加，每 token 平均注意力也会下降。

因此长文本可能更容易触发阈值，产生过度干预。

## 11.5 $M$ 的比值存在数值稳定问题

当：

$$
A_{T_p}^{h,(l,h)}
\approx0
$$

时：

$$
M^{(l,h)}
=
\frac{
A_{T_p}^{r,(l,h)}
}{
A_{T_p}^{h,(l,h)}
}
$$

可能非常大。

更稳定的实现通常应加入：

$$
M^{(l,h)}
=
\frac{
A_{T_p}^{r,(l,h)}
}{
A_{T_p}^{h,(l,h)}+\varepsilon
},
\tag{28}
$$

并可能采用：

$$
M
\leftarrow
\operatorname{clip}
(M,M_{\min},M_{\max}).
\tag{29}
$$

论文正文公式没有详细说明这些数值处理。

## 11.6 工程实现依赖内部注意力访问

AdaIAT 要求访问并修改每层、每头的注意力矩阵，因此更适用于：

- 开源模型；
- 可修改 forward pass 的框架；
- 能插入 attention hook 的实现。

对于闭源 API 或某些不显式存储完整注意力矩阵的 FlashAttention 实现，部署可能较困难。

---

# 十二、未来可以继续创新的方法与方向

## 12.1 从"增强全部历史文本"转向"增强可信历史文本"

可为每个历史 token 定义可靠性：

$$
r_i\in[-1,1].
$$

可靠性可以综合：

- token 生成概率；
- 解码熵；
- 与图像区域的相似度；
- 是否对应可定位实体；
- 视觉验证器判断；
- 与当前预测的语义相关性；
- token 与当前时间步的距离。

随后使用：

$$
A'(i)
\propto
A(i)\exp(\lambda r_i),
\qquad
i\in T_p.
\tag{30}
$$

当：

$$
r_i>0
$$

时增强可信 token；

当：

$$
r_i\approx0
$$

时保持不变；

当：

$$
r_i<0
$$

时抑制疑似幻觉 token。

这会将 AdaIAT 从：

$$
\text{统一历史文本增强}
$$

发展为：

$$
\boxed{
\text{可靠性加权的细粒度记忆增强}
}
$$

## 12.2 联合控制图像注意力与文本注意力

IAT 强调 $T_p$，PAI 强调图像 token。实际上，两者并非互斥。

可以定义：

$$
p_V=\sum_{i\in V}A(i),
\qquad
p_T=\sum_{i\in T_p}A(i).
$$

根据当前生成状态预测两个控制量：

$$
\lambda_V=f_V(x_t),
\qquad
\lambda_T=f_T(x_t).
$$

然后执行：

$$
A'(i)
\propto
\begin{cases}
A(i)e^{\lambda_V},
&
i\in V,\\[4pt]
A(i)e^{\lambda_T},
&
i\in T_p,\\[4pt]
A(i),
&
\text{otherwise}.
\end{cases}
\tag{31}
$$

例如：

- 生成新物体时提高图像注意力；
- 继续描述已确认物体时提高历史文本注意力；
- 生成关系或数量时同时使用两种信息。

这种方法可以形成：

$$
\boxed{
\text{视觉证据—语言记忆动态路由}
}
$$

## 12.3 将启发式增强转化为最小扰动优化

AdaIAT 当前使用乘法增强，可以进一步构造约束优化：

$$
\min_{A'}
D_{\mathrm{KL}}(A'\Vert A),
\tag{32}
$$

满足：

$$
\sum_{i\in R_t}A'(i)
\geq
\tau_t,
\tag{33}
$$

其中 $R_t$ 是当前被判断为可靠的信息集合。

该目标表示：

> 在保证可靠信息获得足够注意力的同时，使修改后的注意力与原始注意力尽可能接近。

其解通常具有指数倾斜形式：

$$
A'(i)
=
\frac{
A(i)
\exp\left(
\lambda\mathbf 1(i\in R_t)
\right)
}{
\sum_j
A(j)
\exp\left(
\lambda\mathbf 1(j\in R_t)
\right)
}.
\tag{34}
$$

相比固定乘法，这种方法可以明确控制对原始模型的扰动程度。

## 12.4 将静态阈值改为样本相关、时间相关阈值

固定阈值：

$$
\mathcal T^{(l)}
$$

可以改为：

$$
\mathcal T_t^{(l)}
=
g_l
\left(
I,U,T_p,
\text{entropy}_t,
A_t
\right).
\tag{35}
$$

头级放大矩阵也可以动态化：

$$
\mathcal M_t^{(l,h)}
=
g_{l,h}
\left(
A_t,
\text{视觉相似度},
\text{token 不确定性}
\right).
\tag{36}
$$

若希望保持 training-free，可采用滑动统计：

$$
\mu_t
=
(1-\gamma)\mu_{t-1}
+
\gamma A_{T_p,t}.
\tag{37}
$$

然后根据当前上下文中的相对异常，而不是固定 COCO 均值，决定是否干预。

## 12.5 设计长度感知的注意力指标

为了减弱：

$$
A_{T_p}
=
\frac{P_{T_p}}{n}
$$

随长度自然下降的问题，可以考虑：

### 使用总注意力

$$
P_{T_p}^{(l,h)}
=
\sum_{i\in T_p}
A^{(l,h)}(i).
\tag{38}
$$

### 使用最近窗口

$$
A_{\mathrm{recent}}^{(l,h)}
=
\frac{1}{w}
\sum_{i=n-w+1}^{n}
A^{(l,h)}(i).
\tag{39}
$$

### 使用距离衰减

$$
A_{\mathrm{decay}}^{(l,h)}
=
\frac{
\sum_{i=1}^{n}
\gamma^{n-i}A^{(l,h)}(i)
}{
\sum_{i=1}^{n}\gamma^{n-i}
}.
\tag{40}
$$

这样可以避免长文本因 token 数量增加而频繁误触发。

## 12.6 在历史文本写入前加入视觉核验

为避免错误自我强化，可以对新生成实体进行验证：

$$
s(o)
=
\operatorname{sim}
\left(
e_{\text{text}}(o),
e_{\text{vision}}(I)
\right).
\tag{41}
$$

若：

$$
s(o)<\delta,
$$

则将该实体标记为低可信，不允许其在后续被增强。

完整流程可以改为：

$$
\text{生成 token}
\rightarrow
\text{视觉核验}
\rightarrow
\text{写入可信记忆}
\rightarrow
\text{后续注意力增强}.
$$

这可以从根本上缓解：

$$
\text{早期幻觉}
\rightarrow
\text{错误记忆强化}
$$

的问题。

## 12.7 建立幻觉类型相关的注意力控制器

论文的主要统计集中在物体幻觉，但视觉幻觉还包括：

- 属性幻觉；
- 关系幻觉；
- 数量幻觉；
- 位置幻觉；
- 动作幻觉。

未来可以构建：

$$
\mathcal M_{\text{object}},
\quad
\mathcal M_{\text{attribute}},
\quad
\mathcal M_{\text{relation}},
\quad
\mathcal M_{\text{count}}.
\tag{42}
$$

不同类型使用不同的注意力区域：

- 对象存在性更多依赖局部图像区域；
- 颜色属性依赖对应对象视觉特征；
- 空间关系依赖多个实体及其位置；
- 数量判断依赖全局图像信息；
- 语言连贯性则更多依赖 $T_p$。

这会使干预从统一幻觉校正发展为任务和语义类型感知的动态控制。

## 12.8 改进层级检测与头级执行之间的粒度不一致

目前方法使用：

$$
\bar A_{T_p}^{(l)}
$$

进行层级判断，但使用：

$$
M^{(l,h)}
$$

进行头级执行。

可能出现少数关键头已经异常，但由于其他头注意力较高，层级均值仍未触发的情况。

可设计头级异常开关：

$$
g_t^{(l,h)}
=
\mathbf 1
\left[
A_{T_p,t}^{(l,h)}
<
\mathcal T^{(l,h)}
\right].
\tag{43}
$$

更新变为：

$$
\hat A_t^{(l,h)}(i)
=
\left[
1+
g_t^{(l,h)}
\alpha M^{(l,h)}
\mathbf 1(i\in T_p)
\right]
A_t^{(l,h)}(i).
\tag{44}
$$

这样可以做到真正的：

$$
\text{头级检测}
+
\text{头级干预}.
$$

---

# 十三、总结：从图像注意力增强走向可信视觉记忆控制

这篇论文的主要贡献可以用一条逻辑链概括：

$$
\text{图像注意力增强会造成重复}
$$

$$
\Downarrow
$$

$$
\text{真实物体生成时更关注历史文本}
$$

$$
\Downarrow
$$

$$
T_p
\text{ 可视为语言化的视觉记忆}
$$

$$
\Downarrow
$$

$$
\text{IAT 增强历史文本注意力}
$$

$$
\Downarrow
$$

$$
\text{AdaIAT 自适应决定何时增强、每个头增强多少}
$$

其完整在线流程为：

$$
QK^\top
\rightarrow
A=\operatorname{softmax}(QK^\top)
$$

$$
\rightarrow
\bar A_{T_p,t}^{(l)}
\overset{?}{<}
\mathcal T^{(l)}
$$

若不满足，则保持原始注意力；若满足，则：

$$
A'(i)
\propto
\begin{cases}
\left(
1+\alpha M^{(l,h)}
\right)A(i),
&
i\in T_p,\\[4pt]
A(i),
&
i\notin T_p.
\end{cases}
$$

最后：

$$
O=A'V_{\text{value}}.
$$

这篇论文最重要的创新并不是单纯提出一种新的 attention scaling 技巧，而是提出了一个新的研究视角：

$$
\boxed{
\text{LVLM 的已生成文本可以成为视觉事实的中间记忆载体}
}
$$

不过，AdaIAT 仍然把全部历史文本视为大致可靠，并依赖静态离线统计。未来真正值得深入的方向，是将方法从：

$$
\text{增强所有历史文本}
$$

进一步发展为：

$$
\boxed{
\text{动态增强经过视觉验证、与当前预测相关且高可信的历史信息}
}
$$

这样才能进一步解决"什么时候增强、增强什么，以及如何避免强化已有错误"这三个更本质的问题。
