# DPO 成对损失推导与实现约定

本文对仓库中的小型自回归 DPO 训练步给出数学定义和实现约定。所有数据和模型均为自造，不下载也不调用预训练模型或现成 DPO trainer。

## 1. 从 KL 正则到成对损失

对 prompt `x`，策略模型 `π_θ(y|x)` 相对固定 reference model `π_ref(y|x)` 的 KL 正则化奖励最大化目标可写为

\[
\max_\pi \mathbb{E}_{x,y\sim\pi}[r(x,y)]
-\beta\,\mathbb{E}_x\!\left[D_{\mathrm{KL}}(\pi_\theta(\cdot|x)\Vert\pi_{\mathrm{ref}}(\cdot|x))\right].
\]

对每个 prompt 做变分优化，最优策略满足

\[
\pi^\star(y|x)\propto \pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta).
\]

整理出只依赖模型对数概率的隐式奖励（忽略对同 prompt 内两个 completion 相消的常数）：

\[
r(x,y)=\beta\left(\log\pi_\theta(y|x)-\log\pi_{\mathrm{ref}}(y|x)\right).
\]

使用 Bradley–Terry 偏好模型：

\[
P(y_c\succ y_r\mid x)
=\sigma\!\left(r(x,y_c)-r(x,y_r)\right),
\]

其中 `y_c` 是 chosen completion，`y_r` 是 rejected completion。代入隐式奖励：

\[
P(y_c\succ y_r\mid x)=\sigma(\beta\Delta_i),
\]

\[
\Delta_i=
\bigl(\log\pi_\theta(y_c|x)-\log\pi_{\mathrm{ref}}(y_c|x)\bigr)
-
\bigl(\log\pi_\theta(y_r|x)-\log\pi_{\mathrm{ref}}(y_r|x)\bigr).
\]

单样本负对数似然就是本实现采用的成对 DPO 损失：

\[
\mathcal L_i=-\log\sigma(\beta\Delta_i).
\]

批量损失是逐样本损失的普通平均值：

\[
\mathcal L=\frac{1}{B}\sum_{i=1}^B \mathcal L_i.
\]

代码中的稳定形式是 PyTorch 的 `F.logsigmoid(βΔ)`；它等价于 `softplus(-βΔ)`，在很大的正/负 margin 下不会显式计算可能上溢的 sigmoid。

## 2. 梯度方向

设

\[
s_c=\log\pi_\theta(y_c|x),\qquad s_r=\log\pi_\theta(y_r|x).
\]

Reference 分数是常数，因此

\[
\nabla_\theta \mathcal L_i
=
-\beta\,\sigma(-\beta\Delta_i)
\left(\nabla_\theta s_c-\nabla_\theta s_r\right).
\]

这给出三个直接结论：

- 对 chosen：梯度项提高 `log πθ(y_c|x)`。
- 对 rejected：梯度项降低 `log πθ(y_r|x)`。
- 当 chosen/rejected 完全相同时 `Δ=0`，两侧模型分数相同，梯度大小相同、方向相反，总和为零。

在 `Δ=0` 时，chosen 标量分数的导数是 `-β/2`，rejected 标量分数的导数是 `+β/2`。所以 β 改变 KL 正则强度，也改变偏好 margin 的尺度和梯度大小；交换 chosen/rejected 会让 `Δ` 变号，并交换上升/下降方向。

## 3. 序列求和而不是 token 平均

正确目标使用 completion 的序列对数概率之和：

\[
\log\pi_\theta(y|x)=\sum_{t=1}^{|y|}
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

这里每个 completion 都显式追加一个 EOS，所以非空 completion 的得分为原 completion tokens 加 EOS；空 completion 也不是零长度，它恰好有一个 EOS 分数。chosen 和 rejected 始终遵循同一规则。

代码里的未移位 `completion_mask` 与 token 坐标对齐：当第 `t` 个输入 token 是 completion token 时，该位置为 1。自回归计分时再做标签移位：

- logits：`logits[:, :-1]`
- targets：`input_ids[:, 1:]`
- target mask：`completion_mask[:, 1:]`

因此，输出位置 `t-1` 预测输入位置 `t` 的 token。Prompt token 可以作为上下文影响后续预测，但 prompt 标签本身不进入偏好损失；右侧 padding 由 attention mask 和 completion mask 同时排除。

不把上面的序列和偷偷除以 completion 长度。token 平均目标是另一个目标：

\[
\frac{1}{|y|}\sum_{t=1}^{|y|}
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

它会给不同 completion 的每个 token 乘上不同的 `1/|y|` 权重；当 chosen/rejected 长度不同时，这不是对序列和损失的统一常数缩放，梯度方向和相对权重都会实质变化。仓库只在 `counterexamples.py` 中提供该函数，并用测试展示其梯度变化。

## 4. 序列求和的长度相关现象

对数概率通常是非正数，序列和会随 token 数累积。因此 DPO 的奖励 margin 比较的是两个完整序列总对数概率相对 reference 的变化，而不是平均每 token 拟合程度。可能出现：

- 长 completion 的分数有更多项，分数尺度和方差随长度变化。
- 模型通过提高多个 token 的概率累积优势，即使单 token 变化较小。
- 长度分布在 chosen/rejected 间不一致时，长度会与偏好信号耦合。
- EOS 被计分能惩罚“不结束”，并让终止决策参与优化，但不能消除非 EOS token 数带来的长度效应。
- 固定 reference 提供相对偏移，能部分抵消模型对分布的公共长度倾向，但不会自动把不同长度序列归一化。

这不是说一次随机小模型更新就能改善通用对齐能力；这里验证的是目标函数、mask、梯度和一次数值更新是否正确。

## 5. Reference 冻结与缓存

`ReferenceScoreCache` 会：

1. 对 reference model 调用 `requires_grad_(False)` 并在 `eval()` 下计分。
2. 使用 `torch.no_grad()`，结果全部 detach。
3. 拒绝共享 policy/reference 参数张量，避免 reference 被训练步更新。
4. 用 SHA-256 指纹绑定：
   - reference `state_dict()` 中所有参数和 buffer；
   - input token；
   - attention mask；
   - completion mask；
   - shape、device、dtype；
   - 调用时模型的 train/eval 状态。

因此 token 或任一 mask 改变、reference 参数被更新后，都不会复用旧分数。

## 6. 自动化验证覆盖

- 自回归标签移位：位置 `t-1` 预测 token `t`。
- prompt masking：prompt 标签不进损失。
- padding masking：右 padding 不进注意力目标和损失。
- EOS：显式追加并计分；空 completion 恰好计分一个 EOS。
- prompt 长度：同一 pair 必须相同；不同 batch 样本可不同。
- chosen/rejected 完全相同：损失有限且训练梯度为零。
- reference：无梯度、参数冻结、共享参数报错。
- 缓存：命中复用；token、mask、reference 权重变化后失效。
- 数值稳定：极大正/负 margin 仍有限。
- 真实更新：小词表随机初始化模型执行一次 optimizer step。
- 批量一致性：逐样本实现与批量实现的损失和全部可训练梯度一致。
- 双精度有限差分：核验 β 的梯度尺度、margin 符号和交换 chosen/rejected 后的方向。
- 反例：单侧 prompt mask 错误会让 prompt 标签获得偏好梯度；token 平均会实质改变长度不均样本的梯度。
