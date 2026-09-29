# DPO Lab

一个从零实现的小型自回归 DPO 训练步示例。项目使用随机初始化的小词表因果 Transformer，数据全部在测试中自造，不依赖预训练模型，也不调用现成 DPO trainer。

## 内容

- `src/dpo_lab/model.py`：单层小型因果 Transformer，显式处理 causal mask 和 key padding。
- `src/dpo_lab/sequences.py`：chosen/rejected 批处理、prompt/completion mask、右 padding、EOS 追加。
- `src/dpo_lab/scoring.py`：自回归序列对数概率求和与稳定 `logsigmoid` DPO 损失。
- `src/dpo_lab/cache.py`：绑定输入、mask 和 reference 参数指纹的 reference 分数缓存。
- `src/dpo_lab/training.py`：冻结 reference、计算梯度并可执行 optimizer step。
- `src/dpo_lab/counterexamples.py`：仅用于测试的 prompt mask 泄漏和 token 平均目标反例。
- `docs/math.md`：KL 正则、DPO 成对损失、梯度方向和长度效应说明。

## EOS 与 mask 规则

每个 completion 都显式追加 EOS，chosen/rejected 遵循同一规则。空 completion 有且仅有一个被计分的 EOS。Prompt token 可作为上下文，但 prompt 标签不进入偏好损失；右 padding 同时被 attention mask 和 completion mask 排除。

正确序列分数是 completion token log-probability 的和，不做 token 平均。token 平均只作为反例，用来展示它会改变长度不均样本的梯度。

## 运行测试

```bash
pytest -q
```

测试包含真实参数更新、逐样本/批量梯度一致性、双精度有限差分、reference 冻结与缓存失效、EOS/空 completion/同长度边界，以及两个错误目标反例。
