# Recurrent Gated Delta Rule

Recurrent GDN 按 request/value head 执行 FP32 递推，返回 BF16 out，并原地写回逐 token State。
完整的入参、公式、边界和异常约定以 [docs/api.md](docs/api.md) 为准；A5 的资源与同步方案见
[docs/design.md](docs/design.md)，验证矩阵与结果见
[ATK README](../../../../../../tests/atk/recurrent_gated_delta_rule/README.md)。

本次 Ascend950 扩展支持核内 Q/K L2 norm、raw scalar gate、beta sigmoid、allow_neg_eigval、
二维 checkpoint 索引、长序列与超过 256 的 head 数，以及带 padding/offset 的 State 视图。
`g` 可空；`gk` 是保留参数，非空时拒绝。三个融合开关默认关闭，scale 默认 1.0。
BF16 State 写回不会截断核内的 FP32 token 递推。

```python
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

out = recurrent_gated_delta_rule(
    query, key, value, state, beta=beta, scale=128 ** -0.5,
    actual_seq_lengths=lengths, ssm_state_indices=indices,
    num_accepted_tokens=accepted, g=raw_g, A_log=A_log, dt_bias=dt_bias,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True,
)
```

Q/K/V/beta 是 BF16，g/A_log/dt_bias 是 FP32，metadata 是 INT32，所有张量位于同一 NPU。
State 是 `[BlockNum,Nv,128,128]` 的 BF16/FP32，最内轴连续且写视图不重叠。
A2/A3 保留默认属性的旧实现，本次新增能力限 A5；未在 A2/A3 硬件上重新验收。
ACLNN 旧符号保持原 ABI，扩展参数使用 V2，调用说明见
[docs/aclnnRecurrentGatedDeltaRule.md](docs/aclnnRecurrentGatedDeltaRule.md)。
