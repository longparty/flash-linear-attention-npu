# Recurrent GDN 扩展接口

本次需求来源：用户提供的 `docs/new gdn recurrent/design_full.md`、流水图和已修改的 CPU 标杆。目标实测设备为 Ascend950PR，CANN 9.2.0-beta.2。既有 BF16 Q/K/V/beta/out、BF16/FP32 State、GQA、packed varlen、逐 token 原地 State 写回和 FP32 recurrence 保持。

```python
recurrent_gated_delta_rule(
    query, key, value, state, *, beta, scale=1.0,
    actual_seq_lengths, ssm_state_indices, num_accepted_tokens=None,
    g=None, gk=None, use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False, A_log=None, dt_bias=None,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
) -> out
```

| 参数 | shape / dtype | 契约 |
| --- | --- | --- |
| query/key | `[T,Nk,128]` BF16 | Nk>0；正 stride 的非连续视图可直接读取 |
| value | `[T,Nv,128]` BF16 | Nv>0、Nv%Nk=0；不限制 head 数为 256 |
| beta | `[T,Nv]` BF16 | 开关关闭时直接参与计算 |
| state | `[BlockNum,Nv,128,128]` BF16/FP32 | 原地读写；最内轴连续，允许行、head、block padding、transpose block/head 和 storage offset；禁止重叠写视图 |
| actual_seq_lengths | `[B+1]` INT32 | `[prefix,L0,...]`，非负、总和为 T；允许零长度请求，不限制 L<=8 |
| ssm_state_indices | `[T]` 或 `[B,W]` INT32 | 一维按 packed token；二维按请求行，W>=L；索引必须在 State pool 内；有效写槽不能跨请求重叠 |
| num_accepted_tokens | 可选 `[B]` INT32 | 非空请求从第 accepted-1 个 checkpoint 读取初始 State；默认 1；二维可选择 L 以外、W 以内的 checkpoint |
| g | 可选 `[T,Nv]` FP32 | 无 g 时衰减因子为 1；raw gate 模式必需 |
| gk | 保留参数 | 目标设计仅支持 scalar gate，非 None 时明确报错 |
| A_log / dt_bias | 可选 `[Nv]` FP32 | raw gate 模式 A_log 必需，dt_bias 可空；开关关闭时不参与计算 |
| scale | 有限 float | 缩放 query，默认 1.0，与已有入口一致 |
| 三个 use_* 属性 | bool，默认 False | 彼此独立 |
| allow_neg_eigval | bool，默认 False | 仅在 beta sigmoid 开启时使用 `2*sigmoid(beta)`；关闭 sigmoid 时忽略 |
| out | `[T,Nv,128]` BF16 | 返回输出；无效前缀不参与 recurrence，测试有效区域排除前缀 |

## 数学与 State 语义

所有中间量 FP32。`hk=hv//(Nv/Nk)`；norm 使用 `x*rsqrt(sum(x*x)+1e-6)`，归一化后不回落 BF16。raw gate 使用 `z=g+dt_bias`、`log_gate=-exp(A_log)*softplus(z)`，softplus 按 `max(z,0)+log1p(exp(-abs(z)))` 稳定求值。gate 与 beta 的 head 参数按 hv 广播。

```text
Sbar = exp(log_gate) * Sprev       # 无 g 时系数为 1
delta = beta_eff * (v - Sbar @ k)
Snew = Sbar + outer(delta,k)
out = Snew @ (scale*q)
state[write_slot,hv] = cast_state_dtype(Snew)
```

下一个 token 使用核内 FP32 Snew，不回读已量化的 BF16 State。不同请求不能读写彼此的 State；同一请求的初始 checkpoint 可同时是本请求写槽。二维表未使用的 checkpoint 不写回。空请求不读取其 accepted 值。

## 材料冲突的采用解释

- 设计开头保留了“仅一维”旧说明，但目标入参表和用户标杆均支持二维；采用一维和二维。
- 设计明确删除 L<=8 和 head<=256；采用 token tile 分段和单 head gate/beta 搬运。
- 设计目标明确拒绝 gk；旧 column gate 数学只保留在修改前快照，不作为新正向用例。
- 设计允许无 g；CPU 标杆的旧“至少一个 gate”检查按目标契约更新。
- 设计批注的 beta 结果乘 2 采用 `allow_neg_eigval`，仅影响 sigmoid 分支。

## ABI 与异常

扩展 ACLNN 入口使用 V2 名称，参数包含 A_log、dt_bias 及四个 bool；保留原 ACLNN 符号，使用默认扩展属性转发。Python Stable-ABI schema、ctypes 及 L0 参数顺序保持一致。结构、dtype、有限 scale、支持的 stride 和可选参数依赖在 host 检查；设备 metadata 必须满足上述合法域。不得用测试输入预处理代替核内新功能。


## SoC 与直接 C 调用

上述新增能力以 Ascend950 为范围。A2/A3 的旧默认路径要求四个新 bool 均为 false、indices 一维、
Nk/Nv<=256、L<=8、State行 stride=128；扩展请求由公开 Python/host 拦截，本轮未做A2/A3硬件验收。
Python两个后端同时校验设备 metadata 值与跨请求竞争；直接 C 调用方必须保证 metadata 满足合法域。
C++层校验 descriptor、dtype、shape、stride、storage bounds 与属性依赖，结构非法返回 ACLNN_ERR_PARAM_INVALID。
