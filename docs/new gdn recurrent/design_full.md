# Recurrent Gated Delta Rule 算子方案设计

方案设计规则版本：`V2`

文档状态：完整目标设计。本文从算子整体出发定义 Recurrent Gated Delta Rule 的接口语义、数学模型、任务切分、单 Stage AIV 实现、非连续视图访问、UB/GM 布局、同步、性能与验收要求。文中明确标为“当前实现”的内容用于说明落地基线；目标能力只有在 API、host、kernel 和测试全部完成并通过验收后才可发布。

## 1. 目标

### 1.1 算子职责

Recurrent Gated Delta Rule（以下简称 Recurrent GDN）对 packed token 序列逐 token 更新矩阵状态，并生成每个 token 的输出。算子同时承担以下完整职责：

- 按 `actual_seq_lengths` 解析无效前缀与各请求的有效区间；
- 按 `ssm_state_indices` 从 State pool 选取初始状态，并把每个 token 更新后的状态原地写回 State pool；
- 完成 GQA 的 query/key head 到 value head 映射；
- 可选在核内完成 Q/K L2Norm；
- 可选把 raw scalar gate 与 `A_log`、`dt_bias` 转换成 step log gate；
- 可选在核内对 beta 做 sigmoid；
- 仅支持 scalar gate `g` ， 不支持column gate `gk`；
- 在 FP32 中执行 recurrent 状态计算，按公开 dtype 写回 `out` 和 `state`；
- 对符合直接访问约束的非连续 Q/K/V/State 视图按 stride 访问，不在主路径物化连续副本。

完整调用链为：

```text
Python / ACLNN
  -> 参数、shape、dtype、stride 与 storage bounds 校验
  -> tiling、TilingKey、blockDim 和 workspace 生成
  -> RecurrentGatedDeltaRule（一个 AIV-only kernel）
       -> 读取序列元数据和初始 State
       -> Q/K L2Norm、gate 和 beta 预处理（按开关执行）
       -> 逐 token 衰减、预测、delta 更新和输出
       -> 逐 token State 原地写回与 out 写回
```

算子只有一个计算 Stage，全部运行在 AIV；不启用 AIC，不使用 L1/L0/Cube，也不把预处理拆为独立 kernel。算法中间量不落 GM workspace，`algorithm_workspace_bytes = 0`。ACLNN executor 可按运行时要求申请 system workspace，但该空间不承载本算法中间结果。

### 1.2 设计依据

接口与实现证据如下：

- 当前公开接口、shape、dtype、SoC 与边界：[`aclnnRecurrentGatedDeltaRule.md`](./aclnnRecurrentGatedDeltaRule.md)；该目录当前没有独立 `docs/api.md`，补齐前由此文件承担已发布接口的事实来源；
- 独立 CPU 数学标杆：[`tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py`](../../../../../../../tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py)；
- ATK 输入生成、调用与比较：[`tests/atk/recurrent_gated_delta_rule/`](../../../../../../../tests/atk/recurrent_gated_delta_rule/)；
- host/tiling：[`op_host/`](../op_host/)；
- AIV kernel：[`op_kernel/`](../op_kernel/)；

CPU golden 是公式和开关组合的独立正确性基准。NPU kernel 不得以自身实现反向定义 golden。当前 golden 还包含二维 `ssm_state_indices[B,W]` 的 CPU-only 提案分支；本设计的公开 NPU 接口只覆盖一维 `ssm_state_indices[T]`，二维提案不进入实现范围。

### 1.3 完整接口语义

下表描述算子完整目标契约。正式发布时，以同步更新后的 API 文档和 L0 原型为最终 ABI；所有入口的顺序、可选性和默认值必须一致。

| 参数 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `query` | `[T, Nk, K]` | BF16 | packed query；允许尾轴非连续 |
| `key` | `[T, Nk, K]` | BF16 | packed key；允许尾轴非连续 |
| `value` | `[T, Nv, V]` | BF16 | packed value；允许尾轴非连续 |
| `state` | `[BlockNum, Nv, V, K]` | BF16/FP32 | State pool；读写参数，返回时包含所有 token 的原地写回； |
| `beta` | `[T, Nv]` | BF16 | delta 强度；可选择在核内解释为 sigmoid 前的 raw 值 |
| `actual_seq_lengths` | `[B+1]` | INT32 | `[P,L_0,...,L_{B-1}]`；`P` 是无效前缀，且 `P+sum(L_b)=T` |
| `ssm_state_indices` | `[T]`/`[B,W]` | INT32 | token/checkpoint 到 State pool 槽号的映射,可传一维或者二维 |
| `g`,可选 | `[T,Nv]` | FP32 | scalar gate；可为 step log gate 或 raw gate |
| `gk`，不可选 | `[T,Nv,K]` | FP32 | 如果传非None则直接报错 |
| `A_log`，可选 | `[Nv]` | FP32 | raw scalar gate 的 head-wise 参数 |
| `dt_bias`，可选 | `[Nv]` | FP32 | raw scalar gate 的 head-wise bias |
| `num_accepted_tokens`，可选 | `[B]` | INT32 | 每个请求选择初始 State 槽的位置；缺省为 1 |
| `scale`，可选 | scalar attr | FP32 | query 输出缩放，必须为有限值 |
| `use_qk_l2norm_in_kernel`，可选 | bool attr | bool | 是否在 FP32 中沿 K 归一化 Q/K；默认 `false` |
| `use_gate_in_kernel`，可选 | bool attr | bool | 是否把 `g` 解释为 raw scalar gate；默认 `false` |
| `use_beta_sigmoid_in_kernel`，可选 | bool attr | bool | 是否使用 `sigmoid(beta)`；默认 `false` |
| `out` | `[T,Nv,V]` | BF16 | packed token 输出 |

约束与组合规则：

1. `Nk > 0`、`Nv > 0`、`Nv % Nk == 0`，GVA group 数 `G=Nv/Nk`。
2. 目标规格保持 `K=V=128`，`Nk,Nv <= 256`；放宽维度前必须重新完成 UB、指令 repeat 和精度论证。（Nk,Nv <= 256校验删除）
3. `g` 可以为空，`gk`必须为空
4. `use_gate_in_kernel=true` 时 `g`、`A_log` 必须非空，`dt_bias` 可为空。
5. `use_gate_in_kernel=false` 时 `g` 已经是 step log gate，`A_log`、`dt_bias` 不参与计算，也不应搬入 UB。
6. `use_beta_sigmoid_in_kernel=false` 时 beta 直接参与公式；开启时先做稳定 sigmoid。（参考org算子，增加一个bool值，使得结果*2）
7. 三个 bool 开关彼此独立，所有合法组合都必须与 CPU golden 一致。
8. `A_log`、`dt_bias` 按 value head `hv` 广播到 token 轴。
10. 取消 `L_b<=8`；限制，目标 kernel 以 token tile 可选 `4/8/16`，推荐为4。（删掉）

## 2. Stage 0 完整详设

### 2.1 全局符号和任务域

符号表：

| 符号 | 含义 |
| --- | --- |
| `T` | packed token 总数 |
| `B` | 有效请求数，即 `actual_seq_lengths.numel()-1` |
| `Nk/Nv` | query-key/value head 数 |
| `K/V` | key/value 维度，目标规格均为 128 |
| `G` | `Nv/Nk`，每个 query-key head 对应的 value head 数 |
| `X` | A5 token tile，主规格取 4 |
| `hv` | value head id |
| `hk` | `hv/G` |
| `S` | 单个 value head 的 `[V,K]` recurrent State |

### 2.2 Stage 结果

唯一 Stage 定义如下：

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| `Stage 0：Vector，逐 token 完成 Recurrent GDN` | AIV | 元数据解析、预处理、State recurrence、out 与 State 写回 | A5 UB 峰值 208 KiB；L1/L0 0 byte；算法 workspace 0 byte |

Stage 0 必须完成从元数据解析到最终 GM 写回的全部逻辑。Host 负责结构性校验和 tiling，不执行数值预处理；Q/K L2Norm、raw gate 与 beta sigmoid 均属于 Stage 0。Stage 输入是全部公开输入、属性和 tiling data，输出是 `out` 与原地更新后的 `state`，无 GM 中间结果、无核间同步。

### 2.3 Stage 0：Vector，逐 token Recurrent GDN 公式

对序列 `b`、value head `hv` 和 token `t`，令 `hk=floor(hv/G)`。所有中间计算使用 FP32。

#### 2.3.1 Q/K 预处理

当 `use_qk_l2norm_in_kernel=false`：

```text
q_hat = FP32(query[t,hk,:])
k_hat = FP32(key[t,hk,:])
```

当开关为 true，沿最后一维 K 归一化：

```text
q_hat = q / sqrt(sum_j(q_j^2) + 1e-6)
k_hat = k / sqrt(sum_j(k_j^2) + 1e-6)
```

平方、reduce、加 epsilon、`rsqrt` 和乘法均在 FP32 完成；归一化后不回落到 BF16 再参与 recurrence。

#### 2.3.2 Scalar gate 预处理

若 `g` 缺省，则 scalar 衰减因子视为 1。

若 `g` 存在且 `use_gate_in_kernel=false`：

```text
g_eff = FP32(g[t,hv])
alpha = exp(g_eff)
```

若 `use_gate_in_kernel=true`：

```text
z     = FP32(g[t,hv]) + (FP32(dt_bias[hv]) if present else 0)
g_eff = -exp(FP32(A_log[hv])) * softplus(z)
alpha = exp(g_eff)
```

`A_log` 必须按 `hv` 取值，因为 raw scalar gate `g` 的 shape 是 `[T,Nv]`，每个 value head 维护独立 State；它不能按 `hk` 广播，也不能用于 shape 为 `[T,Nv,K]`、已经处于 log-space 的 `gk`。

#### 2.3.3 Beta 与 column gate

```text
beta_eff = sigmoid(FP32(beta[t,hv]))    if use_beta_sigmoid_in_kernel
           FP32(beta[t,hv])             otherwise

alpha_k[j] = exp(FP32(gk[t,hv,j]))      if gk is present
             1                          otherwise
```

#### 2.3.4 Recurrent update

令 `S_prev` 为 `[V,K]`：

```text
S_scalar[i,j] = alpha * S_prev[i,j]
S_bar[i,j]    = S_scalar[i,j] * alpha_k[j]
prediction[i] = sum_j S_bar[i,j] * k_hat[j]
delta[i]      = beta_eff * (FP32(value[t,hv,i]) - prediction[i])
S_new[i,j]    = S_bar[i,j] + delta[i] * k_hat[j]
out[t,hv,i]   = sum_j S_new[i,j] * (scale * q_hat[j])
```

scalar gate 先作用于整个 State，column gate 再按 K 列作用。实现不得改变二者的含义，不得把 `gk` 错误地按 V 行缩放。完成 token 后：

```text
state[ssm_state_indices[t], hv, :, :] = cast_state_dtype(S_new)
```

核内用于下一个 token 的 `S_new` 保持 FP32，即使公开 State dtype 是 BF16，也不能从刚写回 GM 的 BF16 State 重新加载继续计算。



### 2.7 A5 UB 空间布局

A5 单 AIV UB 为 248 KiB。`K=V=128` 时一个 FP32 State bank 大小为：

```text
128 * 128 * 4 bytes = 65536 bytes = 64 KiB
```

主方案取 `X=4`，用 bank ring 覆盖 State 计算与写回，采用绝对地址规划。所有 offset 均以单核 UB 起点为 0。

#### 2.7.1 FP32 State

| UB 区间 | 大小 | 对象 | 生命周期 |
| --- | ---: | --- | --- |
| `[0,64 KiB)` | 64 KiB | `stateBank0` FP32 | 当前/下一 token State |
| `[64,128 KiB)` | 64 KiB | `stateBank1` FP32 | 当前/下一 token State |
| `[128,192 KiB)` | 64 KiB | `stateBank2` FP32 | 预取或 MTE3 写回保持 |
| `[192,194 KiB)` | 2 KiB | `qIn[2][X][128]` BF16 | 双缓冲 tile |
| `[194,196 KiB)` | 2 KiB | `kIn[2][X][128]` BF16 | 双缓冲 tile |
| `[196,198 KiB)` | 2 KiB | `vIn[2][X][128]` BF16 | 双缓冲 tile |
| `[202,202.5 KiB)` | 0.5 KiB | `gIn[2][X]` FP32 reserve | scalar gate |
| `[202.5,203 KiB)` | 0.5 KiB | `betaIn[2][X]` | beta 与对齐 reserve |
| `[203,203.5 KiB)` | 0.5 KiB | `A_log`/gate scale slots | per-head scalar reserve |
| `[203.5,204 KiB)` | 0.5 KiB | `dt_bias` slots | per-head scalar reserve |
| `[204,206 KiB)` | 2 KiB | `out[2][X][128]` BF16 | 双缓冲 MTE3 |
| `[206,208 KiB)` | 2 KiB | FP32 vector/reduce scratch | norm、gate、prediction（A5不在UB保存中间结果） |
| `[208,248 KiB)` | 40 KiB | contiguous reserve | event退化、对齐、扩展余量 |

峰值 208 KiB，保留 40 KiB 连续余量。State 三 bank 的角色按 token 轮转：`cur` 供计算读取，`next` 接收更新，`store` 在 MTE3 完成前保持不可复用。只有收到对应 `MTE3_DONE` 后，`store` 才能回到 free 集合。

### 2.8 Stage 0 流程和流水

1. MTE2 一次搬入本 tile Q/K/V/beta/g/state/bias/A_log，只搬运当前HEAD。
2. 等待本tile INPUT_READY，搬运下一个tile
3. 在本tile内进行循环操作：
   1. state decay: S_decay = S_t-1 * a
   2. prediction z = S_decay @ k
   3. WEdelta: delta = Beta @ (z - v)
   4. State update: S_t = S_decay + delta @ k^T
   5. out = State @ (scale*q)
   6. MTE3搬出state和out
4. 整个tile计算完成后，继续MTE2搬运下一个tile

单个 work item 的逻辑流程：

```text
1. 读取 L_b；若为 0，立即结束该 work item
2. 计算 start/end、accepted 位置、initial_slot
3. 从 State view 搬入 initial State；BF16 路径完成 FP32 cast
4. 读取当前 hv 的 A_log/dt_bias（仅 raw gate 路径需要）
5. for tileStart in range(start, end, X):
     5.1 MTE2 一次搬入本 tile Q/K/V/beta/g/state/bias/A_log
     5.2 同时预取下一 tile；等待本 tile INPUT_READY
     5.3 发起一次顶层 VF 调用；VF 内部 for t in tile:
           a. Q/K cast；按开关做 FP32 L2Norm
           b. 按开关计算 beta_eff、alpha、alpha_k
           c. State scalar/column decay
           d. prediction = State @ k
           e. delta 和 outer update 得到下一 FP32 Stateju
           f. out = State @ (scale*q)
           g. 取得 write_slot，异步写 State；异步写 out
           h. 轮转 State/out/input bank
6. 等待该 work item 所有 MTE3 完成，释放 bank
```

同一 State 的 token recurrence 不能跨 token 并行；流水的重叠来自：

- 当前 token vector 计算与下一 tile Q/K/V/gate MTE2；
- 当前 token 计算与上一 token State/out MTE3；
- 不同 AIV 核处理不同 sequence-head。

R12 的一次 Vector 逻辑调用单位定义为 `(work item, token tile)`：该 AIV 先完整搬入自己负责的本 tile 输入，再以一次顶层 VF 调用完成 tile 内全部有效 token 的顺序 recurrence；不能把 L2Norm、gate 或 beta 预处理拆成额外 Stage/VF 调用。State store 地址可能不连续，因此逐 token按 stride 发起。Q/K/V tile 的 direct-view 搬运依据各自 token stride 设置 block count 和 src gap。尾 tile 使用 `tileLen=min(X,end-tileStart)`，无效 slot 不发起计算、通知或写回，禁止读越过 sequence end。

### 2.9 数据搬运汇总

设单个 work 的有效长度为 `L`，State 元素字节数 `sState=2(BF16)/4(FP32)`。下表的次数与字节均以单 work 计；“1次”表示逻辑上只从 GM 读取一次，物理 DMA 可按 `X` 分成 `ceil(L/X)` 个 tile。

| 数据 | GM 访问次数与字节 | UB 驻留/复用 | 批量搬运与 L2 策略 |
| --- | --- | --- | --- |
| Q | 读1次，`L*K*2` B | input ping/pong；BF16→FP32；可做 norm | 以 `X` token、独立 q stride 搬运；同 `hk` 的 G 个 work 可命中 L2，不跨核建 UB 共享 |
| K | 读1次，`L*K*2` B | input ping/pong；BF16→FP32；可做 norm | 同 Q，使用独立 k stride |
| V | 读1次，`L*V*2` B | input ping/pong；BF16→FP32 | 以 `X` token、v stride 搬运；流式读取 |
| beta | 读1次，`L*2` B | input ping/pong；可选 sigmoid | 按当前 `hv` 收集 tile；默认 L2 cache |
| g | 可选读1次，`L*4` B | input ping/pong；raw gate 或直接 exp | 按当前 `hv` 收集 tile；默认 L2 cache |
| gk | 可选读1次，`L*K*4` B | input ping/pong；exp 后按 State 列广播 | 以 `X` token批量搬运；流式读取 |
| A_log | raw gate 时读1次，4 B | 当前 work 全程驻留标量 slot | 连续 hv work 依赖 L2；不物化广播 |
| dt_bias | raw gate 且存在时读1次，4 B | 当前 work 全程驻留标量 slot | 同 A_log |
| initial State | 读1次，`V*K*sState` B | FP32 State bank；跨全部 token 驻留 | 逐 V 行按 State stride 搬运；大块流式访问 |
| updated State | 写 `L` 次，`L*V*K*sState` B | FP32 compute bank 与 store/staging bank 轮转 | 每 token逐 V 行按目标 slot/stride 写回；不从 GM 回读 |
| out | 写 `L` 次，`L*V*2` B | out ping/pong | 每 token连续 V 写回；流式写 |
| actual lengths | 读序列边界所需 INT32 | scalar/寄存器 | 元数据默认 cache；host 可预生成前缀，禁止每 token重扫 B |
| state indices | 初始槽1项、写槽 L 项，各4 B | scalar/寄存器 | 按 token顺序读取；合法性在 host 预检 |
| accepted tokens | 可选读1项，4 B | scalar | 仅非空 work读取 |

当前设计不强制非默认 L2 cache hint：Q/K/A_log/dt_bias 期望从相邻 `hv` work 的共享读中获益，V/g/gk/State/out 主要为流式访问。最终是否使用 cache bypass 或显式 hint 只能依据 profiler 的 cache hit、MTE stall 和总时延决定，并记录到 validation。任何预取都必须在合法 sequence/token 和 State slot 校验之后发生。











### 遗留问题

1.state的非连续视图

2.QKV的非连续视图
