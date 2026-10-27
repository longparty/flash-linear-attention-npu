# Recurrent GDN Ascend 950PR&950DT系列产品设计方案

方案设计规则版本：V2（当前仓库 docs/agents/03；初期 CANNBot V1 校验保留为辅助记录）

## 1. 算子定位与性能模型

实现 api.md 的 scalar gated recurrent GDN 扩展。主模型 case：当前仓库正式性能集的最小 BF16 case；补充 B、head、L 的诊断矩阵，不将其冒充用户模型。性能目标：新增能力正确且默认路径无明显回退；使用同设备、输入、预热与采样次数的 msprof 主 kernel Task Duration(us)，并保留完整 ATK 设备时间。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
| --- | --- | --- |
| API 输入 | Q/K/V/beta；State；g/A_log/dt_bias；metadata | BF16；BF16/FP32；FP32；INT32 |
| API 输出 | out；原地 State | BF16；与 State 输入一致 |
| Vector 计算 | norm、gate、beta、State、dot、delta、out 累加 | FP32 |
| Cube 输入与累加 | 无 Cube 操作 | 不适用 |
| 跨 Stage 交接 | 单 Stage | 不适用 |

## 2. Stage 0–N 完整详设

### 2.1 变量与任务映射

| 符号 | 含义 |
| --- | --- |
| T/B | packed token 数 / 请求数 |
| Nk/Nv/G | key head / value head / Nv/Nk |
| K/V | 128 / 128 |
| X | 输入 tile 4 个 token，尾 tile 只搬有效 token |
| work | 一个 request、一个 value head |
| Nwork | B*Nv |
| blockDim | min(可用 AIV 数, Nwork) |

```text
for work_id = block_idx; work_id < B*Nv; work_id += blockDim:
  b = work_id // Nv; hv = work_id % Nv; hk = hv // G
  start = prefix + sum(lengths[0:b]); L = lengths[b]
  if L == 0: continue
  initial = indices[start+accepted[b]-1]      # [T]
         or indices[b,accepted[b]-1]         # [B,W]
  load initial State once
  for tile in [start,start+L), step X:
    prefetch Q/K/V/beta/g for this head only
    for valid token in tile, sequentially:
      compute in VF; store a distinct State snapshot and out
```

所有元素 offset 使用 64 位；Q/K/V 按独立三维 stride，State 按 block/head/row stride。最内轴连续 Q/K/V 使用批量 MTE2；尾轴非连续使用按 stride 的直接视图收集，避免 GM 连续副本。State 最内轴必须连续，行 padding 可由二维 MTE stride 表达。不同 work 的 State 写集合不相交。

### 2.2 Stage 总览

| Stage | 执行单元 | 主要动作 | 结束条件 |
| --- | --- | --- | --- |
| S0 | AIV，RegBase/VF | metadata、QK norm、gate/beta、逐 token recurrence 和写回 | 本 work 的 MTE3 全部完成 |

不启用 AIC。FP32 State 在 UB 跨 token 保留。每 token 的 VF 在寄存器中完成预处理与行级 recurrence；输出快照经外层 MTE3 写回。输入按 X=4 分块，两份 staging bank 交替；首次使用 bank 时已空闲，tile>=8 回绕时以 V_MTE2 保护旧 VF 消费完成。VF 边界以 token 写回的数据依赖确定：四个完整 FP32 State 快照已需 256 KiB，因此不在一次 VF 结束后同时保留四个快照。此处按固定 CANNBot A5 R12 的实际生命周期规则细化用户草案中的单 tile VF。

### 2.3 Stage 详细设计

#### Stage 0：Vector，融合预处理与递推

```text
q,k = FP32(Q[t,hk]),FP32(K[t,hk])
if norm: q,k *= rsqrt(reduce_sum(square(q,k))+1e-6)
if raw_gate: z=g+dt_bias; g_eff=-exp(A_log)*(max(z,0)+log1p(exp(-abs(z))))
else: g_eff=g
alpha=exp(g_eff) if g exists else 1
beta_eff=sigmoid(beta)*(2 if allow_neg_eigval else 1) if sigmoid else beta
for row in 0..127:
  s_bar = alpha*S[row]
  delta = beta_eff*(V[row]-reduce_sum(s_bar*k))
  s_new = s_bar + delta*k
  out[row] = reduce_sum(s_new*(scale*q))
  S[row]=s_new; snapshot[row]=cast_state_dtype(s_new)
```

norm、stable softplus 和 sigmoid 使用 FP32。softplus 的 x=exp(-abs(z))<1/64 时使用 x-x²/2+x³/3-x⁴/4，避免先求 1+x 抹去小尾部；相对截断误差上界 1.2e-8，其余区域使用 Ln(1+x)。这对 A_log≈-g 的合法相消输入尤其重要。sigmoid 将 beta 截断至 [-80,80] 后计算 `1/(1+exp(-beta))`，避免 exp 溢出；尾部差异小于 2e-35。归约结果在寄存器广播；不为 dot/delta 分别落 UB。BF16 cast 用 RNE。无效 token 不执行 exp、norm、metadata 读取或写回。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| State snapshot ping/pong | UB | UB[64,128) KiB 与 UB[128,192) KiB；单份最大 64 KiB，总 128 KiB | 不涉及 | 两份；VF 写，MTE3 最后消费；wait MTE3 free 后复用，BF16 单份使用低 32 KiB |
| out ping/pong | UB | UB[204,205) KiB 与 UB[205,206) KiB；单份 1 KiB | 不涉及 | BF16 每 token 256 B；MTE3 完成后复用 |
| out | GM（API 输出） | out_base + 2*(t*Nv+hv)*128 B，本次 256 B | 使能 | 每有效 token 一次写，默认缓存 |
| State | GM（API 输出） | state_base + sizeof(State)*(slot*s0+hv*s1) B；本次 32 KiB/BF16 或 64 KiB/FP32 | 使能 | 每 token 一次二维写；不回读当前 token 的 BF16 快照 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| Q/K/V | GM（API 输入） | tensor_base + 2*(t*stride0+h*stride1+j*stride2) B；每 tile 每输入至多 1024 B 有效数据 | 使能 | 每 work 各读一次；QK head group 依赖 L2 复用 |
| beta/g | GM（API 输入） | tensor_base + sizeof(dtype)*(t*Nv+hv) B；每 tile 4/8/16 B 有效值 | 使能 | 单 head 收集，不按 Nv 分配 UB |
| A_log/dt_bias | GM（API 输入） | tensor_base + 4*hv B；单值 4 B | 使能 | 仅 raw gate 路径读取，当前 work 复用 |
| initial State | GM（API 输入） | state_base + sizeof(State)*(initial*s0+hv*s1) B；本次 32 KiB/BF16 或 64 KiB/FP32 | 使能 | 每 work 仅一次读，随后保持 FP32 |
| metadata | GM（API 输入） | lengths_base + 4*b B；indices_base + 4*(t 或 b*W+i) B；accepted_base + 4*b B；每项 4 B | 使能 | 仅合法请求/槽读取；空请求跳过 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| recurrent State | UB | UB[0,64) KiB；单缓冲 64 KiB | 不涉及 | FP32 [128,128]，从 initial cast 后跨全部 token 驻留 |
| input ping/pong | UB | UB[192,195) KiB 与 UB[195,198) KiB；各 3 KiB | 不涉及 | 每份 Q/K/V 各 1 KiB，tile X=4；MTE2 ready 后 VF 读，最后 VF 后 free |
| scalar tiles/head parameters | UB | UB[198,202) KiB；4 KiB | 不涉及 | beta/g 对齐块、A_log/dt_bias；两个 input bank 隔离 |
| scratch | UB | UB[202,204) KiB 与 UB[206,208) KiB；共 4 KiB | 不涉及 | norm、gather 对齐、State 初始转换；使用前验证最大连续范围 |
| 保留 | UB | UB[208,248) KiB；40 KiB | 不涉及 | 编译器和退化余量；峰值 208 KiB |
| L1 | L1 | 不适用 | 不涉及 | AIV-only，无 Cube 数据 |
| L0A | L0A | 不适用 | 不涉及 | 无 AIC |
| L0B | L0B | 不适用 | 不涉及 | 无 AIC |
| L0C | L0C | 不适用 | 不涉及 | 无 Cube 累加 |

每个 VF 同时保留 Q/K 各两个 FP32 register、State 两个、dot/update/output 与 scalar 广播等固定工作集；行循环不展开整 State，避免把 16384 元素留在寄存器。编译后检查 spill 和 UB 警告。tile 尾只搬/计算真实 token。

### 2.4 用户额外 workspace

`workspace_size=0`：无 GM 中间结果、relay 或跨 work record；不包含 API 输入、最终输出、片上空间和固定系统 workspace。

### 2.5 跨 Stage 同步契约

单 Stage、无跨核同步，使用局部 HardEvent 保护 MTE2/VF/MTE3 的地址生命周期。

| 事件 ID/资源 | 第一次语义 | 复用语义 | 被保护的数据 |
| --- | --- | --- | --- |
| input ready/free | MTE2 set ready，VF wait | 最后 VF 读完后发布 free，下一 tile MTE2 wait | input ping/pong |
| snapshot free | 初始化可写；VF 写后 MTE3 wait ready | 下一次 VF 覆盖前 wait MTE3 free | State/out ping/pong |
| work drain | MTE3 完成 | 下一个 work 才覆盖原地址 | 所有输出 bank |
| MTE3_MTE2 | 非首 work 的上一份输出 bank 被 MTE3 消费完 | 下一 work 的 initial State MTE2 才能覆盖 bank 0 | 同址 State 初始读入与输出快照；MTE3_V 不能替代此跨 pipe 依赖 |

```text
MTE2 load -> set ready -> wait ready -> VF compute -> set V_MTE3
-> wait V_MTE3 -> MTE3 store -> set MTE3_V -> wait MTE3_V -> free/reuse
```

事件下一次 set 必须晚于上一代 wait。退出和 work 切换排空全部 bank。不存在零长度 work 的悬空 wait。

### 2.6 V2 规则复核与实际复用采数

采用 CG=1、每个 AIV 对应一个 request/value head、grid-stride 分发。每份 FP32 State 64 KiB，
两个完整 State FP32 输出快照128 KiB，与其它固定区合计208 KiB。CG=2 需要至少416 KiB，
CG=3/4 需要624/832 KiB，超过248 KiB UB；按当前完整快照方案不可达。AIV-only 不占 AIC，
blockDim 使用可用 AIV 数。最小工程性能 case 代入 B=1,Nv=1,Nwork=1,blockDim=1,wave=1。
R21 的 norm/softplus/sigmoid 均在有效 token 上执行，State 和 dot 的所有计算为FP32。

首个 work 不存在旧 MTE3 消费者，跳过 MTE3_MTE2；后续 work 保留该事件。每 work 的输入
bank 0/1 首次写入不与当前 State 转换区重叠；第3个 tile 起 V_MTE2 保护回绕。输入 staging
的就绪由 MTE2_V 保证，输出 staging 的覆盖由各 bank 的 MTE3_V 保证。
本方案的正确性结论使用地址生命周期与实际复用记录；硬件搬运/计算重叠比例不作为已测结论。

调试目标 rgdn_direct_trace 独立编译 RGDN_ENABLE_TRACE，为每个 core 分配4096条独占GM记录，
每条记录 blockDim/core/work/token/bank/event 六个uint64。该区域不与算子数据重叠，不增加
同步。记录 work begin、initial reclaim、input ready、output free wait、VF ready、store issued、
work drain 和 input recycle wait 的实际发出顺序。普通目标与生产 wheel 不定义该宏。
