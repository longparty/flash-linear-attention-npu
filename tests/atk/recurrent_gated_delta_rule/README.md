# Recurrent GDN ATK 工程

接口与数学唯一契约：[api.md](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/api.md)。
实现与资源：[design.md](../../../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/docs/design.md)。
CPU 唯一实现为 `reference.py`，原文件名 `reference_recurrent_gated_delta_rule.py` 仅兼容导入。
本轮按当前仓库 AGENTS.md / docs/agents 五阶段执行，保留最初 PR #833 的工程快照、CANNBot
和 CATLASS 辅助校验记录；这些工具不替代当前仓库的正式 ATK 验收。

## 支持与边界

- Q/K/V/beta/out 为 BF16；State 为 BF16/FP32 `[BlockNum,Nv,128,128]`，原地写回。
- Ascend950 扩展覆盖核内 QK norm、raw scalar gate、beta sigmoid、allow_neg_eigval、二维 checkpoint，
  L>8、head>256、GQA、State 行/head/block padding、transpose 和 offset，以及 QKV 正 stride 视图。
- lengths 为 INT32 `[prefix,L0,...]`，非负且总和 T；零长度请求跳过其 accepted。indices 为 `[T]`
  或 `[B,W]`，二维表允许 accepted>L 且 <=W。跨请求 State 读写不得竞争。
- `g=None` 表示 decay=1；`gk` 保留但非空时拒绝。A_log/dt_bias 为 FP32 `[Nv]`，raw gate 要求 g/A_log。
- 三个融合开关默认 false；allow_neg_eigval 仅在 beta sigmoid 开启时将 sigmoid 结果乘2。
  QK norm 在 FP32 中使用 epsilon=1e-6，不在递推前重新量化 BF16。
- BF16 State 的各 token 写回不截断核内 FP32 recurrence。比较完整 State；out 的无效前缀统一排除。
- 新能力的硬件验收范围为卡0 Ascend950PR；A2/A3 保留旧默认路径，本轮未在 A2/A3 硬件验收。

## 固化精度与来源

用户提供的 CPU 标杆和 `docs/new gdn recurrent/design_full.md` 是语义来源，起始 HEAD 为
`1cd1944bbe6429dd38f71eb201a808bf17cee3f0`。原始源码、用户标杆与旧 wheel 有独立快照。
ATK 使用原生 mixed_tolerance_bm：BF16 rtol/atol=0.0156/0.0156，FP32=0.000977/0.0000153，
matched ratio>=0.99。direct/public 辅助比较还按冻结的 `reference/precision-policy.json` 检查完整输出、
每个 head 和边界区域，max_abs 分别限制 BF16 0.002、FP32 0.0002。

`check_reference.py` 的34项纯 CPU 检查验证独立标量递推、16个开关组合、二维 checkpoint 与 State 不变性。
`check_value_domain.py` 在三个固定种子下进行102组 FP64 校准；开关诊断 case 的 scale 固定为8，
正确结果满足既定阈值，全零 out/State、错误 scale 和遗漏生效融合步骤均失败。性能与旧形状回归输入保留原 scale。
普通随机 QKV 标准差0.05，State标准差0.01，beta默认[0.1,0.9]；融合分支读取 raw beta/g，
边界另覆盖 ±100，以及 A_log=12..25 与 g=-A_log 的相消尾部。阈值未因 NPU 结果而放宽。

CT Tool 固定0.9.1，官方 wheel SHA256 为
`0bdd83003c172f5dac8a010aff889400cc91f2caa6ba49f4c24127980f9b25c8`。
正确/错误值域、历史 BF16 mask、跨 work 同步以及 softplus 尾部故障均保留逐输出的 CT 空间图。
精确 A_log=20/g=-20 复现中，修复前 out/State max_abs 为0.10620117/0.02613516，修复后均为0。

## 正式用例映射

精度 JSON 共834条，来源为实际 ATK gen_cases 生成后逐项核对。每个逻辑 profile 包含固定种子
20261022、20261023、20261024。0–299 对应 RGDR-P001..P100，300–599 对应 RGDR-G001..G100；
每个旧 design_id 对应连续3个种子。旧 gk/both 标签只保留形状和布局，gate_mode 已按新契约改为scalar g。
新增234条的映射如下。

| 逻辑分支 | 精度 case id |
| --- | --- |
| 开关16组合 / bf16 | 600–647 |
| L=1/3/4/8/9/17/33 / bf16 | 648–668 |
| 二维表、accepted>L、prefix、空请求、无 bias | 669–671, 765–767 |
| 无 g | 672–674, 768–770 |
| 全空请求与无效前缀 | 675–677, 771–773 |
| State transpose/head/block/compound/row padding | 678–692, 774–788 |
| QKV/beta/g 尾轴 stride=2 | 693–695, 789–791 |
| 开关16组合 / fp32 | 696–743 |
| L=1/3/4/8/9/17/33 / fp32 | 744–764 |
| A_log 与负 raw gate 相消的小尾部 | 792–794, 810–812 |
| 只读 QKV 重叠视图 | 795–797, 813–815 |
| QKV token/head transpose | 798–800, 816–818 |
| 全部输入非连续与 State row padding | 801–803, 819–821 |
| raw gate/beta ±100 | 804–806, 822–824 |
| 零 Q/K 与 norm epsilon | 807–809, 825–827 |
| 257 heads / GQA 129→258 | 828–833 |

直接调试额外包含强制单核连续8个 work，不把该调试开关暴露为公开属性。
异常场景独立通过真实 C++ host 校验和两种 Python 后端验证：错误 dtype/shape、预留 gk、缺少 raw gate
依赖、非法 bool/scale、metadata 值域、accepted、跨请求竞争、重叠 State 和设备不一致。
Python 异常还检查 State 数值与版本不变，以及同一 metadata 张量原地修改后的缓存失效。

## 性能来源

`atk_recurrent_gated_delta_rule_perf.json` 原样保留仓库原有最小 BF16 工程基线：case 0，T=B=Nk=Nv=1，
K=V=128，scale=0.08838834764831843，seed=20260826，scalar g。用户没有提供新的模型 shape 或逐 case 性能阈值，
本用例用于工程回归。完整 ATK device_perf 包含输入构造等操作，主核时间单独从 msprof op_summary
筛选 OP Type=RecurrentGatedDeltaRule 的20个样本统计，二者不可混用。

## 手工 MSS / 确定性集

12条独立手工用例覆盖全部可达 TilingKey。key10=BF16 State，key30=FP32 State；进一步覆盖融合、
输入回绕、输出bank复用、跨work、长尾、二维表、row padding、offset、QKV尾轴stride和只读重叠。

| MSS case id | TilingKey | 覆盖 |
| --- | --- | --- |
| 0 | 10 | length_bf16_1；L=[1]，Nv=2 |
| 1 | 30 | length_fp32_1；L=[1]，Nv=2 |
| 2 | 10 | flags_bf16_1111；L=[5]，Nv=2 |
| 3 | 30 | fused_fp32_input_bank_wrap；L=[17]，Nv=2 |
| 4 | 10 | all_views_bf16；L=[3, 5]，Nv=2 |
| 5 | 30 | all_views_fp32；L=[3, 5]，Nv=2 |
| 6 | 10 | large_heads_257；L=[1]，Nv=257 |
| 7 | 30 | many_work_one_core；L=[5, 3]，Nv=256 |
| 8 | 30 | table_fp32；L=[0, 5, 1]，Nv=6 |
| 9 | 10 | overlapping_qkv_bf16；L=[5]，Nv=4 |
| 10 | 10 | compound_storage_offset；L=[3, 5]，Nv=2 |
| 11 | 30 | softplus_tail_fp32；L=[5]，Nv=4 |

实际选择以 mssanitizer 日志中的 `RecurrentGatedDeltaRule_<hash>_10/_30` 为证据。调试专用
`rgdn_direct_trace` 记录了实际 blockDim/core/work/token/bank/event：单核连续8个 work、输入bank回绕6次、
输出bank多次复用、6个空work；其3个记录用例也与 CPU 比较通过。生产 wheel 与正常性能目标关闭 trace。

## 复现

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fla_rgdn
source /usr/local/Ascend/cann/set_env.sh
GEN_CASES_DTYPE_NUMBERS=834 bash tests/atk/run_test_cpu.sh \
  -op=recurrent_gated_delta_rule -scope=gen_cases -soc=ascend950
ATK_GM_INIT_MODE=off ATK_OUTPUT_ROOT=<output> bash tests/atk/run_test_cpu.sh \
  -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=all -soc=ascend950
```

ATK26.9.8 的 GM 初始化将字节计数传入默认FP32 torch.empty，30GiB预留实际申请约122GiB并OOM；
因此采用脚本支持的 off 选项。mssanitizer、有限值检查与 State padding 哨兵检查均保留。
当前统一脚本没有向确定性/MSS node 传输出路径，其报告默认落在算子 atk_output；验收后将本轮两份
完整目录归档到相同输出根，再运行 `common/check_atk_result.py --type all` 核查真实计数。
不以“未找到报告（跳过）”作为通过证据。

## 最终记录

2026-10-23，卡0 Ascend950PR_9579，CANN9.2.0-beta.2，Python3.11 / torch2.10.0+cpu / torch_npu2.10.0.post7 / ATK26.9.8。

| 验证 | 结果 |
| --- | --- |
| 独立 CPU 语义 / 值域校准 | 34/34；102/102 |
| 正常直接 kernel / host 非法输入 | 237/237；17/17，另合法host1/1 |
| Stable ABI / ctypes / 公开异常 | 237/237；9/9；58/58 |
| 实际 work/bank 调试记录及其精度 | 3/3 |
| 正式 ATK 精度 | 834/834 |
| 正式确定性（每例50次） | 12/12 |
| 正式 mssanitizer memcheck | 12/12，实际key10/30均无错误 |
| 旧 ABI 与 V2 C++ 示例 | 2/2 执行通过 |
| ABI / mutation / device guard 单元 | 10通过+1跳过；7通过；9通过 |

最终报告独立检查为3/3通过，计数834/12/12；报告完整目录归档后重新执行检查，未依赖默认路径的跳过提示。
同条件旧/新 wheel、相同原始性能JSON与executor、卡0复测：旧20个样本均值 **4.829100us**；
新两轮40个样本均值 **4.974375us**（相对均值 **+3.01%**）。
两个新批次分别为4.889900/5.058850us，
共享卡存在波动；仅作上述小工程case的实测记录，不推广为模型吞吐结论。
修改前夜间基线为4.767750us，完整ATK device_perf为17.4934us；最新正式device_perf为
[18.0923]us，包含其它输入构造操作。

本机完整证据位于 `~/fla_rgdn_workflow/`：`delivery-acceptance/`、`delivery-results-checked.log`、
`performance-delivery-summary.json`、`delivery-identity.json`、`trace-final/`、`ct-final/`和`softplus-tail/ct/`。
构建/安装为 `build-release.log` / `install-release.log`，最终恢复校验为 `verify-delivery.log`。
最终wheel SHA256：`cdc9e9f3b74af373818f4db6d76cb67fe90b37220d6d06b7c1f8e819fa0f9b2d`。wheel版本保留起始HEAD，实际未提交修改的源码和二进制以identity记录为准。
环境为 `~/miniforge3/envs/fla_rgdn`，最终环境依赖见本机 `environment-delivery.lock.txt`。
用户要求的带时间工作记录位于 `docs/new gdn recurrent/WORK_LOG.md`。
