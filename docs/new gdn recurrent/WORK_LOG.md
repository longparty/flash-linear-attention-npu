# Recurrent GDN 适配工作日志

时间均为服务器本地时间（Asia/Shanghai，UTC+08:00）。

## 2026-10-22 21:37 — 任务开始

- 起始分支：`main`；提交：`1cd1944bbe6429dd38f71eb201a808bf17cee3f0`。
- 用户输入：已修改的 `tests/atk/recurrent_gated_delta_rule/reference_recurrent_gated_delta_rule.py`、`docs/new gdn recurrent/design_full.md` 和 `流水图.drawio`。
- 初始工作树仅上述标杆修改及新设计目录；保留用户已有内容。
- 目标：适配 Recurrent GDN NPU 入参与新功能，补全测试，先测旧实现性能，再验证新实现精度、性能、内存和确定性。
- 环境：用户目录内创建 `fla_rgdn`；CANN 使用 `/usr/local/Ascend/cann/set_env.sh`；优先 NPU 0；不修改公共安装。
- 流程来源：用户指定 PR #833 的 `docs/算子自动开发工作流.md` 优先，正在读取并固定版本；本地流程作为工程和验收资料。
- 计划依次完成：接口差异与标杆验收 → 旧版本基线 → 设计与实现 → 构建 → 定向精度 → 正式测试及性能对比 → 归档。

## 2026-10-22 22:20 — 环境与接口差异

- 固定 CANNBot PR #1274：`35974d3499f7f7e94231272ec0cfc2785de28cad`；用户目录已初始化 `operators/recurrent_gdn_catlass`，路由 `linear_attention / catlass-linear-attention-v1`。
- 环境：`~/miniforge3/envs/fla_rgdn`，Python 3.11、torch 2.10.0+cpu、torch_npu 2.10.0.post7、ATK 26.9.8。NPU 0 运行时可用，设备为 Ascend950PR_9579。
- 卡 0 已有其他 atk/python 进程，不清理、不终止；性能采集记录该共享条件。
- 已保存修改前 NPU 源码、ATK 资产、用户标杆和 HEAD 原标杆至 `~/fla_rgdn_workflow/baseline_source/`。
- 已启动旧实现单算子构建，日志 `~/fla_rgdn_workflow/build-baseline.log`。
- 已新增算子 `docs/api.md`，逐项记录新开关、二维索引、长度/head 限制移除、视图约束及设计材料中的旧说明冲突。
- PyPI 仅有 triton-ascend 3.2.0，未冒充满足 CANN 9.x 的 >=3.2.1；本次 Ascend C 路径继续独立准备，若正式预检需要，另获取匹配包。

## 2026-10-23 09:27 — 基线与直调开发

- 旧 NPU 定向精度：2/2 通过；正式 ATK 性能：1/1 执行成功。
- 主 kernel Task Duration 均值 4.767750 us，范围 4.557000–5.113000 us；完整 ATK device_perf 为 17.4934 us，二者范围不同，分别保留。
- 纯 CPU 独立检查 34 项通过，包含 16 个开关组合和三种固定种子校准；关闭 torch 设备插件自动加载运行，未依赖 NPU 数值。
- 标杆已收敛到 tests/atk/recurrent_gated_delta_rule/reference.py，旧文件保留兼容导入；目标契约新增允许无 g、拒绝 gk、allow_neg_eigval；用户原文件有快照。
- CANNBot workflow/reference definition/design 校验通过；正在 implementation。A5 直调工程与新 RegBase kernel 已落盘，首轮编译修复 GM_ADDR 多变量宏、TLA 常量与 Rsqrt 原语差异；尚未宣称 NPU 新功能通过。

## 2026-10-23 11:15 — CANNBot 直调验收完成

- 最终 direct full：195/195 通过；真实 C++ host 非法输入 17/17 拦截，另有合法 host case 1/1 通过。
- BF16 转换使用对应的 BF16 mask；同核跨 work 在 bank 0 初始 State 搬入前增加 MTE3_MTE2 保护。两个故障均保留 actual/golden 和定向诊断，未放宽精度。
- 最终设计/实现同步校验通过；5 次预热后 5 个正式 direct 主核样本均值 3.670000 us。原始数据 direct-prof-final/，摘要 direct-performance-final.json。
- ct viz 不在环境或固定 CANNBot 包中，命令不可用；失败输出已保留，另用 rank-one/恒等状态/线性 key 解析诊断定位，未以替代诊断冒充 ct viz 执行。
- workflow 标记 complete；开始本仓公开 ACLNN/Stable-ABI/Python 适配与 ATK 验收。

## 2026-10-23 15:10 — 公开接口与扩展回归

- 完成 V2 ACLNN、L0/host/tiling、Stable ABI、ctypes 和 Python 入参适配；新 A5 RegBase kernel 已经完整 wheel 构建并安装在 fla_rgdn。
- 公共接口首轮出现 native abort，gdb 定位到 InferShape 的 storage/logical shape 混用；恢复 SetOriginalShape 与 DFX 后，公开冒烟 9/9 通过。
- 最终扩展 direct 225/225、Stable ABI 225/225、ctypes 9/9 通过，包含额外 row padding、QKV transpose、极端 raw gate/beta、零 QK 和每核连续 work。
- 公开异常 58/58 通过；验证 metadata cache 原地更新失效、拒绝非法入参时 State 数值/版本均不变。既有 ABI 单元 11 项（1 skip）、mutation contract 7 项通过。
- ATK gen_cases 实际生成 822 条候选，与已审查矩阵逐项一致，已固化正式精度 JSON；MSS 人工选择 8 条，perf 保持原始 1 条。
- 首轮 scope=all 因 ATK 26.9.8 GM 初始化将字节数传给默认 FP32 torch.empty，30 GiB 预留实际申请约 122 GiB，发生 OOM。已停止仅本轮的 19 个专属进程，不清理他人进程；采用仓库支持 ATK_GM_INIT_MODE=off 重新执行 all，保留原失败日志。
- 全量 ATK、最终性能、确定性、mssanitizer 和 C++ 示例仍在验收，尚未宣称整体完成。

## 2026-10-23 16:18 CST — 数值边界修复与最终归档准备

- 最初用户指定 PR 工作流保留为工程资料；用户补充当前 AGENTS 后，按 docs/agents 的五阶段与 V2 设计规则复核并归档。
- 原小 scale 的全零 out 可能通过 BF16 绝对容差，已回到值域校准阶段；最终开关诊断 scale=8。102组三种种子的CPU/FP64校准通过，能拒绝错误scale、全零结果和遗漏融合步骤。
- CT0.9.1 已从官方release安装并核对SHA256，补做12组值域/历史故障图和4组softplus修复前后图；先前“ct不可用”为历史状态。
- A_log=20/g=-20 的相消边界复现 out/State 误差0.10620117/0.02613516，修复小尾部 log1p 后精确复测两个输出均为0差异。
- 增加 RGDN_ENABLE_TRACE 独立调试目标，实际记录单核8个work、6次输入bank回绕、输出bank复用和6个空work，3个记录用例精度通过；生产目标关闭记录。
- 最终直接kernel237/237、公开Stable ABI237/237、ctypes9/9、公开异常58/58通过。
- 正式精度834/834、12条确定性/内存集上一轮全通过；最终MSS把FP32融合case延长到17，确保实际覆盖输入bank回绕，正在最后一轮全量验收。
- 首 work 与输入bank首次使用不存在旧消费者，跳过冗余等待；跨work和回绕仍保留明确保护。最新主kernel初测4.905us，正在同条件最终复测；不以早前直调时间冒充正式公开路径性能。

## 2026-10-23 16:40 CST — 正式验收完成

- 最后scope=all实测精度834/834、确定性12/12（各50次）、mssanitizer12/12；FP32融合MSS长度17，覆盖输入bank回绕。归档默认输出后独立check_atk_result确认3/3通过，非跳过结论。
- 旧C ABI与V2 C++示例均执行成功；最终生产目标不启用trace；安装后二进制、kernel与Python包装器身份一致。
- 相同原始JSON/原始executor、NPU0、同一时段旧新复测：旧核4.829100us，新两批40样本均值4.974375us，均值差+3.01%。数据与各批次波动全部保留，用户未提供新模型case或硬性能阈值。
- 性能JSON与原始快照逐字节一致，实际scale为0.08838834764831843、seed=20260826。已按实物修正文档并补齐YAML的性能属性域，实际gen_cases再次核对834条与正式精度输入一致。
- 最终恢复交付wheel，SHA256 cdc9e9f3b74af373818f4db6d76cb67fe90b37220d6d06b7c1f8e819fa0f9b2d；专属环境fla_rgdn与Miniforge均位于用户目录，未修改公共CANN或他人任务。
- 最终结果已收敛至算子ATK README，开发期validation.md按当前流程移除；完整构建、测试、CT与性能日志保留在~/fla_rgdn_workflow。

## 2026-10-23 16:56 CST — 最终检查

- 正式设计已按当前仓库V2整理为两章；CANNBot辅助快照按六章格式保存在用户目录（仅标题层级规范化），辅助workflow/reference/design校验通过。
- ABI、mutation、device guard单元回归，Python编译、git diff --check、本次文档链接与源码identity复核均通过。
- 开发期validation.md已移除；最终README、工作日志和用户目录完整证据保留。
