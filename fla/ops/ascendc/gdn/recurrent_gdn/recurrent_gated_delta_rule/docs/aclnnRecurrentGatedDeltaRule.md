# Recurrent GDN ACLNN 调用

输入、数学、dtype/layout、metadata 合法域和别名契约统一见 [api.md](api.md)。
声明位于 [aclnn_recurrent_gated_delta_rule.h](../op_host/op_api/aclnn_recurrent_gated_delta_rule.h)。

旧 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` 和 launch 符号保持 ABI，采用全部融合属性为 false。
A5 扩展使用以下两段式入口；A2/A3 不支持 V2。

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *beta, aclTensor *state,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *A_log, const aclTensor *dt_bias, float scaleValue,
    bool useQkL2NormInKernel, bool useGateInKernel,
    bool useBetaSigmoidInKernel, bool allowNegEigval,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnRecurrentGatedDeltaRuleV2(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

先创建描述符并调用 GetWorkspaceSize，按返回值分配 workspace，再调用 launch，最后同步 stream。
State 必须与输入 pool 同址；输出 out 由调用方分配。Q/K/V 和 State 的真实 storage shape、
stride 和 offset 必须透传，不能把 padded storage 当成紧凑逻辑 shape。
`gk` 必须为 nullptr，raw gate 开关开启时 g 与 A_log 必需，dt_bias 可以为 nullptr。

C++ 层检查 descriptor、dtype、shape、stride 与属性依赖。直接 C 调用方负责保证设备上
lengths、indices、accepted 的值满足 api.md 的合法域和跨请求无数据竞争；Python 两种后端还会
在 launch 前校验 metadata 值。非法 descriptor 返回 ACLNN_ERR_PARAM_INVALID。

[示例](../examples/test_aclnn_recurrent_gated_delta_rule.cpp) 默认验证旧 ABI 的执行，
传入 `--v2` 使用核内 QK norm 与 beta sigmoid。新增功能的数值比较见 ATK 用例。
