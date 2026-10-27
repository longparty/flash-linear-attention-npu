/** Copyright (c) 2025-2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#include "aclnn_recurrent_gated_delta_rule.h"
#include "recurrent_gated_delta_rule.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace op;
namespace {
struct Params {
    const aclTensor *query, *key, *value, *beta;
    aclTensor *state;
    const aclTensor *lengths, *indices, *g, *gk, *accepted, *aLog, *bias;
    float scale;
    bool norm, rawGate, sigmoid, twice;
    aclTensor *out;
};

bool ViewValid(const aclTensor *t)
{
    if (!t) return true;
    const auto &shape = t->GetViewShape();
    const auto &strides = t->GetViewStrides();
    int64_t storage = t->GetStorageShape().GetShapeSize();
    int64_t last = t->GetViewOffset();
    if (storage <= 0 || last < 0 || last >= storage || strides.size() != shape.GetDimNum()) return false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t n = shape.GetDim(i), stride = strides[i];
        if (n <= 0 || stride <= 0 || (n > 1 && stride > (storage - 1 - last) / (n - 1))) return false;
        last += (n - 1) * stride;
    }
    return true;
}

bool Shape(const aclTensor *t, std::initializer_list<int64_t> sizes)
{
    if (!t || t->GetViewShape().GetDimNum() != sizes.size()) return false;
    size_t i = 0;
    for (auto size : sizes) if (t->GetViewShape().GetDim(i++) != size) return false;
    return true;
}

bool Type(const aclTensor *t, std::initializer_list<DataType> types)
{
    return !t || std::find(types.begin(), types.end(), t->GetDataType()) != types.end();
}

bool Check(const Params &p)
{
    if (!p.query || !p.key || !p.value || !p.beta || !p.state || !p.lengths || !p.indices || !p.out || p.gk)
        return false;
    if (!std::isfinite(p.scale) || (p.rawGate && (!p.g || !p.aLog))) return false;
    for (auto t : {p.query, p.key, p.value, p.beta, static_cast<const aclTensor *>(p.out)}) if (!Type(t, {DataType::DT_BF16})) return false;
    if (!Type(p.state, {DataType::DT_BF16, DataType::DT_FLOAT})) return false;
    for (auto t : {p.lengths, p.indices, p.accepted}) if (!Type(t, {DataType::DT_INT32})) return false;
    for (auto t : {p.g, p.aLog, p.bias}) if (!Type(t, {DataType::DT_FLOAT})) return false;
    if (p.query->GetViewShape().GetDimNum() != 3 || p.value->GetViewShape().GetDimNum() != 3 ||
        p.state->GetViewShape().GetDimNum() != 4 || p.lengths->GetViewShape().GetDimNum() != 1) return false;
    const auto &q = p.query->GetViewShape(), &v = p.value->GetViewShape(), &s = p.state->GetViewShape();
    int64_t t = q.GetDim(0), nk = q.GetDim(1), nv = v.GetDim(1), blocks = s.GetDim(0);
    int64_t b = p.lengths->GetViewShape().GetDim(0) - 1;
    if (t <= 0 || nk <= 0 || nv <= 0 || blocks <= 0 || b <= 0 || nv % nk || q.GetDim(2) != 128 || v.GetDim(2) != 128)
        return false;
    if (!Shape(p.key, {t,nk,128}) || !Shape(p.value, {t,nv,128}) || !Shape(p.beta, {t,nv}) ||
        !Shape(p.state, {blocks,nv,128,128}) || !Shape(p.out, {t,nv,128})) return false;
    if (p.indices->GetViewShape().GetDimNum() == 1) {
        if (!Shape(p.indices, {t})) return false;
    } else if (p.indices->GetViewShape().GetDimNum() == 2) {
        if (p.indices->GetViewShape().GetDim(0) != b || p.indices->GetViewShape().GetDim(1) <= 0) return false;
    } else return false;
    if ((p.g && !Shape(p.g, {t,nv})) || (p.aLog && !Shape(p.aLog, {nv})) ||
        (p.bias && !Shape(p.bias, {nv})) || (p.accepted && !Shape(p.accepted, {b}))) return false;
    for (auto x : {p.query,p.key,p.value,p.beta,static_cast<const aclTensor *>(p.state),p.lengths,p.indices,
                   p.g,p.accepted,p.aLog,p.bias,static_cast<const aclTensor *>(p.out)}) if (!ViewValid(x)) return false;
    const auto &stride = p.state->GetViewStrides();
    if (stride[3] != 1 || stride[2] < 128) return false;
    std::vector<std::pair<int64_t,int64_t>> axes;
    for (size_t i = 0; i < 4; ++i) if (s.GetDim(i) > 1) axes.emplace_back(stride[i], s.GetDim(i));
    std::sort(axes.begin(), axes.end());
    int64_t span = 1;
    for (auto [step, count] : axes) { if (step < span) return false; span += (count - 1) * step; }
    return true;
}

aclnnStatus Workspace(const Params &p, bool extended, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!workspaceSize || !executor || !Check(p)) return ACLNN_ERR_PARAM_INVALID;
    const char *soc = aclrtGetSocName();
    bool a5 = soc && std::strstr(soc, "950");
    if (extended && !a5) return ACLNN_ERR_PARAM_INVALID;
    auto exec = CREATE_EXECUTOR();
    CHECK_RET(exec.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // aclCreateTensor may describe padded storage as a flat allocation. Shape
    // inference consumes the logical original shape, rather than that storage.
    for (auto t : {p.query, p.key, p.value, p.beta, static_cast<const aclTensor *>(p.state),
                   p.lengths, p.indices, p.g, p.accepted, p.aLog, p.bias}) {
        if (t) t->SetOriginalShape(t->GetViewShape());
    }
    auto view = [&](const aclTensor *t) -> const aclTensor * {
        if (IsContiguous(t)) return t;
        return exec->CreateView(t, t->GetViewShape(), t->GetStorageShape(), t->GetViewStrides(), t->GetViewOffset());
    };
    auto q = a5 ? view(p.query) : l0op::Contiguous(p.query, exec.get());
    auto k = a5 ? view(p.key) : l0op::Contiguous(p.key, exec.get());
    auto v = a5 ? view(p.value) : l0op::Contiguous(p.value, exec.get());
    auto beta = l0op::Contiguous(p.beta, exec.get());
    auto lengths = l0op::Contiguous(p.lengths, exec.get());
    auto indices = l0op::Contiguous(p.indices, exec.get());
    auto g = p.g ? l0op::Contiguous(p.g, exec.get()) : nullptr;
    auto accepted = p.accepted ? l0op::Contiguous(p.accepted, exec.get()) : nullptr;
    auto aLog = p.rawGate ? l0op::Contiguous(p.aLog, exec.get()) : nullptr;
    auto bias = p.rawGate && p.bias ? l0op::Contiguous(p.bias, exec.get()) : nullptr;
    aclTensor *state = p.state;
    if (!IsContiguous(state)) state = exec->CreateView(state, state->GetViewShape(), state->GetStorageShape(),
                                                     state->GetViewStrides(), state->GetViewOffset());
    CHECK_RET(q && k && v && beta && lengths && indices && state, ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::RecurrentGatedDeltaRule(q,k,v,beta,state,lengths,indices,g,nullptr,accepted,
        aLog,bias,p.scale,p.norm,p.rawGate,p.sigmoid,p.twice,exec.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result, p.out, exec.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = exec->GetWorkspaceSize();
    exec.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
}

extern "C" {
aclnnStatus aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize(
    const aclTensor *query,const aclTensor *key,const aclTensor *value,const aclTensor *beta,
    aclTensor *state,const aclTensor *lengths,const aclTensor *indices,const aclTensor *g,const aclTensor *gk,
    const aclTensor *accepted,const aclTensor *aLog,const aclTensor *bias,float scale,
    bool norm,bool rawGate,bool sigmoid,bool twice,aclTensor *out,uint64_t *workspaceSize,aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRuleV2,
        DFX_IN(query,key,value,beta,state,lengths,indices,g,gk,accepted,aLog,bias,scale,norm,rawGate,sigmoid,twice),
        DFX_OUT(state,out));
    return Workspace({query,key,value,beta,state,lengths,indices,g,gk,accepted,aLog,bias,scale,norm,rawGate,sigmoid,twice,out},
                     true,workspaceSize,executor);
}
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,const aclTensor *key,const aclTensor *value,const aclTensor *beta,
    aclTensor *state,const aclTensor *lengths,const aclTensor *indices,const aclTensor *g,const aclTensor *gk,
    const aclTensor *accepted,float scale,aclTensor *out,uint64_t *workspaceSize,aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRule,
        DFX_IN(query,key,value,beta,state,lengths,indices,g,gk,accepted,scale), DFX_OUT(state,out));
    return Workspace({query,key,value,beta,state,lengths,indices,g,gk,accepted,nullptr,nullptr,scale,false,false,false,false,out},
                     false,workspaceSize,executor);
}
aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace,uint64_t workspaceSize,aclOpExecutor *executor,aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRule);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}
aclnnStatus aclnnRecurrentGatedDeltaRuleV2(void *workspace,uint64_t workspaceSize,aclOpExecutor *executor,aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRuleV2);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}
}
