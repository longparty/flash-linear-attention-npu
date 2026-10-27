/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling.cpp
 * \brief
 */
#include "recurrent_gated_delta_rule_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

#include <cmath>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule, RecurrentGatedDeltaRuleTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t CUSEQLENS_INDEX = 5;
const size_t SSM_STATE_INDICES_INDEX = 6;
const size_t G_INDEX = 7;
const size_t GK_INDEX = 8;
const size_t ACC_TO_INDEX = 9;
const size_t A_LOG_INDEX = 10;
const size_t DT_BIAS_INDEX = 11;

void RecurrentGatedDeltaRuleTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.isA5 = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.vectorCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

RecurrentGatedDeltaRuleTilingContext RecurrentGatedDeltaRuleTiling::BuildProcessorContext() const
{
    RecurrentGatedDeltaRuleTilingContext ctx;
    ctx.nodeName = context_->GetNodeName();
    ctx.queryShape = &context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    ctx.keyShape = &context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    ctx.valueShape = &context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    ctx.betaShape = &context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    ctx.stateShape = &context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    ctx.cuSeqlensShape = &context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    ctx.ssmStateShape = &context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();
    ctx.aivNum = compileInfo_.aivNum;
    ctx.ubSize = compileInfo_.ubSize;
    ctx.stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    ctx.scale = tilingData_.scale;
    ctx.hasGama = tilingData_.hasGama;
    ctx.hasGamaK = tilingData_.hasGamaK;
    ctx.hasAcceptedTokens = tilingData_.hasAcceptedTokens;
    return ctx;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetStateStrides() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetStateStrides."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoOpTiling()
{
    OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoLibApiTiling()
{
    const auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    const uint64_t stateDtypeKey = stateDtype == ge::DT_FLOAT ? RGDR_TPL_FP32 : RGDR_TPL_BF16;
    tilingKey_ = GET_TPL_TILING_KEY(stateDtypeKey);
    return ge::GRAPH_SUCCESS;
};

uint64_t RecurrentGatedDeltaRuleTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(RGDR_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RecurrentGatedDeltaRuleTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(BETA_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(BETA_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(STATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(STATE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(CUSEQLENS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(CUSEQLENS_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(SSM_STATE_INDICES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(SSM_STATE_INDICES_INDEX));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "query dtype, key dtype and value dtype should be bfloat16"),
                return ge::GRAPH_FAILED);

    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(betaDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "beta dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != ge::DT_FLOAT && stateDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "state dtype should be bfloat16 or float32"),
                return ge::GRAPH_FAILED);
    auto cuSeqlensDtype = context_->GetInputDesc(CUSEQLENS_INDEX)->GetDataType();
    auto ssmStateIndicesDtype = context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    OP_CHECK_IF(cuSeqlensDtype != ge::DT_INT32 || ssmStateIndicesDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "cuSeqlens dtype and ssmStateIndices dtype should be int32"),
                return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
        auto gamaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gamaDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "gama dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }

    if (context_->GetOptionalInputDesc(GK_INDEX) != nullptr) {
        auto gamaKDtype = context_->GetOptionalInputDesc(GK_INDEX)->GetDataType();
        OP_CHECK_IF(gamaKDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "gamaK dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }

    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        auto numAcceptedTokensDtype = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetDataType();
        OP_CHECK_IF(numAcceptedTokensDtype != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(), "numAcceptedTokens dtype should be int32"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    RecurrentGatedDeltaRuleTilingProcessor processor(BuildProcessorContext());
    return processor.ProcessShapes(tilingData_);
}

bool RecurrentGatedDeltaRuleTiling::CheckFormat(ge::Format format, const std::string &Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "state") ||
        !CheckFormat(context_->GetInputDesc(CUSEQLENS_INDEX)->GetStorageFormat(), "actual_seq_lengths") ||
        !CheckFormat(context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetStorageFormat(), "ssm_state_indices")) {
        return ge::GRAPH_FAILED;
    }

    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
        auto gamaFormat = context_->GetOptionalInputDesc(G_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "gama format not support NZ"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(GK_INDEX) != nullptr) {
        auto gamaKFormat = context_->GetOptionalInputDesc(GK_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaKFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "gamaK format not support NZ"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetStorageFormat();
        OP_CHECK_IF(numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context_->GetNodeName(), "numAcceptedTokens format not support NZ"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float *scalePtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_IF(scalePtr == nullptr || !std::isfinite(*scalePtr),
                OP_LOGE(context_->GetNodeName(), "scaleValue must be finite"), return ge::GRAPH_FAILED);
    float scaleValue = *scalePtr;
    tilingData_.scale = scaleValue;
    const bool *norm = attrs->GetAttrPointer<bool>(1);
    const bool *gate = attrs->GetAttrPointer<bool>(2);
    const bool *sigmoid = attrs->GetAttrPointer<bool>(3);
    const bool *twice = attrs->GetAttrPointer<bool>(4);
    tilingData_.useQkL2NormInKernel = norm && *norm;
    tilingData_.useGateInKernel = gate && *gate;
    tilingData_.useBetaSigmoidInKernel = sigmoid && *sigmoid;
    tilingData_.allowNegEigval = twice && *twice;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetStateStrides()
{
    auto read = [&](size_t index, size_t rank, uint64_t *out) {
        auto stride = context_->GetInputStride(index);
        uint64_t dense = 1;
        const auto &shape = context_->GetInputShape(index)->GetOriginShape();
        for (size_t i = rank; i > 0; --i) {
            int64_t value = stride && stride->GetDimNum() == rank ? stride->GetStride(i - 1) : dense;
            if (value <= 0) return false;
            out[i - 1] = static_cast<uint64_t>(value);
            dense *= shape.GetDim(i - 1);
        }
        return true;
    };
    OP_CHECK_IF(!read(QUERY_INDEX,3,tilingData_.queryStrides) || !read(KEY_INDEX,3,tilingData_.keyStrides) ||
                !read(VALUE_INDEX,3,tilingData_.valueStrides) || !read(STATE_INDEX,4,tilingData_.stateStrides),
                OP_LOGE(context_->GetNodeName(), "strides must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.stateStrides[3] != 1 || tilingData_.stateStrides[2] < tilingData_.dk,
                OP_LOGE(context_->GetNodeName(), "State inner axis must be contiguous"), return ge::GRAPH_FAILED);
    tilingData_.stateStride0 = static_cast<uint32_t>(tilingData_.stateStrides[0]);
    tilingData_.stateStride1 = static_cast<uint32_t>(tilingData_.stateStrides[1]);
    tilingData_.stateStride2 = static_cast<uint32_t>(tilingData_.stateStrides[2]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetOptionalInput()
{
    tilingData_.hasGama = context_->GetOptionalInputDesc(G_INDEX) != nullptr;
    tilingData_.hasGamaK = 0;
    tilingData_.hasAcceptedTokens = context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr;
    tilingData_.hasDtBias = context_->GetOptionalInputDesc(DT_BIAS_INDEX) != nullptr;
    OP_CHECK_IF(context_->GetOptionalInputDesc(GK_INDEX) != nullptr,
                OP_LOGE(context_->GetNodeName(), "gk is reserved"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.useGateInKernel && (!tilingData_.hasGama || !context_->GetOptionalInputDesc(A_LOG_INDEX)),
                OP_LOGE(context_->GetNodeName(), "raw gate requires g and A_log"), return ge::GRAPH_FAILED);
    for (size_t index : {A_LOG_INDEX,DT_BIAS_INDEX}) {
        auto desc = context_->GetOptionalInputDesc(index);
        if (!desc) continue;
        auto shape = context_->GetOptionalInputShape(index);
        OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT || !shape ||
                    shape->GetOriginShape().GetDimNum() != 1 || shape->GetOriginShape().GetDim(0) != tilingData_.nv,
                    OP_LOGE(context_->GetNodeName(), "A_log/dt_bias must be FP32 [Nv]"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void RecurrentGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "stateOutBufferNum: [%u]", tilingData_.stateOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "attnOutBufferNum: [%u]", tilingData_.attnOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "hasGama: [%u]", tilingData_.hasGama);
    OP_LOGD(context_->GetNodeName(), "hasGamaK: [%u]", tilingData_.hasGamaK);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CalUbSize()
{
    if (compileInfo_.isA5) {
        OP_CHECK_IF(compileInfo_.ubSize < 208U * 1024U,
                    OP_LOGE(context_->GetNodeName(), "A5 UB must hold 208 KiB"), return ge::GRAPH_FAILED);
        tilingData_.ubCalSize = 208U * 1024U;
        tilingData_.ubRestBytes = 0;
        tilingData_.vStep = 128;
        tilingData_.stateOutBufferNum = 2;
        tilingData_.attnOutBufferNum = 2;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(tilingData_.useQkL2NormInKernel || tilingData_.useGateInKernel ||
                tilingData_.useBetaSigmoidInKernel || tilingData_.allowNegEigval || tilingData_.stateIndicesWidth ||
                tilingData_.nk > 256 || tilingData_.nv > 256 || tilingData_.stateStrides[2] != 128,
                OP_LOGE(context_->GetNodeName(), "extended Recurrent GDN requires Ascend950"), return ge::GRAPH_FAILED);
    RecurrentGatedDeltaRuleTilingProcessor processor(BuildProcessorContext());
    return processor.ProcessUb(tilingData_);
}

static ge::graphStatus RecurrentGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentGatedDeltaRule(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentGatedDeltaRuleCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRule)
    .Tiling(RecurrentGatedDeltaRuleTilingFunc)
    .TilingParse<RecurrentGatedDeltaRuleCompileInfo>(TilingPrepareForRecurrentGatedDeltaRule);
} // namespace optiling
