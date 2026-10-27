/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef RECURRENT_GDN_ARCH35_V2_H
#define RECURRENT_GDN_ARCH35_V2_H

#include "kernel_operator.h"
#include "../recurrent_gated_delta_rule_struct.h"

namespace RecurrentGatedDeltaRule {
struct RGDRV2InitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR state;
    GM_ADDR lengths;
    GM_ADDR indices;
    GM_ADDR gate;
    GM_ADDR accepted;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR out;
#ifdef RGDN_ENABLE_TRACE
    GM_ADDR trace;
#endif
};
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_utils/vector/regbase.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "../recurrent_gated_delta_rule_struct.h"

namespace RecurrentGatedDeltaRule {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr CastTrait kRgdnLoadEven = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kRgdnLoadOdd = {
    RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kRgdnStoreEven = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, RoundMode::CAST_RINT};
constexpr CastTrait kRgdnStoreOdd = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void RgdnLoadMatrix(LocalTensor<T> dst, GlobalTensor<T> src,
                                     uint32_t rows, uint32_t cols, uint64_t srcStride,
                                     uint32_t dstStride)
{
    auto gm = tla::MakeTensor(src,
        tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(srcStride, tla::Int<1>{})),
        Catlass::Arch::PositionGM{});
    auto ub = tla::MakeTensor(dst,
        tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(dstStride, tla::Int<1>{})),
        Catlass::Arch::PositionUB{});
    Catlass::Epilogue::Tile::CopyGm2UbTla<Catlass::Arch::Ascend950, decltype(gm), decltype(ub)>{}(ub, gm);
}

template <typename T>
__aicore__ inline void RgdnStoreMatrix(GlobalTensor<T> dst, LocalTensor<T> src,
                                      uint32_t rows, uint32_t cols, uint64_t dstStride,
                                      uint32_t srcStride)
{
    auto ub = tla::MakeTensor(src,
        tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(srcStride, tla::Int<1>{})),
        Catlass::Arch::PositionUB{});
    auto gm = tla::MakeTensor(dst,
        tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(dstStride, tla::Int<1>{})),
        Catlass::Arch::PositionGM{});
    Catlass::Epilogue::Tile::CopyUb2GmTla<Catlass::Arch::Ascend950, decltype(ub), decltype(gm)>{}(gm, ub);
}

template <HardEvent Event>
__aicore__ inline void RgdnSync()
{
    auto id = GetTPipePtr()->FetchEventID(Event);
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

template <typename State>
__simd_vf__ inline void RgdnLoadStateVF(__ubuf__ float *dst, __ubuf__ State *src)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t i = 0; i < 256; ++i) {
        RegTensor<float> value;
        if constexpr (std::is_same_v<State, float>) {
            DataCopy(value, src + i * 64);
        } else {
            RegTensor<State> packed;
            DataCopy<State, LoadDist::DIST_UNPACK_B16>(packed, src + i * 64);
            Cast<float, State, kRgdnLoadEven>(value, packed, mask);
        }
        DataCopy(dst + i * 64, value, mask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_callee__ inline void RgdnNormalize(RegTensor<float> &x0, RegTensor<float> &x1, MaskReg &mask)
{
    RegTensor<float> sum0, sum1, sum, broadcast;
    Mul(sum0, x0, x0, mask);
    Mul(sum1, x1, x1, mask);
    Add(sum0, sum0, sum1, mask);
    ReduceSum(sum, sum0, mask);
    Duplicate(broadcast, sum, mask);
    Adds(broadcast, broadcast, 1.0e-6f, mask);
    RegTensor<float> one;
    Duplicate(one, 1.0f, mask);
    Sqrt(broadcast, broadcast, mask);
    Div(broadcast, one, broadcast, mask);
    Mul(x0, x0, broadcast, mask);
    Mul(x1, x1, broadcast, mask);
}

template <typename State>
__simd_vf__ inline void RgdnTokenVF(
    __ubuf__ float *state, __ubuf__ State *snapshot, __ubuf__ bfloat16_t *out,
    __ubuf__ bfloat16_t *q, __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *v,
    __ubuf__ bfloat16_t *beta, __ubuf__ float *gate, __ubuf__ float *outTmp,
    float scale, float aLog, float bias, uint32_t hasGate,
    uint32_t normalize, uint32_t rawGate, uint32_t sigmoid, uint32_t allowNegative)
{
    MaskReg fp = CreateMask<float, MaskPattern::ALL>();
    MaskReg bf = CreateMask<bfloat16_t, MaskPattern::ALL>();
    RegTensor<bfloat16_t> qb, kb, vb, bb;
    RegTensor<float> q0, q1, k0, k1, betaEff, alpha;
    DataCopy(qb, q);
    DataCopy(kb, k);
    Cast<float, bfloat16_t, kRgdnLoadEven>(q0, qb, bf);
    Cast<float, bfloat16_t, kRgdnLoadOdd>(q1, qb, bf);
    Cast<float, bfloat16_t, kRgdnLoadEven>(k0, kb, bf);
    Cast<float, bfloat16_t, kRgdnLoadOdd>(k1, kb, bf);
    if (normalize) {
        RgdnNormalize(q0, q1, fp);
        RgdnNormalize(k0, k1, fp);
    }
    Muls(q0, q0, scale, fp);
    Muls(q1, q1, scale, fp);
    DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(bb, beta);
    Cast<float, bfloat16_t, kRgdnLoadEven>(betaEff, bb, bf);
    if (sigmoid) {
        // Bound exp's argument; the discarded sigmoid tail is below 2e-35.
        RegTensor<float> one;
        Duplicate(one, 1.0f, fp);
        Maxs(betaEff, betaEff, -80.0f, fp);
        Mins(betaEff, betaEff, 80.0f, fp);
        Muls(betaEff, betaEff, -1.0f, fp);
        Exp(betaEff, betaEff, fp);
        Adds(betaEff, betaEff, 1.0f, fp);
        Div(betaEff, one, betaEff, fp);
        if (allowNegative) Muls(betaEff, betaEff, 2.0f, fp);
    }
    Duplicate(alpha, 1.0f, fp);
    if (hasGate) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(alpha, gate);
        if (rawGate) {
            RegTensor<float> positive, negative, logScale, smallLog;
            Adds(alpha, alpha, bias, fp);
            Maxs(positive, alpha, 0.0f, fp);
            Abs(negative, alpha, fp);
            Muls(negative, negative, -1.0f, fp);
            Exp(negative, negative, fp);
            // log(1+x) loses tiny x before Ln. For x<1/64, log1p(x)'s
            // fourth-order polynomial has relative truncation error <1.2e-8.
            MaskReg small;
            CompareScalar<float, CMPMODE::LT>(small, negative, 0.015625f, fp);
            Muls(smallLog, negative, -0.25f, fp);
            Adds(smallLog, smallLog, 1.0f / 3.0f, fp);
            Mul(smallLog, smallLog, negative, fp);
            Adds(smallLog, smallLog, -0.5f, fp);
            Mul(smallLog, smallLog, negative, fp);
            Adds(smallLog, smallLog, 1.0f, fp);
            Mul(smallLog, smallLog, negative, fp);
            Adds(negative, negative, 1.0f, fp);
            Ln(negative, negative, fp);
            Select(negative, smallLog, negative, small);
            Add(alpha, positive, negative, fp);
            Duplicate(logScale, aLog, fp);
            Exp(logScale, logScale, fp);
            Mul(alpha, alpha, logScale, fp);
            Muls(alpha, alpha, -1.0f, fp);
        }
        Exp(alpha, alpha, fp);
    }
    for (uint16_t row = 0; row < 128; ++row) {
        RegTensor<float> s0, s1, p0, p1, sum, prediction, value, delta;
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(s0, s1, state + row * 128);
        Mul(s0, s0, alpha, fp);
        Mul(s1, s1, alpha, fp);
        Mul(p0, s0, k0, fp);
        Mul(p1, s1, k1, fp);
        Add(p0, p0, p1, fp);
        ReduceSum(sum, p0, fp);
        Duplicate(prediction, sum, fp);
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(vb, v + row);
        Cast<float, bfloat16_t, kRgdnLoadEven>(value, vb, bf);
        Sub(delta, value, prediction, fp);
        Mul(delta, delta, betaEff, fp);
        Mul(p0, delta, k0, fp);
        Mul(p1, delta, k1, fp);
        Add(s0, s0, p0, fp);
        Add(s1, s1, p1, fp);
        DataCopy<float, StoreDist::DIST_INTLV_B32>(state + row * 128, s0, s1, fp);
        if constexpr (std::is_same_v<State, float>) {
            DataCopy<float, StoreDist::DIST_INTLV_B32>(snapshot + row * 128, s0, s1, fp);
        } else {
            RegTensor<State> packed;
            Cast<State, float, kRgdnStoreOdd>(packed, s1, fp);
            Cast<State, float, kRgdnStoreEven>(packed, s0, fp);
            StoreAlign(snapshot + row * 128, packed, bf);
        }
        Mul(p0, s0, q0, fp);
        Mul(p1, s1, q1, fp);
        Add(p0, p0, p1, fp);
        ReduceSum(sum, p0, fp);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outTmp + row, sum, fp);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> o0, o1;
    RegTensor<bfloat16_t> output;
    DataCopy<float, LoadDist::DIST_DINTLV_B32>(o0, o1, outTmp);
    Cast<bfloat16_t, float, kRgdnStoreOdd>(output, o1, fp);
    Cast<bfloat16_t, float, kRgdnStoreEven>(output, o0, fp);
    StoreAlign(out, output, bf);
}

template <typename State>
class RGDRV2 {
public:
    __aicore__ inline RGDRV2(const RecurrentGatedDeltaRuleTilingData *tiling) : td_(*tiling) {}

    __aicore__ inline void Init(const RGDRV2InitParams &p, TPipe *pipe)
    {
        q_.SetGlobalBuffer((__gm__ bfloat16_t *)p.query);
        k_.SetGlobalBuffer((__gm__ bfloat16_t *)p.key);
        v_.SetGlobalBuffer((__gm__ bfloat16_t *)p.value);
        beta_.SetGlobalBuffer((__gm__ bfloat16_t *)p.beta);
        state_.SetGlobalBuffer((__gm__ State *)p.state);
        lengths_.SetGlobalBuffer((__gm__ int32_t *)p.lengths);
        indices_.SetGlobalBuffer((__gm__ int32_t *)p.indices);
        accepted_.SetGlobalBuffer((__gm__ int32_t *)p.accepted);
        gate_.SetGlobalBuffer((__gm__ float *)p.gate);
        aLog_.SetGlobalBuffer((__gm__ float *)p.aLog);
        bias_.SetGlobalBuffer((__gm__ float *)p.dtBias);
        out_.SetGlobalBuffer((__gm__ bfloat16_t *)p.out);
#ifdef RGDN_ENABLE_TRACE
        trace_.SetGlobalBuffer((__gm__ uint64_t *)p.trace);
#endif
        pipe->InitBuffer(ub_, 208 * 1024);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t work = GetBlockIdx(); work < uint64_t(td_.b) * td_.nv; work += GetBlockNum()) {
            Trace(work, 0, 0, 0);
            uint32_t batch = work / td_.nv, head = work % td_.nv;
            int64_t start = lengths_.GetValue(0);
            for (uint32_t b = 0; b < batch; ++b) start += lengths_.GetValue(b + 1);
            int32_t length = lengths_.GetValue(batch + 1);
            if (length <= 0) continue;
            int32_t accepted = td_.hasAcceptedTokens ? accepted_.GetValue(batch) : 1;
            uint64_t indexBase = td_.stateIndicesWidth ? uint64_t(batch) * td_.stateIndicesWidth : start;
            int32_t initial = indices_.GetValue(indexBase + accepted - 1);
            auto stateFp32 = At<float>(0);
            auto inputState = At<State>(64 * 1024);
            // The initial-state DMA reuses output bank 0. MTE3_V protects a
            // future VF, but cannot order an MTE2 writer of that same bank.
            if (work != GetBlockIdx()) RgdnSync<HardEvent::MTE3_MTE2>();
            Trace(work, 0, 0, 1);
            RgdnLoadMatrix(inputState, state_[StateOffset(initial, head)], 128, 128, td_.stateStrides[2], 128);
            RgdnSync<HardEvent::MTE2_V>();
            RgdnLoadStateVF<State>((__ubuf__ float *)stateFp32.GetPhyAddr(),
                                   (__ubuf__ State *)inputState.GetPhyAddr());
            float aLog = td_.useGateInKernel ? aLog_.GetValue(head) : 0.0f;
            float bias = td_.useGateInKernel && td_.hasDtBias ? bias_.GetValue(head) : 0.0f;
            bool pending[2] = {false, false};
            for (int32_t tile = 0; tile < length; tile += 4) {
                uint32_t count = length - tile < 4 ? length - tile : 4;
                uint32_t inputBank = (tile / 4) & 1;
                LoadInputs(start + tile, head, count, inputBank, tile >= 8);
                if (tile >= 8) Trace(work, tile, inputBank, 7);
                RgdnSync<HardEvent::MTE2_V>();
                Trace(work, tile, inputBank, 2);
                for (uint32_t t = 0; t < count; ++t) {
                    uint32_t bank = (tile + t) & 1;
                    if (pending[bank]) WaitFlag<HardEvent::MTE3_V>(bank);
                    Trace(work, tile + t, bank, 3);
                    auto snapshot = At<State>((64 + bank * 64) * 1024);
                    auto output = At<bfloat16_t>((204 + bank) * 1024);
                    uint32_t base = (192 * 1024) + inputBank * 3072;
                    auto q = At<bfloat16_t>(base + t * 256);
                    auto k = At<bfloat16_t>(base + 1024 + t * 256);
                    auto v = At<bfloat16_t>(base + 2048 + t * 256);
                    auto beta = At<bfloat16_t>(198 * 1024 + inputBank * 512 + t * 32);
                    auto gate = At<float>(198 * 1024 + inputBank * 512 + 128 + t * 32);
                    auto tmp = At<float>(206 * 1024);
                    RgdnTokenVF<State>((__ubuf__ float *)stateFp32.GetPhyAddr(),
                        (__ubuf__ State *)snapshot.GetPhyAddr(), (__ubuf__ bfloat16_t *)output.GetPhyAddr(),
                        (__ubuf__ bfloat16_t *)q.GetPhyAddr(), (__ubuf__ bfloat16_t *)k.GetPhyAddr(),
                        (__ubuf__ bfloat16_t *)v.GetPhyAddr(), (__ubuf__ bfloat16_t *)beta.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(), (__ubuf__ float *)tmp.GetPhyAddr(),
                        td_.scale, aLog, bias, td_.hasGama, td_.useQkL2NormInKernel,
                        td_.useGateInKernel, td_.useBetaSigmoidInKernel, td_.allowNegEigval);
                    SetFlag<HardEvent::V_MTE3>(bank);
                    WaitFlag<HardEvent::V_MTE3>(bank);
                    Trace(work, tile + t, bank, 4);
                    uint64_t token = start + tile + t;
                    int32_t slot = indices_.GetValue(td_.stateIndicesWidth ? indexBase + tile + t : token);
                    RgdnStoreMatrix(state_[StateOffset(slot, head)], snapshot, 128, 128, td_.stateStrides[2], 128);
                    RgdnStoreMatrix(out_[(token * td_.nv + head) * 128], output, 1, 128, 128, 128);
                    SetFlag<HardEvent::MTE3_V>(bank);
                    pending[bank] = true;
                    Trace(work, tile + t, bank, 5);
                }
            }
            for (uint32_t bank = 0; bank < 2; ++bank) if (pending[bank]) WaitFlag<HardEvent::MTE3_V>(bank);
            Trace(work, length, 0, 6);
        }
    }

private:
    __aicore__ inline void Trace(uint64_t work, uint64_t token, uint64_t bank, uint64_t event)
    {
#ifdef RGDN_ENABLE_TRACE
        // One disjoint 4096-record segment per AIV. Debug only, no extra sync.
        if (traceCount_ < 4096) {
            uint64_t offset = (uint64_t(GetBlockIdx()) * 4096 + traceCount_++) * 6;
            trace_.SetValue(offset, GetBlockNum());
            trace_.SetValue(offset + 1, GetBlockIdx());
            trace_.SetValue(offset + 2, work);
            trace_.SetValue(offset + 3, token);
            trace_.SetValue(offset + 4, bank);
            trace_.SetValue(offset + 5, event + 1);
        }
#endif
    }
#ifdef RGDN_ENABLE_TRACE
    GlobalTensor<uint64_t> trace_;
    uint32_t traceCount_ = 0;
#endif
    template <typename T> __aicore__ inline LocalTensor<T> At(uint32_t offset)
    { return ub_.GetWithOffset<T>((208 * 1024 - offset) / sizeof(T), offset); }

    __aicore__ inline uint64_t StateOffset(uint64_t slot, uint64_t head)
    { return slot * td_.stateStrides[0] + head * td_.stateStrides[1]; }

    __aicore__ inline void LoadQkv(LocalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src,
                                    uint64_t token, uint64_t head, uint32_t count, const uint64_t *stride)
    {
        uint64_t base = token * stride[0] + head * stride[1];
        // Multi-row DMA encodes the gap after each row as an unsigned byte
        // count. Read-only views may overlap rows; gather those directly.
        if (stride[2] == 1 && stride[0] >= 128 && stride[0] - 128 <= UINT32_MAX / sizeof(bfloat16_t)) {
            RgdnLoadMatrix(dst, src[base], count, 128, stride[0], 128);
        } else {
            // Direct view fallback: each source element is read at its original
            // address. No GM contiguous tensor or out-of-bounds wider DMA.
            RgdnSync<HardEvent::V_S>();
            for (uint32_t t = 0; t < count; ++t)
                for (uint32_t j = 0; j < 128; ++j)
                    dst.SetValue(t * 128 + j, src.GetValue(base + t * stride[0] + j * stride[2]));
            RgdnSync<HardEvent::S_V>();
        }
    }

    __aicore__ inline void LoadInputs(uint64_t token, uint32_t head, uint32_t count, uint32_t bank, bool reuse)
    {
        // Both input banks start free. Reuse begins at tile 8; a later work
        // already drained its predecessor through MTE3_MTE2 before loading State.
        if (reuse) RgdnSync<HardEvent::V_MTE2>();
        uint32_t base = 192 * 1024 + bank * 3072;
        uint32_t keyHead = head / (td_.nv / td_.nk);
        LoadQkv(At<bfloat16_t>(base), q_, token, keyHead, count, td_.queryStrides);
        LoadQkv(At<bfloat16_t>(base + 1024), k_, token, keyHead, count, td_.keyStrides);
        LoadQkv(At<bfloat16_t>(base + 2048), v_, token, head, count, td_.valueStrides);
        RgdnLoadMatrix(At<bfloat16_t>(198 * 1024 + bank * 512), beta_[token * td_.nv + head],
                       count, 1, td_.nv, 16);
        if (td_.hasGama) RgdnLoadMatrix(At<float>(198 * 1024 + bank * 512 + 128),
            gate_[token * td_.nv + head], count, 1, td_.nv, 8);
    }

    RecurrentGatedDeltaRuleTilingData td_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<bfloat16_t> q_, k_, v_, beta_, out_;
    GlobalTensor<State> state_;
    GlobalTensor<int32_t> lengths_, indices_, accepted_;
    GlobalTensor<float> gate_, aLog_, bias_;
};
} // namespace RecurrentGatedDeltaRule
#endif
#endif
