// Copyright (c) 2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "acl/acl.h"
#include "../op_kernel/arch35/recurrent_gated_delta_rule_v2.h"

using namespace RecurrentGatedDeltaRule;

template <typename State>
__global__ __aicore__ void rgdn_direct_kernel(RGDRV2InitParams inputs,
                                             RecurrentGatedDeltaRuleTilingData tiling)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    RGDRV2<State> op(&tiling);
    op.Init(inputs, &pipe);
    op.Process();
#endif
}

static void Check(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

static std::vector<uint8_t> Read(const std::string &path, bool optional = false)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        if (optional) return {};
        throw std::runtime_error("cannot read " + path);
    }
    auto size = file.tellg();
    if (size < 0) throw std::runtime_error("invalid size " + path);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!bytes.empty() && !file.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
        throw std::runtime_error("short read " + path);
    return bytes;
}

static void Validate(const std::string &dir, const RecurrentGatedDeltaRuleTilingData &td, bool fp32 = false)
{
    auto require = [](bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); };
    require(td.t && td.b && td.sBlockNum && td.nk && td.nv && td.vectorCoreNum, "dimensions must be positive");
    require(td.dk == 128 && td.dv == 128, "K and V must be 128");
    require(td.nv % td.nk == 0, "Nv must be divisible by Nk");
    require(std::isfinite(td.scale), "scale must be finite");
    for (auto flag : {td.hasGama, td.hasAcceptedTokens, td.useQkL2NormInKernel,
                     td.useGateInKernel, td.useBetaSigmoidInKernel, td.allowNegEigval, td.hasDtBias})
        require(flag <= 1, "flags must be bool");
    require(!std::filesystem::exists(dir + "/gk.bin"), "gk is reserved");
    auto size = [&](const char *name) {
        auto path = dir + "/" + name + ".bin";
        return std::filesystem::exists(path) ? std::filesystem::file_size(path) : uintmax_t(0);
    };
    auto footprint = [&](const char *name, std::vector<uint64_t> shape,
                         const uint64_t *stride, uint32_t elementBytes) {
        uint64_t limit = size(name) / elementBytes;
        require(limit > 0, "missing input or empty storage");
        uint64_t last = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            require(stride[i] > 0, "strides must be positive");
            if (shape[i] > 1) {
                require(stride[i] <= (limit - 1 - last) / (shape[i] - 1), "view exceeds storage bounds");
                last += stride[i] * (shape[i] - 1);
            }
        }
    };
    footprint("query", {td.t, td.nk, 128}, td.queryStrides, 2);
    footprint("key", {td.t, td.nk, 128}, td.keyStrides, 2);
    footprint("value", {td.t, td.nv, 128}, td.valueStrides, 2);
    uint64_t scalarStride[] = {td.nv, 1};
    footprint("beta", {td.t, td.nv}, scalarStride, 2);
    require(td.stateStrides[3] == 1 && td.stateStrides[2] >= 128, "State inner axis must be contiguous");
    footprint("state", {td.sBlockNum, td.nv, 128, 128}, td.stateStrides, fp32 ? 4 : 2);
    std::vector<std::pair<uint64_t, uint64_t>> axes;
    uint64_t shape[] = {td.sBlockNum, td.nv, 128, 128};
    for (size_t i = 0; i < 4; ++i) if (shape[i] > 1) axes.emplace_back(td.stateStrides[i], shape[i]);
    std::sort(axes.begin(), axes.end());
    uint64_t span = 1;
    for (auto [stride, dimension] : axes) {
        require(stride >= span, "State view must not overlap");
        require(stride <= (UINT64_MAX - span) / (dimension - 1), "State stride overflow");
        span += stride * (dimension - 1);
    }
    if (td.hasGama) footprint("g", {td.t, td.nv}, scalarStride, 4);
    if (td.useGateInKernel) {
        require(td.hasGama && size("A_log") == uint64_t(td.nv) * 4, "raw gate requires g and A_log");
        if (td.hasDtBias) require(size("dt_bias") == uint64_t(td.nv) * 4, "dt_bias shape mismatch");
    }
    auto metadata = [&](const char *name, uint64_t count) {
        require(size(name) % 4 == 0 && size(name) / 4 == count, "metadata shape mismatch");
        auto bytes = Read(dir + "/" + name + ".bin");
        std::vector<int32_t> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    };
    auto lengths = metadata("actual_seq_lengths", uint64_t(td.b) + 1);
    int64_t sum = 0;
    for (auto n : lengths) { require(n >= 0, "sequence lengths must be nonnegative"); sum += n; }
    require(sum == td.t, "sequence lengths must sum to T");
    auto indices = metadata("ssm_state_indices", td.stateIndicesWidth ? uint64_t(td.b) * td.stateIndicesWidth : td.t);
    for (auto index : indices) require(index >= 0 && uint32_t(index) < td.sBlockNum, "State index out of range");
    auto accepted = td.hasAcceptedTokens ? metadata("num_accepted_tokens", td.b) : std::vector<int32_t>(td.b, 1);
    std::unordered_map<int32_t, uint32_t> writers;
    std::vector<std::pair<int32_t, uint32_t>> readers;
    uint64_t start = lengths[0];
    for (uint32_t b = 0; b < td.b; ++b) {
        uint32_t length = lengths[b + 1];
        uint32_t width = td.stateIndicesWidth ? td.stateIndicesWidth : length;
        require(!td.stateIndicesWidth || length <= width, "sequence exceeds table width");
        if (length) {
            require(accepted[b] >= 1 && uint32_t(accepted[b]) <= width, "accepted checkpoint out of range");
            uint64_t base = td.stateIndicesWidth ? uint64_t(b) * width : start;
            readers.emplace_back(indices[base + accepted[b] - 1], b);
            for (uint32_t t = 0; t < length; ++t) {
                auto [it, inserted] = writers.emplace(indices[base + t], b);
                require(inserted || it->second == b, "State writes overlap across requests");
            }
        }
        start += length;
    }
    for (auto [slot, b] : readers) {
        auto it = writers.find(slot);
        require(it == writers.end() || it->second == b, "initial State races another request");
    }
}

struct Buffer {
    void *ptr = nullptr;
    size_t size = 0;
    explicit Buffer(const std::vector<uint8_t> &bytes) : size(bytes.size())
    {
        if (!size) return;
        Check(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY), "malloc");
        Check(aclrtMemcpy(ptr, size, bytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    }
    Buffer(const Buffer &) = delete;
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    void Save(const std::string &path)
    {
        std::vector<uint8_t> bytes(size);
        Check(aclrtMemcpy(bytes.data(), size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
        std::ofstream file(path, std::ios::binary);
        if (!file.write(reinterpret_cast<const char *>(bytes.data()), size))
            throw std::runtime_error("cannot write " + path);
    }
};

static void RunCase(const std::string &dir, bool fp32, int iterations, aclrtStream stream)
{
    if (iterations <= 0) throw std::runtime_error("iterations must be positive");
    auto tdBytes = Read(dir + "/tiling.bin");
    if (tdBytes.size() != sizeof(RecurrentGatedDeltaRuleTilingData))
        throw std::runtime_error("TilingData byte count mismatch");
    RecurrentGatedDeltaRuleTilingData td{};
    std::memcpy(&td, tdBytes.data(), sizeof(td));
    Validate(dir, td, fp32);
    Buffer q(Read(dir + "/query.bin")), k(Read(dir + "/key.bin")), v(Read(dir + "/value.bin"));
    Buffer beta(Read(dir + "/beta.bin")), state(Read(dir + "/state.bin"));
    Buffer lengths(Read(dir + "/actual_seq_lengths.bin")), indices(Read(dir + "/ssm_state_indices.bin"));
    Buffer gate(Read(dir + "/g.bin", true)), accepted(Read(dir + "/num_accepted_tokens.bin", true));
    Buffer aLog(Read(dir + "/A_log.bin", true)), bias(Read(dir + "/dt_bias.bin", true));
    Buffer out(std::vector<uint8_t>(uint64_t(td.t) * td.nv * 128 * 2, 0));
    RGDRV2InitParams p{(GM_ADDR)q.ptr, (GM_ADDR)k.ptr, (GM_ADDR)v.ptr, (GM_ADDR)beta.ptr,
        (GM_ADDR)state.ptr, (GM_ADDR)lengths.ptr, (GM_ADDR)indices.ptr, (GM_ADDR)gate.ptr,
        (GM_ADDR)accepted.ptr, (GM_ADDR)aLog.ptr, (GM_ADDR)bias.ptr, (GM_ADDR)out.ptr};
#ifdef RGDN_ENABLE_TRACE
    Buffer trace(std::vector<uint8_t>(uint64_t(td.vectorCoreNum) * 4096 * 6 * sizeof(uint64_t), 0));
    p.trace = (GM_ADDR)trace.ptr;
#endif
    for (int i = 0; i < iterations; ++i) {
        if (fp32) rgdn_direct_kernel<float><<<td.vectorCoreNum, nullptr, stream>>>(p, td);
        else rgdn_direct_kernel<bfloat16_t><<<td.vectorCoreNum, nullptr, stream>>>(p, td);
        Check(aclrtSynchronizeStream(stream), "kernel synchronize");
    }
    out.Save(dir + "/out.bin");
    state.Save(dir + "/state_out.bin");
#ifdef RGDN_ENABLE_TRACE
    trace.Save(dir + "/trace.bin");
#endif
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "usage: rgdn_direct CASE_DIRECTORY bf16|fp32 [iterations=1] [device=0]\n";
        return 2;
    }
    try {
        if (std::string(argv[1]) == "--validate") {
            auto bytes = Read(std::string(argv[2]) + "/tiling.bin");
            if (bytes.size() != sizeof(RecurrentGatedDeltaRuleTilingData))
                throw std::invalid_argument("TilingData byte count mismatch");
            RecurrentGatedDeltaRuleTilingData td{};
            std::memcpy(&td, bytes.data(), sizeof(td));
            Validate(argv[2], td, argc > 3 && std::string(argv[3]) == "fp32");
            std::cout << "host contract validated\n";
            return 0;
        }
        bool server = std::string(argv[1]) == "--server";
        int device = server ? std::stoi(argv[2]) : (argc > 4 ? std::stoi(argv[4]) : 0);
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "set device");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        if (server) {
            std::cout << "RGDN_READY" << std::endl;
            std::string line;
            while (std::getline(std::cin, line)) {
                std::istringstream command(line);
                std::string dir, dtype;
                int iterations = 1;
                if (!(command >> dir >> dtype >> iterations)) throw std::runtime_error("invalid server command");
                RunCase(dir, dtype == "fp32", iterations, stream);
                std::cout << "RGDN_DONE" << std::endl;
            }
        } else {
            RunCase(argv[1], std::string(argv[2]) == "fp32", argc > 3 ? std::stoi(argv[3]) : 1, stream);
        }
        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtResetDevice(device), "reset device");
        Check(aclFinalize(), "aclFinalize");
        std::cout << "direct kernel completed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
