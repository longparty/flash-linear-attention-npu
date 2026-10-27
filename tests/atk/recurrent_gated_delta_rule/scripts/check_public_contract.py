"""Exercise public error handling and metadata-cache invalidation on real NPU tensors."""
from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(60, exit=True)

import torch
import torch_npu  # noqa: F401

from check_direct import build_inputs
from check_public import to_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule
    from fla_npu.ops.ascendc._aclnn_ctypes import npu_recurrent_gated_delta_rule

    torch.set_num_threads(1)
    print("contract: initialize NPU", flush=True)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    profile = dict(seq_lengths=[2, 1], prefix_tokens=0, HK=1, HV=2, K=128, V=128,
                   T=3, B=2, block_num=10, state_dtype="fp32", state_layout="contiguous",
                   input_layout="contiguous", gate_mode="g", state_indices_width=4,
                   accepted_tokens=[1, 1], use_gate_in_kernel=True, scale=128**-0.5, seed=20261022)

    def replace(name, value):
        return lambda inputs: inputs.__setitem__(name, value(inputs) if callable(value) else value)

    def change_element(name, index, value):
        def mutate(inputs):
            inputs[name][index] = value
        return mutate

    mutations = {
        "reserved_gk": replace("gk", lambda x: x["key"].float()),
        "missing_gate": replace("g", None),
        "missing_A_log": replace("A_log", None),
        "non_boolean_switch": replace("use_qk_l2norm_in_kernel", 1),
        "nan_scale": replace("scale", float("nan")),
        "infinite_scale": replace("scale", float("inf")),
        "wrong_query_dtype": replace("query", lambda x: x["query"].float()),
        "wrong_state_dtype": replace("state", lambda x: x["state"].half()),
        "wrong_metadata_dtype": replace("actual_seq_lengths", lambda x: x["actual_seq_lengths"].long()),
        "wrong_key_shape": replace("key", lambda x: x["key"][..., :64]),
        "wrong_value_tokens": replace("value", lambda x: x["value"][:1]),
        "wrong_beta_shape": replace("beta", lambda x: x["beta"][:, :1]),
        "wrong_gate_shape": replace("g", lambda x: x["g"][:, :1]),
        "wrong_A_log_shape": replace("A_log", lambda x: x["A_log"][:1]),
        "wrong_bias_shape": replace("dt_bias", lambda x: x["dt_bias"][:1]),
        "wrong_accepted_shape": replace("num_accepted_tokens", lambda x: x["num_accepted_tokens"][:1]),
        "wrong_table_batch": replace("ssm_state_indices", lambda x: x["ssm_state_indices"][:1]),
        "short_table": replace("ssm_state_indices", lambda x: x["ssm_state_indices"][:, :1]),
        "negative_length": change_element("actual_seq_lengths", 1, -1),
        "bad_length_sum": change_element("actual_seq_lengths", 1, 3),
        "negative_index": change_element("ssm_state_indices", (0, 0), -1),
        "large_index": change_element("ssm_state_indices", (0, 0), 10),
        "zero_accepted": change_element("num_accepted_tokens", 0, 0),
        "large_accepted": change_element("num_accepted_tokens", 0, 5),
        "cross_request_write": change_element("ssm_state_indices", (1, 0), 0),
        "cross_request_read": change_element("ssm_state_indices", (1, 0), 2),
        "state_inner_stride": replace("state", lambda x: torch.zeros(10, 2, 128, 256, device=device)[..., ::2]),
        "overlapping_state": replace("state", lambda x: x["state"].as_strided(x["state"].shape, (16384, 16384, 128, 1))),
        "cpu_tensor": replace("beta", lambda x: x["beta"].cpu()),
    }
    reports = []
    for backend, function in (("stable", recurrent_gated_delta_rule), ("ctypes", npu_recurrent_gated_delta_rule)):
        for name, mutate in mutations.items():
            print(backend, name, "START", flush=True)
            faulthandler.dump_traceback_later(60, exit=True)
            cpu = build_inputs(profile, torch.device("cpu"))
            inputs = {n: to_npu(x, device) for n, x in cpu.items()}
            # Populate the successful metadata cache before mutating the same tensors.
            function(**inputs)
            torch.npu.synchronize()
            if name == "cross_request_read":
                inputs["num_accepted_tokens"][0] = 3
            mutate(inputs)
            before = inputs["state"].cpu().clone()
            version = inputs["state"]._version
            try:
                function(**inputs)
            except (ValueError, TypeError, NotImplementedError) as error:
                torch.npu.synchronize()
                assert torch.equal(before, inputs["state"].cpu()), name
                assert inputs["state"]._version == version, name
                reports.append(dict(backend=backend, name=name, error=type(error).__name__, passed=True))
                print(backend, name, "PASS", flush=True)
            else:
                raise AssertionError(f"{backend}: {name} unexpectedly accepted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(cases=len(reports), passed=len(reports), reports=reports), indent=2))
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
