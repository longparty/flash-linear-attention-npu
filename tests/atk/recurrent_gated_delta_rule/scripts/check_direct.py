"""CANNBot direct-kernel precision runner; uses the ATK input builder and unique golden."""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

OP_TESTS = Path(__file__).resolve().parents[1]
REPO = OP_TESTS.parents[2]
OP = REPO / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule"
sys.path.insert(0, str(OP_TESTS))
from executor_recurrent_gated_delta_rule import build_inputs
from reference import recurrent_gated_delta_rule_reference


class Tiling(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in (
        "vectorCoreNum", "ubCalSize", "ubRestBytes", "t", "nk", "dk", "nv", "dv", "sBlockNum", "b",
        "vStep", "stateOutBufferNum", "attnOutBufferNum")]
    _fields_ += [("scale", ctypes.c_float)]
    _fields_ += [(n, ctypes.c_uint32) for n in (
        "hasGama", "hasGamaK", "hasAcceptedTokens", "stateStride0", "stateStride1", "stateStride2")]
    _fields_ += [(n, ctypes.c_uint64 * size) for n, size in (
        ("queryStrides", 3), ("keyStrides", 3), ("valueStrides", 3), ("stateStrides", 4))]
    _fields_ += [(n, ctypes.c_uint32) for n in (
        "stateIndicesWidth", "useQkL2NormInKernel", "useGateInKernel", "useBetaSigmoidInKernel",
        "allowNegEigval", "hasDtBias")]


from case_matrix import build_extension_profiles as specs


def storage_view(tensor: torch.Tensor) -> torch.Tensor:
    size = tensor.untyped_storage().nbytes() // tensor.element_size()
    offset = tensor.storage_offset()
    return tensor.as_strided((size - offset,), (1,), offset)


def prepare(directory: Path, inputs: dict, vector_cores: int = 64) -> Tiling:
    directory.mkdir(parents=True, exist_ok=True)
    td = Tiling()
    q, v, state = inputs["query"], inputs["value"], inputs["state"]
    td.t, td.nk, td.dk = q.shape
    td.nv, td.dv = v.shape[1:]
    td.b = inputs["actual_seq_lengths"].numel() - 1
    td.sBlockNum = state.shape[0]
    td.vectorCoreNum = min(vector_cores, td.b * td.nv)
    td.scale = inputs["scale"]
    td.hasGama = inputs["g"] is not None
    td.hasAcceptedTokens = inputs["num_accepted_tokens"] is not None
    td.hasDtBias = inputs["dt_bias"] is not None
    for attr, name in (("queryStrides", "query"), ("keyStrides", "key"),
                       ("valueStrides", "value"), ("stateStrides", "state")):
        getattr(td, attr)[:] = inputs[name].stride()
    td.stateStride0, td.stateStride1, td.stateStride2 = state.stride()[:3]
    if inputs["ssm_state_indices"].ndim == 2:
        td.stateIndicesWidth = inputs["ssm_state_indices"].shape[1]
    for attr, name in (("useQkL2NormInKernel", "use_qk_l2norm_in_kernel"),
                       ("useGateInKernel", "use_gate_in_kernel"),
                       ("useBetaSigmoidInKernel", "use_beta_sigmoid_in_kernel"),
                       ("allowNegEigval", "allow_neg_eigval")):
        setattr(td, attr, inputs[name])
    (directory / "tiling.bin").write_bytes(bytes(td))
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name not in {"query", "key", "value", "state"}:
            value = value.contiguous()
        (directory / f"{name}.bin").write_bytes(storage_view(value).view(torch.uint8).numpy().tobytes())
    return td


def compare(actual, golden, inputs, compare_case, policy, name):
    prefix = int(inputs["actual_seq_lengths"][0])
    mask = np.ones(actual[0].shape, dtype=bool)
    mask[:prefix] = False
    actual_named = {"state": actual[1]}
    golden_named = {"state": golden[1]}
    masks = {"state": np.ones(actual[1].shape, dtype=bool)}
    regions = {"state": {}}
    if mask.any():
        actual_named["out"], golden_named["out"] = actual[0], golden[0]
        masks["out"] = mask
        boundary = np.zeros_like(mask)
        boundary[prefix], boundary[-1] = True, True
        regions["out"] = {"boundary": boundary}
    expected = {n: {"shape": list(a.shape), "dtype": str(a.dtype).removeprefix("torch."),
                    "required_regions": list(regions[n])} for n, a in actual_named.items()}
    report = compare_case(actual_named, golden_named, case=name, policy=policy,
                          expected_outputs=expected, valid_masks=masks, regions=regions)
    if not report["pass"]:
        return report
    # Independently enforce the same policy per value head without keeping
    # O(H^2) full-State region masks resident for large-head cases.
    for head in range(actual[0].shape[1]):
        a = {n: x[:, head] for n, x in actual_named.items()}
        g = {n: x[:, head] for n, x in golden_named.items()}
        m = {n: x[:, head] for n, x in masks.items()}
        exp = {n: {"shape": list(x.shape), "dtype": str(x.dtype).removeprefix("torch."),
                   "required_regions": []} for n, x in a.items()}
        per_head = compare_case(a, g, case=f"{name}/head_{head}", policy=policy,
            expected_outputs=exp, valid_masks=m, regions={n: {} for n in a})
        if not per_head["pass"]:
            return per_head
    report["heads_checked"] = actual[0].shape[1]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", help="Only run cases whose names contain this string")
    parser.add_argument("--cannbot", type=Path, default=Path.home() / "fla_rgdn_workflow/cannbot-skills/plugins-official/catlass-op-generator/catlass-cpp-generator")
    args = parser.parse_args()
    sys.path.insert(0, str(args.cannbot / "skills/catlass-cpp-test/scripts"))
    from compare_precision import compare_case
    policy = json.loads((OP / "reference/precision-policy.json").read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    host_log = (args.output / "host.log").open("w")
    process = subprocess.Popen([str(args.binary), "--server", str(args.device)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=host_log)
    pending = b""

    def response(marker):
        nonlocal pending
        deadline = time.monotonic() + 60
        while True:
            if b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                host_log.write(line.decode(errors="replace") + "\n")
                if line.decode(errors="replace").strip() == marker:
                    return
                continue
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([process.stdout], [], [], wait)[0]:
                process.kill()
                process.wait()
                raise TimeoutError("direct host did not respond within 60 seconds")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"direct host exited with {process.poll()}; see host.log")
            pending += chunk

    response("RGDN_READY")
    reports = []
    seeds = (20261022,) if args.scope == "smoke" else (20261022, 20261023, 20261024)
    for spec in specs(args.scope):
        if args.case and args.case not in spec["name"]:
            continue
        for seed in seeds:
            spec = dict(spec, seed=seed)
            name = f"{spec['name']}_{seed}"
            directory = args.output / name
            inputs = build_inputs(spec, torch.device("cpu"))
            prepare(directory, inputs, int(spec.get("cores", 64)))
            before = inputs["state"].clone()
            golden = recurrent_gated_delta_rule_reference(**inputs)
            assert torch.equal(before, inputs["state"])
            start = time.monotonic()
            try:
                process.stdin.write(f"{directory} {spec['state_dtype']} 1\n".encode())
                process.stdin.flush()
                response("RGDN_DONE")
                output = torch.frombuffer(bytearray((directory / "out.bin").read_bytes()), dtype=torch.bfloat16).reshape(inputs["value"].shape)
                raw_state = torch.frombuffer(bytearray((directory / "state_out.bin").read_bytes()), dtype=before.dtype)
                state = raw_state.as_strided(before.shape, inputs["state"].stride())
                report = compare((output, state), golden, inputs, compare_case, policy, name)
                if not report["pass"]:
                    torch.save({"out": output, "state": state}, directory / "actual.pt")
                    torch.save({"out": golden[0], "state": golden[1]}, directory / "golden.pt")
            except Exception as exc:
                report = {"case": name, "status": "ERROR", "pass": False, "error": str(exc)}
            report["elapsed_s"] = time.monotonic() - start
            reports.append(report)
            (directory / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print(name, report["status"], flush=True)
            if not report["pass"]:
                print(json.dumps(report, indent=2, default=str), flush=True)
                break
        if reports and not reports[-1]["pass"]:
            break
    summary = {"cases": len(reports), "passed": sum(r["pass"] for r in reports), "reports": reports}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    process.stdin.close()
    process.wait(timeout=60)
    host_log.close()
    raise SystemExit(0 if reports and all(r["pass"] for r in reports) else 1)


if __name__ == "__main__":
    main()
