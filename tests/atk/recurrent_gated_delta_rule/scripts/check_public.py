"""End-to-end checks for the installed Stable-ABI and ctypes Recurrent GDN APIs."""
from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

from check_direct import OP, build_inputs, compare, specs
from reference import recurrent_gated_delta_rule_reference


def to_npu(tensor, device):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    count = tensor.untyped_storage().nbytes() // tensor.element_size()
    base = torch.full((count,), 0.125 if tensor.is_floating_point() else 0,
                      dtype=tensor.dtype, device=device)
    view = base.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())
    view.copy_(tensor)
    return view


def padding_unchanged(state):
    count = state.untyped_storage().nbytes() // state.element_size()
    mask = torch.zeros(count, dtype=torch.bool)
    mask.as_strided(state.shape, state.stride(), state.storage_offset()).fill_(True)
    if not (~mask).any():
        return True
    flat = state.as_strided((count,), (1,), 0).cpu()
    expected = torch.full_like(flat[~mask], 0.125)
    return torch.equal(flat[~mask], expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", choices=("stable", "ctypes"), default="stable")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cannbot", type=Path, default=Path.home() / "fla_rgdn_workflow/cannbot-skills/plugins-official/catlass-op-generator/catlass-cpp-generator")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.cannbot / "skills/catlass-cpp-test/scripts"))
    from compare_precision import compare_case
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, exit=True)
    from fla_npu.ops import ascendc
    if args.backend == "stable":
        function = ascendc.recurrent_gated_delta_rule
    else:
        from fla_npu.ops.ascendc import _aclnn_ctypes
        function = _aclnn_ctypes.npu_recurrent_gated_delta_rule
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    policy = json.loads((OP / "reference/precision-policy.json").read_text())
    profiles = specs(args.scope)
    if args.scope == "smoke":
        profiles += [p for p in specs("full") if p["name"] in
                     {"all_views_bf16", "all_views_fp32", "table_bf16", "ungated_fp32", "large_heads_257"}]
    seeds = (20261022,) if args.scope == "smoke" else (20261022, 20261023, 20261024)
    reports = []
    for profile in profiles:
        for seed in seeds:
            name = f"{profile['name']}_{seed}"
            cpu = build_inputs(dict(profile, seed=seed), torch.device("cpu"))
            golden = recurrent_gated_delta_rule_reference(**cpu)
            inputs = {n: to_npu(x, device) for n, x in cpu.items()}
            version = inputs["state"]._version
            print(args.backend, name, "START", flush=True)
            faulthandler.dump_traceback_later(60, exit=True)
            output = function(**inputs)
            torch.npu.synchronize()
            faulthandler.cancel_dump_traceback_later()
            if args.backend == "stable":
                assert inputs["state"]._version == version + 1, "State mutation version must bump exactly once"
            actual = (output.cpu(), inputs["state"].cpu())
            report = compare(actual, golden, cpu, compare_case, policy, name)
            report["padding_unchanged"] = padding_unchanged(inputs["state"])
            report["pass"] = report["pass"] and report["padding_unchanged"]
            reports.append(report)
            print(args.backend, name, report["status"], "padding", report["padding_unchanged"], flush=True)
            if not report["pass"]:
                directory = args.output / name
                directory.mkdir(exist_ok=True)
                torch.save(dict(zip(("out", "state"), actual)), directory / "actual.pt")
                torch.save(dict(zip(("out", "state"), golden)), directory / "golden.pt")
                (directory / "report.json").write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps(report, indent=2, default=str), flush=True)
                break
        if reports and not reports[-1]["pass"]:
            break
    result = {"backend": args.backend, "cases": len(reports), "passed": sum(r["pass"] for r in reports),
              "reports": reports}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if reports and all(r["pass"] for r in reports) else 1)


if __name__ == "__main__":
    main()
