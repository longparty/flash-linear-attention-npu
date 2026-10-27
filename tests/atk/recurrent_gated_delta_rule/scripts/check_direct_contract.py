"""Negative host-contract tests through the real C++ direct executable."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

import torch
from check_direct import Tiling, build_inputs, prepare, specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(dict(specs("smoke")[0], seed=20261022), torch.device("cpu"))
    base = args.output / "valid"
    td = prepare(base, inputs)
    checks = []

    def check(name, mutate, message):
        target = args.output / name
        shutil.copytree(base, target, dirs_exist_ok=True)
        modified = Tiling.from_buffer_copy(bytes(td))
        mutate(modified, target)
        (target / "tiling.bin").write_bytes(bytes(modified))
        result = subprocess.run([str(args.binary), "--validate", str(target)],
                                capture_output=True, text=True, timeout=10)
        ok = result.returncode != 0 and message in result.stderr
        checks.append({"case": name, "pass": ok, "returncode": result.returncode,
                       "stderr": result.stderr.strip()})
        assert ok, checks[-1]

    valid = subprocess.run([str(args.binary), "--validate", str(base)], capture_output=True, text=True, timeout=10)
    assert valid.returncode == 0, valid.stderr
    check("invalid_k", lambda t, p: setattr(t, "dk", 64), "K and V")
    check("invalid_v", lambda t, p: setattr(t, "dv", 64), "K and V")
    check("zero_heads", lambda t, p: setattr(t, "nk", 0), "positive")
    check("head_ratio", lambda t, p: setattr(t, "nk", 3), "divisible")
    check("nonfinite_scale", lambda t, p: setattr(t, "scale", float("nan")), "finite")
    check("nonbool_switch", lambda t, p: setattr(t, "useGateInKernel", 2), "bool")
    check("reserved_gk", lambda t, p: (p / "gk.bin").write_bytes(b"x"), "reserved")
    check("raw_without_a_log", lambda t, p: setattr(t, "useGateInKernel", 1), "A_log")
    check("negative_stride", lambda t, p: t.queryStrides.__setitem__(2, 0), "positive")
    check("qkv_bounds", lambda t, p: t.queryStrides.__setitem__(2, 100000), "storage")
    check("state_column_stride", lambda t, p: t.stateStrides.__setitem__(3, 2), "contiguous")
    check("state_overlap", lambda t, p: t.stateStrides.__setitem__(1, 1), "overlap")
    check("state_bounds", lambda t, p: t.stateStrides.__setitem__(0, 100000000), "storage")
    check("negative_lengths", lambda t, p: (p / "actual_seq_lengths.bin").write_bytes(
        torch.tensor([0, -1], dtype=torch.int32).numpy().tobytes()), "nonnegative")
    check("length_sum", lambda t, p: (p / "actual_seq_lengths.bin").write_bytes(
        torch.tensor([0, 2], dtype=torch.int32).numpy().tobytes()), "sum to T")
    check("index_bounds", lambda t, p: (p / "ssm_state_indices.bin").write_bytes(
        torch.tensor([3], dtype=torch.int32).numpy().tobytes()), "out of range")

    def accepted(t, p):
        t.hasAcceptedTokens = 1
        (p / "num_accepted_tokens.bin").write_bytes(torch.tensor([0], dtype=torch.int32).numpy().tobytes())
    check("accepted_zero", accepted, "checkpoint")
    report = {"status": "PASS", "valid_cases": 1, "negative_cases": len(checks), "checks": checks}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
