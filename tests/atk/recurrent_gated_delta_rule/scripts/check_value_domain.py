"""CPU-only calibration: correct FP32/BF16 results pass; representative bugs fail."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from case_matrix import build_extension_profiles
from executor_recurrent_gated_delta_rule import build_inputs
from reference import recurrent_gated_delta_rule_reference as reference


def matched(actual, golden, dtype):
    rtol, atol = ((.0156, .0156) if dtype == torch.bfloat16 else (.000977, .0000153))
    return float(torch.isclose(actual.double(), golden.double(), rtol=rtol, atol=atol).double().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    reports = []
    profiles = [p for p in build_extension_profiles("full") if p["name"].startswith(("flags_", "softplus_tail_"))]
    for profile in profiles:
        for seed in (20261022, 20261023, 20261024):
            inputs = build_inputs(dict(profile, seed=seed), torch.device("cpu"))
            actual = reference(**inputs)
            precise = reference(**{n: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                                   for n, v in inputs.items()})
            name = profile["name"] + "_" + str(seed)
            report = dict(name=name, scale=inputs["scale"], outputs={})
            for label, a, g in zip(("out", "state"), actual, precise):
                ratio = matched(a, g, a.dtype)
                error = float((a.double() - g).abs().max())
                assert ratio >= .99, (name, label, ratio)
                assert error <= (.002 if a.dtype == torch.bfloat16 else .0002), (name, label, error)
                zero_ratio = matched(torch.zeros_like(a), g, a.dtype)
                assert zero_ratio < .99, (name, label, "zero output was not rejected")
                report["outputs"][label] = dict(correct_matched_ratio=ratio, correct_max_abs=error,
                                                zero_matched_ratio=zero_ratio)
                if label == "out":
                    scale_ratio = matched(2 * a, g, a.dtype)
                    assert scale_ratio < .99, (name, "wrong scale was not rejected")
                    report["outputs"][label]["double_scale_matched_ratio"] = scale_ratio
                if seed == 20261022 and profile["name"] in {"flags_bf16_0000", "flags_fp32_1111"}:
                    # CT consumes plain tensors, and the public dtype policy is
                    # tracked in the JSON rather than inferred from these files.
                    for role, tensor in (("correct", a), ("golden", g), ("zero", torch.zeros_like(a))):
                        torch.save(tensor.float(), args.output / f"{name}_{label}_{role}.pt")
            # Only active switches change the mathematical result. Verify the
            # chosen domain detects omission of each active preprocessing step.
            for flag in ("use_qk_l2norm_in_kernel", "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval"):
                if not inputs[flag] or (flag == "allow_neg_eigval" and not inputs["use_beta_sigmoid_in_kernel"]):
                    continue
                wrong = reference(**dict(inputs, **{flag: False}))[0]
                ratio = matched(wrong, precise[0], actual[0].dtype)
                assert ratio < .99, (name, flag, "omission was not rejected", ratio)
                report[flag + "_omitted_ratio"] = ratio
            reports.append(report)
    result = dict(cases=len(reports), passed=len(reports), seeds=[20261022, 20261023, 20261024], reports=reports)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print("CPU value-domain calibration:", len(reports), "PASS", flush=True)


if __name__ == "__main__":
    main()
