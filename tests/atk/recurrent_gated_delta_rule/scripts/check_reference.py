"""CPU contract checks and FP32/BF16 versus FP64 policy calibration."""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import recurrent_gated_delta_rule_reference as reference


def run() -> dict:
    torch.set_num_threads(1)
    checked = 0
    # Independent scalar recurrence: only [row=0,col=0] can be nonzero.
    for norm, raw, sigmoid, twice in itertools.product((False, True), repeat=4):
        q = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
        q[..., 0] = 1
        v = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
        v[:, :, 0] = torch.tensor([[0.5, -0.25], [0.25, 0.75]])
        state = torch.zeros(3, 2, 128, 128)
        state[2, :, 0, 0] = torch.tensor([0.125, -0.125])
        before = state.clone()
        beta = torch.full((2, 2), 0.5, dtype=torch.bfloat16)
        gate = torch.tensor([[-0.25, 0.25], [0.125, -0.5]])
        a_log = torch.tensor([0.0, 0.5])
        bias = torch.tensor([0.125, -0.25])
        out, updated = reference(q, q, v, state, beta, 0.5,
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([3], dtype=torch.int32),
            g=gate, A_log=a_log, dt_bias=bias,
            use_qk_l2norm_in_kernel=norm, use_gate_in_kernel=raw,
            use_beta_sigmoid_in_kernel=sigmoid, allow_neg_eigval=twice)
        assert torch.equal(state, before), "CPU golden mutated its input State"
        effective_q = 1 / math.sqrt(1 + 1e-6) if norm else 1.0
        beta_value = (1 / (1 + math.exp(-0.5))) * (2 if twice else 1) if sigmoid else 0.5
        expected = torch.zeros_like(out)
        for h in range(2):
            s = float(state[2, h, 0, 0])
            for t in range(2):
                g = float(gate[t, h])
                if raw:
                    z = g + float(bias[h])
                    g = -math.exp(float(a_log[h])) * (max(z, 0) + math.log1p(math.exp(-abs(z))))
                s *= math.exp(g)
                s += beta_value * (float(v[t, h, 0]) - s * effective_q) * effective_q
                expected[t, h, 0] = s * effective_q * 0.5
                assert abs(float(updated[t, h, 0, 0]) - s) < 2e-7
        assert torch.equal(out, expected), (norm, raw, sigmoid, twice)
        assert torch.equal(updated[2], state[2])
        checked += 1

    maxima = {"out": 0.0, "state": 0.0}
    for seed in (20261022, 20261023, 20261024):
        torch.manual_seed(seed)
        for length in (1, 5, 17):
            for dtype in (torch.bfloat16, torch.float32):
                q = (torch.randn(length, 1, 128) * 0.1).bfloat16()
                k = (torch.randn(length, 1, 128) * 0.1).bfloat16()
                v = (torch.randn(length, 2, 128) * 0.1).bfloat16()
                s = (torch.randn(length, 2, 128, 128) * 0.01).to(dtype)
                args = dict(query=q, key=k, value=v, state=s,
                    beta=torch.randn(length, 2).bfloat16(), scale=128**-0.5,
                    actual_seq_lengths=torch.tensor([0, length], dtype=torch.int32),
                    ssm_state_indices=torch.arange(length, dtype=torch.int32),
                    g=torch.randn(length, 2), A_log=torch.randn(2) * 0.5,
                    dt_bias=torch.randn(2) * 0.2,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
                actual = reference(**args)
                precise = reference(**{n: x.double() if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                                       for n, x in args.items()})
                for n, a, g in zip(("out", "state"), actual, precise):
                    maxima[n] = max(maxima[n], float((a.double() - g).abs().max()))
                checked += 1
    assert maxima["out"] < 0.002 and maxima["state"] < 0.002, maxima
    return {"status": "PASS", "cases": checked, "seeds": [20261022, 20261023, 20261024],
            "max_abs_vs_fp64": maxima,
            "policy_max_abs": {"bfloat16": 0.002, "float32": 0.0002}}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
