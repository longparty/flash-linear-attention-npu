"""Shared, pure-Python profiles for direct and ATK extended-contract tests."""
import itertools

def build_extension_profiles(scope: str) -> list[dict]:
    cases = []

    def add(name, lengths=(1,), **overrides):
        s = dict(name=name, seq_lengths=list(lengths), prefix_tokens=0, HK=1, HV=2,
                 K=128, V=128, gate_mode="g", state_dtype="bf16", state_layout="contiguous",
                 input_layout="contiguous", scale=128**-0.5)
        s.update(overrides)
        s["B"] = len(s["seq_lengths"])
        s["T"] = s["prefix_tokens"] + sum(s["seq_lengths"])
        s["block_num"] = max(s["T"], s["B"] * int(s.get("state_indices_width", 0))) + 2
        s["design_id"] = "RGDN-V2-" + name
        cases.append(s)

    if scope == "smoke":
        add("default_bf16")
        add("default_fp32", state_dtype="fp32")
        add("fused_bf16", (5,), use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
        add("fused_fp32", (5,), state_dtype="fp32", state_indices_width=7, accepted_tokens=[7],
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    else:
        for dtype in ("bf16", "fp32"):
            for norm, gate, beta, twice in itertools.product((False, True), repeat=4):
                add(f"flags_{dtype}_{int(norm)}{int(gate)}{int(beta)}{int(twice)}", (5,),
                    state_dtype=dtype, scale=8.0, use_qk_l2norm_in_kernel=norm, use_gate_in_kernel=gate,
                    use_beta_sigmoid_in_kernel=beta, allow_neg_eigval=twice)
            for length in (1, 3, 4, 8, 9, 17, 33):
                add(f"length_{dtype}_{length}", (length,), state_dtype=dtype,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
            add(f"table_{dtype}", (0, 5, 1), prefix_tokens=2, state_dtype=dtype,
                state_indices_width=8, accepted_tokens=[0, 8, 6], HK=2, HV=6,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True, use_dt_bias=False)
            add(f"ungated_{dtype}", (3, 0, 2), state_dtype=dtype, gate_mode="none")
            add(f"empty_{dtype}", (0, 0), prefix_tokens=3, state_dtype=dtype)
            for layout in ("noncontiguous", "head_padded", "block_padded", "head_block_padded", "row_padded"):
                add(f"state_{dtype}_{layout}", (3, 5), state_dtype=dtype, state_layout=layout,
                    HK=2, HV=4, use_qk_l2norm_in_kernel=True)
            add(f"strided_qkv_{dtype}", (5,), state_dtype=dtype,
                input_layout="noncontiguous_qkv_beta_g", use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)
        for dtype in ("bf16", "fp32"):
            add(f"softplus_tail_{dtype}", (5,), HK=2, HV=4, state_dtype=dtype,
                gate_profile="cancelling_tail", beta_profile="zero", scale=8.0,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)
            add(f"overlapping_qkv_{dtype}", (5,), HK=2, HV=4, state_dtype=dtype,
                input_layout="overlapping_qkv", use_qk_l2norm_in_kernel=True)
            add(f"transposed_qkv_{dtype}", (5,), HK=2, HV=4, state_dtype=dtype,
                input_layout="token_head_transposed", use_qk_l2norm_in_kernel=True)
            add(f"all_views_{dtype}", (3, 5), state_dtype=dtype, state_layout="row_padded",
                input_layout="noncontiguous_all", use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
            add(f"extreme_raw_{dtype}", (5,), state_dtype=dtype, gate_profile="extreme",
                beta_profile="extreme", use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
            add(f"zero_qk_{dtype}", (5,), state_dtype=dtype, data_profile="zero_qk",
                use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True)
        add("large_heads_257", HK=257, HV=257, state_dtype="bf16")
        add("large_heads_gqa", HK=129, HV=258, state_dtype="fp32")
        add("many_work_one_core", (5, 3), HK=2, HV=4, state_dtype="fp32", cores=1,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    return cases

