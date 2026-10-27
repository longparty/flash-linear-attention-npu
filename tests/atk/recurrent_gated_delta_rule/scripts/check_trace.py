"""Check actual AIV work/bank records from the separate debug kernel target."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from check_direct import OP, build_inputs, compare, prepare, specs
from reference import recurrent_gated_delta_rule_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cannbot", type=Path, default=Path.home() / "fla_rgdn_workflow/cannbot-skills/plugins-official/catlass-op-generator/catlass-cpp-generator")
    args = parser.parse_args()
    sys.path.insert(0, str(args.cannbot / "skills/catlass-cpp-test/scripts"))
    from compare_precision import compare_case
    torch.set_num_threads(1)
    policy = json.loads((OP / "reference/precision-policy.json").read_text())
    reports = []
    for profile in specs("full"):
        if profile["name"] not in {"many_work_one_core", "length_fp32_17", "table_fp32"}:
            continue
        directory = args.output / profile["name"]
        inputs = build_inputs(dict(profile, seed=20261022), torch.device("cpu"))
        td = prepare(directory, inputs, int(profile.get("cores", 64)))
        with (directory / "host.log").open("w") as log:
            subprocess.run([str(args.binary), str(directory), "fp32", "1", str(args.device)],
                           stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
        trace = np.fromfile(directory / "trace.bin", dtype=np.uint64).reshape(td.vectorCoreNum, 4096, 6)
        observed = {}
        for core, segment in enumerate(trace):
            rows = segment[segment[:, 5] != 0].tolist()
            assert len(rows) < 4096, "trace overflow"
            previous_work = None
            for block_dim, recorded_core, work, token, bank, event in rows:
                assert block_dim == td.vectorCoreNum and recorded_core == core
                assert work % td.vectorCoreNum == core and work < td.b * td.nv
                if previous_work is not None:
                    assert work >= previous_work, "work traversal went backwards"
                previous_work = work
                observed.setdefault(work, []).append((token, bank, event))
        assert set(observed) == set(range(td.b * td.nv)), "missing work"
        recycled_inputs = reused_outputs = empty = 0
        for work, events in observed.items():
            length = profile["seq_lengths"][work // td.nv]
            assert events[0][2] == 1, "work did not start"
            if not length:
                assert len(events) == 1, "empty work issued synchronization"
                empty += 1
                continue
            assert events[-1] == (length, 0, 7), "work did not drain"
            ready = [(token, bank) for token, bank, event in events if event == 3]
            assert ready == [(t, (t // 4) % 2) for t in range(0, length, 4)]
            recycle = [(token, bank) for token, bank, event in events if event == 8]
            assert recycle == [(t, (t // 4) % 2) for t in range(8, length, 4)]
            recycled_inputs += len(recycle)
            for token in range(length):
                lifecycle = [(bank, event) for t, bank, event in events if t == token and event in (4, 5, 6)]
                assert lifecycle == [(token % 2, event) for event in (4, 5, 6)], "output bank lifecycle"
                if token >= 2:
                    reused_outputs += 1
        out = torch.frombuffer(bytearray((directory / "out.bin").read_bytes()), dtype=torch.bfloat16).reshape(inputs["value"].shape)
        state = torch.frombuffer(bytearray((directory / "state_out.bin").read_bytes()), dtype=inputs["state"].dtype).as_strided(inputs["state"].shape, inputs["state"].stride())
        golden = recurrent_gated_delta_rule_reference(**inputs)
        accuracy = compare((out, state), golden, inputs, compare_case, policy, profile["name"])
        assert accuracy["pass"], accuracy
        report = dict(case=profile["name"], block_dim=td.vectorCoreNum, actual_work=len(observed),
                      same_core_multiple_work=max(sum(work % td.vectorCoreNum == core for work in observed)
                                                  for core in range(td.vectorCoreNum)) > 1,
                      input_bank_recycles=recycled_inputs, output_bank_reuses=reused_outputs,
                      empty_work=empty, accuracy_pass=True)
        reports.append(report)
        (directory / "trace.json").write_text(json.dumps(dict(report=report, records=trace[trace[:, :, 5] != 0].tolist()), indent=2))
        print(report, flush=True)
    assert len(reports) == 3 and any(r["same_core_multiple_work"] for r in reports)
    assert any(r["input_bank_recycles"] for r in reports) and any(r["empty_work"] for r in reports)
    (args.output / "summary.json").write_text(json.dumps(dict(cases=3, passed=3, reports=reports), indent=2))


if __name__ == "__main__":
    main()
