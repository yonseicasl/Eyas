# Energy minimization, dense layer, embedded device
import os
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import re
import pandas as pd

# Check cuda device
print(f"CUDA: {tvm.cuda().exist}")
print(f"Device name: {tvm.cuda(0).device_name}")

# Define a IRModule that contains only one dense layer
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def dense3(lv76: T.Buffer((T.int64(30), T.int64(5120)), "float32"),
               B: T.Buffer((T.int64(1280), T.int64(5120)), "float32"),
               T_matmul_NT: T.Buffer((T.int64(30), T.int64(1280)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(30), T.int64(1280), T.int64(5120)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv76[v_i0, v_k], B[v_i1, v_k])
                T.writes(T_matmul_NT[v_i0, v_i1])
                with T.init():
                    T_matmul_NT[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NT[v_i0, v_i1] = T_matmul_NT[v_i0, v_i1] + lv76[v_i0, v_k] * B[v_i1, v_k]

    @R.function
    def main(lv76: R.Tensor((30, 5120), dtype="float32"),
             B: R.Tensor((1280, 5120), dtype="float32"),
             ) -> R.Tensor((30, 1280), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.dense3, (lv76, B,), out_sinfo=R.Tensor((30, 1280), dtype="float32"))
            gv: R.Tensor((30, 1280), dtype="float32") = lv
            R.output(gv)
        return gv

# Create a IRModule and run metaschedule on it
mod = Module
total_trials = 8000
target = tvm.target.Target("nvidia/jetson-agx-orin-32gb")

# Baseline
os.environ["TVMP_CONSTRUCT_DATASET"] = "0"
os.environ["TVMP_POWER_CAP"] = "0"
os.environ["TVMP_POWER_EXP"] = "0"
os.environ["TVMP_LATENCY_EXP"] = "1"
work_dir = "energy_minimization_baseline"
mod_1 = relax.get_pipeline("static_shape_tuning", work_dir=work_dir, target=target, total_trials=total_trials)(mod)

# With Eyas
os.environ["TVMP_POWER_EXP"] = "1"
work_dir = "energy_minimization_eyas"
mod_2 = relax.get_pipeline("static_shape_tuning", work_dir=work_dir, target=target, total_trials=total_trials)(mod)

# Parse results
reg_exp = r"\s*(\d+)\s*\|\s*(.+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+)\s*\|\s*(.+)\s*\|\s*(.+)\s*\|\s*(.+)\s*\|\s*(\d+)\s*\|\s*(.+)\s*"
with open('./energy_minimization_baseline/logs/tvm.meta_schedule.logging.task_scheduler.log') as f:
    for line in f.readlines()[-10:]:
        if match := re.match(reg_exp, line):
            baseline_power, baseline_latency = float(match.group(5)), float(match.group(6))
            baseline_energy = baseline_power * baseline_latency / 1000 # Convert to mJ
            break

with open('./energy_minimization_eyas/logs/tvm.meta_schedule.logging.task_scheduler.log') as f:
    for line in f.readlines()[-10:]:
        if match := re.match(reg_exp, line):
            eyas_power, eyas_latency = float(match.group(5)), float(match.group(6))
            eyas_energy = eyas_power * eyas_latency / 1000 # Convert to mJ
            break

data = {
    "Target Metric":        ["Min. latency", "Min. energy", "-"],
    "Total Trials":         [total_trials, total_trials, "-"],
    "Final Latency (us)":   [f"{baseline_latency:.1f}", f"{eyas_latency:.1f}", f"{(eyas_latency / baseline_latency) * 100 - 100:+.1f}%"],
    "Final Power (W)" :     [f"{baseline_power:.1f}", f"{eyas_power:.1f}", f"{(eyas_power / baseline_power) * 100 - 100:+.1f}%"],
    "Final Energy (mJ)":    [f"{baseline_energy:.1f}", f"{eyas_energy:.1f}", f"{(eyas_energy / baseline_energy) * 100 - 100:+.1f}%"],
}

df = pd.DataFrame(data, index=["Baseline", "TVM + Eyas", "Difference"])
table_str = df.to_markdown(tablefmt="grid")
table_width = len(table_str.split('\n')[0])
title = "Layer-Level Energy Minimization Results"
print()
print(title.center(table_width))
print(table_str)

# Reset environment variables
os.environ["TVMP_CONSTRUCT_DATASET"] = "0"
os.environ["TVMP_POWER_CAP"] = "0"
os.environ["TVMP_POWER_EXP"] = "0"
os.environ["TVMP_LATENCY_EXP"] = "1"
