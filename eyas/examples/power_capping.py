# Power capping, GPT-2, CPU
import os
import onnx
import time
import numpy as np
import tvm
from tvm import relax
from tvm.relay.frontend import from_onnx
from tvm.relax.testing import relay_translator

# Set configuration
total_trials = 20000
target = "llvm -num-cores 32"

# Load onnx model
onnx_model = onnx.load("../models/test/gpt2/decoder_model.onnx")

# Convert to tvm relax module
mod, params = from_onnx(model=onnx_model, shape={'input_ids': (64, 16), 'attention_mask': (64, 16)})
mod = relay_translator.from_relay(func=mod["main"], target=target, relay_params=params)

# Baseline
os.environ["TVMP_CONSTRUCT_DATASET"] = "0"
os.environ["TVMP_POWER_CAP"] = "0"
os.environ["TVMP_POWER_EXP"] = "0"
os.environ["TVMP_LATENCY_EXP"] = "1"
work_dir = "power_capping_baseline"
mod_baseline = relax.get_pipeline("static_shape_tuning", work_dir=work_dir, target=target, total_trials=total_trials)(mod)

# With Eyas
os.environ["TVMP_POWER_CAP"] = "150"
work_dir = "power_capping_eyas"
mod_eyas = relax.get_pipeline("static_shape_tuning", work_dir=work_dir, target=target, total_trials=total_trials)(mod)

# Allocate input data
dev = tvm.device(target)
input_ids = tvm.nd.array(np.random.rand(64, 16).astype("int64"), dev)
attention_mask = tvm.nd.array(np.random.rand(64, 16).astype("int64"), dev)

# Build both modules
ex_baseline = relax.build(mod_baseline, target=target)
vm_baseline = relax.VirtualMachine(ex_baseline, dev)
ex_eyas = relax.build(mod_eyas, target=target)
vm_eyas = relax.VirtualMachine(ex_eyas, dev)

# Measure power consumption and latency of each module
MEASURE_TIME = 60
def read_energy():
    with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
        return int(f.read().strip())

for desc, vm in (('Baseline', vm_baseline), ('TVM + Eyas', vm_eyas)):
    start_time = time.time()
    start_energy = read_energy()
    num_executions = 0
    while time.time() - start_time < MEASURE_TIME:
        cpu_out = vm["main"](input_ids, attention_mask)
        num_executions += 1
    end_time = time.time()
    end_energy = read_energy()

    total_energy = (end_energy - start_energy) * 1e-6
    total_time = end_time - start_time
    average_power = total_energy / total_time
    average_latency = total_time / num_executions

    print()
    print(f"--------------- {desc} ---------------")
    print(f"Average Power Consumption (W): {average_power:.1f}")
    print(f"Average Latency (s): {average_latency:.1f}")

# Reset environment variables
os.environ["TVMP_CONSTRUCT_DATASET"] = "0"
os.environ["TVMP_POWER_CAP"] = "0"
os.environ["TVMP_POWER_EXP"] = "0"
os.environ["TVMP_LATENCY_EXP"] = "1"
