import sys
import onnx
import tvm
from tvm import relax
from tvm.relay.frontend import from_onnx
from tvm.relax.testing import relay_translator

# Read configuration
if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <num_trials> <target>")
    sys.exit(1)
TOTAL_TRIALS, TARGET = int(sys.argv[1]), sys.argv[2]

# Load onnx model
onnx_model = onnx.load("./model.onnx")

# Convert to tvm relax module
mod, params = from_onnx(model=onnx_model)
try:
    mod_1 = relay_translator.from_relay(func=mod["main"], target=TARGET, relay_params=params)
    # Run metaschedule
    mod_1 = relax.get_pipeline("static_shape_tuning", work_dir="tuning_logs", target=TARGET, total_trials=TOTAL_TRIALS)(mod_1)
except:
    print("Conversion failed. Retrying...")
    mod_2 = relay_translator.from_relay(func=mod["main"], target="llvm", relay_params=params)
    # Run metaschedule
    mod_2 = relax.get_pipeline("static_shape_tuning", work_dir="tuning_logs_2", target=TARGET, total_trials=TOTAL_TRIALS)(mod_2)
