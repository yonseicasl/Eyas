import sys
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# Read configuration
if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <num_trials> <target>")
    sys.exit(1)
TOTAL_TRIALS, TARGET = int(sys.argv[1]), sys.argv[2]

# Load onnx model
onnx_model = onnx.load("./model.onnx")

# Convert to tvm relax module
mod = from_onnx(model=onnx_model, shape_dict={"input_ids": [64, 64], "attention_mask": [64, 64]})

# Run metaschedule
mod = relax.get_pipeline("static_shape_tuning", work_dir="tuning_logs", target=TARGET, total_trials=TOTAL_TRIALS)(mod)
