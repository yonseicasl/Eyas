import sys
import torch
from torch.export import export
from torchvision.models import resnet50, ResNet50_Weights
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Read configuration
if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <num_trials> <target>")
    sys.exit(1)
TOTAL_TRIALS, TARGET = int(sys.argv[1]), sys.argv[2]

# Load torch model
torch_model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

# Convert to tvm relax module
example_args = (torch.randn(64, 3, 224, 224, dtype=torch.float32),)
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(exported_program, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)

# Run metaschedule
mod = relax.get_pipeline("static_shape_tuning", work_dir="tuning_logs", target=TARGET, total_trials=TOTAL_TRIALS)(mod)
