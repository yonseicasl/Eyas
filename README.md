# Eyas: Fast and Accurate Power Estimation at Compile Time for Machine Learning Applications

## Setup
### 1. Install [TVM](https://github.com/apache/tvm)
#### Description
* `build_tvm.sh` builds and installs TVM.
#### Usage
####  
```
sh build_tvm.sh
```

### 2. Construct Dataset for Eyas
#### Description
* `eyas/construct_dataset.sh` downloads necessary ML models, and constructs both training and test datasets for Eyas.
* ML models are executed on `<target_hw>`, and`<num_trials>` samples are collected for each ML model.
* `<target_hw>` must match the current hardware environment.
* If `[models]` is provided, the script executes only those specific ML models.
* If model execution underutilizes the target hardware or fails to reach peak power draw, increase the input size within the `eyas/models/training/<model_name>/main.py` and `eyas/models/test/<model_name>/main.py` files.
* Ensure the working directory is set to `eyas` before executing the script.

#### Usage
```
sh construct_dataset.sh <num_trials> <target_hw> [models]
```
#### Examples
```
sh construct_dataset.sh 10000 "llvm -num-cores 16"
sh construct_dataset.sh 20000 "nvidia/geforce-rtx-2080-ti"
sh construct_dataset.sh 30000 "nvidia/jetson-agx-orin-32gb" "bert,gpt2,resnet50,inception_v3"
```

### 3. Train Eyas
#### Description
* `eyas/eyas.py` trains and tests Eyas, the power prediction model.
* After training, the newly trained model is automatically integrated into TVM. 
* Set `<train?>` to 1 to train Eyas, and `<test?>` to 1 to test Eyas. 
* Ensure the working directory is set to `eyas` before executing the script.

#### Usage
```
python3 eyas.py <train?> <test?>
```
#### Examples
```
python3 eyas.py 1 0
python3 eyas.py 0 1
python3 eyas.py 1 1
```

## How to Use
### 1. Configure *TVM + Eyas*
* *TVM + Eyas* is configured with 4 environment variables.
* *TVM + Eyas* aims to minimize the target metric $T = \text{power}^{P} \times \text{latency}^{L}$ for a given ML model, under a user-specified power limit.   
* For example, set $P=0$ and $L=1$ to minimize latency, or set $P=1$ and $L=1$ to minimize energy consumption.
* For example, set `TVMP_POWER_CAP=150` to enable a 150W power limit.

| Environment Variables  |    Supported Values     | Description                                                                                            |
|------------------------|:-----------------------:|--------------------------------------------------------------------------------------------------------|
| TVMP_CONSTRUCT_DATASET |       `0` or `1`        | Disables or enables the dataset construction mode.<br/>Should be set to `0` when optimizing ML models. |
| TVMP_POWER_CAP         | `0` or positive integer | Sets the power limit (in watts) for ML models.<br/>If set to `0`, power capping is disabled.           |
| TVMP_POWER_EXP         | `0` or positive integer | Sets the value of $P$ in the target metric $T$.                                                        |
| TVMP_LATENCY_EXP       | `0` or positive integer | Sets the value of $L$ in the target metric $T$.                                                        |

### 2. Optimize and Deploy ML Models
* TVM + Eyas uses the same API and workflow as TVM for importing, optimizing, and deploying models.
* For detailed usage, refer to the [TVM v0.19.0 documentation](https://github.com/apache/tvm/tree/v0.19.0/docs/).

## Examples
### 1. Model-Level Power Capping
#### Description
* `examples/power_capping.py` minimizes the latency of the GPT-2 model on an Intel Xeon Gold 6346 CPU under a 150W power limit.
* Ensure the working directory is set to `examples` before executing the script.

#### Usage
```
python3 power_capping.py
```


### 2. Layer-Level Energy Minimization
#### Description
* `examples/energy_minimization.py` minimizes the energy consumption of a dense (fully-connected) layer on an NVIDIA Jetson AGX Orin embedded device.
* Ensure the working directory is set to `examples` before executing the script.

#### Usage
```
python3 energy_minimization.py
```
