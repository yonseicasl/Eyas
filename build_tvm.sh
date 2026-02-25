#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Create tvm/build and cd to it
echo "--- Creating build directory ---"
mkdir -p build && cd build

# 2. Copy config.cmake from the root (parent) directory
echo "--- Copying config.cmake ---"
if [ -f "../cmake/config.cmake" ]; then
    cp ../cmake/config.cmake .
else
    echo "Error: config.cmake not found in the root directory."
    exit 1
fi

# 3. Append default settings
{
    echo ""
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)"
    echo 'set(USE_LLVM "llvm-config --ignore-libllvm --link-static")'
    echo "set(HIDE_PRIVATE_SYMBOLS ON)"
    echo "set(USE_METAL  OFF)"
    echo "set(USE_VULKAN OFF)"
    echo "set(USE_OPENCL OFF)"
    echo "set(USE_CUTLASS OFF)"
} >> config.cmake

# 4. Check for nvcc and append CUDA settings
if which nvcc > /dev/null 2>&1; then
    echo "NVCC found. Enabling CUDA support..."
    {
        echo "set(USE_CUDA ON)"
        echo "set(USE_CUBLAS ON)"
        echo "set(USE_CUDNN ON)"
    } >> config.cmake
else
    echo "NVCC not found. Disabling CUDA support..."
    {
        echo "set(USE_CUDA OFF)"
        echo "set(USE_CUBLAS OFF)"
        echo "set(USE_CUDNN OFF)"
    } >> config.cmake
fi

# 5. Run cmake and build
echo "--- Starting CMake and Build ---"
cmake .. && cmake --build . --parallel $(nproc)

# 6. Install python package
echo "--- Installing TVM ---"
cd .. && pip3 install -e ./python

# 7. Validate installation
echo "--- Validating installation ---"
python3 -c "import tvm; print(tvm.__version__)"
