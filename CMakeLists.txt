# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.16)

# Define the project name and specify C++ as the language
project(MicrogradCpp VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake that our header files are in the 'include' directory.
# This makes #include "micrograd/value.hpp" work correctly.
include_directories(include)

# --- Define Executables ---
# An executable is a runnable program. We'll create one for each test file.

# 1. Define the 'test_value' executable
add_executable(
    test_value
    tests/test_value.cpp
    src/value.cpp
)

# 2. Define the 'test_nn' executable
add_executable(
    test_nn
    tests/test_nn.cpp
    src/value.cpp
    src/neuron.cpp
    src/layer.cpp
    src/mlp.cpp
)

# --- Optional: Print a message after configuration ---
message(STATUS "Project configured. Ready to build with 'make' or 'cmake --build .'")