cmake_minimum_required(VERSION 3.18)
project(cuda_finance LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Order Book executable
add_executable(order_book
    order-book/order_book.cu
    order-book/order_book.cuh
    order-book/main.cu
)

# Parallel Sort executable
add_executable(parallel_sort
    parallel-sort/parallel_sort.cu
    parallel-sort/parallel_sort.cuh
    parallel-sort/main.cu
)

# Add important CUDA compiler flags including extended lambda support
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extended-lambda")

# Set CUDA architecture flags - adjust based on your GPU
set_target_properties(order_book PROPERTIES 
    CUDA_ARCHITECTURES "50;52;60;61;70;75;80"
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(parallel_sort PROPERTIES 
    CUDA_ARCHITECTURES "50;52;60;61;70;75;80"
    CUDA_SEPARABLE_COMPILATION ON
)

# Add compiler options
target_compile_options(order_book PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(parallel_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Generate compile_commands.json for better VS Code integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)