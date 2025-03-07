# Market Data Parallel Sort

This component demonstrates high-performance sorting of financial market data using different parallel algorithms and compares them with sequential CPU sorting.

## Implementation Overview

The implementation provides three different sorting approaches:
1. **CPU Sort**: Standard sequential sort using STL
2. **Thrust Sort**: High-level GPU sorting using the Thrust library
3. **Bitonic Sort**: Custom low-level GPU sorting implementation

## Technical Details

### Market Data Structure

The core data being sorted is time-series market data with these properties:
- Nanosecond-precision timestamps (critical for HFT)
- Price and volume information
- Trade and quote flags

### Sorting Algorithms

#### CPU Sort
- Uses standard `std::sort` with custom comparator
- Provides baseline performance for comparison
- O(n log n) complexity with optimized branch prediction

#### Thrust Sort
- High-level abstraction using Thrust parallel algorithms
- Automatically selects optimal GPU sorting implementation
- Handles memory transfers implicitly
- Balances programmer productivity and performance

#### Bitonic Sort
- Custom implementation of the bitonic sort network
- Highly parallelizable O(log² n) algorithm
- Each comparison-swap operation runs on a separate CUDA thread
- Explicitly handles power-of-2 padding requirements

## Performance Analysis

![Parallel Sort Benchmark](https://github.com/Shikha-code36/cuda-hft-fundamentals/blob/main/screenshots/parallel_sort_execution.png)

In benchmark tests with 1 million data points:
- CPU Sort: ~560ms
- Thrust Sort: ~20ms (28× speedup)
- Bitonic Sort: ~97ms (5.8× speedup)

### Why GPU Sorting Outperforms CPU
1. **Massive Parallelism**: Thousands of comparisons happen simultaneously
2. **Memory Bandwidth**: GPUs have significantly higher memory bandwidth
3. **Specialized Hardware**: CUDA cores are optimized for data-parallel operations

### Thrust vs. Bitonic Performance
- **Thrust** generally outperforms our custom bitonic implementation due to:
  - Advanced optimization techniques
  - Radix sort for appropriate data types
  - Memory transfer optimizations
- **Bitonic Sort** demonstrates the core principles of GPU parallelism but with:
  - More explicit control
  - Educational value for understanding parallel sorting

## Use Cases in HFT

Fast sorting of market data is critical for:
1. **Time-Series Analysis**: Processing market data in chronological order
2. **Event Reconstruction**: Recreating the exact sequence of market events
3. **Backtesting**: Historical simulations of trading strategies
4. **Market Surveillance**: Detecting irregular trading patterns

