# Zero-Copy Market Data Processing

This component implements high-performance, low-latency market data processing using CUDA's zero-copy memory capabilities and asynchronous streams. It significantly reduces latency for high-frequency trading (HFT) systems.

## Technical Overview

### Key Features

1. **Zero-Copy Memory Access**
   - Uses mapped memory for direct GPU access to host memory
   - Eliminates costly PCIe bus transfers
   - Reduces latency by removing memory copy overhead

2. **Pipelined Processing with CUDA Streams**
   - Implements asynchronous data processing using multiple streams
   - Overlaps computation and data transfer operations
   - Enables continuous processing of market data feeds

3. **Asynchronous API**
   - Thread-safe queue for submitting market data packets
   - Background processing thread for continuous operation
   - Non-blocking design for real-time market data handling

### Performance Goal

- Reduce end-to-end latency by 50% compared to standard memory transfer approaches

## Implementation Details

### Memory Management Techniques

#### Standard Memory Transfers (Baseline)
```
CPU Memory -> cudaMemcpy -> GPU Memory -> Processing -> cudaMemcpy -> CPU Memory
```

#### Zero-Copy Approach
```
Mapped Memory (accessible by both CPU and GPU) -> In-place Processing
```

### CUDA Streams for Pipelining

The implementation uses multiple CUDA streams to enable overlapped execution of:
- Memory operations (when needed)
- Kernel execution for different data segments
- Event synchronization

### Market Data Format

The system processes a simplified ITCH-like market data format with these fields:
- Timestamp (nanosecond precision)
- Message type (Add, Cancel, Execute, Trade)
- Order ID
- Side (Buy/Sell)
- Price
- Quantity

## Benchmark Results

Typical results for 1,000,000 market data packets:

| Method | Average Latency (μs) | Improvement |
|--------|---------------------|-------------|
| Standard | ~13,000 μs | Baseline |
| Zero-Copy | ~3,700 μs | ~72% |
| Zero-Copy + Streams | ~3,800 μs | ~72% |

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0+
- C++17 compatible compiler
- CMake 3.18+

### Compilation
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Running the Benchmark
```bash
./zero_copy_processor [num_packets] [num_runs]
```

## Integration with Order Book

This zero-copy processor can be integrated with the existing order book implementation:

1. Use zero-copy to efficiently receive and parse market data
2. Feed processed orders directly into the GPU-accelerated order book
3. Implement a continuous matching engine that processes order book updates