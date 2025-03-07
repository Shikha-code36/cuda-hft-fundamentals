#ifndef PARALLEL_SORT_CUH
#define PARALLEL_SORT_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

// Market data structure
struct MarketData {
    unsigned long long timestamp; // nanosecond timestamp
    double price;
    int volume;
    int trade_id;
    bool is_trade; // true for trades, false for quotes
    
    // Default constructor
    __host__ __device__
    MarketData() : timestamp(0), price(0.0), volume(0), trade_id(0), is_trade(false) {}
    
    // Constructor
    __host__ __device__
    MarketData(unsigned long long ts, double p, int vol, int id, bool trade) 
        : timestamp(ts), price(p), volume(vol), trade_id(id), is_trade(trade) {}
};

// Function declarations
void generateRandomMarketData(std::vector<MarketData>& data, int size);
void cpuSort(std::vector<MarketData>& data);
void thrustSort(thrust::device_vector<MarketData>& data);
void bitonicSort(thrust::device_vector<MarketData>& data);
void benchmarkSorting(int data_size);

// GPU kernel for bitonic sort
__global__ void bitonicSortKernel(MarketData* data, int j, int k);

#endif // PARALLEL_SORT_CUH