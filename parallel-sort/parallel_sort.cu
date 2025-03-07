#include "parallel_sort.cuh"

// Generate random market data
void generateRandomMarketData(std::vector<MarketData>& data, int size) {
    data.clear();
    data.reserve(size);
    
    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long> time_dist(0, 86400000000000); // nanoseconds in a day
    std::uniform_real_distribution<double> price_dist(1000.0, 2000.0);
    std::uniform_int_distribution<int> volume_dist(1, 1000);
    std::uniform_int_distribution<int> id_dist(1, 1000000);
    std::bernoulli_distribution trade_dist(0.3); // 30% trades, 70% quotes
    
    for (int i = 0; i < size; i++) {
        unsigned long long timestamp = time_dist(gen);
        double price = price_dist(gen);
        int volume = volume_dist(gen);
        int trade_id = id_dist(gen);
        bool is_trade = trade_dist(gen);
        
        data.emplace_back(timestamp, price, volume, trade_id, is_trade);
    }
}

// CPU sort implementation
void cpuSort(std::vector<MarketData>& data) {
    std::sort(data.begin(), data.end(), [](const MarketData& a, const MarketData& b) {
        return a.timestamp < b.timestamp;
    });
}

// Thrust sort implementation
void thrustSort(thrust::device_vector<MarketData>& data) {
    thrust::sort(data.begin(), data.end(), 
        [] __host__ __device__ (const MarketData& a, const MarketData& b) {
            return a.timestamp < b.timestamp;
        });
}

// Bitonic sort kernel
__global__ void bitonicSortKernel(MarketData* data, int j, int k) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int ixj = i ^ j;
    
    if (ixj > i) {
        // Sort in ascending or descending order based on the direction bit
        bool ascending = ((i & k) == 0);
        
        if ((ascending && data[i].timestamp > data[ixj].timestamp) ||
            (!ascending && data[i].timestamp < data[ixj].timestamp)) {
            // Swap elements
            MarketData temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

// Bitonic sort implementation
void bitonicSort(thrust::device_vector<MarketData>& d_data) {
    int n = d_data.size();
    
    // Round up to the next power of 2
    int pow2 = 1;
    while (pow2 < n) pow2 *= 2;
    
    // Resize data to power of 2 with "infinite" timestamp values
    int original_size = n;
    d_data.resize(pow2, MarketData(ULLONG_MAX, 0.0, 0, 0, false));
    
    // Get raw pointer to device data
    MarketData* raw_ptr = thrust::raw_pointer_cast(d_data.data());
    
    // Determine grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (pow2 + threadsPerBlock - 1) / threadsPerBlock;
    
    // Perform bitonic sort
    for (int k = 2; k <= pow2; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(raw_ptr, j, k);
            cudaDeviceSynchronize();
        }
    }
    
    // Resize back to original size
    d_data.resize(original_size);
}

// Benchmark function
void benchmarkSorting(int data_size) {
    std::cout << "Benchmarking sorting with " << data_size << " market data points..." << std::endl;
    
    // Generate random data
    std::vector<MarketData> h_data;
    generateRandomMarketData(h_data, data_size);
    
    // Copy for each algorithm
    std::vector<MarketData> cpu_data = h_data;
    thrust::device_vector<MarketData> thrust_data = h_data;
    thrust::device_vector<MarketData> bitonic_data = h_data;
    
    // CPU sort
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cpuSort(cpu_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "CPU sort completed in " << duration.count() << "ms" << std::endl;
    }
    
    // Thrust sort
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        thrustSort(thrust_data);
        
        // Ensure all CUDA operations are complete
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Thrust sort completed in " << duration.count() << "ms" << std::endl;
    }
    
    // Bitonic sort
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bitonicSort(bitonic_data);
        
        // Ensure all CUDA operations are complete
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Bitonic sort completed in " << duration.count() << "ms" << std::endl;
    }
    
    // Verify results
    std::vector<MarketData> thrust_result(thrust_data.size());
    thrust::copy(thrust_data.begin(), thrust_data.end(), thrust_result.begin());
    
    std::vector<MarketData> bitonic_result(bitonic_data.size());
    thrust::copy(bitonic_data.begin(), bitonic_data.end(), bitonic_result.begin());
    
    bool thrust_correct = true;
    bool bitonic_correct = true;
    
    for (size_t i = 0; i < cpu_data.size(); i++) {
        if (cpu_data[i].timestamp != thrust_result[i].timestamp) {
            thrust_correct = false;
            break;
        }
        
        if (cpu_data[i].timestamp != bitonic_result[i].timestamp) {
            bitonic_correct = false;
            break;
        }
    }
    
    std::cout << "Thrust sort results are " << (thrust_correct ? "correct" : "incorrect") << std::endl;
    std::cout << "Bitonic sort results are " << (bitonic_correct ? "correct" : "incorrect") << std::endl;
}