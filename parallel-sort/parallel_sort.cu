#include "parallel_sort.cuh"

// ------------------------------------------------------------------
// DATA GENERATION: Create realistic market data for testing
// ------------------------------------------------------------------
void generateRandomMarketData(std::vector<MarketData>& data, int size) {
    data.clear();
    data.reserve(size);
    
    // Initialize random number generators with appropriate distributions for financial data
    std::random_device rd;
    std::mt19937 gen(rd());
    // Time distribution spans a full trading day in nanoseconds (HFT-scale precision)
    std::uniform_int_distribution<unsigned long long> time_dist(0, 86400000000000); // nanoseconds in a day
    // Price distribution models a reasonable range for a liquid stock or ETF
    std::uniform_real_distribution<double> price_dist(1000.0, 2000.0);
    // Volume distribution accounts for both small and medium-sized orders
    std::uniform_int_distribution<int> volume_dist(1, 1000);
    std::uniform_int_distribution<int> id_dist(1, 1000000);
    // Trade vs. quote ratio models realistic market data composition
    // Typically there are more quotes than actual trades in real markets
    std::bernoulli_distribution trade_dist(0.3); // 30% trades, 70% quotes
    
    // Generate data points with realistic properties
    for (int i = 0; i < size; i++) {
        unsigned long long timestamp = time_dist(gen);
        double price = price_dist(gen);
        int volume = volume_dist(gen);
        int trade_id = id_dist(gen);
        bool is_trade = trade_dist(gen);
        
        data.emplace_back(timestamp, price, volume, trade_id, is_trade);
    }
}

// ------------------------------------------------------------------
// CPU IMPLEMENTATION: Baseline sequential sort for comparison
// ------------------------------------------------------------------
void cpuSort(std::vector<MarketData>& data) {
    // Standard C++ STL sort using a lambda to sort by timestamp
    // In HFT, market data must be processed in strict time order
    // This establishes our baseline performance for comparison
    std::sort(data.begin(), data.end(), [](const MarketData& a, const MarketData& b) {
        return a.timestamp < b.timestamp;
    });
}

// ------------------------------------------------------------------
// THRUST IMPLEMENTATION: High-level GPU sort using Thrust library
// ------------------------------------------------------------------
void thrustSort(thrust::device_vector<MarketData>& data) {
    // Thrust provides a high-level interface to CUDA algorithms
    // This is analogous to using std::sort, but on the GPU
    // The lambda needs both __host__ and __device__ annotations to work with Thrust
    thrust::sort(data.begin(), data.end(), 
        [] __host__ __device__ (const MarketData& a, const MarketData& b) {
            return a.timestamp < b.timestamp;
        });
    // Note: Thrust handles all memory management and kernel launches internally
    // This provides a good balance of performance and programmer productivity
}

// ------------------------------------------------------------------
// BITONIC SORT KERNEL: Core comparison and swap operation
// ------------------------------------------------------------------
__global__ void bitonicSortKernel(MarketData* data, int j, int k) {
    // Calculate global thread index
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Calculate the comparison partner index using XOR operation
    // This is the key to the bitonic sort algorithm's parallel structure
    int ixj = i ^ j;  // Bitwise XOR to find comparison partner
    
    // Only perform comparison if partner is in higher half of dataset
    // This ensures we don't duplicate work (other thread will handle the case)
    if (ixj > i) {
        // Determine sort direction based on the k-bit of the index
        // - If (i & k) is 0, sort in ascending order
        // - If (i & k) is 1, sort in descending order
        // This alternating pattern is what creates the bitonic sequences
        bool ascending = ((i & k) == 0);
        
        // Perform comparison and conditional swap based on sort direction
        if ((ascending && data[i].timestamp > data[ixj].timestamp) ||
            (!ascending && data[i].timestamp < data[ixj].timestamp)) {
            // Swap elements to ensure correct ordering
            MarketData temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
        // Note: This swap is performed in parallel across many threads
        // Each thread handles exactly one comparison/swap operation
    }
}

// ------------------------------------------------------------------
// BITONIC SORT IMPLEMENTATION: Custom parallel sort algorithm
// ------------------------------------------------------------------
void bitonicSort(thrust::device_vector<MarketData>& d_data) {
    int n = d_data.size();
    
    // Bitonic sort requires a power-of-2 sized array
    // Round up to the next power of 2 for the algorithm to work correctly
    int pow2 = 1;
    while (pow2 < n) pow2 *= 2;
    
    // Save original size and pad array with "infinite" timestamp values
    // This ensures the actual data is sorted correctly without altering algorithm
    int original_size = n;
    d_data.resize(pow2, MarketData(ULLONG_MAX, 0.0, 0, 0, false)); // Pad with "infinity"
    
    // Get raw pointer for CUDA kernel access
    MarketData* raw_ptr = thrust::raw_pointer_cast(d_data.data());
    
    // Configure kernel execution parameters
    // This determines how threads are grouped and dispatched to the GPU
    int threadsPerBlock = 256;  // Number of threads per block (GPU warp alignment)
    int blocksPerGrid = (pow2 + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division
    
    // ------------------------------------------------------------------
    // BITONIC SORT ALGORITHM (PARALLEL IMPLEMENTATION)
    // ------------------------------------------------------------------
    // The algorithm consists of two nested loops:
    // - The outer loop (k) defines the size of the bitonic sequences
    // - The inner loop (j) performs comparisons within each sequence
    //
    // For each pass, we launch thousands of threads to perform comparisons
    // in parallel, with each thread handling a specific element pair
    for (int k = 2; k <= pow2; k *= 2) {         // k doubles each iteration (2,4,8,...)
        for (int j = k / 2; j > 0; j /= 2) {     // j halves each iteration (k/2,k/4,...)
            // Launch GPU kernel with calculated grid dimensions
            bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(raw_ptr, j, k);
            
            // Synchronize to ensure this pass is complete before next iteration
            // This is necessary as each pass depends on the results of the previous one
            cudaDeviceSynchronize();
        }
    }
    
    // Restore original size by removing padding elements
    d_data.resize(original_size);
    
    // At this point, the first 'original_size' elements are sorted as required
}

// ------------------------------------------------------------------
// BENCHMARK FUNCTION: Compare performance of all sorting implementations
// ------------------------------------------------------------------
void benchmarkSorting(int data_size) {
    std::cout << "Benchmarking sorting with " << data_size << " market data points..." << std::endl;
    
    // Generate test data for the benchmarks
    std::vector<MarketData> h_data;
    generateRandomMarketData(h_data, data_size);
    
    // Make copies for each sorting implementation to ensure fair comparison
    std::vector<MarketData> cpu_data = h_data;
    thrust::device_vector<MarketData> thrust_data = h_data;  // Implicitly copies to GPU
    thrust::device_vector<MarketData> bitonic_data = h_data; // Implicitly copies to GPU
    
    // ------------------------------------------------------------------
    // CPU SORT BENCHMARK
    // ------------------------------------------------------------------
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cpuSort(cpu_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "CPU sort completed in " << duration.count() << "ms" << std::endl;
    }
    
    // ------------------------------------------------------------------
    // THRUST SORT BENCHMARK
    // ------------------------------------------------------------------
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        thrustSort(thrust_data);
        
        // Ensure all GPU operations are complete before stopping the timer
        // This is crucial for accurate GPU benchmarking
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Thrust sort completed in " << duration.count() << "ms" << std::endl;
    }
    
    // ------------------------------------------------------------------
    // BITONIC SORT BENCHMARK
    // ------------------------------------------------------------------
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bitonicSort(bitonic_data);
        
        // Ensure all GPU operations are complete before stopping the timer
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Bitonic sort completed in " << duration.count() << "ms" << std::endl;
    }
    
    // ------------------------------------------------------------------
    // RESULT VERIFICATION
    // ------------------------------------------------------------------
    // Copy GPU results back to host for verification
    std::vector<MarketData> thrust_result(thrust_data.size());
    thrust::copy(thrust_data.begin(), thrust_data.end(), thrust_result.begin());
    
    std::vector<MarketData> bitonic_result(bitonic_data.size());
    thrust::copy(bitonic_data.begin(), bitonic_data.end(), bitonic_result.begin());
    
    // Compare CPU (reference) results with GPU implementations
    bool thrust_correct = true;
    bool bitonic_correct = true;
    
    for (size_t i = 0; i < cpu_data.size(); i++) {
        // Check if thrust sort matches CPU sort
        if (cpu_data[i].timestamp != thrust_result[i].timestamp) {
            thrust_correct = false;
            break;
        }
        
        // Check if bitonic sort matches CPU sort
        if (cpu_data[i].timestamp != bitonic_result[i].timestamp) {
            bitonic_correct = false;
            break;
        }
    }
    
    // Report correctness results
    std::cout << "Thrust sort results are " << (thrust_correct ? "correct" : "incorrect") << std::endl;
    std::cout << "Bitonic sort results are " << (bitonic_correct ? "correct" : "incorrect") << std::endl;
    
    // In HFT systems, both performance and correctness are critical
    // Incorrect sorting could lead to trade sequencing errors and financial losses
}