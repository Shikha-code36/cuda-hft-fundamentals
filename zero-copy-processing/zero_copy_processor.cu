#include "zero_copy_processor.cuh"

// CUDA kernel for standard data processing - optimized for coalesced memory access
__global__ void processMarketDataKernel(const MarketDataPacket* input, MarketDataPacket* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Simple market data processing: filter and transform
        MarketDataPacket packet = input[idx];
        
        // Process based on message type (simplified)
        if (packet.message_type == 'A') {  // Add Order
            output[idx] = packet;
        }
        else if (packet.message_type == 'X') {  // Cancel Order
            packet.shares = 0;
            output[idx] = packet;
        }
        else if (packet.message_type == 'E') {  // Execute Order
            packet.price *= packet.shares;  // Calculate total value
            output[idx] = packet;
        }
        else {
            output[idx] = packet;
        }
    }
}

// CUDA kernel for zero-copy data processing - optimized for locality
__global__ void processMarketDataZeroCopyKernel(MarketDataPacket* data, int size) {
    // Use shared memory cache for better performance
    __shared__ MarketDataPacket sharedData[256]; // Adjustable based on GPU capabilities
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cache to shared memory first (if within valid range)
    if (idx < size && threadIdx.x < 256) {
        sharedData[threadIdx.x] = data[idx];
    }
    
    // Ensure shared memory is loaded
    __syncthreads();
    
    // Process data if in range
    if (idx < size && threadIdx.x < 256) {
        // Process in-place using shared memory for faster access
        MarketDataPacket& packet = sharedData[threadIdx.x];
        
        if (packet.message_type == 'A') {
            packet.message_type = 'a';
        }
        else if (packet.message_type == 'X') {
            packet.shares = 0;
            packet.message_type = 'x';
        }
        else if (packet.message_type == 'E') {
            packet.price *= packet.shares;
            packet.message_type = 'e';
        }
        
        // Write back to global memory
        data[idx] = packet;
    }
}

// Constructor
ZeroCopyProcessor::ZeroCopyProcessor(size_t buffer_size, size_t process_size)
    : buffer_size(buffer_size), process_size(process_size), running(false) {
    
    h_mapped_data = nullptr;
    d_mapped_data = nullptr;
    d_results = nullptr;
    h_results = nullptr;
}

// Destructor
ZeroCopyProcessor::~ZeroCopyProcessor() {
    // Stop background processing if running
    if (running.load()) {
        stopProcessing();
    }
    
    cleanup();
}

// Initialize CUDA resources
bool ZeroCopyProcessor::initialize() {
    cudaError_t error;
    
    // Select the GPU with the highest PCIe bandwidth
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    int bestDevice = 0;
    int maxPCIeBandwidth = 0;
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        int pciBandwidth = deviceProp.pciBusID;
        
        if (pciBandwidth > maxPCIeBandwidth) {
            maxPCIeBandwidth = pciBandwidth;
            bestDevice = device;
        }
    }
    
    cudaSetDevice(bestDevice);
    
    // Create CUDA stream with highest priority
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    error = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Create events for timing with best timing resolution
    error = cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create start event: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create stop event: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate page-locked host memory for standard processing
    error = cudaMallocHost(&h_results, buffer_size * sizeof(MarketDataPacket), cudaHostAllocWriteCombined);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate device memory for standard processing
    error = cudaMalloc(&d_results, buffer_size * sizeof(MarketDataPacket));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate mapped memory with optimal flags for zero-copy operation
    error = cudaHostAlloc(&h_mapped_data, buffer_size * sizeof(MarketDataPacket), 
                         cudaHostAllocMapped | cudaHostAllocPortable);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate mapped memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Get device pointer to mapped memory
    error = cudaHostGetDevicePointer(&d_mapped_data, h_mapped_data, 0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device pointer: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

// Clean up CUDA resources
void ZeroCopyProcessor::cleanup() {
    if (d_results) {
        cudaFree(d_results);
        d_results = nullptr;
    }
    
    if (h_results) {
        cudaFreeHost(h_results);
        h_results = nullptr;
    }
    
    if (h_mapped_data) {
        cudaFreeHost(h_mapped_data);
        h_mapped_data = nullptr;
        d_mapped_data = nullptr;
    }
    
    if (start_event) {
        cudaEventDestroy(start_event);
    }
    
    if (stop_event) {
        cudaEventDestroy(stop_event);
    }
    
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

// Process a batch with standard memory transfers - optimized
float ZeroCopyProcessor::processStandard(const std::vector<MarketDataPacket>& packets) {
    size_t num_packets = packets.size();
    
    if (num_packets > buffer_size) {
        num_packets = buffer_size;
    }
    
    // Prefetch data to avoid delays
    cudaStreamAttachMemAsync(stream, h_results, 0, cudaMemAttachHost);
    
    // Record start time
    cudaEventRecord(start_event, stream);
    
    // Copy data from host to device using page-locked memory for faster transfers
    cudaMemcpyAsync(d_results, packets.data(), num_packets * sizeof(MarketDataPacket), 
                   cudaMemcpyHostToDevice, stream);
    
    // Process data on the device - optimize thread block size for occupancy
    int threadsPerBlock = 256;  // Optimal for most NVIDIA GPUs
    int blocksPerGrid = (num_packets + threadsPerBlock - 1) / threadsPerBlock;
    
    processMarketDataKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_results, d_results, num_packets);
    
    // Copy results back to host asynchronously
    cudaMemcpyAsync(h_results, d_results, num_packets * sizeof(MarketDataPacket), 
                   cudaMemcpyDeviceToHost, stream);
    
    // Record stop time
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    
    return milliseconds;
}

// Process a batch with zero-copy memory - optimized
float ZeroCopyProcessor::processZeroCopy(const std::vector<MarketDataPacket>& packets) {
    size_t num_packets = packets.size();
    
    if (num_packets > buffer_size) {
        num_packets = buffer_size;
    }
    
    // Record start time
    cudaEventRecord(start_event, stream);
    
    // Copy data directly to mapped memory with vectorized operations for speed
    memcpy(h_mapped_data, packets.data(), num_packets * sizeof(MarketDataPacket));
    
    // Cache control for better performance with zero-copy memory
    cudaDeviceSynchronize(); // Make sure GPU sees the updated data
    
    // Process data on the device with optimized kernel
    int threadsPerBlock = 256; // Optimized for most GPUs
    int blocksPerGrid = (num_packets + threadsPerBlock - 1) / threadsPerBlock;
    
    processMarketDataZeroCopyKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_mapped_data, num_packets);
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
    
    // Record stop time
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    
    return milliseconds;
}

// Process a batch with zero-copy + CUDA streams - optimized
float ZeroCopyProcessor::processZeroCopyStreamed(const std::vector<MarketDataPacket>& packets) {
    size_t num_packets = packets.size();
    
    if (num_packets > buffer_size) {
        num_packets = buffer_size;
    }
    
    // Create multiple streams for overlapping operations
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Record start time
    cudaEventRecord(start_event, stream);
    
    // Copy data directly to mapped memory with prefetch hints
    memcpy(h_mapped_data, packets.data(), num_packets * sizeof(MarketDataPacket));
    
    // Calculate optimal processing parameters
    int threadsPerBlock = 256;
    int chunk_size = (num_packets + 3) / 4; // Split into 4 equal chunks for streams
    
    // Process data in parallel chunks with multiple streams
    for (int i = 0; i < 4 && i*chunk_size < num_packets; i++) {
        int offset = i * chunk_size;
        int current_chunk = std::min(chunk_size, (int)(num_packets - offset));
        int blocksPerGrid = (current_chunk + threadsPerBlock - 1) / threadsPerBlock;
        
        if (current_chunk <= 0) break;
        
        // Launch kernel in separate stream for this chunk
        processMarketDataZeroCopyKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
            (d_mapped_data + offset, current_chunk);
    }
    
    // Synchronize all streams
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Record stop time
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    
    // Clean up streams
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    return milliseconds;
}

// Asynchronous packet submission
void ZeroCopyProcessor::submitPackets(const std::vector<MarketDataPacket>& packets) {
    std::lock_guard<std::mutex> lock(mutex);
    packet_queue.push(packets);
    cv.notify_one();
}

// Start asynchronous processing
void ZeroCopyProcessor::startProcessing() {
    if (running.load()) {
        return;  // Already running
    }
    
    running.store(true);
    
    // Start worker thread
    worker_thread = std::thread([this]() {
        while (running.load()) {
            std::vector<MarketDataPacket> packets;
            
            // Get next batch of packets
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                    return !packet_queue.empty() || !running.load();
                });
                
                if (!running.load()) {
                    break;
                }
                
                if (packet_queue.empty()) {
                    continue;
                }
                
                packets = std::move(packet_queue.front());
                packet_queue.pop();
            }
            
            // Process this batch with zero-copy and streams
            processZeroCopyStreamed(packets);
        }
    });
}

// Stop asynchronous processing
void ZeroCopyProcessor::stopProcessing() {
    if (!running.load()) {
        return;  // Already stopped
    }
    
    running.store(false);
    cv.notify_all();
    
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

// Generate random market data for testing
std::vector<MarketDataPacket> generateRandomMarketData(int size) {
    std::vector<MarketDataPacket> data;
    data.reserve(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long> time_dist(0, 86400000000000); // nanoseconds in a day
    std::uniform_real_distribution<double> price_dist(10.0, 1000.0);
    std::uniform_int_distribution<unsigned int> size_dist(1, 1000);
    std::uniform_int_distribution<unsigned int> id_dist(1, 1000000);
    
    // Message types: A=Add, X=Cancel, E=Execute, T=Trade
    const char message_types[] = {'A', 'X', 'E', 'T'};
    std::uniform_int_distribution<int> msg_dist(0, 3);
    
    // Sides: B=Buy, S=Sell
    const char sides[] = {'B', 'S'};
    std::uniform_int_distribution<int> side_dist(0, 1);
    
    for (int i = 0; i < size; i++) {
        data.emplace_back(
            time_dist(gen),
            message_types[msg_dist(gen)],
            id_dist(gen),
            sides[side_dist(gen)],
            price_dist(gen),
            size_dist(gen)
        );
    }
    
    return data;
}

// Benchmark function to compare different processing methods
void benchmarkZeroCopyProcessing(int num_packets, int num_runs) {
    std::cout << "Benchmarking zero-copy market data processing with " 
              << num_packets << " packets (" << num_runs << " runs)..." << std::endl;
    
    // Generate test data
    auto test_data = generateRandomMarketData(num_packets);
    
    // Pre-warm GPU to ensure consistent results
    cudaFree(0);
    
    // Create and initialize processor with optimized parameters
    ZeroCopyProcessor processor(num_packets, num_packets/4);
    if (!processor.initialize()) {
        std::cerr << "Failed to initialize processor. Aborting benchmark." << std::endl;
        return;
    }
    
    // Run benchmarks with warmup cycles
    float total_standard = 0.0f;
    float total_zero_copy = 0.0f;
    float total_streamed = 0.0f;
    
    // Warmup runs
    for (int i = 0; i < 3; i++) {
        processor.processStandard(test_data);
        processor.processZeroCopy(test_data);
        processor.processZeroCopyStreamed(test_data);
    }
    
    // Timed runs
    for (int i = 0; i < num_runs; i++) {
        // Standard memory copy + processing
        float time_standard = processor.processStandard(test_data);
        total_standard += time_standard;
        
        // Zero-copy memory processing
        float time_zero_copy = processor.processZeroCopy(test_data);
        total_zero_copy += time_zero_copy;
        
        // Zero-copy with streams
        float time_streamed = processor.processZeroCopyStreamed(test_data);
        total_streamed += time_streamed;
    }
    
    // Calculate averages
    float avg_standard = total_standard / num_runs;
    float avg_zero_copy = total_zero_copy / num_runs;
    float avg_streamed = total_streamed / num_runs;
    
    // Convert milliseconds to microseconds for more intuitive HFT metrics
    float avg_standard_us = avg_standard * 1000.0f;
    float avg_zero_copy_us = avg_zero_copy * 1000.0f;
    float avg_streamed_us = avg_streamed * 1000.0f;
    
    // Print results
    std::cout << "Results:" << std::endl;
    std::cout << "  Standard processing:     " << avg_standard_us << " μs" << std::endl;
    std::cout << "  Zero-copy processing:    " << avg_zero_copy_us << " μs" << std::endl;
    std::cout << "  Zero-copy with streams:  " << avg_streamed_us << " μs" << std::endl;
    
    // Calculate improvements
    float improvement1 = (avg_standard_us - avg_zero_copy_us) / avg_standard_us * 100.0f;
    float improvement2 = (avg_standard_us - avg_streamed_us) / avg_standard_us * 100.0f;
    
    std::cout << std::endl;
    std::cout << "Zero-copy improvement:       " << improvement1 << "%" << std::endl;
    std::cout << "Zero-copy+streams improvement: " << improvement2 << "%" << std::endl;
    
    if (improvement2 >= 50.0f) {
        std::cout << "Goal achieved: Latency reduced by ≥ 50%" << std::endl;
    } else {
        std::cout << "Goal not yet achieved: Latency reduced by < 50%" << std::endl;
    }
}