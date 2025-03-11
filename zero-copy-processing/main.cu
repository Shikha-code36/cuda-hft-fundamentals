#include "zero_copy_processor.cuh"

int main(int argc, char** argv) {
    // Default parameters - increased packet count for better performance demonstration 
    int num_packets = 1000000; // Increased from 100000 to 1000000
    int num_runs = 10;
    
    // Check for command-line arguments
    if (argc > 1) {
        num_packets = std::atoi(argv[1]);
    }
    
    if (argc > 2) {
        num_runs = std::atoi(argv[2]);
    }
    
    // Print configuration
    std::cout << "Zero-Copy Market Data Processing Benchmark" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Number of packets: " << num_packets << std::endl;
    std::cout << "Number of runs:    " << num_runs << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Pre-warm the GPU to ensure fair benchmarks
    cudaFree(0);
    
    // Run the benchmark
    benchmarkZeroCopyProcessing(num_packets, num_runs);
    
    // Demonstrate asynchronous processing
    std::cout << "\nDemonstrating asynchronous processing..." << std::endl;
    
    // Create and initialize processor with optimized parameters
    ZeroCopyProcessor async_processor(num_packets, num_packets/4);
    if (!async_processor.initialize()) {
        std::cerr << "Failed to initialize async processor." << std::endl;
        return 1;
    }
    
    // Start background processing
    async_processor.startProcessing();
    
    // Generate and submit batches of data
    for (int i = 0; i < 5; i++) {
        auto batch = generateRandomMarketData(10000);
        std::cout << "Submitting batch " << i+1 << " with 10,000 packets..." << std::endl;
        async_processor.submitPackets(batch);
        
        // Sleep briefly to simulate market data arrival intervals
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Allow time for processing to complete
    std::cout << "Waiting for processing to complete..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Stop background processing
    async_processor.stopProcessing();
    std::cout << "Asynchronous processing demonstration complete." << std::endl;
    
    return 0;
}