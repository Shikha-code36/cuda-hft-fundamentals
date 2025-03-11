#ifndef ZERO_COPY_PROCESSOR_CUH
#define ZERO_COPY_PROCESSOR_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <random>  // Added for random number generation

// Market data packet structure - simplified ITCH-like format
struct MarketDataPacket {
    unsigned long long timestamp;    // Timestamp in nanoseconds
    char message_type;               // Message type identifier
    unsigned int order_id;           // Order reference number
    char side;                       // Buy/Sell indicator
    double price;                    // Price in fixed-point format
    unsigned int shares;             // Number of shares
    
    // Default constructor
    __host__ __device__
    MarketDataPacket() : timestamp(0), message_type(0), order_id(0), 
                        side(0), price(0.0), shares(0) {}
    
    // Constructor
    __host__ __device__
    MarketDataPacket(unsigned long long ts, char type, unsigned int id, 
                    char s, double p, unsigned int sh) 
        : timestamp(ts), message_type(type), order_id(id), 
          side(s), price(p), shares(sh) {}
};

// Zero-Copy Processor using CUDA Streams and Mapped Memory
class ZeroCopyProcessor {
private:
    // Host-mapped memory pointers
    MarketDataPacket* h_mapped_data;      // Host pointer to mapped memory
    MarketDataPacket* d_mapped_data;      // Device pointer to same memory
    
    // Stream for asynchronous processing
    cudaStream_t stream;
    
    // Processing buffers and results
    MarketDataPacket* d_results;          // Device results buffer
    MarketDataPacket* h_results;          // Host results buffer
    
    // Configuration
    size_t buffer_size;                   // Size of data buffers
    size_t process_size;                  // Number of packets to process at once
    
    // Events for timing and synchronization
    cudaEvent_t start_event, stop_event;
    
    // Concurrency control
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<std::vector<MarketDataPacket>> packet_queue;
    std::atomic<bool> running;
    std::thread worker_thread;

public:
    // Constructor and destructor
    ZeroCopyProcessor(size_t buffer_size = 1024, size_t process_size = 512);
    ~ZeroCopyProcessor();
    
    // Initialize CUDA resources
    bool initialize();
    
    // Process a batch of market data with standard memory transfers (for baseline comparison)
    float processStandard(const std::vector<MarketDataPacket>& packets);
    
    // Process a batch with zero-copy memory
    float processZeroCopy(const std::vector<MarketDataPacket>& packets);
    
    // Process a batch with zero-copy + CUDA streams
    float processZeroCopyStreamed(const std::vector<MarketDataPacket>& packets);
    
    // Asynchronous API
    void submitPackets(const std::vector<MarketDataPacket>& packets);
    void startProcessing();
    void stopProcessing();
    
    // Helpers
    void cleanup();
};

// Generate random market data for testing (exported for main.cu)
std::vector<MarketDataPacket> generateRandomMarketData(int size);

// Benchmark function to compare standard vs. zero-copy approaches
void benchmarkZeroCopyProcessing(int num_packets, int num_runs);

// Internal CUDA kernels
__global__ void processMarketDataKernel(const MarketDataPacket* input, MarketDataPacket* output, int size);
__global__ void processMarketDataZeroCopyKernel(MarketDataPacket* data, int size);

#endif // ZERO_COPY_PROCESSOR_CUH