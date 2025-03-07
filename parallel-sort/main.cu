#include "parallel_sort.cuh"

int main(int argc, char** argv) {
    // Default number of market data points
    int data_size = 1000000;
    
    // Check if a custom size was provided
    if (argc > 1) {
        data_size = std::atoi(argv[1]);
    }
    
    // Run the benchmark
    benchmarkSorting(data_size);
    
    return 0;
}