#include "order_book.cuh"

int main(int argc, char** argv) {
    // Default number of orders
    int num_orders = 100000;
    
    // Check if a custom number was provided
    if (argc > 1) {
        num_orders = std::atoi(argv[1]);
    }
    
    // Run the benchmark
    benchmarkOrderBook(num_orders);
    
    return 0;
}