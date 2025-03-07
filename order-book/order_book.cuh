#ifndef ORDER_BOOK_CUH
#define ORDER_BOOK_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

// Order types
enum OrderType {
    LIMIT_BUY,
    LIMIT_SELL,
    CANCEL
};

// Order structure
struct Order {
    unsigned long long id;
    OrderType type;
    double price;
    int quantity;
    unsigned long long timestamp;

    // Default constructor
    __host__ __device__
    Order() : id(0), type(LIMIT_BUY), price(0.0), quantity(0), timestamp(0) {}

    // Constructor
    __host__ __device__
    Order(unsigned long long _id, OrderType _type, double _price, int _qty, unsigned long long _ts) 
        : id(_id), type(_type), price(_price), quantity(_qty), timestamp(_ts) {}
};

// CUDA Order Book Implementation
class CUDAOrderBook {
private:
    // Device vectors to store buy and sell orders
    thrust::device_vector<Order> d_buy_orders;
    thrust::device_vector<Order> d_sell_orders;
    
    // Maps to track order positions (for quick cancellation)
    std::unordered_map<unsigned long long, int> buy_order_map;
    std::unordered_map<unsigned long long, int> sell_order_map;
    
    // Current best bid/ask
    double best_bid;
    double best_ask;

public:
    CUDAOrderBook();
    ~CUDAOrderBook();
    
    // Process an order
    void processOrder(const Order& order);
    
    // Add a limit order
    void addLimitOrder(const Order& order);
    
    // Cancel an order
    void cancelOrder(unsigned long long order_id);
    
    // Match orders (crosses the spread)
    void matchOrders();
    
    // Print the current state of the order book
    void printOrderBook() const;
    
    // Getters for bid/ask
    double getBestBid() const { return best_bid; }
    double getBestAsk() const { return best_ask; }
};

// CPU Order Book Implementation (for comparison)
class CPUOrderBook {
private:
    // Maps for price-time priority
    std::map<double, std::vector<Order>, std::greater<double>> buy_orders; // Higher prices first
    std::map<double, std::vector<Order>> sell_orders; // Lower prices first
    
    // Maps to track order positions
    std::unordered_map<unsigned long long, std::pair<double, int>> order_map;
    
    // Current best bid/ask
    double best_bid;
    double best_ask;

public:
    CPUOrderBook();
    ~CPUOrderBook();
    
    // Process an order
    void processOrder(const Order& order);
    
    // Add a limit order
    void addLimitOrder(const Order& order);
    
    // Cancel an order
    void cancelOrder(unsigned long long order_id);
    
    // Match orders
    void matchOrders();
    
    // Print the current state of the order book
    void printOrderBook() const;
    
    // Getters for bid/ask
    double getBestBid() const { return best_bid; }
    double getBestAsk() const { return best_ask; }
};

// Benchmark function to compare CPU and GPU implementations
void benchmarkOrderBook(int num_orders);

#endif // ORDER_BOOK_CUH