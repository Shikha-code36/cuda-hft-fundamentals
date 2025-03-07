#include "order_book.cuh"

// GPU kernel to add a buy order
__global__ void addBuyOrderKernel(Order* orders, int* size, Order new_order) {
    if (threadIdx.x == 0) {
        orders[*size] = new_order;
        *size += 1;
    }
}

// GPU kernel to add a sell order
__global__ void addSellOrderKernel(Order* orders, int* size, Order new_order) {
    if (threadIdx.x == 0) {
        orders[*size] = new_order;
        *size += 1;
    }
}

// GPU kernel to cancel an order
__global__ void cancelOrderKernel(Order* orders, int* size, int index) {
    if (threadIdx.x == 0 && index < *size) {
        // Move the last order to the position of the canceled order
        orders[index] = orders[*size - 1];
        *size -= 1;
    }
}

// CUDA Order Book Implementation
CUDAOrderBook::CUDAOrderBook() : best_bid(0.0), best_ask(std::numeric_limits<double>::max()) {
    // Reserve space for orders
    d_buy_orders.reserve(1000000);
    d_sell_orders.reserve(1000000);
}

CUDAOrderBook::~CUDAOrderBook() {
    // Cleanup (handled by thrust)
}

void CUDAOrderBook::processOrder(const Order& order) {
    if (order.type == CANCEL) {
        cancelOrder(order.id);
    } else if (order.type == LIMIT_BUY || order.type == LIMIT_SELL) {
        addLimitOrder(order);
        matchOrders();
    }
}

void CUDAOrderBook::addLimitOrder(const Order& order) {
    if (order.type == LIMIT_BUY) {
        // Add order to device vector
        d_buy_orders.push_back(order);
        
        // Get the index of the newly added order
        int index = d_buy_orders.size() - 1;
        
        // Get a host copy of the order to access its ID
        Order host_order = d_buy_orders[index];
        
        // Store index in the map
        buy_order_map[host_order.id] = index;
        
        // Update best bid if necessary
        if (host_order.price > best_bid) {
            best_bid = host_order.price;
        }
    } 
    else if (order.type == LIMIT_SELL) {
        // Add order to device vector
        d_sell_orders.push_back(order);
        
        // Get the index of the newly added order
        int index = d_sell_orders.size() - 1;
        
        // Get a host copy of the order to access its ID
        Order host_order = d_sell_orders[index];
        
        // Store index in the map
        sell_order_map[host_order.id] = index;
        
        // Update best ask if necessary
        if (host_order.price < best_ask) {
            best_ask = host_order.price;
        }
    }
}

void CUDAOrderBook::cancelOrder(unsigned long long order_id) {
    // Check buy orders
    auto buy_it = buy_order_map.find(order_id);
    if (buy_it != buy_order_map.end()) {
        int index = buy_it->second;
        
        if (d_buy_orders.empty()) {
            return; // Safety check
        }
        
        // Swap with the last element and pop_back for O(1) removal
        if (index < d_buy_orders.size() - 1) {
            // Get a host copy of the last order
            Order last_order = d_buy_orders.back();
            
            // Copy it to the position of the canceled order
            d_buy_orders[index] = last_order;
            
            // Update the map with the new position
            buy_order_map[last_order.id] = index;
        }
        
        // Make sure not to pop from an empty vector
        if (!d_buy_orders.empty()) {
            d_buy_orders.pop_back();
        }
        
        buy_order_map.erase(order_id);
        
        // Recalculate best bid if necessary
        if (d_buy_orders.empty()) {
            best_bid = 0.0;
        } else {
            // Find the new best bid
            best_bid = 0.0;
            
            // Copy to host for iteration
            thrust::host_vector<Order> h_buy_orders = d_buy_orders;
            for (const auto& order : h_buy_orders) {
                if (order.price > best_bid) {
                    best_bid = order.price;
                }
            }
        }
        return;
    }
    
    // Check sell orders
    auto sell_it = sell_order_map.find(order_id);
    if (sell_it != sell_order_map.end()) {
        int index = sell_it->second;
        
        if (d_sell_orders.empty()) {
            return; // Safety check
        }
        
        // Swap with the last element and pop_back for O(1) removal
        if (index < d_sell_orders.size() - 1) {
            // Get a host copy of the last order
            Order last_order = d_sell_orders.back();
            
            // Copy it to the position of the canceled order
            d_sell_orders[index] = last_order;
            
            // Update the map with the new position
            sell_order_map[last_order.id] = index;
        }
        
        // Make sure not to pop from an empty vector
        if (!d_sell_orders.empty()) {
            d_sell_orders.pop_back();
        }
        
        sell_order_map.erase(order_id);
        
        // Recalculate best ask if necessary
        if (d_sell_orders.empty()) {
            best_ask = std::numeric_limits<double>::max();
        } else {
            // Find the new best ask
            best_ask = std::numeric_limits<double>::max();
            
            // Copy to host for iteration
            thrust::host_vector<Order> h_sell_orders = d_sell_orders;
            for (const auto& order : h_sell_orders) {
                if (order.price < best_ask) {
                    best_ask = order.price;
                }
            }
        }
    }
}

void CUDAOrderBook::matchOrders() {
    // Check if there are any matches (bid >= ask)
    if (d_buy_orders.empty() || d_sell_orders.empty() || best_bid < best_ask) {
        return;
    }
    
    // Sort buy orders by price (descending) and time (ascending)
    thrust::sort(d_buy_orders.begin(), d_buy_orders.end(), 
        [] __host__ __device__ (const Order& a, const Order& b) {
            if (a.price != b.price) return a.price > b.price;
            return a.timestamp < b.timestamp;
        });
    
    // Sort sell orders by price (ascending) and time (ascending)
    thrust::sort(d_sell_orders.begin(), d_sell_orders.end(), 
        [] __host__ __device__ (const Order& a, const Order& b) {
            if (a.price != b.price) return a.price < b.price;
            return a.timestamp < b.timestamp;
        });
    
    // Match orders
    bool matched = true;
    while (matched && !d_buy_orders.empty() && !d_sell_orders.empty()) {
        matched = false;
        
        // Safety check for empty vectors
        if (d_buy_orders.empty() || d_sell_orders.empty()) {
            break;
        }
        
        // Get copies of the best bid and ask orders
        Order best_buy = d_buy_orders[0];
        Order best_sell = d_sell_orders[0];
        
        // Check if they match
        if (best_buy.price >= best_sell.price) {
            matched = true;
            
            // Determine the executed quantity
            int exec_qty = std::min(best_buy.quantity, best_sell.quantity);
            
            // Update order quantities
            best_buy.quantity -= exec_qty;
            best_sell.quantity -= exec_qty;
            
            // Update or remove orders in device memory
            if (best_buy.quantity == 0) {
                buy_order_map.erase(best_buy.id);
                // Check if vector is not empty before erasing
                if (!d_buy_orders.empty()) {
                    d_buy_orders.erase(d_buy_orders.begin());
                }
            } else {
                // Update the order in device memory if vector is not empty
                if (!d_buy_orders.empty()) {
                    d_buy_orders[0] = best_buy;
                }
            }
            
            if (best_sell.quantity == 0) {
                sell_order_map.erase(best_sell.id);
                // Check if vector is not empty before erasing
                if (!d_sell_orders.empty()) {
                    d_sell_orders.erase(d_sell_orders.begin());
                }
            } else {
                // Update the order in device memory if vector is not empty
                if (!d_sell_orders.empty()) {
                    d_sell_orders[0] = best_sell;
                }
            }
            
            // Update best bid/ask
            if (d_buy_orders.empty()) {
                best_bid = 0.0;
            } else {
                Order front_buy = d_buy_orders[0];
                best_bid = front_buy.price;
            }
            
            if (d_sell_orders.empty()) {
                best_ask = std::numeric_limits<double>::max();
            } else {
                Order front_sell = d_sell_orders[0];
                best_ask = front_sell.price;
            }
        }
    }
    
    // Rebuild the order maps
    buy_order_map.clear();
    thrust::host_vector<Order> h_buy_orders = d_buy_orders;
    for (size_t i = 0; i < h_buy_orders.size(); i++) {
        buy_order_map[h_buy_orders[i].id] = i;
    }
    
    sell_order_map.clear();
    thrust::host_vector<Order> h_sell_orders = d_sell_orders;
    for (size_t i = 0; i < h_sell_orders.size(); i++) {
        sell_order_map[h_sell_orders[i].id] = i;
    }
}

void CUDAOrderBook::printOrderBook() const {
    std::cout << "---- Order Book State ----" << std::endl;
    
    // Copy data from device to host for printing
    thrust::host_vector<Order> h_buy_orders = d_buy_orders;
    thrust::host_vector<Order> h_sell_orders = d_sell_orders;
    
    // Print sell orders (highest to lowest)
    std::cout << "SELL:" << std::endl;
    for (auto it = h_sell_orders.rbegin(); it != h_sell_orders.rend(); ++it) {
        std::cout << "  " << (*it).price << " - " << (*it).quantity << std::endl;
    }
    
    // Print spread
    std::cout << "SPREAD: " << (best_ask - best_bid) << std::endl;
    
    // Print buy orders (highest to lowest)
    std::cout << "BUY:" << std::endl;
    for (auto it = h_buy_orders.begin(); it != h_buy_orders.end(); ++it) {
        std::cout << "  " << (*it).price << " - " << (*it).quantity << std::endl;
    }
    
    std::cout << "------------------------" << std::endl;
}

// CPU Order Book Implementation
CPUOrderBook::CPUOrderBook() : best_bid(0.0), best_ask(std::numeric_limits<double>::max()) {}

CPUOrderBook::~CPUOrderBook() {}

void CPUOrderBook::processOrder(const Order& order) {
    if (order.type == CANCEL) {
        cancelOrder(order.id);
    } else if (order.type == LIMIT_BUY || order.type == LIMIT_SELL) {
        addLimitOrder(order);
        matchOrders();
    }
}

void CPUOrderBook::addLimitOrder(const Order& order) {
    if (order.type == LIMIT_BUY) {
        buy_orders[order.price].push_back(order);
        order_map[order.id] = {order.price, buy_orders[order.price].size() - 1};
        
        // Update best bid if necessary
        if (order.price > best_bid) {
            best_bid = order.price;
        }
    } else if (order.type == LIMIT_SELL) {
        sell_orders[order.price].push_back(order);
        order_map[order.id] = {order.price, sell_orders[order.price].size() - 1};
        
        // Update best ask if necessary
        if (order.price < best_ask) {
            best_ask = order.price;
        }
    }
}

void CPUOrderBook::cancelOrder(unsigned long long order_id) {
    auto it = order_map.find(order_id);
    if (it == order_map.end()) {
        return;
    }
    
    double price = it->second.first;
    int index = it->second.second;
    
    // Check if it's a buy or sell order
    if (buy_orders.find(price) != buy_orders.end() && index < buy_orders[price].size()) {
        // Check if the order at this position matches the ID
        if (buy_orders[price][index].id == order_id) {
            // Remove the order
            buy_orders[price].erase(buy_orders[price].begin() + index);
            
            // If no more orders at this price, remove the price level
            if (buy_orders[price].empty()) {
                buy_orders.erase(price);
                
                // Update best bid if necessary
                if (price == best_bid) {
                    best_bid = buy_orders.empty() ? 0.0 : buy_orders.begin()->first;
                }
            }
            
            // Update indices for remaining orders at this price
            for (int i = index; i < buy_orders[price].size(); i++) {
                order_map[buy_orders[price][i].id].second = i;
            }
        }
    } else if (sell_orders.find(price) != sell_orders.end() && index < sell_orders[price].size()) {
        // Check if the order at this position matches the ID
        if (sell_orders[price][index].id == order_id) {
            // Remove the order
            sell_orders[price].erase(sell_orders[price].begin() + index);
            
            // If no more orders at this price, remove the price level
            if (sell_orders[price].empty()) {
                sell_orders.erase(price);
                
                // Update best ask if necessary
                if (price == best_ask) {
                    best_ask = sell_orders.empty() ? std::numeric_limits<double>::max() : sell_orders.begin()->first;
                }
            }
            
            // Update indices for remaining orders at this price
            for (int i = index; i < sell_orders[price].size(); i++) {
                order_map[sell_orders[price][i].id].second = i;
            }
        }
    }
    
    // Remove the order from the map
    order_map.erase(order_id);
}

void CPUOrderBook::matchOrders() {
    // Check if there are any matches
    if (buy_orders.empty() || sell_orders.empty() || best_bid < best_ask) {
        return;
    }
    
    bool matched = true;
    while (matched && !buy_orders.empty() && !sell_orders.empty() && best_bid >= best_ask) {
        matched = false;
        
        // Get the best bid and ask prices
        double bid_price = buy_orders.begin()->first;
        double ask_price = sell_orders.begin()->first;
        
        if (bid_price >= ask_price) {
            matched = true;
            
            // Safety check to ensure vectors are not empty before accessing front
            if (buy_orders[bid_price].empty() || sell_orders[ask_price].empty()) {
                break;
            }
            
            // Get the oldest orders at these prices
            Order& buy_order = buy_orders[bid_price].front();
            Order& sell_order = sell_orders[ask_price].front();
            
            // Determine the executed quantity
            int exec_qty = std::min(buy_order.quantity, sell_order.quantity);
            
            // Update order quantities
            buy_order.quantity -= exec_qty;
            sell_order.quantity -= exec_qty;
            
            // Remove filled orders
            if (buy_order.quantity == 0) {
                order_map.erase(buy_order.id);
                
                // Safety check before erasing
                if (!buy_orders[bid_price].empty()) {
                    buy_orders[bid_price].erase(buy_orders[bid_price].begin());
                }
                
                // Update indices for remaining orders at this price
                for (int i = 0; i < buy_orders[bid_price].size(); i++) {
                    order_map[buy_orders[bid_price][i].id].second = i;
                }
                
                // If no more orders at this price, remove the price level
                if (buy_orders[bid_price].empty()) {
                    buy_orders.erase(bid_price);
                    
                    // Update best bid
                    best_bid = buy_orders.empty() ? 0.0 : buy_orders.begin()->first;
                }
            }
            
            if (sell_order.quantity == 0) {
                order_map.erase(sell_order.id);
                
                // Safety check before erasing
                if (!sell_orders[ask_price].empty()) {
                    sell_orders[ask_price].erase(sell_orders[ask_price].begin());
                }
                
                // Update indices for remaining orders at this price
                for (int i = 0; i < sell_orders[ask_price].size(); i++) {
                    order_map[sell_orders[ask_price][i].id].second = i;
                }
                
                // If no more orders at this price, remove the price level
                if (sell_orders[ask_price].empty()) {
                    sell_orders.erase(ask_price);
                    
                    // Update best ask
                    best_ask = sell_orders.empty() ? std::numeric_limits<double>::max() : sell_orders.begin()->first;
                }
            }
        }
    }
}

void CPUOrderBook::printOrderBook() const {
    std::cout << "---- Order Book State ----" << std::endl;
    
    // Print sell orders (highest to lowest)
    std::cout << "SELL:" << std::endl;
    for (auto it = sell_orders.rbegin(); it != sell_orders.rend(); ++it) {
        std::cout << "  " << it->first << " - ";
        int total_qty = 0;
        for (const auto& order : it->second) {
            total_qty += order.quantity;
        }
        std::cout << total_qty << " (" << it->second.size() << " orders)" << std::endl;
    }
    
    // Print spread
    std::cout << "SPREAD: " << (best_ask - best_bid) << std::endl;
    
    // Print buy orders (highest to lowest)
    std::cout << "BUY:" << std::endl;
    for (auto it = buy_orders.begin(); it != buy_orders.end(); ++it) {
        std::cout << "  " << it->first << " - ";
        int total_qty = 0;
        for (const auto& order : it->second) {
            total_qty += order.quantity;
        }
        std::cout << total_qty << " (" << it->second.size() << " orders)" << std::endl;
    }
    
    std::cout << "------------------------" << std::endl;
}

// Benchmark function
void benchmarkOrderBook(int num_orders) {
    std::cout << "Benchmarking with " << num_orders << " orders..." << std::endl;
    
    // Generate random orders
    std::vector<Order> orders;
    orders.reserve(num_orders);
    
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    for (int i = 0; i < num_orders; i++) {
        OrderType type;
        
        // 10% chance of a cancel order
        if (i > 100 && std::rand() % 100 < 10) {
            type = CANCEL;
            // Get a random order ID from the first 90% of orders
            int random_index = std::rand() % (i - 10);
            unsigned long long random_id = orders[random_index].id;
            
            orders.emplace_back(random_id, type, 0.0, 0, i);
        } else {
            // 50% chance of buy vs sell
            type = (std::rand() % 2 == 0) ? LIMIT_BUY : LIMIT_SELL;
            
            // Random price between 90 and 110
            double price = 90.0 + (std::rand() % 2000) / 100.0;
            
            // Random quantity between 1 and 100
            int quantity = 1 + (std::rand() % 100);
            
            orders.emplace_back(i, type, price, quantity, i);
        }
    }
    
    // Benchmark CPU implementation
    {
        CPUOrderBook cpu_book;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& order : orders) {
            cpu_book.processOrder(order);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double orders_per_second = static_cast<double>(num_orders) / (duration.count() / 1000.0);
        
        std::cout << "CPU order book processed " << num_orders << " orders in " 
                  << duration.count() << "ms (" << orders_per_second << " orders/sec)" << std::endl;
    }
    
    // Benchmark GPU implementation
    {
        CUDAOrderBook gpu_book;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& order : orders) {
            gpu_book.processOrder(order);
        }
        
        // Ensure all CUDA operations are complete
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double orders_per_second = static_cast<double>(num_orders) / (duration.count() / 1000.0);
        
        std::cout << "GPU order book processed " << num_orders << " orders in " 
                  << duration.count() << "ms (" << orders_per_second << " orders/sec)" << std::endl;
    }
}