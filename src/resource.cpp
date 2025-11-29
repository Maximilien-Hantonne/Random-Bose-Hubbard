#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <omp.h>

#include "resource.hpp"

// Static variable definitions
namespace Resource {
    std::chrono::high_resolution_clock::time_point start_time;
    bool is_running = false;
}



// MEMORY USAGE :

/* Get the memory usage of the program in KB and optionally print it */
long Resource::get_memory_usage(bool print) {
    std::ifstream statm_file("/proc/self/statm");
    if (!statm_file.is_open()) {
        if (print) {
            std::cerr << "Error reading memory usage from /proc/self/statm." << std::endl;
        }
        return -1;
    }
    
    long size, resident, share, text, lib, data, dt;
    if (!(statm_file >> size >> resident >> share >> text >> lib >> data >> dt)) {
        if (print) {
            std::cerr << "Error parsing memory usage data." << std::endl;
        }
        return -1;
    }
    
    static const long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
    const long memory_usage = resident * page_size_kb; // Convert pages to KB
    
    if (print) {
        std::cout << "Memory usage: " << memory_usage << " KB";
    }
    return memory_usage;
}

/* Get the available memory in KB */
long Resource::get_available_memory() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        std::cerr << "Error getting system info." << std::endl;
        return -1;
    }
    return static_cast<long>(info.freeram / 1024);
}

/* Estimate the memory usage of a sparse matrix */
size_t Resource::estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix) {
    const size_t numNonZeros = static_cast<size_t>(matrix.nonZeros());
    const size_t numCols = static_cast<size_t>(matrix.cols());
    
    size_t memoryUsage = numNonZeros * sizeof(double);  // Values
    memoryUsage += numNonZeros * sizeof(int);           // Row indices
    memoryUsage += (numCols + 1) * sizeof(int);         // Column pointers
    
    return memoryUsage;
}


// TIMER :

/* Timer function to measure the duration of the calculation */
void Resource::timer() {
    if (!is_running) {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    } else {
        const auto end_time = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> duration = end_time - start_time;
        is_running = false;
        
        const double total_seconds = duration.count();
        if (total_seconds > 60.0) {
            const int minutes = static_cast<int>(total_seconds) / 60;
            const double seconds = total_seconds - (minutes * 60.0);
            std::cout << "Duration: " << minutes << "m " << seconds << "s";
        } else {
            std::cout << "Duration: " << total_seconds << "s";
        }
    }
}


// OPENMP THREADS :

/* Set the number of threads for OpenMP parallelization */
void Resource::set_omp_threads(const Eigen::SparseMatrix<double>& matrix, int nb_matrix) {
    const int max_threads = omp_get_max_threads();
    
    if (nb_matrix <= 0 || matrix.nonZeros() == 0) {
        omp_set_num_threads(max_threads);
        return;
    }
    
    const size_t memory_per_matrix = estimateSparseMatrixMemoryUsage(matrix);
    const long available_memory = get_available_memory();
    
    if (available_memory <= 0) {
        omp_set_num_threads(std::max(1, max_threads / 2));
        return;
    }
    
    constexpr double memory_usage_factor = 0.8; // Use 80% of available memory
    const size_t usable_memory = static_cast<size_t>(available_memory) * 1024 * memory_usage_factor;
    const size_t total_memory = memory_per_matrix * static_cast<size_t>(nb_matrix);
    
    const int memory_threads = (total_memory > 0) ? static_cast<int>(usable_memory / total_memory) : max_threads;
    const int num_threads = std::max(1, std::min(memory_threads, max_threads));
    
    omp_set_num_threads(num_threads);
}