#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <omp.h>

#include "resource.hpp"



// MEMORY USAGE :

/* Get the memory usage of the program in KB and optionally print it */
long Resource::get_memory_usage(bool print) {
    std::ifstream statm_file("/proc/self/statm");
    long memory_usage = -1;
    if (statm_file.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm_file >> size >> resident >> share >> text >> lib >> data >> dt;
        memory_usage = resident * (sysconf(_SC_PAGESIZE) / 1024); // Convert pages to KB
    }
    if (memory_usage == -1) {
        std::cerr << "Error reading memory usage from /proc/self/statm." << std::endl;
    } else if (print) {
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
    return info.freeram / 1024;
}

/* Estimate the memory usage of a sparse matrix */
size_t Resource::estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix) {
    size_t numNonZeros = matrix.nonZeros();
    size_t numCols = matrix.cols();
    size_t memoryUsage = numNonZeros * sizeof(double);
    memoryUsage += numNonZeros * sizeof(int); 
    memoryUsage += (numCols + 1) * sizeof(int);
    return memoryUsage;
}


// TIMER :

/* Timer function to measure the duration of the calculation */
void Resource::timer() {
    if (!is_running) {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    } else {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        is_running = false;
        if (duration.count() > 60) {
            int minutes = static_cast<int>(duration.count()) / 60;
            double seconds = duration.count() - (minutes * 60);
            std::cout << "Duration: " << minutes << "m " << seconds << "s";
        } else {
            std::cout << "Duration: " << duration.count() << "s";
        }
    }
}


// OPENMP THREADS :

/* Set the number of threads for OpenMP parallelization */
void Resource::set_omp_threads(const Eigen::SparseMatrix<double>& matrix, int nb_matrix) {
    int max_threads = omp_get_max_threads();
    if (nb_matrix <= 0 || matrix.nonZeros() == 0) {
        omp_set_num_threads(max_threads);
        return;
    }
    size_t memory_per_matrix = estimateSparseMatrixMemoryUsage(matrix);
    long available_memory = get_available_memory();
    size_t usable_memory = static_cast<size_t>(available_memory) * 1024 * 0.8;
    size_t total_memory = memory_per_matrix * nb_matrix;
    int memory_threads = static_cast<int>(usable_memory / total_memory);
    int num_threads = std::max(6, std::min(memory_threads, max_threads));
    omp_set_num_threads(num_threads);
    std::cout << num_threads << " threads";
}