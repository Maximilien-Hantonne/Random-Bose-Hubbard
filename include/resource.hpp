#pragma once

#include <chrono>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <Eigen/SparseCore>


namespace Resource
{
    long get_memory_usage(bool print = false);
    long get_available_memory();
    size_t estimateSparseMatrixMemoryUsage(const Eigen::SparseMatrix<double>& matrix);
    void timer();
    void set_omp_threads(const Eigen::SparseMatrix<double>& matrix1, int nb_matrix);

    static std::chrono::high_resolution_clock::time_point start_time;
    static bool is_running = false;

}

