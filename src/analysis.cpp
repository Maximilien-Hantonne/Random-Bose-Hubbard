#include <cmath>
#include <vector>
#include <limits>
#include <complex>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/SparseCore/SparseMatrix.h"
#include "operator.hpp"
#include "analysis.hpp"
#include "hamiltonian.hpp"
#include "resource.hpp"
#include "neighbours.hpp"


        /* MAIN FUNCTIONS */

/*main function for exact calculations parameters*/
void Analysis::exact_parameters(int m, int n, double T,double U, double mu, double s, double r, std::string fixed_param, double sigma_t, double sigma_U, double sigma_u, int realizations) {
    
    // Prerequisites
    if (std::abs(T-0.0) < std::numeric_limits<double>::epsilon() && std::abs(U-0.0) < std::numeric_limits<double>::epsilon() && std::abs(mu-0.0) < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Error: At least one of the parameters T, U, mu must be different from zero." << std::endl;
        return;
    }

    // Start of the calculations
    Resource::timer();

    // Set the geometry of the lattice
    const std::vector<std::vector<int>> nei = Neighbours::chain_neighbours(m);
    // const std::vector<std::vector<int>> nei = Neighbours::square_neighbours(m);

    // // Set the matrices for each term of the Hamiltonian in the Fock states from 1 to n bosons
    // int n_min = 1, n_max = n;
    // auto [tags, basis] = BH::max_set_basis(m, n);
    // Eigen::SparseMatrix<double> JH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 1, 0, 0);
    // Eigen::SparseMatrix<double> UH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 1, 0);
    // Eigen::SparseMatrix<double> uH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 0, 1);
    
    // Set the Fock states basis with a fixed number of bosons with their tags
    auto [tags, basis] = BH::fixed_set_basis(m, n);
    Eigen::SparseMatrix<double> TH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 1, 0, 0);
    Eigen::SparseMatrix<double> UH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 1, 0);
    Eigen::SparseMatrix<double> uH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 0, 1);

    // Set the number of threads for the calculations
    Resource::set_omp_threads(TH, 3);

    // Set the range of parameters for the calculations
    double T_min = T, T_max = T + r, mu_min = mu, mu_max = mu + r, U_min = U, U_max = U + r;

    // Calculate the exact parameters
    calculate_and_save(basis, tags, TH, UH, uH, fixed_param, T, U, mu, T_min, T_max, U_min, U_max, mu_min, mu_max, s, s, sigma_t, sigma_U, sigma_u, realizations, m, n);

    // End of the calculations
    std::cout << " - ";
    Resource::timer();
    std::cout << " - ";
    Resource::get_memory_usage(true);
    std::cout << std::endl;
}


/* calculate and save gap ratio and other quantities */
void Analysis::calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double>& TH, const Eigen::SparseMatrix<double>& UH, const Eigen::SparseMatrix<double>& uH, const std::string& fixed_param, const double T, const double U, const double mu, const double T_min, const double T_max, const double U_min, const double U_max, const double mu_min, const double mu_max, const double param1_step, const double param2_step, const double sigma_T, const double sigma_U, const double sigma_u, const int realizations, const int m, const int n) {
    
    // Save the fixed parameter and value in a file
    std::ofstream file("phase.txt");
    file << fixed_param << " ";
    if (fixed_param == "T") {
        file << T << std::endl;
    } else if (fixed_param == "U") {
        file << U << std::endl;
    } else {
        file << mu << std::endl;
    }
    file << "m " << m << " n " << n << " R " << realizations << std::endl;
    
    // Parameters for the calculations
    const int nb_eigen = 20;

    // Hoist fixed_param comparisons to avoid string comparison in loop
    const bool is_T_fixed = (fixed_param == "T");
    const bool is_U_fixed = (fixed_param == "U");

    // Varying parameters
    const double param1_min = is_T_fixed ? U_min : (is_U_fixed ? T_min : T_min);
    const double param1_max = is_T_fixed ? U_max : (is_U_fixed ? T_max : T_max);
    const double param2_min = is_T_fixed ? mu_min : (is_U_fixed ? mu_min : U_min);
    const double param2_max = is_T_fixed ? mu_max : (is_U_fixed ? mu_max : U_max);
    
    // Matrices initialization
    const int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
    const int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
    const int total_size = num_param1 * num_param2;
    std::vector<double> param1_values(total_size);
    std::vector<double> param2_values(total_size);
    std::vector<double> gap_ratios_values(total_size);
    std::vector<double> condensate_fraction_values(total_size);
    std::vector<double> fluctuations_values(total_size);
    std::vector<double> qEA_values(total_size);
    const int diagonal_size = std::min(num_param1, num_param2);
    std::vector<double> diagonal_ratios(0);
    diagonal_ratios.reserve(diagonal_size);
    std::vector<Eigen::VectorXcd> diagonal_eigenvalues(0);
    diagonal_eigenvalues.reserve(diagonal_size);

    // Progress tracking
    std::atomic<int> progress_counter(0);
    const int num_threads = omp_get_max_threads();

    // Spacing parameters
    const double log_param1_min = std::log10(param1_min);
    const double log_param1_max = std::log10(param1_max);
    const double log_param2_min = std::log10(param2_min);
    const double log_param2_max = std::log10(param2_max);
    const double log_param1_step = (log_param1_max - log_param1_min) / (num_param1 - 1);
    const double log_param2_step = (log_param2_max - log_param2_min) / (num_param2 - 1);

    // Main loop for the calculations with parallelization
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < num_param1; ++i) {
        for (int j = 0; j < num_param2; ++j) {

            const double param1 = std::pow(10.0, log_param1_min + i * log_param1_step);
            const double param2 = std::pow(10.0, log_param2_min + j * log_param2_step);
            
            const double T_val = is_T_fixed ? T : param1;
            const double U_val = is_U_fixed ? U : (is_T_fixed ? param1 : param2);
            const double mu_val = is_T_fixed ? param2 : (is_U_fixed ? param2 : mu);

            const int index = i * num_param2 + j;
            
            // Initialization of variables
            double sum_gap_ratio = 0.0, sum_condensate_fraction = 0.0, sum_fluctuations = 0.0;
            Eigen::VectorXd q1 = (realizations > 1) ? Eigen::VectorXd::Zero(m) : Eigen::VectorXd();
            Eigen::VectorXd q2 = (realizations > 1) ? Eigen::VectorXd::Zero(m) : Eigen::VectorXd();
            const bool is_diagonal = (j == num_param2 - i - 1);
            Eigen::VectorXcd sum_eigenvalues = is_diagonal ? Eigen::VectorXcd::Zero(nb_eigen) : Eigen::VectorXcd();
            
            // Pre-compute thread hash once per parameter point
            const size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
            
            // Loop over disorder realizations
            for (int real = 0; real < realizations; ++real) {
                const unsigned int seed = static_cast<unsigned int>(
                    thread_hash ^ (static_cast<size_t>(i) << 32) ^ (static_cast<size_t>(j) << 16) ^ static_cast<size_t>(real)
                );
                const Eigen::SparseMatrix<double> H = BH::random_hamiltonian(TH, T_val, sigma_T, UH, U_val, sigma_U, uH, mu_val, sigma_u, seed);

                // Diagonalization
                Eigen::MatrixXcd eigenvectors;
                Eigen::VectorXcd eigenvalues = Op::IRLM_eigen(H, nb_eigen, eigenvectors);
                Op::sort_eigen(eigenvalues, eigenvectors);

                // Gap ratios
                sum_gap_ratio += gap_ratios(eigenvalues, nb_eigen).mean();

                // SPDM
                const Eigen::MatrixXcd spdm = SPDM(basis, tags, eigenvectors.col(0));

                // Condensate fraction
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(spdm, Eigen::EigenvaluesOnly);
                sum_condensate_fraction += solver.eigenvalues().maxCoeff() / spdm.trace().real();
                
                // Fluctuations and Edwards-Anderson parameter
                const auto [mean_ni, mean_ni_sq, site_ni] = mean_occupations(eigenvectors.col(0), basis);
                sum_fluctuations += mean_ni_sq - mean_ni * mean_ni;
                if (realizations > 1) {
                    q1 += site_ni.array().square().matrix();
                    q2 += site_ni;
                }                        
                
                // Eigenvalues on the antidiagonal
                if (is_diagonal) {
                    sum_eigenvalues += eigenvalues;
                }
            }
            
            // Averages over disorder realizations
            const double inv_r = 1.0 / realizations;
            param1_values[index] = param1;
            param2_values[index] = param2;
            gap_ratios_values[index] = sum_gap_ratio * inv_r;
            condensate_fraction_values[index] = sum_condensate_fraction * inv_r;
            fluctuations_values[index] = sum_fluctuations * inv_r;
            qEA_values[index] = (realizations > 1) ? (((q1 * inv_r) - (q2 * inv_r).array().square().matrix()).sum() / m) : 0.0;
            
            if (is_diagonal) {
                #pragma omp critical
                {
                    diagonal_eigenvalues.push_back(sum_eigenvalues * inv_r);
                    diagonal_ratios.push_back(param1 / param2);
                }
            }
            
            // Progress
            const int local_count = progress_counter.fetch_add(1) + 1;
            if (local_count % 10 == 0 || local_count == total_size) {
                const int percent = (local_count * 100) / total_size;
                std::cout << "\r" << num_threads << " threads - " << percent << "%" << std::flush;
            }
        }
    }

    // Save the results to a file
    for (int i = 0; i < total_size; ++i) {
        file << param1_values[i] << " " << param2_values[i] << " " << gap_ratios_values[i] << " " << condensate_fraction_values[i] << " " << fluctuations_values[i] << " " << qEA_values[i] << std::endl;
    }
    file.close();

    // Save eigenvalues
    std::ofstream eigen_file("eigenvalues_diagonal.txt");
    const std::string param1_name = is_T_fixed ? "U" : (is_U_fixed ? "T" : "T");
    const std::string param2_name = is_T_fixed ? "mu" : (is_U_fixed ? "mu" : "U");
    eigen_file << "# Ratio " << param1_name << "/" << param2_name << " Eigenvalues" << std::endl;
    for (size_t k = 0; k < diagonal_eigenvalues.size(); ++k) {
        eigen_file << diagonal_ratios[k];
        for (int e = 0; e < diagonal_eigenvalues[k].size(); ++e) {
            eigen_file << " " << diagonal_eigenvalues[k][e].real();
        }
        eigen_file << std::endl;
    }
    eigen_file.close();
}


        /* GAP RATIOS */

/* Calculate the energy gap ratios of the system */
Eigen::VectorXd Analysis::gap_ratios(const Eigen::VectorXcd& eigenvalues, int nb_eigen) {
    Eigen::VectorXd gap_ratios(nb_eigen - 2);

    // Sort the eigenvalues by their real part
    std::vector<double> sorted_eigenvalues(nb_eigen);
    for (int i = 0; i < nb_eigen; ++i) {
        sorted_eigenvalues[i] = eigenvalues[i].real();
    }
    std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end());

    // Calculate the gap ratios
    for (int i = 1; i < nb_eigen - 1; ++i) {
        const double E_prev = sorted_eigenvalues[i - 1];
        const double E_curr = sorted_eigenvalues[i];
        const double E_next = sorted_eigenvalues[i + 1];
        const double gap1 = E_next - E_curr;
        const double gap2 = E_curr - E_prev;
        const double min_gap = std::min(gap1, gap2);
        const double max_gap = std::max(gap1, gap2);
        gap_ratios[i - 1] = (max_gap != 0.0) ? (min_gap / max_gap) : 0.0;
        }

    return gap_ratios;
}


        /* SPDM FUNCTIONS */

/* Calculate the single-particle density matrix of the system */
Eigen::MatrixXcd Analysis::SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::VectorXcd& phi0) {
    const int m = basis.rows();
    const int D = basis.cols();
    Eigen::MatrixXcd spdm = Eigen::MatrixXcd::Zero(m, m);
    Eigen::VectorXd probs(D);
    for (int k = 0; k < D; ++k) {
        probs[k] = std::norm(phi0[k]);
    }
    std::unordered_map<double, int> tag_index;
    tag_index.reserve(D);
    for (int k = 0; k < D; ++k) {
        tag_index.emplace(tags[k], k);
    }
    static constexpr int primes[] = { 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 };
    static const std::vector<int> primes_vec(std::begin(primes), std::end(primes));
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            std::complex<double> sum = 0.0;
            if (i == j) {
                for (int k = 0; k < D; ++k) {
                    sum += probs[k] * basis(i, k);
                }
            } else {
                for (int k = 0; k < D; ++k) {
                    const double ni = basis(i, k);
                    const double nj = basis(j, k);
                    if (nj <= 0.0) continue;
                    const double factor = std::sqrt((ni + 1.0) * nj);
                    Eigen::VectorXd state = basis.col(k);
                    state(i) += 1.0;
                    state(j) -= 1.0;
                    const double target_tag = BH::calculate_tag(state, primes_vec);
                    const auto it = tag_index.find(target_tag);
                    if (it != tag_index.end()) {
                        const int l = it->second;
                        sum += std::conj(phi0[k]) * phi0[l] * factor;
                    }
                }
            }
            spdm(i, j) = sum;
            if (i != j) spdm(j, i) = std::conj(sum);
        }
    }
    return spdm;
}

/* Calculate site occupations: returns (spatial_avg_ni, spatial_avg_ni_sq, per_site_ni) */
std::tuple<double, double, Eigen::VectorXd> Analysis::mean_occupations(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis){
    const int m = basis.rows();
    const int D = basis.cols();
    double sum_ni = 0.0;
    double sum_ni_sq = 0.0;
    Eigen::VectorXd site_ni = Eigen::VectorXd::Zero(m);
    for (int k = 0; k < D; ++k) {
        const double prob = std::norm(phi0[k]);
        for (int i = 0; i < m; ++i) {
            const double ni = basis(i, k);
            const double prob_ni = prob * ni;
            sum_ni += prob_ni;
            sum_ni_sq += prob * ni * ni;
            site_ni[i] += prob_ni;
        }
    }
    return {sum_ni / m, sum_ni_sq / m, site_ni};
}