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
void Analysis::exact_parameters(const int m, const int n, const double T, const double U, const double mu, const double s, const double r, const std::string& fixed_param, const double sigma_t, const double delta_U, const double delta_u, const int realizations, const std::string& scale) {
    
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
    // const std::vector<std::vector<int>> nei = Neighbours::cube_neighbours(m);

    // // Set the matrices for each term of the Hamiltonian in the Fock states from 1 to n bosons
    // int n_min = 1, n_max = n;
    // auto [tags, basis] = BH::max_set_basis(m, n);
    // Eigen::SparseMatrix<double> TH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 1, 0, 0);
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
    calculate_and_save(basis, tags, TH, UH, uH, fixed_param, T, U, mu, T_min, T_max, U_min, U_max, mu_min, mu_max, s, s, scale, sigma_t, delta_U, delta_u, realizations, m, n);

    // End of the calculations
    std::cout << " - ";
    Resource::timer();
    std::cout << " - ";
    Resource::get_memory_usage(true);
    std::cout << std::endl;
}


/* calculate and save gap ratio and other quantities */
void Analysis::calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double>& TH, const Eigen::SparseMatrix<double>& UH, const Eigen::SparseMatrix<double>& uH, const std::string& fixed_param, const double T, const double U, const double mu, const double T_min, const double T_max, const double U_min, const double U_max, const double mu_min, const double mu_max, const double param1_step, const double param2_step, const std::string& scale, const double sigma_T, const double delta_U, const double delta_u, const int realizations, const int m, const int n) {
    
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
    file << "m " << m << " n " << n << " R " << realizations << " scale " << scale << std::endl;
    
    // Parameters for the calculations
    constexpr int nb_eigen = 20;

    const bool is_T_fixed = (fixed_param == "T");
    const bool is_U_fixed = (fixed_param == "U");
    const bool compute_qEA = (realizations > 1);
    const bool has_disorder = (sigma_T > 0.0 || delta_U > 0.0 || delta_u > 0.0);

    const double inv_r = 1.0 / realizations;
    const double inv_m = 1.0 / m;
    const ScaleType scale_type = parse_scale(scale);

    // Varying parameters
    const double param1_min = is_T_fixed ? U_min : (is_U_fixed ? T_min : T_min);
    const double param1_max = is_T_fixed ? U_max : (is_U_fixed ? T_max : T_max);
    const double param2_min = is_T_fixed ? mu_min : (is_U_fixed ? mu_min : U_min);
    const double param2_max = is_T_fixed ? mu_max : (is_U_fixed ? mu_max : U_max);
    
    // Grid sizes
    const int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
    const int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
    
    // Parameter values
    const std::vector<double> param1_precomputed = precompute_params(param1_min, param1_max, num_param1, scale_type);
    const std::vector<double> param2_precomputed = precompute_params(param2_min, param2_max, num_param2, scale_type);

    // Matrices initialization
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
    std::vector<Eigen::VectorXd> diagonal_eigenvalues(0);
    diagonal_eigenvalues.reserve(diagonal_size);

    // Progress tracking
    std::atomic<int> progress_counter(0);
    const int num_threads = omp_get_max_threads();

    // Main loop for the calculations with parallelization
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < num_param1; ++i) {
        for (int j = 0; j < num_param2; ++j) {

            const double param1 = param1_precomputed[i];
            const double param2 = param2_precomputed[j];
            const double T_val = is_T_fixed ? T : param1;
            const double U_val = is_U_fixed ? U : (is_T_fixed ? param1 : param2);
            const double mu_val = is_T_fixed ? param2 : (is_U_fixed ? param2 : mu);

            const int index = i * num_param2 + j;
            
            // Initialization of variables
            double sum_gap_ratio = 0.0, sum_condensate_fraction = 0.0, sum_fluctuations = 0.0;
            const bool is_diagonal = (j == num_param2 - i - 1);
            Eigen::VectorXd sum_eigenvalues;
            if (is_diagonal) {
                sum_eigenvalues = Eigen::VectorXd::Zero(nb_eigen);
            }
            Eigen::VectorXd q1, q2;
            if (compute_qEA) {
                q1 = Eigen::VectorXd::Zero(m);
                q2 = Eigen::VectorXd::Zero(m);
            }
            size_t thread_hash = 0;
            if (has_disorder) {
                thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
            }
            
            // Loop over disorder realizations
            for (int real = 0; real < realizations; ++real) {
                const unsigned int seed = has_disorder 
                    ? static_cast<unsigned int>(thread_hash ^ (static_cast<size_t>(i) << 32) ^ (static_cast<size_t>(j) << 16) ^ static_cast<size_t>(real))
                    : 0u;
                const Eigen::SparseMatrix<double> H = BH::random_hamiltonian(TH, T_val, sigma_T, UH, U_val, delta_U, uH, mu_val, delta_u, seed);

                // Diagonalization
                Eigen::MatrixXcd eigenvectors;
                Eigen::VectorXd eigenvalues = Op::IRLM_eigen(H, nb_eigen, eigenvectors);
                Op::sort_eigen(eigenvalues, eigenvectors);

                // Gap ratios
                sum_gap_ratio += gap_ratios(eigenvalues, nb_eigen);

                // Ground state
                const Eigen::VectorXcd& ground_state = eigenvectors.col(0);

                // SPDM
                const Eigen::MatrixXcd spdm = SPDM(basis, tags, ground_state);

                // Condensate fraction
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(spdm, Eigen::EigenvaluesOnly);
                sum_condensate_fraction += solver.eigenvalues().maxCoeff() / spdm.trace().real();
                
                // Fluctuations and Edwards-Anderson parameter
                const auto [mean_ni, mean_ni_sq, site_ni] = mean_occupations(ground_state, basis);
                sum_fluctuations += mean_ni_sq - mean_ni * mean_ni;
                if (compute_qEA) {
                    q1 += site_ni.array().square().matrix();
                    q2 += site_ni;
                }                        
                
                // Eigenvalues on the antidiagonal
                if (is_diagonal) {
                    sum_eigenvalues += eigenvalues;
                }
            }
            
            // Averages over disorder realizations
            param1_values[index] = param1;
            param2_values[index] = param2;
            gap_ratios_values[index] = sum_gap_ratio * inv_r;
            condensate_fraction_values[index] = sum_condensate_fraction * inv_r;
            fluctuations_values[index] = sum_fluctuations * inv_r;
            if (compute_qEA) {
                const Eigen::VectorXd q2_avg = q2 * inv_r;
                qEA_values[index] = ((q1 * inv_r - q2_avg.array().square().matrix()).sum() * inv_m);
            } else {
                qEA_values[index] = 0.0;
            }
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
            eigen_file << " " << diagonal_eigenvalues[k][e];
        }
        eigen_file << std::endl;
    }
    eigen_file.close();
}

/* Compute parameter value at a given index based on scale type */
double Analysis::compute_param(double p_min, double p_max, int idx, int num_points, ScaleType scale) {
    if (num_points <= 1) return p_min;
    
    switch (scale) {
        case ScaleType::Logarithmic: {
            const double log_min = std::log10(p_min);
            const double log_max = std::log10(p_max);
            const double log_step = (log_max - log_min) / (num_points - 1);
            return std::pow(10.0, log_min + idx * log_step);
        }
        case ScaleType::Linear:
        default: {
            const double lin_step = (p_max - p_min) / (num_points - 1);
            return p_min + idx * lin_step;
        }
    }
}

/* Precompute all parameter values for a range */
std::vector<double> Analysis::precompute_params(double p_min, double p_max, int num_points, ScaleType scale) {
    std::vector<double> params(num_points);
    if (num_points <= 1) {
        if (num_points == 1) params[0] = p_min;
        return params;
    }
    switch (scale) {
        case ScaleType::Logarithmic: {
            const double log_min = std::log10(p_min);
            const double log_step = (std::log10(p_max) - log_min) / (num_points - 1);
            for (int i = 0; i < num_points; ++i) {
                params[i] = std::pow(10.0, log_min + i * log_step);
            }
            break;
        }
        case ScaleType::Linear:
        default: {
            const double lin_step = (p_max - p_min) / (num_points - 1);
            for (int i = 0; i < num_points; ++i) {
                params[i] = p_min + i * lin_step;
            }
            break;
        }
    }
    return params;
}

        /* GAP RATIOS */

/* Calculate the energy gap ratios of the system and return their sum */
double Analysis::gap_ratios(const Eigen::VectorXd& eigenvalues, int nb_eigen) {
    double sum = 0.0;
    
    // Calculate the gap ratios and sum them
    for (int i = 1; i < nb_eigen - 1; ++i) {
        const double E_prev = eigenvalues[i - 1];
        const double E_curr = eigenvalues[i];
        const double E_next = eigenvalues[i + 1];
        const double gap1 = E_next - E_curr;
        const double gap2 = E_curr - E_prev;
        const double min_gap = std::min(gap1, gap2);
        const double max_gap = std::max(gap1, gap2);
        sum += (max_gap != 0.0) ? (min_gap / max_gap) : 0.0;
    }
    return sum / (nb_eigen - 2);
}


        /* SPDM FUNCTIONS */

/* Calculate the single-particle density matrix of the system */
Eigen::MatrixXcd Analysis::SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::VectorXcd& phi0) {
    const int m = basis.rows();
    const int D = basis.cols();
    Eigen::MatrixXcd spdm = Eigen::MatrixXcd::Zero(m, m);
    
    // Precompute probabilities
    Eigen::VectorXd probs(D);
    for (int k = 0; k < D; ++k) {
        probs[k] = std::norm(phi0[k]);
    }
    
    // Build hash map once for O(1) lookup
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