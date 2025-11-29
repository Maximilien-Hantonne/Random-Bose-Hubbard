#include <cmath>
#include <vector>
#include <limits>
#include <complex>
#include <fstream>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <array>
#include <map>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

#include "Eigen/src/Core/Matrix.h"
#include "operator.hpp"
#include "analysis.hpp"
#include "hamiltonian.hpp"
#include "resource.hpp"
#include "neighbours.hpp"


        /* MAIN FUNCTIONS */

/**
 * @brief Main function for exact calculations parameters.
 *
 * This function performs exact calculations for the Bose-Hubbard model parameters.
 * It sets up the lattice geometry, constructs the Hamiltonian matrices, and calculates
 * various physical quantities over a range of parameters. The results are saved to files.
 *
 * @param m The number of lattice sites.
 * @param n The number of bosons.
 * @param J The hopping parameter.
 * @param U The on-site interaction parameter.
 * @param mu The chemical potential.
 * @param s The step size for the parameter sweep.
 * @param r The range for the parameter sweep.
 * @param fixed_param The parameter to be fixed during the calculations ("J", "U", or "mu").
 */

/*main function for exact calculations parameters*/
void Analysis::exact_parameters(int m, int n, double J, double U, double mu, double s, double r, const std::string& fixed_param) {
    
    // Prerequisites
    constexpr double EPSILON = std::numeric_limits<double>::epsilon();
    if (std::abs(J) < EPSILON && std::abs(U) < EPSILON && std::abs(mu) < EPSILON) {
        std::cerr << "Error: At least one of the parameters J, U, mu must be different from zero." << std::endl;
        return;
    }

    // Start of the calculations
    Resource::timer();

    // Set the geometry of the lattice
    Neighbours neighbours(m);
    neighbours.chain_neighbours();
    // neighbours.square_neighbours();
    const std::vector<std::vector<int>>& nei = neighbours.getNeighbours();

    // // Set the matrices for each term of the Hamiltonian in the Fock states from 1 to n bosons
    // int n_min = 1, n_max = n;
    // auto [tags, basis] = BH::max_set_basis(m, n);
    // Eigen::SparseMatrix<double> JH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 1, 0, 0);
    // Eigen::SparseMatrix<double> UH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 1, 0);
    // Eigen::SparseMatrix<double> uH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 0, 1);
    
    // Set the Fock states basis with a fixed number of bosons with their tags
    const auto [tags, basis] = BH::fixed_set_basis(m, n);

    // Set the matrices for each term of the Hamiltonian in the Fock states with n bosons
    const Eigen::SparseMatrix<double> JH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 1, 0, 0);
    const Eigen::SparseMatrix<double> UH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 1, 0);
    const Eigen::SparseMatrix<double> uH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 0, 1);

    // Set the number of threads for the calculations
    Resource::set_omp_threads(JH, 3);

    // Set the range of parameters for the calculations
    double J_min = J, J_max = J + r, mu_min = mu, mu_max = mu + r, U_min = U, U_max = U + r;

    // Calculate the exact parameters
    if (fixed_param == "J") {
        const Eigen::SparseMatrix<double> scaled_JH = JH * J;
        calculate_and_save(basis, tags, scaled_JH, UH, uH, fixed_param, J, J_min, J_max, mu_min, mu_max, s, s);
    }
    else if (fixed_param == "U") {
        const Eigen::SparseMatrix<double> scaled_UH = UH * U;
        calculate_and_save(basis, tags, scaled_UH, JH, uH, fixed_param, U, J_min, J_max, U_min, U_max, s, s);
    }
    else{
        const Eigen::SparseMatrix<double> scaled_uH = uH * mu;
        calculate_and_save(basis, tags, scaled_uH, JH, UH, fixed_param, mu, J_min, J_max, mu_min, mu_max, s, s);
    }

    // End of the calculations
    std::cout << " - ";
    Resource::timer();
    std::cout << " - ";
    Resource::get_memory_usage(true);
    std::cout << std::endl;
}


/* calculate and save gap ratio and other quantities */
void Analysis::calculate_and_save(const Eigen::MatrixXi& basis, const std::vector<boost::multiprecision::cpp_int>& tags, const Eigen::SparseMatrix<double>& H_fixed, const Eigen::SparseMatrix<double>& H1, const Eigen::SparseMatrix<double>& H2, const std::string& fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step) {
    
    // Save the fixed parameter and value in a file
    std::ofstream file("phase.txt");
    file << fixed_param << " ";
    file << fixed_value << std::endl;
    
    // Parameters for the calculations
    int nb_eigen = 20;

    // Matrices initialization
    const int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
    const int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
    const int total_elements = num_param1 * num_param2;
    
    std::vector<double> param1_values(total_elements);
    std::vector<double> param2_values(total_elements);
    std::vector<double> gap_ratios_values(total_elements);
    std::vector<double> condensate_fraction_values(total_elements);
    std::vector<double> coherence_values(total_elements);
    Eigen::MatrixXcd eigenvectors;
    Eigen::MatrixXd matrix_ratios(total_elements, nb_eigen - 2);
    std::vector<Eigen::MatrixXd> spdm_matrices;

    // Progress tracking
    std::atomic<int> progress_counter(0);
    const int total_iterations = total_elements;

    // Threshold for the loop
    constexpr double variance_threshold_percent = 1e-8;

    // Main loop for the calculations with parallelization
    while (true) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < num_param1; ++i) {
            for (int j = 0; j < num_param2; ++j) {

                // Parameters
                double param1 = param1_min + i * param1_step;
                double param2 = param2_min + j * param2_step;

                // Hamiltonian
                Eigen::SparseMatrix<double> H = H_fixed + H1 * param1 + H2 * param2;

                // Diagonalization
                Eigen::VectorXcd eigenvalues = Op::IRLM_eigen(H, nb_eigen, eigenvectors);

                // Gap ratios
                Eigen::VectorXd vec_ratios = gap_ratios(eigenvalues, nb_eigen);
                double gap_ratio = vec_ratios.size() > 0 ? vec_ratios.sum() / vec_ratios.size() : 0.0;

                // SPDM
                Eigen::MatrixXcd spdm;
                spdm = SPDM(basis, tags, eigenvectors);

                // Condensate fraction
                Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(spdm);
                const double lambda_0 = solver.eigenvalues().real().maxCoeff();
                // std::cout << "lambda " << lambda_0 << std::endl;
                const double N = std::real(spdm.trace());
                // std::cout << "N" << N << std::endl;
                double condensate_fraction = (std::abs(N) > std::numeric_limits<double>::epsilon()) 
                                           ? lambda_0 / N : 0.0;

                // Coherence
                double K;
                K = coherence(spdm);

                const int index = i * num_param2 + j;
                
                param1_values[index] = param1;
                param2_values[index] = param2;
                gap_ratios_values[index] = gap_ratio;
                condensate_fraction_values[index] = condensate_fraction;
                coherence_values[index] = K;
                matrix_ratios.row(index) = vec_ratios;
                
                #pragma omp critical
                {
                    if(j == num_param2 - i - 1){
                        spdm_matrices.push_back(spdm.real());
                    }
                }
                int local_count = progress_counter.fetch_add(1) + 1;
                if (local_count % 10 == 0 || local_count == total_iterations) {
                    int percent = (local_count * 100) / total_iterations;
                    std::cout << "\r" << percent << "%" << std::flush;
                }
            }
        }

        // Calculate the mean of the gap ratios
        double mean_gap_ratio = std::accumulate(gap_ratios_values.begin(), gap_ratios_values.end(), 0.0) / gap_ratios_values.size();
        
        // Calculate the variance of the gap ratios
        double variance_gap_ratio = 0.0;
        for (const auto& value : gap_ratios_values) {
            variance_gap_ratio += (value - mean_gap_ratio) * (value - mean_gap_ratio);
        }
        variance_gap_ratio /= gap_ratios_values.size();

        // Continue in the loop if the variance is below the threshold
        if (variance_gap_ratio > variance_threshold_percent * mean_gap_ratio) {
            break;
        }

        // Increase the number of eigenvalues and resize the matrix
        nb_eigen += 5;
        matrix_ratios.resize(total_elements, nb_eigen - 2);  
    }

    // Save the results to a file
    for (int i = 0; i < total_elements; ++i) {
        file << param1_values[i] << " " << param2_values[i] << " " << gap_ratios_values[i] << " " << condensate_fraction_values[i] << " " << coherence_values[i] << std::endl;
    }
    file.close();
    
    // // PCA, dispersion, and clustering initialization
    // std::vector<Eigen::MatrixXd> pca_matrices;
    // std::vector<double> dispersions;
    // std::vector<Eigen::VectorXi> cluster_labels;

    // // Main loop for the PCA, dispersion, and clustering
    // int num_rows = 3;
    // for (int i = 0; i < param1_max; ++i) {

    //     // Choose a subset of the matrix to analyze
    //     int start_row = static_cast<int>(std::max(param2_max - num_rows - i * (param2_max - num_rows) / param1_max, 0.0));
    //     int end_row = static_cast<int>(std::max(param2_max - i * (param2_max - num_rows) / param1_max, 0.0 + num_rows));
    //     Eigen::MatrixXd sub_matrix = matrix_ratios.block(i + start_row, 0, i + end_row, matrix_ratios.cols());

    //     // PCA
    //     sub_matrix = standardize_matrix(sub_matrix);
    //     sub_matrix = (sub_matrix.adjoint() * sub_matrix) / double(sub_matrix.rows() - 1);
    //     Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(sub_matrix);
    //     Eigen::MatrixXd eigenvectors2 = eigensolver.eigenvectors().real().rowwise().reverse();
    //     Eigen::MatrixXd projected_data = sub_matrix * eigenvectors2.leftCols(2);
    //     pca_matrices.push_back(projected_data);

    //     // Dispersion
    //     double dispersion = calculate_dispersion(projected_data);
    //     dispersions.push_back(dispersion);
    // }

    // // Save the results to a file
    // save_matrices_to_csv("spdm_matrices.csv", spdm_matrices, "Matrix");
    // save_matrices_to_csv("pca_results.csv", pca_matrices, "PCA");
    // save_dispersions("dispersions.csv", dispersions);
}


        /* GAP RATIOS */

/* Calculate the energy gap ratios of the system */
Eigen::VectorXd Analysis::gap_ratios(Eigen::VectorXcd eigenvalues, int nb_eigen) {
    Eigen::VectorXd gap_ratios(nb_eigen - 2);

    // Sort the eigenvalues by their real part
    std::vector<double> sorted_eigenvalues;
    sorted_eigenvalues.reserve(nb_eigen);
    for (int i = 0; i < nb_eigen; ++i) {
        sorted_eigenvalues.emplace_back(std::real(eigenvalues[i]));
    }
    std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end());

    // Calculate the gap ratios
    for (int i = 1; i < nb_eigen - 1; ++i) {
        double E_prev = sorted_eigenvalues[i - 1];
        double E_curr = sorted_eigenvalues[i];
        double E_next = sorted_eigenvalues[i + 1];
        double min_gap = std::min(E_next - E_curr, E_curr - E_prev);
        double max_gap = std::max(E_next - E_curr, E_curr - E_prev);
        gap_ratios[i - 1] = (max_gap != 0) ? (min_gap / max_gap) : 0;
        }

    return gap_ratios;
}


        /* SPDM FUNCTIONS */

/* Calculate the single-particle density matrix of the system */
Eigen::MatrixXcd Analysis::SPDM(const Eigen::MatrixXi& basis, const std::vector<boost::multiprecision::cpp_int>& tags, Eigen::MatrixXcd& eigenvectors) {
    int m = basis.rows();
    int D = basis.cols();
    Eigen::MatrixXcd spdm = Eigen::MatrixXcd::Zero(m, m);

    Eigen::VectorXcd phi0 = eigenvectors.col(0);

    std::map<boost::multiprecision::cpp_int, int> tag_index;
    for (int k = 0; k < D; ++k) {
        tag_index.emplace(tags[k], k);
    }
    static const std::vector<int> primes = { 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 };

    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            std::complex<double> sum = 0.0;

            if (i == j) {
                for (int k = 0; k < D; ++k) {
                    double ni = static_cast<double>(basis(i, k));
                    double prob = std::norm(phi0[k]);
                    sum += prob * ni;
                }
            } else {
                for (int k = 0; k < D; ++k) {
                    int ni = basis(i, k);
                    int nj = basis(j, k);
                    if (nj <= 0) continue;
                    double factor = std::sqrt(double(ni + 1) * double(nj));
                    Eigen::VectorXi state = basis.col(k);
                    state(i) += 1;
                    state(j) -= 1;
                    boost::multiprecision::cpp_int target_tag = BH::calculate_tag(state, primes);
                    auto it = tag_index.find(target_tag);
                    if (it != tag_index.end()) {
                        int l = it->second;
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

/* Calculate the mean value of the operator <phi0|ai+ aj|phi0> */
std::complex<double> Analysis::braket(const Eigen::VectorXcd& phi0, const Eigen::MatrixXi& basis, const std::vector<boost::multiprecision::cpp_int>& tags, int i, int j){
    std::complex<double> val = 0.0;
    const int D = basis.cols();
    if (i == j) {
        for (int k = 0; k < D; ++k) {
            double ni = static_cast<double>(basis(i, k));
            double prob = std::norm(phi0[k]);
            val += prob * ni;
        }
        return val;
    }
    static const std::vector<int> primes = { 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 };
    for (int k = 0; k < D; ++k) {
        int ni = basis(i, k);
        int nj = basis(j, k);
        if (nj < 1) continue;                     
        double factor = std::sqrt(double(ni + 1) * double(nj));
        Eigen::VectorXi state = basis.col(k);
        state(i) += 1;
        state(j) -= 1;
        boost::multiprecision::cpp_int new_tag = BH::calculate_tag(state, primes);
        int idx = BH::search_tag(tags, new_tag);
        if (idx >= 0) {
            val += std::conj(phi0[k]) * phi0[idx] * factor;
        }
    }
    return val;
}


/* Calculate the fluctuations of the system */
double Analysis::coherence(const Eigen::MatrixXcd& spdm) {
    double sum_all = 0.0;
    double sum_diag = 0.0;

    // Calculate the sum of the squares of the elements of the spdm
    for (int i = 0; i < spdm.rows(); ++i) {
        const double diag_norm = std::norm(spdm(i, i));
        sum_diag += diag_norm;
        sum_all += diag_norm;
        for (int j = i + 1; j < spdm.cols(); ++j) {
            sum_all += 2.0 * std::norm(spdm(i, j));
        }
    }
    return (sum_all - sum_diag) / sum_all;
}

                    ///// UTILITY FUNCTIONS /////


        /* PCA FUNCTIONS */

/* calculate the dispersion of the projected points */
double Analysis::calculate_dispersion(const Eigen::MatrixXd& projected_data) {
    if (projected_data.rows() <= 1) {
        return 0.0;
    }

    // Calculate the mean of the projected data
    const Eigen::VectorXd mean = projected_data.colwise().mean();

    // Calculate the variances directly without creating centered matrix
    double total_variance = 0.0;
    for (int col = 0; col < projected_data.cols(); ++col) {
        double variance = 0.0;
        for (int row = 0; row < projected_data.rows(); ++row) {
            const double diff = projected_data(row, col) - mean(col);
            variance += diff * diff;
        }
        total_variance += variance / (projected_data.rows() - 1);
    }

    return total_variance;
}


        /* STANDARDIZATION */

/* standardize a matrix */
Eigen::MatrixXd Analysis::standardize_matrix(const Eigen::MatrixXd& matrix) {
    if (matrix.rows() <= 1) {
        return matrix;
    }

    // Calculate the mean and standard deviation of the matrix
    const Eigen::VectorXd mean = matrix.colwise().mean();
    
    // Calculate standard deviation more efficiently
    Eigen::VectorXd variance = Eigen::VectorXd::Zero(matrix.cols());
    for (int col = 0; col < matrix.cols(); ++col) {
        for (int row = 0; row < matrix.rows(); ++row) {
            const double diff = matrix(row, col) - mean(col);
            variance(col) += diff * diff;
        }
        variance(col) /= (matrix.rows() - 1);
    }
    const Eigen::VectorXd stddev = variance.cwiseSqrt();
    
    // Create standardized matrix
    Eigen::MatrixXd standardized_matrix = matrix;
    for (int col = 0; col < matrix.cols(); ++col) {
        if (stddev(col) > std::numeric_limits<double>::epsilon()) {
            for (int row = 0; row < matrix.rows(); ++row) {
                standardized_matrix(row, col) = (matrix(row, col) - mean(col)) / stddev(col);
            }
        }
    }
    
    return standardized_matrix;
}


        /* SAVE TO CSV */

/* Save the real part of matrices to a CSV file */
void Analysis::save_matrices_to_csv(const std::string& filename, const std::vector<Eigen::MatrixXd>& matrices, const std::string& label) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    
    for (size_t k = 0; k < matrices.size(); ++k) {
        file << label << " " << k << '\n';
        const Eigen::MatrixXd& matrix = matrices[k];
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                file << matrix(i, j);
                if (j < matrix.cols() - 1) {
                    file << ',';
                }
            }
            file << '\n';
        }
        file << '\n';
    }
}

/* Save the dispersions to a file */
void Analysis::save_dispersions(const std::string& filename, const std::vector<double>& dispersions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    
    for (size_t k = 0; k < dispersions.size(); ++k) {
        file << "Dispersion " << k << '\n'
             << dispersions[k] << "\n\n";
    }
}