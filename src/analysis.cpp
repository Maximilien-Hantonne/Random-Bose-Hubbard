#include <cmath>
#include <vector>
#include <limits>
#include <complex>
#include <fstream>
#include <iostream>
#include <algorithm>

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

/**
 * @brief Main function for exact calculations parameters.
 *
 * This function performs exact calculations for the Bose-Hubbard model parameters.
 * It sets up the lattice geometry, constructs the Hamiltonian matrices, and calculates
 * various physical quantities over a range of parameters. The results are saved to files.
 *
 * @param m The number of lattice sites.
 * @param n The number of bosons.
 * @param T The hopping parameter.
 * @param U The on-site interaction parameter.
 * @param mu The chemical potential.
 * @param s The step size for the parameter sweep.
 * @param r The range for the parameter sweep.
 * @param fixed_param The parameter to be fixed during the calculations ("T", "U", or "mu").
 */

/*main function for exact calculations parameters*/
void Analysis::exact_parameters(int m, int n, double T,double U, double mu, double s, double r, std::string fixed_param, double sigma_t, double sigma_U, double sigma_u) {
    
    // Prerequisites
    if (std::abs(T-0.0) < std::numeric_limits<double>::epsilon() && std::abs(U-0.0) < std::numeric_limits<double>::epsilon() && std::abs(mu-0.0) < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Error: At least one of the parameters T, U, mu must be different from zero." << std::endl;
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
    auto [tags, basis] = BH::fixed_set_basis(m, n);
    Eigen::SparseMatrix<double> TH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 1, 0, 0);
    Eigen::SparseMatrix<double> UH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 1, 0);
    Eigen::SparseMatrix<double> uH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 0, 1);

    // Set the number of threads for the calculations
    Resource::set_omp_threads(TH, 3);

    // Set the range of parameters for the calculations
    double T_min = T, T_max = T + r, mu_min = mu, mu_max = mu + r, U_min = U, U_max = U + r;

    // Calculate the exact parameters
    calculate_and_save(basis, tags, TH, UH, uH, fixed_param, T, U, mu, T_min, T_max, U_min, U_max, mu_min, mu_max, s, s, sigma_t, sigma_U, sigma_u);

    // End of the calculations
    std::cout << " - ";
    Resource::timer();
    std::cout << " - ";
    Resource::get_memory_usage(true);
    std::cout << std::endl;
}


/* calculate and save gap ratio and other quantities */
void Analysis::calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double>& TH, const Eigen::SparseMatrix<double>& UH, const Eigen::SparseMatrix<double>& uH, const std::string& fixed_param, const double T, const double U, const double mu, const double T_min, const double T_max, const double U_min, const double U_max, const double mu_min, const double mu_max, const double param1_step, const double param2_step, const double sigma_T, const double sigma_U, const double sigma_u) {
    
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
    std::vector<double> param1_values(num_param1 * num_param2);
    std::vector<double> param2_values(num_param1 * num_param2);
    std::vector<double> gap_ratios_values(num_param1 * num_param2);
    std::vector<double> condensate_fraction_values(num_param1 * num_param2);
    std::vector<double> fluctuations_values(num_param1 * num_param2);
    Eigen::MatrixXd matrix_ratios(num_param1 * num_param2, nb_eigen -2);
    std::vector<Eigen::MatrixXd> spdm_matrices;
    std::vector<double> diagonal_ratios;
    std::vector<Eigen::VectorXcd> diagonal_eigenvalues;

    // Progress tracking
    std::atomic<int> progress_counter(0);
    const int total_iterations = num_param1 * num_param2;

    // Spacing parameters
    const double log_param1_min = std::log10(param1_min);
    const double log_param1_max = std::log10(param1_max);
    const double log_param2_min = std::log10(param2_min);
    const double log_param2_max = std::log10(param2_max);
    const double log_param1_step = (log_param1_max - log_param1_min) / (num_param1 - 1);
    const double log_param2_step = (log_param2_max - log_param2_min) / (num_param2 - 1);

    // Main loop for the calculations with parallelization
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < num_param1; ++i) {
        for (int j = 0; j < num_param2; ++j) {

            const double param1 = std::pow(10.0, log_param1_min + i * log_param1_step);
            const double param2 = std::pow(10.0, log_param2_min + j * log_param2_step);
            
            const double T_val = is_T_fixed ? T : param1;
            const double U_val = is_U_fixed ? U : (is_T_fixed ? param1 : param2);
            const double mu_val = is_T_fixed ? param2 : (is_U_fixed ? param2 : mu);

            const Eigen::SparseMatrix<double> H = BH::random_hamiltonian(TH, T_val, sigma_T, UH, U_val, sigma_U, uH, mu_val, sigma_u);

            // Diagonalization
            Eigen::MatrixXcd eigenvectors;
            Eigen::VectorXcd eigenvalues = Op::IRLM_eigen(H, nb_eigen, eigenvectors);
            Op::sort_eigen(eigenvalues, eigenvectors);

            // Gap ratios
            const Eigen::VectorXd vec_ratios = gap_ratios(eigenvalues, nb_eigen);
            const double gap_ratio = vec_ratios.sum() / vec_ratios.size();

            // Extract ground state once for both SPDM and fluctuations
            const Eigen::VectorXcd& phi0 = eigenvectors.col(0);

            // SPDM
            const Eigen::MatrixXcd spdm = SPDM(basis, tags, phi0);

            // Condensate fraction
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(spdm);
            const double lambda_0 = solver.eigenvalues().maxCoeff();
            const double N = spdm.trace().real();
            const double condensate_fraction = lambda_0 / N; 

            // Local fluctuation
            const double fluctuations = local_fluctu(phi0, basis);

            const int index = i * num_param2 + j;
                
            param1_values[index] = param1;
            param2_values[index] = param2;
            gap_ratios_values[index] = gap_ratio;
            condensate_fraction_values[index] = condensate_fraction;
            fluctuations_values[index] = fluctuations;
            matrix_ratios.row(index) = vec_ratios;
                
            #pragma omp critical
            {
                if(j == num_param2 - i - 1){
                    spdm_matrices.push_back(spdm.real());
                    diagonal_eigenvalues.push_back(eigenvalues);
                    diagonal_ratios.push_back(param1 / param2);
                }
            }
            const int local_count = progress_counter.fetch_add(1) + 1;
            if (local_count % 10 == 0 || local_count == total_iterations) {
                const int percent = (local_count * 100) / total_iterations;
                std::cout << "\r" << percent << "%" << std::flush;
            }
        }
    }

    // Save the results to a file
    for (int i = 0; i < num_param1 * num_param2; ++i) {
        file << param1_values[i] << " " << param2_values[i] << " " << gap_ratios_values[i] << " " << condensate_fraction_values[i] << " " << fluctuations_values[i] << std::endl;
    }
    file.close();

    // Save diagonal eigenvalues to file
    std::ofstream eigen_file("eigenvalues_diagonal.txt");
    std::string param1_name, param2_name;
    if (fixed_param == "T") {
        param1_name = "U";
        param2_name = "mu";
    } else if (fixed_param == "U") {
        param1_name = "T";
        param2_name = "mu";
    } else {
        param1_name = "T";
        param2_name = "U";
    }
    eigen_file << "# Ratio " << param1_name << "/" << param2_name << " Eigenvalues" << std::endl;
    for (size_t k = 0; k < diagonal_eigenvalues.size(); ++k) {
        eigen_file << diagonal_ratios[k];
        for (int e = 0; e < diagonal_eigenvalues[k].size(); ++e) {
            eigen_file << " " << diagonal_eigenvalues[k][e].real();
        }
        eigen_file << std::endl;
    }
    eigen_file.close();
    
    // // PCA, dispersion
    // std::vector<Eigen::MatrixXd> pca_matrices;
    // std::vector<double> dispersions;

    // // Main loop for the PCA and dispersion
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
                    const double ni = basis(i, k);
                    const double prob = std::norm(phi0[k]);
                    sum += prob * ni;
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

/* Calculate the mean value of the operator <phi0|ai+ aj|phi0> */
std::complex<double> Analysis::braket(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int i, int j){
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
        double ni = basis(i, k);
        double nj = basis(j, k);
        if (nj < 1) continue;                     
        double factor = std::sqrt((ni + 1) * nj);
        Eigen::VectorXd state = basis.col(k);
        state(i) += 1;
        state(j) -= 1;
        double new_tag = BH::calculate_tag(state, primes);
        int idx = BH::search_tag(tags, new_tag);
        if (idx >= 0) {
            val += std::conj(phi0[k]) * phi0[idx] * factor;
        }
    }
    return val;
}

double Analysis::local_fluctu(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis){
    const int m = basis.rows();
    const int D = basis.cols();
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double mean_n = 0.0;
        double mean_n2 = 0.0;
        for (int k = 0; k < D; ++k) {
            const double prob = std::norm(phi0[k]);
            const double ni = basis(i, k);
            mean_n += prob * ni;
            mean_n2 += prob * ni * ni;
        }
        sum += mean_n2 - mean_n * mean_n;
    }
    return sum / m;
}

                    ///// UTILITY FUNCTIONS /////


        /* PCA FUNCTIONS */

/* calculate the dispersion of the projected points */
double Analysis::calculate_dispersion(const Eigen::MatrixXd& projected_data) {

    // Calculate the mean of the projected data
    Eigen::VectorXd mean = projected_data.colwise().mean();

    // Center the projected data
    Eigen::MatrixXd centered = projected_data.rowwise() - mean.transpose();

    // Calculate the variances of the projected data
    Eigen::VectorXd variances = (centered.array().square().colwise().sum() / (centered.rows() - 1)).matrix();
    double dispersion = variances.sum();

    return dispersion;
}


        /* STANDARDIZATION */

/* standardize a matrix */
Eigen::MatrixXd Analysis::standardize_matrix(const Eigen::MatrixXd& matrix) {

    // Initialize the standardized matrix
    Eigen::MatrixXd standardized_matrix = matrix;

    // Calculate the mean and standard deviation of the matrix
    Eigen::VectorXd mean = matrix.colwise().mean();
    Eigen::VectorXd stddev = ((matrix.rowwise() - mean.transpose()).array().square().colwise().sum() / (matrix.rows() - 1)).sqrt();
    
    // Standardize the matrix
    standardized_matrix = (matrix.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();
    return standardized_matrix;
}


        /* SAVE TO CSV */

/* Save the real part of matrices to a CSV file */
void Analysis::save_matrices_to_csv(const std::string& filename, const std::vector<Eigen::MatrixXd>& matrices, const std::string& label) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t k = 0; k < matrices.size(); ++k) {
            file << label << " " << k << std::endl;
            const Eigen::MatrixXd& matrix = matrices[k];
            for (int i = 0; i < matrix.rows(); ++i) {
                for (int j = 0; j < matrix.cols(); ++j) {
                    file << matrix(i, j);
                    if (j < matrix.cols() - 1) {
                        file << ",";
                    }
                }
                file << std::endl;
            }
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

/* Save the dispersions to a file */
void Analysis::save_dispersions(const std::string& filename, const std::vector<double>& dispersions) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t k = 0; k < dispersions.size(); ++k) {
            file << "Dispersion " << k << std::endl;
            file << dispersions[k] << std::endl;
            file << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}