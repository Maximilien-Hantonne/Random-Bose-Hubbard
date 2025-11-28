#include <cmath>
#include <vector>
#include <limits>
#include <complex>
#include <fstream>
#include <numeric>
#include <iostream>
#include <algorithm>

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
void Analysis::exact_parameters(int m, int n, double J,double U, double mu, double s, double r, std::string fixed_param) {
    
    // Prerequisites
    if (std::abs(J-0.0) < std::numeric_limits<double>::epsilon() && std::abs(U-0.0) < std::numeric_limits<double>::epsilon() && std::abs(mu-0.0) < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Error: At least one of the parameters J, U, mu must be different from zero." << std::endl;
        return;
    }

    // Start of the calculations
    Resource::timer();

    // Set the geometry of the lattice
    Neighbours neighbours(m);
    neighbours.chain_neighbours();
    const std::vector<std::vector<int>>& nei = neighbours.getNeighbours();

    // // Set the matrices for each term of the Hamiltonian in the Fock states from 1 to n bosons
    // int n_min = 1, n_max = n;
    // auto [tags, basis] = BH::max_set_basis(m, n);
    // Eigen::SparseMatrix<double> JH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 1, 0, 0);
    // Eigen::SparseMatrix<double> UH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 1, 0);
    // Eigen::SparseMatrix<double> uH = BH::max_bosons_hamiltonian(nei, m, n_min, n_max, 0, 0, 1);
    
    // Set the Fock states basis with a fixed number of bosons with their tags
    auto [tags, basis] = BH::fixed_set_basis(m, n);

    // Set the matrices for each term of the Hamiltonian in the Fock states with n bosons
    Eigen::SparseMatrix<double> JH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 1, 0, 0);
    Eigen::SparseMatrix<double> UH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 1, 0);
    Eigen::SparseMatrix<double> uH = BH::fixed_bosons_hamiltonian(nei, basis, tags, m, n, 0, 0, 1);

    // Set the number of threads for the calculations
    Resource::set_omp_threads(JH, 3);

    // Set the range of parameters for the calculations
    double J_min = J, J_max = J + r, mu_min = mu, mu_max = mu + r, U_min = U, U_max = U + r;

    // Calculate the exact parameters
    if (fixed_param == "J") {
        JH = JH * J;
        calculate_and_save(basis, tags, JH, UH, uH, fixed_param, J, J_min, J_max, mu_min, mu_max, s, s);
    }
    else if (fixed_param == "U") {
        UH = UH * U;
        calculate_and_save(basis, tags, UH, JH, uH, fixed_param, U, J_min, J_max, U_min, U_max, s, s);
    }
    else{
        uH = uH * mu;
        calculate_and_save(basis, tags, uH, JH, UH, fixed_param, mu, J_min, J_max, mu_min, mu_max, s, s);
    }

    // End of the calculations
    std::cout << " - ";
    Resource::timer();
    std::cout << " - ";
    Resource::get_memory_usage(true);
    std::cout << std::endl;
}


/* calculate and save gap ratio and other quantities */
void Analysis::calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double> H_fixed, const Eigen::SparseMatrix<double> H1, const Eigen::SparseMatrix<double> H2, std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step) {
    
    // Save the fixed parameter and value in a file
    std::ofstream file("phase.txt");
    file << fixed_param << " ";
    file << fixed_value << std::endl;
    
    // Parameters for the calculations
    int nb_eigen = 20;

    // Matrices initialization
    int num_param1 = static_cast<int>((param1_max - param1_min) / param1_step) + 1;
    int num_param2 = static_cast<int>((param2_max - param2_min) / param2_step) + 1;
    std::vector<double> param1_values(num_param1 * num_param2);
    std::vector<double> param2_values(num_param1 * num_param2);
    std::vector<double> gap_ratios_values(num_param1 * num_param2);
    std::vector<double> condensate_fraction_values(num_param1 * num_param2);
    std::vector<double> coherence_values(num_param1 * num_param2);
    Eigen::MatrixXcd eigenvectors;
    Eigen::MatrixXd matrix_ratios(num_param1 * num_param2, nb_eigen -2);
    std::vector<Eigen::MatrixXd> spdm_matrices;

    // Progress tracking
    std::atomic<int> progress_counter(0);
    int total_iterations = num_param1 * num_param2;

    // Threshold for the loop
    double variance_threshold_percent = 1e-8;

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
                Eigen::EigenSolver<Eigen::MatrixXd> solver(spdm.real());
                double max_eigenval = solver.eigenvalues().real().maxCoeff();
                double condensate_fraction = std::abs(max_eigenval / spdm.trace());

                // Coherence
                double K;
                K = coherence(spdm);

                int index = i * num_param1 + j;
                
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
        matrix_ratios.resize(num_param1 * num_param2, nb_eigen - 2);  
    }

    // Save the results to a file
    for (int i = 0; i < num_param1 * num_param2; ++i) {
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
    // save_cluster_labels("cluster_labels.csv", cluster_labels);
}


        /* GAP RATIOS */

/* Calculate the energy gap ratios of the system */
Eigen::VectorXd Analysis::gap_ratios(Eigen::VectorXcd eigenvalues,int nb_eigen) {
    Eigen::VectorXd gap_ratios(nb_eigen - 2);

    // Sort the eigenvalues by their real part
    std::vector<double> sorted_eigenvalues(nb_eigen);
    for (int i = 0; i < nb_eigen; ++i) {
        sorted_eigenvalues[i] = std::real(eigenvalues[i]);
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
Eigen::MatrixXcd Analysis::SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, Eigen::MatrixXcd& eigenvectors) {
    
    // Initialize the single-particle density matrix
    int m = basis.rows();
    Eigen::MatrixXcd spdm = Eigen::MatrixXcd::Zero(m, m);

    // Calculate the braket for the ground state
    Eigen::VectorXd phi0 = eigenvectors.col(0).real();
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            spdm(i, j) = braket(phi0, basis, tags, i, j);
        }
        for (int j = 0; j < i; ++j) {
            spdm(i, j) = std::conj(spdm(j, i));
        }
    }
    return spdm/ eigenvectors.cols();
}

/* Calculate the fluctuations of the system */
double Analysis::coherence(const Eigen::MatrixXcd& spdm) {
    double sum_all = 0;
    double sum_diag = 0;

    // Calculate the sum of the squares of the elements of the spdm
    for (int i = 0; i < spdm.rows(); ++i) {
        double diag_val = std::real(spdm(i, i) * std::conj(spdm(i, i)));
        sum_diag += diag_val;
        sum_all += diag_val;
        for (int j = i + 1; j < spdm.cols(); ++j) {
            sum_all += 2.0 * std::real(spdm(i, j) * std::conj(spdm(i, j)));
        }
    }
    return (sum_all - sum_diag)/sum_all;
}


        /* BRAKET CALCULATIONS */

/* Calculate the mean value of the operator <phi0|ai+ aj|phi0> */
std::complex<double> Analysis::braket(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int i, int j) {
    
    // Initialization
    std::complex<double> braket_value = 0 ;
    static const std::vector<int> primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
   
   // Loop over the Fock states basis
    for (int k = 0; k < basis.cols(); ++k) {

        // If the coefficient of the decomposition of the state is different from zero
        if (std::abs(phi0[k]) > std::numeric_limits<double>::epsilon()) {
            Eigen::VectorXd state = basis.col(k);

            // If the state has at least one boson at site j
            if (state[i] >= 0 && state[j] >= 1) {
                state[i] += 1;
                state[j] -= 1;

                // Calculate the tag of the new state after applying (ai+ aj)
                double x = BH::calculate_tag(state, primes, i);

                // Search the index of the new state in the basis
                int index = BH::search_tag(tags, x);

                // If the coefficient of the decomposition of the new state is different from zero
                if (std::abs(phi0[index]) > std::numeric_limits<double>::epsilon()) {
                    braket_value += std::conj(phi0[k]) * phi0[index] * sqrt(state[i] * state[j]);
                }
            }
        }
    }
    return braket_value;
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