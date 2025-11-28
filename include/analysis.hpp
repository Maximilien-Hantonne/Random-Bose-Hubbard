#pragma once

#include <complex>
#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace Analysis
{
            ///// EXACT CALCULATIONS /////

/* Calculate the gap ratios, spdm, boson density, and compressibility for a range of parameters */
static void calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double> H_fixed, const Eigen::SparseMatrix<double> H1, const Eigen::SparseMatrix<double> H2, std::string fixed_param, double fixed_value, double param1_min, double param1_max, double param2_min, double param2_max, double param1_step, double param2_step);

// GAP RATIOS

/* Calculate the gap ratios of the system */
static Eigen::VectorXd gap_ratios(Eigen::VectorXcd eigenvalues, int nb_eigen);

// SPDM FUNCTIONS

/* Calculate the single-particle density matrix of the system */
static Eigen::MatrixXcd SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, Eigen::MatrixXcd& eigenvectors);

/* Calculate the compressibility of the system */
static double coherence(const Eigen::MatrixXcd& spdm);

// MEAN VALUE CALCULATIONS

/* Calculate the mean value of the annihilation operator <n|ai+ aj|n> */
static std::complex<double> braket(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int i, int j);

// PCA FUNCTIONS

/* calculate the dispersion of the projected points */
static double calculate_dispersion(const Eigen::MatrixXd& projected_data);

// UTILITY FUNCTIONS

/* standardize a matrix */
static Eigen::MatrixXd standardize_matrix(const Eigen::MatrixXd& matrix);

/* save a matrix to a csv file */
static void save_matrices_to_csv(const std::string& filename, const std::vector<Eigen::MatrixXd>& matrices, const std::string& label);

/* save the dispersions to a file */
static void save_dispersions(const std::string& filename, const std::vector<double>& dispersions);

void exact_parameters(int m, int n, double J, double U, double mu, double s, double r, std::string fixed_param);
}