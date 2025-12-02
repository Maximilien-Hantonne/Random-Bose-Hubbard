#pragma once

#include <complex>
#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace Analysis
{
            ///// EXACT CALCULATIONS /////

/* Calculate the gap ratios, spdm, boson density, and compressibility for a range of parameters */
static void calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double>& TH, const Eigen::SparseMatrix<double>& UH, const Eigen::SparseMatrix<double>& uH, const std::string& fixed_param, const double T, const double U, const double mu, const double T_min, const double T_max, const double U_min, const double U_max, const double mu_min, const double mu_max, const double param1_step, const double param2_step, const double sigma_T = 0.0, const double sigma_U = 0.0, const double sigma_u = 0.0);

// GAP RATIOS

/* Calculate the gap ratios of the system */
static Eigen::VectorXd gap_ratios(const Eigen::VectorXcd& eigenvalues, int nb_eigen);

// SPDM FUNCTIONS

/* Calculate the single-particle density matrix of the system */
static Eigen::MatrixXcd SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::VectorXcd& phi0);

/* Calculate the average local fluctuation of the system */
static double local_fluctu(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis);

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

void exact_parameters(int m, int n, double T, double U, double mu, double s, double r, std::string fixed_param, double sigma_T = 0.0, double sigma_U = 0.0, double sigma_u = 0.0);
}