#pragma once

#include <tuple>
#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace Analysis
{
            ///// EXACT CALCULATIONS /////

/* Calculate the gap ratios, spdm, boson density, and compressibility for a range of parameters */
static void calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double>& TH, const Eigen::SparseMatrix<double>& UH, const Eigen::SparseMatrix<double>& uH, const std::string& fixed_param, const double T, const double U, const double mu, const double T_min, const double T_max, const double U_min, const double U_max, const double mu_min, const double mu_max, const double param1_step, const double param2_step, const double sigma_T = 0.0, const double delta_U = 0.0, const double delta_u = 0.0, const int realizations = 1, const int m = 0, const int n = 0);

// GAP RATIOS

/* Calculate the gap ratios of the system */
static Eigen::VectorXd gap_ratios(const Eigen::VectorXcd& eigenvalues, int nb_eigen);

// SPDM FUNCTIONS

/* Calculate the single-particle density matrix of the system */
static Eigen::MatrixXcd SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::VectorXcd& phi0);

// MEAN OCCUPATIONS

/* Calculate site occupations: returns (spatial_avg_ni, spatial_avg_ni_sq, per_site_ni) */
static std::tuple<double, double, Eigen::VectorXd> mean_occupations(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis);


void exact_parameters(int m, int n, double T, double U, double mu, double s, double r, std::string fixed_param, double sigma_T = 0.0, double delta_U = 0.0, double delta_u = 0.0, int realizations = 1);
}