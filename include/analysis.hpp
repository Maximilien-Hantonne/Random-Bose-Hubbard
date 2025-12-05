#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace Analysis
{

// Scale type enum for faster dispatch (avoid string comparison in hot loops)
enum class ScaleType { Linear, Logarithmic };

// Convert string to ScaleType
inline ScaleType parse_scale(const std::string& scale) {
    return (scale == "log") ? ScaleType::Logarithmic : ScaleType::Linear;
}

            ///// EXACT CALCULATIONS /////

// GAP RATIOS

/* Calculate the gap ratios of the system and return their mean */
static double gap_ratios(const Eigen::VectorXd& eigenvalues, int nb_eigen);

// SPDM FUNCTIONS

/* Calculate the single-particle density matrix of the system */
static Eigen::MatrixXcd SPDM(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::VectorXcd& phi0);

// MEAN OCCUPATIONS

/* Calculate site occupations: returns (spatial_avg_ni, spatial_avg_ni_sq, per_site_ni) */
static std::tuple<double, double, Eigen::VectorXd> mean_occupations(const Eigen::VectorXcd& phi0, const Eigen::MatrixXd& basis);


// MAIN FUNCTIONS

/* Main function for exact calculations parameters */
void exact_parameters(int m, int n, double T, double U, double mu, double s, double r, const std::string& fixed_param, double sigma_T = 0.0, double delta_U = 0.0, double delta_u = 0.0, int realizations = 1, const std::string& scale = "log");

/* Calculate the gap ratios, spdm, boson density, and compressibility for a range of parameters */
static void calculate_and_save(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const Eigen::SparseMatrix<double>& TH, const Eigen::SparseMatrix<double>& UH, const Eigen::SparseMatrix<double>& uH, const std::string& fixed_param, const double T, const double U, const double mu, const double T_min, const double T_max, const double U_min, const double U_max, const double mu_min, const double mu_max, const double param1_step, const double param2_step, const std::string& scale = "log", const double sigma_T = 0.0, const double delta_U = 0.0, const double delta_u = 0.0, const int realizations = 1, const int m = 0, const int n = 0);

/* Compute parameter value at a given index based on scale type */
double compute_param(double p_min, double p_max, int idx, int num_points, ScaleType scale);

/* Precompute all parameter values for a range */
std::vector<double> precompute_params(double p_min, double p_max, int num_points, ScaleType scale);
}