#pragma once

#include<vector>
#include<Eigen/Dense>
#include<Eigen/SparseCore> 
#include "Eigen/src/Core/Matrix.h"

namespace BH
{

// DIMENSION OF THE HILBERT SPACE

    /* calculate the dimension of the Hilbert space for n bosons on m sites */
    static int binomial(int n, int k); // Binomial coefficient
    static int dimension(int m, int n); // Dimension of the Hilbert space

// ELEMENTARY FUNCTIONS

    /* calculate the sum of the elements of a vector between 2 index */
    static int sum(const Eigen::VectorXd& state, int index1, int index2); 

// INITIALIZE THE HILBERT SPACE BASIS

    /* calculate the next state of the Hilbert space in lexicographic order */
    static bool next_lexicographic(Eigen::VectorXd& state, int m, int n); 

    /* creates a matrix that has the vectors of the Hilbert space basis in columns */
    static Eigen::MatrixXd init_lexicographic(int m, int n); 

// SORT THE HILBERT SPACE BASIS TO FACILITATE CALCULUS

    /* calculate the unique tag of a state */
    double calculate_tag(const Eigen::VectorXd& state, const std::vector<int>& primes);

    /* calculate and store the tags of each state of the Hilbert space basis */
    Eigen::VectorXd calculate_tags(const Eigen::MatrixXd& basis, const std::vector<int>& primes);

    /* sort the states of the Hilbert space by ascending order compared by their tags */
    static void sort_basis(Eigen::VectorXd& tags, Eigen::MatrixXd& basis); 

    /* gives the index of the wanted tag x by the Newton method */
    int search_tag(const Eigen::VectorXd& tags, double x);

// FILL THE HAMILTONIAN OF THE SYSTEM

    /* fill the hopping term of the Hamiltonian */
    static void fill_hopping(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const std::vector<std::vector<int>>& neighbours, const std::vector<int>& primes, Eigen::SparseMatrix<double>& hmatrix, double t);

    /* fill the interaction term of the Hamiltonian */
    static void fill_interaction(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double U); 

    /* fill the chemical potential term of the Hamiltonian */
    static void fill_chemical(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double mu); 

// BASIS

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> fixed_set_basis(int m, int n);
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> max_set_basis(int m, int n);
    
// HAMILTONIAN MATRICES

    Eigen::SparseMatrix<double> fixed_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int m, int n, double t, double U, double mu);
    Eigen::SparseMatrix<double> max_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, int m, int n_min, int n_max, double t, double U, double mu);

// RANDOMIZE HAMILTONIAN

    /* Random Hamiltonian */
    Eigen::SparseMatrix<double> random_hamiltonian(const Eigen::SparseMatrix<double>& tH, const double t, const double sigma_t,
                                        const Eigen::SparseMatrix<double>& UH, const double U, const double delta_U,
                                        const Eigen::SparseMatrix<double>& uH, const double u, const double delta_u,
                                        const unsigned int seed);

} 

