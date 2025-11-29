#pragma once

#include<vector>
#include<Eigen/Dense>
#include<Eigen/SparseCore> 
#include "Eigen/src/Core/Matrix.h"
#include<boost/multiprecision/cpp_int.hpp>

namespace BH
{

// DIMENSION OF THE HILBERT SPACE

    /* calculate the dimension of the Hilbert space for n bosons on m sites */
    static int binomial(int n, int k); // Binomial coefficient
    static int dimension(int m, int n); // Dimension of the Hilbert space

// ELEMENTARY FUNCTIONS

    /* calculate the sum of the elements of a vector between 2 index */
    static int sum(const Eigen::VectorXi& state, int index1, int index2); 

// INITIALIZE THE HILBERT SPACE BASIS

    /* calculate the next state of the Hilbert space in lexicographic order */
    static bool next_lexicographic(Eigen::VectorXi& state, int m, int n); 

    /* creates a matrix that has the vectors of the Hilbert space basis in columns */
    static Eigen::MatrixXi init_lexicographic(int m, int n); 

// SORT THE HILBERT SPACE BASIS TO FACILITATE CALCULUS

    /* calculate the unique tag of the kth column of the matrix */
    boost::multiprecision::cpp_int calculate_tag(const Eigen::MatrixXi& basis, const std::vector<int>& primes, int k);
    /* calculate the unique tag of a state */
    boost::multiprecision::cpp_int calculate_tag(const Eigen::VectorXi& state, const std::vector<int>& primes);

    /* calculate and store the tags of each state of the Hilbert space basis */
    std::vector<boost::multiprecision::cpp_int> calculate_tags(const Eigen::MatrixXi& basis, const std::vector<int>& primes);

    /* sort the states of the Hilbert space by ascending order compared by their tags */
    static void sort_basis(std::vector<boost::multiprecision::cpp_int>& tags, Eigen::MatrixXi& basis); 

    /* gives the index of the wanted tag x by the Newton method */
    int search_tag(const std::vector<boost::multiprecision::cpp_int>& tags, const boost::multiprecision::cpp_int& x);

// FILL THE HAMILTONIAN OF THE SYSTEM

    /* fill the hopping term of the Hamiltonian */
    static void fill_hopping(const Eigen::MatrixXi& basis, const std::vector<boost::multiprecision::cpp_int>& tags, const std::vector<std::vector<int>>& neighbours, const std::vector<int>& primes, Eigen::SparseMatrix<double>& hmatrix, double J);

    /* fill the interaction term of the Hamiltonian */
    static void fill_interaction(const Eigen::MatrixXi& basis, Eigen::SparseMatrix<double>& hmatrix, double U); 

    /* fill the chemical potential term of the Hamiltonian */
    static void fill_chemical(const Eigen::MatrixXi& basis, Eigen::SparseMatrix<double>& hmatrix, double mu); 

// BASIS

    std::pair<std::vector<boost::multiprecision::cpp_int>, Eigen::MatrixXi> fixed_set_basis(int m, int n);
    std::pair<std::vector<boost::multiprecision::cpp_int>, Eigen::MatrixXi> max_set_basis(int m, int n);
    
// HAMILTONIAN MATRICES

    Eigen::SparseMatrix<double> fixed_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, const Eigen::MatrixXi& basis, const std::vector<boost::multiprecision::cpp_int>& tags, int m, int n, double J, double U, double mu);
    Eigen::SparseMatrix<double> max_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, int m, int n_min, int n_max, double J, double U, double mu);

} 

