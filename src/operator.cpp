#include <cmath>
#include <numeric>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "operator.hpp"
#include "Eigen/src/Core/Matrix.h"

using namespace Spectra;



///// DIAGONALIZATION /////

    /* SORT EIGENVALUES AND EIGENVECTORS IN DESCENDING ORDER */

/* Sort eigenvalues and eigenvectors in ascending order by eigenvalue */
void Op::sort_eigen(Eigen::VectorXd& eigenvalues, Eigen::MatrixXcd& eigenvectors) {
    const int n = eigenvalues.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&eigenvalues](int i, int j) { return eigenvalues[i] < eigenvalues[j]; });
    
    // In-place permutation using cycle detection
    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; ++i) {
        if (visited[i] || indices[i] == i) continue;
        int j = i;
        while (!visited[j]) {
            visited[j] = true;
            int next = indices[j];
            if (next != i) {
                std::swap(eigenvalues[j], eigenvalues[next]);
                eigenvectors.col(j).swap(eigenvectors.col(next));
            }
            j = next;
        }
    }
}

    /* IMPLICITLY RESTARTED LANCZOS METHOD (IRLM) */

/* implement the IRLM for a sparse symmetric matrix to find the smallest nb_eigen eigenvalues */
Eigen::VectorXd Op::IRLM_eigen(const Eigen::SparseMatrix<double>& O, int nb_eigen, Eigen::MatrixXcd& eigenvectors, bool& success) {
    SparseSymMatProd<double> op(O); // create a matrix operation object for symmetric matrix
    SymEigsSolver<SparseSymMatProd<double>> eigs(op, nb_eigen, 2 * nb_eigen+1); 
    eigs.init();
    [[maybe_unused]] int nconv = eigs.compute(Spectra::SortRule::SmallestAlge); // find smallest algebraic eigenvalues
    Eigen::VectorXd eigenvalues;
    if (eigs.info() != Spectra::CompInfo::Successful) { // verify if the eigen search is a success
        success = false;
        std::cerr << "Warning: Eigenvalue computation failed." << std::endl;
        eigenvalues = Eigen::VectorXd::Constant(nb_eigen, -1.0);
        eigenvectors = Eigen::MatrixXcd::Constant(O.rows(), nb_eigen, std::complex<double>(-1.0, 0.0));
        return eigenvalues;
    }
    success = true;
    eigenvalues = eigs.eigenvalues(); // eigenvalues are real for symmetric matrices
    eigenvectors = eigs.eigenvectors(); // eigenvectors of the hamiltonian
    return eigenvalues;
}


    /* EXACT DIAGONALIZATION */

/* Calculate the exact eigenvalues and eigenvectors of the hamiltonian by an exact diagonalization */
Eigen::VectorXd Op::exact_eigen(const Eigen::SparseMatrix<double>& O, Eigen::MatrixXd& eigenvectors, bool& success) {
    Eigen::MatrixXd dense_smat = Eigen::MatrixXd(O); // convert sparse matrix to dense matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_smat); // solve the eigen problem for the hamiltonian
    Eigen::VectorXd eigenvalues;
    if (eigensolver.info() != Eigen::Success) { // verify if the eigen search is a success
        success = false;
        std::cerr << "Warning: Eigenvalue computation failed." << std::endl;
        eigenvalues = Eigen::VectorXd::Constant(O.rows(), -1.0);
        eigenvectors = Eigen::MatrixXd::Constant(O.rows(), O.cols(), -1.0);
        return eigenvalues;
    }
    success = true;
    eigenvectors = eigensolver.eigenvectors(); // eigenvectors of the hamiltonian
    eigenvalues = eigensolver.eigenvalues(); // eigenvalues of the hamiltonian
    return eigenvalues;
}