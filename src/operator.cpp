#include <cmath>
#include <numeric>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include "operator.hpp"
#include "Eigen/src/Core/Matrix.h"

using namespace Spectra;



///// DIAGONALIZATION /////

    /* SORT EIGENVALUES AND EIGENVECTORS IN DESCENDING ORDER */

/* Sort eigenvalues and eigenvectors in ascending order by real part of eigenvalue */
void Op::sort_eigen(Eigen::VectorXcd& eigenvalues, Eigen::MatrixXcd& eigenvectors) {
    const int n = eigenvalues.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&eigenvalues](int i, int j) { return eigenvalues[i].real() < eigenvalues[j].real(); });
    Eigen::VectorXcd temp_eigenvalues = eigenvalues;
    Eigen::MatrixXcd temp_eigenvectors = eigenvectors;
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = temp_eigenvalues[indices[i]];
        eigenvectors.col(i) = temp_eigenvectors.col(indices[i]);
    }
}

    /* IMPLICITLY RESTARTED LANCZOS METHOD (IRLM) */

/* implement the IRLM for a sparse matrix to find the smallest nb_eigen eigenvalues of a sparse matrix */
Eigen::VectorXcd Op::IRLM_eigen(Eigen::SparseMatrix<double> O,int nb_eigen, Eigen::MatrixXcd& eigenvectors) {
    SparseGenMatProd<double> op(O); // create a compatible matrix object
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, nb_eigen, 2 * nb_eigen+1); // create an eigen solver object
    eigs.init();
    [[maybe_unused]] int nconv = eigs.compute(Spectra::SortRule::SmallestReal); // solve the eigen problem
    if (eigs.info() != Spectra::CompInfo::Successful) { // verify if the eigen search is a success
        throw std::runtime_error("Eigenvalue computation failed.");
    }
    Eigen::VectorXcd eigenvalues = eigs.eigenvalues(); // eigenvalues of the hamiltonian
    eigenvectors = eigs.eigenvectors(); // eigenvectors of the hamiltonian
    return eigenvalues;
}


    /* EXACT DIAGONALIZATION */

/* Calculate the exact eigenvalues and eigenvectors of the hamiltonian by an exact diagonalization */
Eigen::VectorXd Op::exact_eigen(Eigen::SparseMatrix<double> O, Eigen::MatrixXd& eigenvectors) {
    Eigen::MatrixXd dense_smat = Eigen::MatrixXd(O); // convert sparse matrix to dense matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_smat); // solve the eigen problem for the hamiltonian
    if (eigensolver.info() != Eigen::Success) { // verify if the eigen search is a success
        throw std::runtime_error("Eigenvalue computation failed.");
    }
    eigenvectors = eigensolver.eigenvectors(); // eigenvectors of the hamiltonian
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues(); // eigenvalues of the hamiltonian
    return eigenvalues;
}