#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>


namespace Op
{

// DIAGONALIZATION : 

    /* sort eigenvalues and eigenvectors in descending order */
    void sort_eigen(Eigen::VectorXd& eigenvalues, Eigen::MatrixXcd& eigenvectors);


// DIAGONALIZATION : 

    Eigen::VectorXd IRLM_eigen(Eigen::SparseMatrix<double> O, int nb_eigen, Eigen::MatrixXcd& eigenvectors);
    Eigen::VectorXd exact_eigen(Eigen::SparseMatrix<double> O, Eigen::MatrixXd& eigenvectors);

}