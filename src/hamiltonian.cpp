#include<vector>
#include<iostream>
#include<random>
#include<utility>
#include<unordered_map>
#include<Eigen/Dense>
#include<Eigen/SparseCore>

#include "hamiltonian.hpp"


/////  IMPLEMENTATION OF THE BH CLASS METHODS  /////

    
    /* ELEMENTARY FUNCTIONS */

/* Calculate the sum of the elements of a vector between 2 index */
int BH::sum(const Eigen::VectorXd& state, int index1, int index2) { 
	int s = 0;
	for (int i = index1; i <= index2; i++) {
		s += state[i];
	}
	return s;
}


    /* DIMENSION OF THE HILBERT SPACE */

/* Calculate the binomial coefficient */
int BH::binomial(int n, int k){
	if (k==0 || k==n){
		return 1;
	}
	if (k > n/2) {
		return binomial(n,n-k);
	}
	else{
		return n*binomial(n-1,k-1)/k;
	}
}

/* Calculate the dimension of the Hilbert space for n bosons on m sites */
int BH::dimension(int m, int n) {
	return binomial(m + n - 1, n);
}


    /* INITIALIZE THE HILBERT SPACE BASIS */

/* Calculate the next Fock state of the Hilbert space in lexicographic order */
bool BH::next_lexicographic(Eigen::VectorXd& state, int m, int n) {
	for (int k = m - 2; k > -1; k--) {
		if (state[k] != 0) {
			state[k] -= 1;
			state[k + 1] = n - sum(state, 0, k);
			for (int i = k + 2; i < m; i++) {
				state[i] = 0;
			}
			return true;
		}
	}
	return false;
}

/* Create the matrix that has the Fock states of the Hilbert space basis in columns */
Eigen::MatrixXd BH::init_lexicographic(int m, int n) {
    int D = dimension(m, n);
    Eigen::MatrixXd basis(m, D);
    Eigen::VectorXd state = Eigen::VectorXd::Zero(m);
    state(0) = n;
    int col = 0;
    do {
        basis.col(col++) = state;
    } while (next_lexicographic(state, m, n));
    return basis;
}


    /* SORT THE HILBERT SPACE BASIS TO FACILITATE CALCULUS */

/* Calculate the unique tag of the kth column of the matrix */
double BH::calculate_tag(const Eigen::MatrixXd& basis, const std::vector<int>& primes, int k) {
	double tag = 0;
	for (int i = 0; i < basis.rows(); i++) {
		tag += basis.coeff(i, k) * log(primes[i]);
	}
	return tag;
}

/* Calculate the unique tag of a state */
double BH::calculate_tag(const Eigen::VectorXd& state, const std::vector<int>& primes) {
    double tag = 0;
    for (int i = 0; i < state.size(); i++) {
        tag += state[i] * log(primes[i]);
    }
    return tag;
}


/* Calculate and store the tags of each state of the Hilbert space basis */
Eigen::VectorXd BH::calculate_tags(const Eigen::MatrixXd& basis, const std::vector<int>& primes) {
	Eigen::VectorXd tags(basis.cols());
	for (int i = 0; i < basis.cols(); i++) {
		tags[i] = calculate_tag(basis, primes, i);
	}
	return tags;
}

/* Sort the states of the Hilbert space by ascending order compared by their tags*/
void BH::sort_basis(Eigen::VectorXd& tags, Eigen::MatrixXd& basis) {
    std::vector<int> indices(tags.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&tags](int a, int b) {return tags[a] < tags[b];});
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        while (indices[i] != i) {
            int j = indices[i];
            std::swap(tags[i], tags[j]);
            basis.col(i).swap(basis.col(j));
            std::swap(indices[i], indices[j]);
        }
    }
}

/* Gives the index of the wanted tag x by the Newton method */
int BH::search_tag(const Eigen::VectorXd& tags, double x) {
	int a = 0;
	int b = tags.size() - 1;
	int m = (a + b) / 2;
	while (fabs(tags[m] - x) > 1e-3 && a <= b) {
		if (tags[m] < x) {
			a = m + 1;
		}
		else {
			b = m - 1;
		}
		m = (a + b) / 2;
	}
	return m;
}

/* Create the matrix that has the Fock states of the Hilbert space basis in columns sorted by tags with their unique tag */
std::pair<Eigen::VectorXd, Eigen::MatrixXd> BH::fixed_set_basis(int m, int n) {
    std::vector<int> primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
    Eigen::MatrixXd basis = init_lexicographic(m, n);
    Eigen::VectorXd tags = calculate_tags(basis, primes);
    sort_basis(tags, basis);
    return std::make_pair(tags, basis);
}

/* Create the matrix that has the Fock states of the Hilbert space basis in columns sorted by tags with their unique tag */
std::pair<Eigen::VectorXd, Eigen::MatrixXd> BH::max_set_basis(int m, int n) {
    Eigen::VectorXd tags;
    Eigen::MatrixXd basis(m, 0);
    for (int bosons = 1; bosons <= n; ++bosons) {
        auto [fixed_tags, fixed_basis] = fixed_set_basis(m, bosons);
        int old_size = tags.size();
        tags.conservativeResize(old_size + fixed_tags.size());
        tags.segment(old_size, fixed_tags.size()) = fixed_tags;
        int old_cols = basis.cols();
        basis.conservativeResize(Eigen::NoChange, old_cols + fixed_basis.cols());
        basis.block(0, old_cols, basis.rows(), fixed_basis.cols()) = fixed_basis;
    }
    sort_basis(tags, basis);
    return std::make_pair(tags, basis);
}
    /* FILL THE HAMILTONIAN OF THE SYSTEM */

/* Fill the hopping term of the Hamiltonian */
void BH::fill_hopping(const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, const std::vector<std::vector<int>>& neighbours, const std::vector<int>& primes, Eigen::SparseMatrix<double>& hmatrix, double T) {
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(basis.cols() * basis.rows() * neighbours.size());
    for (int k = 0; k < basis.cols(); k++) {
        for (int i = 0; i < static_cast<int>(neighbours.size()); i++) {
            for (int j = 0; j < static_cast<int>(neighbours[i].size()); j++) {
                Eigen::VectorXd state = basis.col(k);
                if (basis.coeff(i, k) >= 0 && basis.coeff(j, k) >= 1) {
                    state[i] += 1;
                    state[j] -= 1;
                    double x = calculate_tag(state, primes, i);
                    int index = search_tag(tags, x);
                    assert(index >= 0 && index < tags.size()); // Add assertion to check index bounds
                    double value = sqrt((basis.coeff(i, k) + 1) * basis.coeff(j, k));
                    tripletList.push_back(Eigen::Triplet<double>(index, k, -T * value));
                    tripletList.push_back(Eigen::Triplet<double>(k, index, -T * value));
                }
            }
        }
    }
    hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

/* Fill the interaction term of the Hamiltonian */
void BH::fill_interaction(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double U) {
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(hmatrix.nonZeros() + basis.cols());
	for (int k = 0; k < hmatrix.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(hmatrix, k); it; ++it) {
			tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		}
	}
	for (int k = 0; k < basis.cols(); k++) {
		double value = 0;
		for (int i = 0; i < basis.rows(); i++) {
			double ni = basis.coeff(i, k);
			value += (ni - 1) * ni;
		}
		tripletList.push_back(Eigen::Triplet<double>(k, k, 0.5 * U * value));
	}
	hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

/* Fill the chemical potential term of the Hamiltonian */
void BH::fill_chemical(const Eigen::MatrixXd& basis, Eigen::SparseMatrix<double>& hmatrix, double mu) {
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(hmatrix.nonZeros() + basis.cols());
	for (int k = 0; k < hmatrix.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(hmatrix, k); it; ++it) {
			tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		}
	}
	for (int k = 0; k < basis.cols(); k++) {
		double value = 0;
		for (int i = 0; i < basis.rows(); i++) {
			double ni = basis.coeff(i, k);
			value += ni;
		}
		tripletList.push_back(Eigen::Triplet<double>(k, k, -mu * value));
	}
	hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}


    /* HAMILTONIAN MATRICES */

/* Create the Hamiltonian with a fixed number of bosons */
Eigen::SparseMatrix<double> BH::fixed_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, const Eigen::MatrixXd& basis, const Eigen::VectorXd& tags, int m, int n, double T, double U, double mu) {
    int D = dimension(m, n);
    Eigen::SparseMatrix<double> H(D,D);
    H.setZero();
    if (std::abs(T-0.0) > std::numeric_limits<double>::epsilon()) {
        std::vector<int> primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
        fill_hopping(basis, tags, neighbours, primes, H, T);
    }
    else if  (std::abs(U-0.0) > std::numeric_limits<double>::epsilon()) {
        fill_interaction(basis, H, U);
    }
    else if (std::abs(mu-0.0) > std::numeric_limits<double>::epsilon()) {
        fill_chemical(basis, H, mu);
    }
    else{
        std::cerr << "Error: At least one of the parameters T, U, mu must be different from zero." << std::endl;
    }
    return H;
}


/* Create the Hamiltonian with Fock states from 1 to n bosons */
Eigen::SparseMatrix<double> BH::max_bosons_hamiltonian(const std::vector<std::vector<int>>& neighbours, int m, int n_min, int n_max, double T, double U, double mu) {
    int total_dimension = 0;
    std::vector<Eigen::SparseMatrix<double>> hamiltonians;
    if (n_min < 0) {
        n_min = 0;
    }
    if (n_max < n_min) {
        n_max = n_min;
    }
    for (int bosons = n_min; bosons <= n_max; ++bosons) {
        auto [fixed_tags, fixed_basis] = fixed_set_basis(m, bosons);
        Eigen::SparseMatrix<double> hmatrix = fixed_bosons_hamiltonian(neighbours, fixed_basis, fixed_tags, m, bosons, T, U, mu);
        hamiltonians.push_back(hmatrix);
        total_dimension += hmatrix.rows();
    }
    Eigen::SparseMatrix<double> combined_hamiltonian(total_dimension, total_dimension);
    std::vector<Eigen::Triplet<double>> tripletList;
    int offset = 0;
    for (const auto& hmatrix : hamiltonians) {
        for (int k = 0; k < hmatrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(hmatrix, k); it; ++it) {
                tripletList.push_back(Eigen::Triplet<double>(it.row() + offset, it.col() + offset, it.value()));
            }
        }
        offset += hmatrix.rows();
    }
    combined_hamiltonian.setFromTriplets(tripletList.begin(), tripletList.end());
    return combined_hamiltonian;
}


    /* RANDOMIZE HAMILTONIAN */

Eigen::SparseMatrix<double> BH::random_hamiltonian(const Eigen::SparseMatrix<double>& TH, const double T, const double sigma_T,
                                                      const Eigen::SparseMatrix<double>& UH, const double U, const double sigma_U,
                                                      const Eigen::SparseMatrix<double>& uH, const double u, const double sigma_u,
                                                      const unsigned int seed) {
    const int n = TH.rows();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(TH.nonZeros() + UH.nonZeros() + uH.nonZeros());
    
    const bool has_T_disorder = (sigma_T > 0.0);
    const bool has_U_disorder = (sigma_U > 0.0);
    const bool has_u_disorder = (sigma_u > 0.0);
    std::mt19937 gen(seed);
    
    // Hopping term
    if (has_T_disorder) {
        const double stddev_T = (T != 0.0) ? sigma_T * std::abs(T) : sigma_T;
        std::normal_distribution<double> dist_T(T, stddev_T);
        std::unordered_map<int64_t, double> coeff_map;
        coeff_map.reserve(TH.nonZeros() / 2); 
        
        for (int k = 0; k < TH.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(TH, k); it; ++it) {
                const int row = it.row();
                const int col = it.col();
                const int r = std::min(row, col);
                const int c = std::max(row, col);
                const int64_t key = (static_cast<int64_t>(r) << 32) | c;
                const auto found = coeff_map.find(key);
                const double coeff = (found != coeff_map.end()) ? found->second : (coeff_map[key] = dist_T(gen));
                triplets.emplace_back(row, col, coeff * it.value());
            }
        }
    } else {
        for (int k = 0; k < TH.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(TH, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), T * it.value());
            }
        }
    }
    
    // Interaction term
    if (has_U_disorder) {
        const double stddev_U = (U != 0.0) ? sigma_U * std::abs(U) : sigma_U;
        std::normal_distribution<double> dist_U(U, stddev_U);
        for (int k = 0; k < UH.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(UH, k); it; ++it) {
                const double coeff = dist_U(gen);
                triplets.emplace_back(it.row(), it.col(), coeff * it.value());
            }
        }
    } else {
        for (int k = 0; k < UH.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(UH, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), U * it.value());
            }
        }
    }
    
    // Potential term
    if (has_u_disorder) {
        const double stddev_u = (u != 0.0) ? sigma_u * std::abs(u) : sigma_u;
        std::normal_distribution<double> dist_u(u, stddev_u);
        for (int k = 0; k < uH.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(uH, k); it; ++it) {
                const double coeff = dist_u(gen);
                triplets.emplace_back(it.row(), it.col(), coeff * it.value());
            }
        }
    } else {
        for (int k = 0; k < uH.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(uH, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), u * it.value());
            }
        }
    }
    
    Eigen::SparseMatrix<double> H(n, n);
    H.setFromTriplets(triplets.begin(), triplets.end());
    return H;
}
