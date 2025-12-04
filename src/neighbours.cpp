#include <vector>
#include <stdexcept>
#include <cmath>

#include "neighbours.hpp"


///// IMPLEMENTATION OF THE NEIGHBOURS NAMESPACE FUNCTIONS /////


    /* 1D */

/* generate the list of neighbours for a 1D chain */
std::vector<std::vector<int>> Neighbours::chain_neighbours(int m, bool closed) { // by default closed = true for periodic boundary conditions, closed = false for open boundary conditions
	std::vector<std::vector<int>> neighbours(m);
	for (int i = 0; i < m; ++i) {
		if (i > 0) {
			neighbours[i].push_back(i - 1); // Left neighbour
		}
		if (i < m - 1) {
			neighbours[i].push_back(i + 1); // Right neighbour
		}
	}
	if (closed) { // Periodic boundary conditions
		neighbours[0].push_back(m - 1); 
		neighbours[m - 1].push_back(0); 
	}
	return neighbours;
}


    /* 2D */

/* generate the list of neighbours for a 2D square lattice */
std::vector<std::vector<int>> Neighbours::square_neighbours(int m, bool closed) { // by default closed = true for periodic boundary conditions, closed = false for open boundary conditions
    int side = static_cast<int>(std::sqrt(m));
    if (side * side != m) {
        throw std::invalid_argument("The number of sites (m) must be a perfect square.");
    }
    std::vector<std::vector<int>> neighbours(m);
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            int index = i * side + j;
            if (j > 0) {
                neighbours[index].push_back(index - 1); // Left neighbour
            } else if (closed) {
                neighbours[index].push_back(index + side - 1);
            }
            if (j < side - 1) {
                neighbours[index].push_back(index + 1); // Right neighbour
            } else if (closed) {
                neighbours[index].push_back(index - side + 1);
            }
            if (i > 0) {
                neighbours[index].push_back(index - side); // Top neighbour
            } else if (closed) {
                neighbours[index].push_back(index + (side - 1) * side);
            }
            if (i < side - 1) {
                neighbours[index].push_back(index + side); // Bottom neighbour
            } else if (closed) {
                neighbours[index].push_back(index - (side - 1) * side);
            }
        }
    }
    return neighbours;
}


    /* 3D */ 
    
/* generate the list of neighbours for a 3D cubic lattice */
std::vector<std::vector<int>> Neighbours::cube_neighbours(int m, bool closed){ // by default closed = true for periodic boundary conditions, closed = false for open boundary conditions
    int side = static_cast<int>(std::cbrt(m));
    if (side * side * side != m) {
        throw std::invalid_argument("The number of sites (m) must be a perfect cube.");
    }
    std::vector<std::vector<int>> neighbours(m);
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            for (int k = 0; k < side; ++k) {
                int index = i * side * side + j * side + k;
                if (k > 0) {
                    neighbours[index].push_back(index - 1); // Left neighbour
                } else if (closed) {
                    neighbours[index].push_back(index + side - 1);
                }
                if (k < side - 1) {
                    neighbours[index].push_back(index + 1); // Right neighbour
                } else if (closed) {
                    neighbours[index].push_back(index - side + 1);
                }
                if (j > 0) {
                    neighbours[index].push_back(index - side); // Top neighbour
                } else if (closed) {
                    neighbours[index].push_back(index + (side - 1) * side);
                }
                if (j < side - 1) {
                    neighbours[index].push_back(index + side); // Bottom neighbour
                } else if (closed) {
                    neighbours[index].push_back(index - (side - 1) * side);
                }
                if (i > 0) {
                    neighbours[index].push_back(index - side * side); // Front neighbour
                } else if (closed) {
                    neighbours[index].push_back(index + (side - 1) * side * side);
                }
                if (i < side - 1) {
                    neighbours[index].push_back(index + side * side); // Back neighbour
                } else if (closed) {
                    neighbours[index].push_back(index - (side - 1) * side * side);
                }
            }
        }
    }
    return neighbours;
}