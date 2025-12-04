#pragma once

#include <cmath>
#include <vector>

namespace Neighbours {
    std::vector<std::vector<int>> chain_neighbours(int m, bool closed = true);
    std::vector<std::vector<int>> square_neighbours(int m, bool closed = true);
    std::vector<std::vector<int>> cube_neighbours(int m, bool closed = true);
}