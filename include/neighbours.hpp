#pragma once

#include <cmath>
#include <vector>

class Neighbours {
public:

    Neighbours(int m);
    ~Neighbours();

    void chain_neighbours(bool closed = true);
    void square_neighbours(bool closed = true);
    void cube_neighbours(bool closed = true);
    std::vector<std::vector<int>> getNeighbours() const;
    
private:
    int m;
    std::vector<std::vector<int>> neighbours;
};