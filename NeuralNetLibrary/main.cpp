#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "networkOps.hpp"

int main(){
    Eigen::MatrixXf m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);

    std::cout << m.size() << std::endl;

    return (0);
}