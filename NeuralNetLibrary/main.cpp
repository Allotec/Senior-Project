#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

int main(){
    Eigen::MatrixXf m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);

    Eigen::Map<Eigen::VectorXf> v1(m.data(), m.size());
    std::cout << "v1:" << std::endl
              << v1 << std::endl;
    v1.transpose();
    std::cout << "v1:" << std::endl
              << v1 << std::endl;

    std::cout << m << std::endl;

    return (0);
}