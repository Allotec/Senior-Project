#include "networkOps.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

int main() {
  Eigen::MatrixXf m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  // Array:
  // 3  -1
  // 2.5 1.5

  std::cout << "Size of Initial Matrix m: " << m.size() << std::endl;
  std::cout << m << std::endl;
  std::cout << std::endl;

  // Testing Flattening
  Eigen::MatrixXf output1 = flatten(m);
  std::cout << "Flattened Result: \n" << output1 << std::endl;

  // Testing Dense Layers
  // function assumes that input matrix has been flattened
  // in this case input is output1 from  previous test
  Eigen::MatrixXf weights(4, 6),
      bias(1, 6); // input is 1x4, weight is 4x6 so product is 1x6
  weights << 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7,
      8, 9;
  bias << 1, 1, 1, -50, 1, 1;
  std::cout << "\nWeights Matrix: \n" << weights << std::endl;
  std::cout << "\nBias Matrix: \n" << bias << std::endl;
  Eigen::MatrixXf output2 = dense(output1, weights, bias, "linear");
  std::cout << "\nDense Layer Result: \n" << output2 << std::endl;
  Eigen::MatrixXf output3 = dense(output1, weights, bias, "relu");
  std::cout << "\nDense Layer Result: \n" << output3 << std::endl;

  Eigen::MatrixXf output4 = dense(output1, weights, bias, "softmax");
  std::cout << "\nDense Layer Result: \n" << output4 << std::endl;
  return (0);
}
