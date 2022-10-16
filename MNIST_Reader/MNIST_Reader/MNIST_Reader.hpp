#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <utility>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfRM;

std::vector<std::pair<MatrixXfRM, int>>* read_Mnist_Images(std::string images, std::string labels);
int reverseInt(int i); //Goes from big endian to little endian or vice versa
