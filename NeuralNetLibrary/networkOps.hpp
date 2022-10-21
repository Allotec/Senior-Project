#pragma once
#include<Eigen/Dense>
#include<vector>
#include "../File_Parsing_Library/File_Parsing_Library/LayerConstants.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfRM;

//Convolution operation Raaina
MatrixXfRM convolution(
    MatrixXfRM input, //Input matrix
    int padding, //Padding type (valid or same)
    MatrixXfRM kernel, //Kernel matrix
    std::vector<int> stride, //Stride vector (x movement, y movement, ...)
    float bias, //Bias matrix
    int activation //Activation function (relu, sigmoid, tanh, softmax, linear, ...)
    ); 

//Pooling operation Raaina
MatrixXfRM pooling(
    MatrixXfRM input, //Input matrix
    int padding, //Padding type (valid or same)
    std::vector<int> kernel, //dimensions of kernel (x, y)
    int poolingType, //Pooling type (max, average, min...)
    std::vector<int> stride //Stride vector (x movement, y movement, ...)
    );

//Dense Layer operation
MatrixXfRM dense(
    MatrixXfRM input, //Input matrix
    MatrixXfRM weights, //Weights matrix
    MatrixXfRM bias, //Bias matrix
    int activation //Activation function (relu, sigmoid, tanh, softmax, linear, ...)
    );

//Calculates the Softmax of a matrix
MatrixXfRM Softmax(MatrixXfRM input);

//Calculates the relu of a matrix
MatrixXfRM Relu(MatrixXfRM input);

