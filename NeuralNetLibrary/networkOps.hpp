#pragma once
#include "../File_Parsing_Library/File_Parsing_Library/LayerConstants.hpp"
#include <Eigen/Dense>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXfRM;

// Convolution operation Raaina
MatrixXfRM convolution(
    MatrixXfRM input,        // Input matrix
    int padding,             // Padding type (valid or same)
    MatrixXfRM kernel,       // Kernel matrix
    std::vector<int> stride, // Stride vector (x movement, y movement, ...)
    float bias,              // Bias matrix
    int activation // Activation function (relu, sigmoid, tanh, softmax, linear,
                   // ...)
);

// Pooling operation Raaina
MatrixXfRM
pooling(MatrixXfRM input,        // Input matrix
        int padding,             // Padding type (valid or same)
        std::vector<int> kernel, // dimensions of kernel (x, y)
        int poolingType,         // Pooling type (max, average, min...)
        std::vector<int> stride  // Stride vector (x movement, y movement, ...)
);

// Dense Layer operation
MatrixXfRM dense(MatrixXfRM input,   // Input matrix
                 MatrixXfRM weights, // Weights matrix
                 MatrixXfRM bias,    // Bias matrix
                 int activation // Activation function (relu, sigmoid, tanh,
                                // softmax, linear, ...)
);

// Calculates the Softmax of a matrix
MatrixXfRM Softmax(MatrixXfRM input);

// Calculates the relu of a matrix
MatrixXfRM Relu(MatrixXfRM input);

std::vector<MatrixXfRM> *conv2d(std::vector<MatrixXfRM> input,
                                std::vector<MatrixXfRM *> filter, int stride,
                                MatrixXfRM bias, int num_input_channels,
                                int num_output_channels);
std::vector<MatrixXfRM> *ReluMatrix(std::vector<MatrixXfRM> *input);
