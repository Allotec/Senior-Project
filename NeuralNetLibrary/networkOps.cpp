#include "networkOps.hpp"

//Convolution operation
Eigen::MatrixXf convolution(
    Eigen::MatrixXf input, //Input matrix
    std::string padding, //Padding type (valid or same)
    Eigen::MatrixXf kernel, //Kernel matrix
    std::vector<int> stride, //Stride vector (x movement, y movement, ...)
    Eigen::MatrixXf bias, //Bias matrix
    std::string activation //Activation function (relu, sigmoid, tanh, softmax, linear, ...)
    ){
        //Retun empty matrix
        Eigen::MatrixXf output;

        //Return output
        return output;
    }

//Pooling operation
Eigen::MatrixXf pooling(
    Eigen::MatrixXf input, //Input matrix
    std::string padding, //Padding type (valid or same)
    std::string poolingType, //Pooling type (max, average, min...)
    std::vector<int> stride //Stride vector (x movement, y movement, ...)
    ){
        //Retun empty matrix
        Eigen::MatrixXf output;

        //Return output
        return output;
    }

//Dense Layer operation
Eigen::MatrixXf dense(
    Eigen::MatrixXf input, //Input matrix
    Eigen::MatrixXf weights, //Weights matrix
    Eigen::MatrixXf bias, //Bias matrix
    std::string activation //Activation function (relu, sigmoid, tanh, softmax, linear, ...)
    ){
        //Retun empty matrix
        Eigen::MatrixXf output;

        //Return output
        return output;
    }

//Flatten operation
Eigen::MatrixXf flatten(
    Eigen::MatrixXf input //Input matrix
    ){
        //Retun empty matrix
        Eigen::MatrixXf output;

        //Return output
        return output;
    }