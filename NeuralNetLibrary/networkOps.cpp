#include "networkOps.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>


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
        
        Eigen::MatrixXf hidden, output;
        hidden = (input * weights) + bias;// H = f(I*W + B)

        std::transform(activation.begin(), activation.end(), activation.begin(), ::toupper); //convert to all caps to ignore case
        std::cout << "\nActivation: " << activation << std::endl;
        if (activation.compare("SOFTMAX") == 0) {
            //loop to loop thru all values in hidden
            //den = e^input
            Eigen::MatrixXf temp(1, hidden.cols()), numer(1, hidden.cols());
            double num = 0;
            for (int i = 0; i < hidden.cols(); i++) {//gets denominator 
                temp(0, i) = exp(hidden(0, i));
                std::cout << "iteration " << i << std::endl;
                num += temp(0, i);
                std::cout << "sum: " << num << std::endl;
            }
            
            //output = temp;

               
            //loop through all values in hidden
            //num = e^input
            //output for that iteration = num/den
            for (int i = 0; i < hidden.cols(); i++) {
                numer(0, i) = exp(hidden(0, i));
            }
            
            output = numer / num;
        }
        else if (activation.compare("RELU") == 0) {
            
            for (int i = 0; i < hidden.cols(); i++) {//assuming row vector
                if (hidden(0, i) < 0) {//output = 0 if input is negative; else return input
                    hidden(0, i) = 0;
                }
            }
            output = hidden;
        }
        else {
            std::cout << "Error: Invalid Activation Selected!" << std::endl;
            output = hidden * 0;//output will return all 0s for all items in vector
        }
        
        //output = hidden + bias;
        return output;//Return output
    }

//Flatten operation
Eigen::MatrixXf flatten(
    Eigen::MatrixXf input //Input matrix
    ){
        //std::cout << "Test Flatten: " << input.size() << std::endl;
        Eigen::MatrixXf output;
        //output = m;
        output = input.reshaped(1, input.size());
        //output = input;
        
        //Eigen::Map<RowVectorXf> output(input.data(), 1, input.size());
        //Return output
        return output;
    }
