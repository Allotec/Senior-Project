#include <iostream>
#include <float.h>
#include "networkOps.hpp"
#include <vector>
#include <cmath>

//Convolution operation
Eigen::MatrixXf convolution(
    Eigen::MatrixXf input, //Input matrix
    std::string padding, //Padding type (valid or same)
    Eigen::MatrixXf kernel, //Kernel matrix
    std::vector<int> stride, //Stride vector (x movement, y movement, ...)
    Eigen::MatrixXf bias, //Bias matrix
    std::string activation //Activation function (relu, sigmoid, tanh, softmax, linear, ...)
    ){
        int output_rows, output_col;
        long incx = 0, incy = 0;
        long krow = 0, kcol = 0;

        if(padding == "valid"){
            input = input;
        }

        else if(padding == "same"){
            Eigen::MatrixXf clone;
            //add layers of 0 based on
            int p_rows = ceil((kernel.rows() - 1) /2.0);
            //std::cout << "p rows:\n" << p_rows << std::endl;
            int p_cols = ceil((kernel.cols() - 1)/2.0);
            //std::cout << "p cols:\n" << p_cols << std::endl;
            int input_rows = input.rows() + 2 * p_rows;
            int input_col = input.cols() + 2 * p_cols;
            clone.resize(input_rows,input_col);
            clone = Eigen::MatrixXf::Zero(input_rows, input_col);
            
            for(int i = 0; i < input.cols(); i++){
                for(int j = 0; j < input.rows(); j++){
                    //std::cout << "input(i,j):\n" << input(i,j) << std::endl;
                    clone(i+p_rows,j+p_cols) = input(i,j);
                }
            }
            input = clone;
            //std::cout << "Here is the matrix c:\n" << clone << std::endl;
        }
        
        Eigen::MatrixXf output;
        output_rows = floor((input.rows() - kernel.rows())/stride.at(0)) + 1;
        output_col = floor((input.cols() - kernel.cols())/stride.at(1)) + 1;
        output.resize(output_rows,output_col);

        for(int y = 0; y < input.cols(); y = y + stride.at(1)){
            if(input.cols() < y + kernel.cols()) break;
            std::cout << "\n y is " << y << "stride is " << stride.at(0);
            for(int x = 0; x < input.rows(); x = x + stride.at(0)){
                if(input.rows() < x + kernel.rows()) break;
                std::cout << "\n x is " << x << "stride is " << stride.at(0);
                //finding max_value
                for(int i = y; i < kernel.rows() + y; i++){
                    for(int j = x; j < kernel.cols() + x; j++){
                        //std::cout << "\n kernel at 1+x is " << kernel.at(1) + x;
                        std::cout << "\n i is " << i << " j is " << j << " input_val " << input(i,j);
                        output(incy,incx) += input(i, j) * kernel(krow,kcol);
                        std::cout << "\nHere is the matrix o:\n" << output << std::endl;
                        kcol++;
                    }
                    kcol = 0;
                    krow++;
                }
                //std::cout << "\n max is " << max;
                //max = 0;
                kcol = 0;
                krow = 0;
                incx++;
            }
            incy++;
            incx = 0;
        }
        
        output = output + bias;

        if (activation.compare("SOFTMAX") == 0) {
            //loop to loop thru all values in hidden
            //den = e^input
            Eigen::MatrixXf temp(1, output.cols()), numer(1, output.cols());
            double num = 0;
            for (int i = 0; i < output.cols(); i++) {//gets denominator 
                temp(0, i) = exp(output(0, i));
                std::cout << "iteration " << i << std::endl;
                num += temp(0, i);
                std::cout << "sum: " << num << std::endl;
            }
            for (int i = 0; i < output.cols(); i++) {
                numer(0, i) = exp(output(0, i));
            }
            output = numer / num;
        }
        else if (activation.compare("RELU") == 0) {
            
            for (int i = 0; i < output.cols(); i++) {//assuming row vector
                if (output(0, i) < 0) {//output = 0 if input is negative; else return input
                    output(0, i) = 0;
                }
            }
            //output = hidden;
        }
        //Return output
        return output;
    }

//Pooling operation
Eigen::MatrixXf pooling(
    Eigen::MatrixXf input, //Input matrix
    std::string padding, //Padding type (valid or same)
    std::vector<int> kernel, //dimensions of kernel (x, y)
    std::string poolingType, //Pooling type (max, average, min...)
    std::vector<int> stride //Stride vector (x movement, y movement, ...)
    ){
        //std::cout << "i got here";
        long output_rows, output_col;
        float max = 0.0;
        long incx = 0, incy = 0;
        
        if(padding == "valid"){
            input = input;
        }
        else if (padding == "same"){
            Eigen::MatrixXf clone;
            //add layers of 0 based on
            int p_rows = ceil((kernel.at(0) - 1) /2.0);
            //std::cout << "p rows:\n" << p_rows << std::endl;
            int p_cols = ceil((kernel.at(1) - 1)/2.0);
            //std::cout << "p cols:\n" << p_cols << std::endl;
            int input_rows = input.rows() + 2 * p_rows;
            int input_col = input.cols() + 2 * p_cols;
            clone.resize(input_rows,input_col);
            clone = Eigen::MatrixXf::Zero(input_rows, input_col);
            
            for(int i = 0; i < input.cols(); i++){
                for(int j = 0; j < input.rows(); j++){
                    //std::cout << "input(i,j):\n" << input(i,j) << std::endl;
                    clone(i+p_rows,j+p_cols) = input(i,j);
                }
            }
            input = clone;
            //std::cout << "Here is the matrix c:\n" << clone << std::endl;
        }

        //Retun empty matrix
        Eigen::MatrixXf output;
        output_rows = floor((input.rows() - kernel.at(0))/stride.at(0)) + 1;
        output_col = floor((input.cols() - kernel.at(1))/stride.at(1)) + 1;
        output.resize(output_rows,output_col);

        if(poolingType == "max"){
            for(int y = 0; y < input.cols(); y = y + stride.at(1)){
                if(input.cols() < y + kernel.at(1)) break;
                //std::cout << "\n y is " << y << "stride is " << stride.at(0);
                for(int x = 0; x < input.rows(); x = x + stride.at(0)){
                    if(input.rows() < x + kernel.at(0)) break;
                    //std::cout << "\n x is " << x << "stride is " << stride.at(0);
                    //finding max_value
                    for(int i = y; i < kernel.at(0) + y; i++){
                        for(int j = x; j < kernel.at(1) + x; j++){
                            //std::cout << "\n kernel at 1+x is " << kernel.at(1) + x;
                            //std::cout << "\n i is " << i << " j is " << j << " input_val " << input(i,j);
                            if(input(i,j) > max){
                                max = input(i,j);
                                output(incy,incx) = max;
                            }
                        }
                    }
                    //std::cout << "\n max is " << max;
                    max = 0;
                    incx++;
                }
                incy++;
                incx = 0;
            }
        } 
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