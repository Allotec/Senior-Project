#include <iostream>
#include "networkOps.hpp"
#include <vector>

//Convolution operation
Eigen::MatrixXf convolution(
    Eigen::MatrixXf input, //Input matrix
    std::string padding, //Padding type (valid or same)
    Eigen::MatrixXf kernel, //Kernel matrix
    std::vector<int> stride, //Stride vector (x movement, y movement, ...)
    Eigen::MatrixXf bias, //Bias matrix
    std::string activation //Activation function (relu, sigmoid, tanh, softmax, linear, ...)
    ){
        int input_rows, input_col, kernel_rows, kernel_col, output_rows, output_col;
        //Retun empty matrix
        Eigen::MatrixXf output;

        

        if(padding == "valid"){
            //output <= image based on kernel and stride
            output_rows = input_rows - kernel_rows + 1;
            output_col = input_col - kernel_col + 1;
            output(output_rows,output_col);
        }
        else if(padding == "same"){
            //output size == input size when stride = 1
            int p = (kernel_rows - 1)/2;
            output_rows = input_rows + 2 * p - kernel_rows + 1;
            output_col = input_col + 2 * p - kernel_col + 1;
            output(output_rows,output_col);
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
        int output_rows, output_col;
        int max = 0;
        int incx = 0, incy = 0;
        //Retun empty matrix
        Eigen::MatrixXf output;
        output_rows = floor((input.rows() - kernel.at(0))/stride.at(0)) + 1;
        output_col = floor((input.cols() - kernel.at(1))/stride.at(1)) + 1;
        output.resize(output_rows,output_col);  
        //std::cout << "output" << output.size() << " ";
        //std::cout << "i got here";

        if(padding == "valid"){
            input = input;
        }
        else if (padding == "same"){
            //add layers of 0 based on
            //int p = (kernel.at(0)- 1)/2; // this many 0 rows
            //int p = (kernel.at(1)- 1)/2; // this many 0 columns
        }
        std::cout << "i got here";

        if(poolingType == "max"){
            for(int y = 0; y < input.cols(); y = y + stride.at(1)){
                if(input.cols() < y + kernel.at(1)) break;
                std::cout << "\n y is " << y << "stride is " << stride.at(0);
                for(int x = 0; x < input.rows(); x = x + stride.at(0)){
                    if(input.rows() < x + kernel.at(0)) break;
                    std::cout << "\n x is " << x << "stride is " << stride.at(0);
                    //finding max_value
                    for(int i = y; i < kernel.at(0) + y; i++){
                        for(int j = x; j < kernel.at(1) + x; j++){
                            std::cout << "\n kernel at 1+x is " << kernel.at(1) + x;
                            std::cout << "\n i is " << i << " j is " << j << " input_val " << input(i,j);
                            if(input(i,j) > max){
                                max = input(i,j);
                                output(incy,incx) = max;
                                //std::cout << "\n max is " << max;
                            }
                            //if(j > output_col) break;
                        }
                        //if(i > output_rows) break;
                    }
                    std::cout << "\n max is " << max;
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