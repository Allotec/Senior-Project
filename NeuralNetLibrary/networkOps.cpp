#include <iostream>
#include <float.h>
#include "networkOps.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <Eigen/Dense>
#include <cmath>

//Convolution operation
MatrixXfRM convolution(MatrixXfRM input, int padding, MatrixXfRM kernel, std::vector<int> stride, float bias, int activation ){
    int output_rows, output_col;
    long incx = 0, incy = 0;
    long krow = 0, kcol = 0;

	/*std::cout << "Padding- " << (padding == VALID ? "VALID" : "SAME") << std::endl;
	std::cout << "Activation- " << (activation == RELU ? "RELU" : "IDK") << std::endl;
	for (auto i : stride)
		std::cout << "Stride- " << i << std::endl;
	std::cout << "Input- " << std::endl << input << std::endl << 
		"Kernel- " << std::endl << kernel << std::endl <<
		"Bias- " << bias << std::endl;*/
        
    //Check for same if valid do nothing
    if(padding == SAME){
        MatrixXfRM clone;
        //add layers of 0 based on
        int p_rows = ceil((kernel.rows() - 1) / 2.0);
        int p_cols = ceil((kernel.cols() - 1) / 2.0);

        int input_rows = input.rows() + 2 * p_rows;
        int input_col = input.cols() + 2 * p_cols;
        clone.resize(input_rows,input_col);
        clone = MatrixXfRM::Zero(input_rows, input_col);
            
        for(int i = 0; i < input.cols(); i++){
            for(int j = 0; j < input.rows(); j++){
                clone(i + p_rows,j + p_cols) = input(i, j);
            }
        }

        input = clone;
    }
        
    MatrixXfRM output = MatrixXfRM::Zero(
        (int)floor((input.rows() - kernel.rows()) / stride.at(0)) + 1, 
        (int)floor((input.cols() - kernel.cols()) / stride.at(1)) + 1
    );

	//This is more correct I think
    for(int outputRows = 0, x = 0; outputRows < output.rows() && x < input.rows(); outputRows++, x += stride.at(0)) {
        for(int outputCols = 0, y = 0; outputCols < output.cols() && y < input.cols(); outputCols++, y += stride.at(1)) {
            //Weighted sum 
            for(int i = 0; i < kernel.rows(); i++){
                for(int j = 0; j < kernel.cols(); j++){
                    output(outputRows, outputCols) += input(x + i, y + j) * kernel(i, j);
                }
            }
        }
    }

    //Apply the bias
    //output += MatrixXfRM::Constant(output.rows(), output.cols(), bias);

    //Add the activation function
    if (activation == SOFTMAX) {
        return(Softmax(output));
    }
    else if (activation == RELU) {
        return(Relu(output));
    }

    //Return output
    return output;
}

std::vector<MatrixXfRM>* conv2d(std::vector<MatrixXfRM> input, std::vector<MatrixXfRM*> filter, int stride, MatrixXfRM bias, int num_input_channels, int num_output_channels) {
    // Get the dimensions of the input and filter tensors
    int input_rows = input[0].rows();
    int input_cols = input[0].cols();
    int filter_rows = filter.at(0)->rows();
    int filter_cols = filter.at(0)->cols();

    // Calculate the dimensions of the output tensor
    int output_rows = (input_rows - filter_rows) / stride + 1;
    int output_cols = (input_cols - filter_cols) / stride + 1;

    // Initialize the output tensor
    std::vector<MatrixXfRM>* output = new std::vector<MatrixXfRM>;

    // Perform the convolution
    int filter_index = 0;
    for (int oc = 0; oc < num_output_channels; oc++) {
        output->push_back(MatrixXfRM::Zero(output_rows, output_cols));
        for (int ic = 0; ic < num_input_channels; ic++) {
            for (int i = 0; i < output_rows; i++) {
                for (int j = 0; j < output_cols; j++) {
                    output->at(oc)(i, j) += (input.at(ic).block(i * stride, j * stride, filter_rows, filter_cols).cwiseProduct(*filter.at(filter_index))).sum();
                }
            }
            filter_index++;
        }

        output->at(oc) += MatrixXfRM::Constant(output->at(oc).rows(), output->at(oc).cols(), bias(0, oc));
    }

    return(ReluMatrix(output));
}

#include<fstream>

//Pooling operation
MatrixXfRM pooling(MatrixXfRM input, int padding, std::vector<int> kernel, int poolingType, std::vector<int> stride ){
    long output_rows, output_col;
    float max = 0.0;
    long incx = 0, incy = 0;
        
    
    if (padding == SAME){
        MatrixXfRM clone;
        //add layers of 0 based on
        int p_rows = ceil((kernel.at(0) - 1) /2.0);
        int p_cols = ceil((kernel.at(1) - 1)/2.0);
        
        int input_rows = input.rows() + 2 * p_rows;
        int input_col = input.cols() + 2 * p_cols;

        clone.resize(input_rows,input_col);
        clone = MatrixXfRM::Zero(input_rows, input_col);
            
        for(int i = 0; i < input.cols(); i++){
            for(int j = 0; j < input.rows(); j++){
                clone(i + p_rows,j + p_cols) = input(i,j);
            }
        }
        input = clone;
    }

    //Retun empty matrix
    MatrixXfRM output = MatrixXfRM::Zero(//Doesn't work for non square strides
        (int)floor((input.rows() - kernel.at(0)) / stride.at(0)) + 1,
        (int)floor((input.cols() - kernel.at(1)) / stride.at(1)) + 1
    );
    //std::ofstream file;
    //file.open("pooling.txt");
    //file << input << "\n\n";
    //Zero for Max Layer
    if(poolingType == 0){
        /*for(int y = 0; (y < input.cols() && input.cols() >= y + kernel.at(1)); y += stride.at(1)){
            for(int x = 0; x < input.rows() && input.rows() >= x + kernel.at(0); x += stride.at(0)){
                for(int i = y; i < kernel.at(0) + y; i++){
                    for(int j = x; j < kernel.at(1) + x; j++){
                        if(input(i,j) > max){
                            max = input(i,j);
                            output(incy,incx) = max;
                        }
                    }
                }
                max = 0;
                incx++;
            }
            incy++;
            incx = 0;
        }*/

        int i = 0;

        for (int outputRows = 0, x = 0; outputRows < output.rows() && x < input.rows(); outputRows++, x += stride.at(0)) {
            for (int outputCols = 0, y = 0; outputCols < output.cols() && y < input.cols(); outputCols++, y += stride.at(1)) {
                //Weighted sum 
                max = 0;
                //file << "Element- " << i++ << "\n";
                for (int i = 0; i < kernel.at(0); i++) {
                    for (int j = 0; j < kernel.at(1); j++) {
                        if (input(x + i, y + j) > max) {
                            max = input(x + i, y + j);
                            output(outputRows, outputCols) = max;
                        }
                        //file << input(x + i, y + j) << " ";
                    }
                    //file << "\n";
                }

                //file << "Max- " << max << "\n\n";
            }
        }
    } 
    //Average pooling
    else if (poolingType == 1) {

    }

    
    return(output);
}

//Dense Layer operation
MatrixXfRM dense(MatrixXfRM input, MatrixXfRM weights, MatrixXfRM bias, int activation){
    //Inputs and Weights arent matching up because of the convolution filter stuff
    
    if (activation == SOFTMAX) {
        return(Softmax((input * weights) + bias));
    }
    else if (activation == RELU) {
        return(Relu((input * weights) + bias));
    }
    else {
        std::cout << "Error: Invalid Activation Function using default!" << std::endl;
    }

    return(Softmax((input * weights) + bias));
}


//Calculates the Softmax of a matrix
MatrixXfRM Softmax(MatrixXfRM input) {
    //loop to loop thru all values in hidden
    //den = e^input
    MatrixXfRM temp(1, input.cols()), numer(1, input.cols());
    double num = 0;
    for (int i = 0; i < input.cols(); i++) {//gets denominator 
        temp(0, i) = exp(input(0, i));
        num += temp(0, i);
    }

    //loop through all values in hidden
    //num = e^input
    //output for that iteration = num/den
    for (int i = 0; i < input.cols(); i++) {
        numer(0, i) = exp(input(0, i));
    }
	
    return(numer / num);
}

//Calculates the relu of a matrix
MatrixXfRM Relu(MatrixXfRM input) {
    //assuming row vector
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            //output = 0 if input is negative; else return input
            if (input(i, j) < 0) {
                input(i, j) = 0;
            }
        }
    }
    return(input);
}

std::vector<MatrixXfRM>* ReluMatrix(std::vector<MatrixXfRM>* input) {
    //assuming row vector
    for (int k = 0; k < input->size(); k++) {
        for (int i = 0; i < input->at(k).rows(); i++) {
            for (int j = 0; j < input->at(k).cols(); j++) {
                //output = 0 if input is negative; else return input
                if (input->at(k)(i, j) < 0) {
                    input->at(k)(i, j) = 0;
                }
            }
        }
    }

    return(input);
}

