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
        
        //Calculate the convolution
        for(int y = 0; y < input.cols() && input.cols() >= y + kernel.cols(); y += stride.at(1)){
            for(int x = 0; x < input.rows() && input.rows() >= x + kernel.rows(); x += stride.at(0)){
                //Weighted sum 
                for(int i = y; i < kernel.rows() + y; i++){
                    for(int j = x; j < kernel.cols() + x; j++){
                        output(incy,incx) += input(i, j) * kernel(krow,kcol);
                        kcol++;
                    }
                    kcol = 0;
                    krow++;
                }
                kcol = 0;
                krow = 0;
                incx++;
            }
            incy++;
            incx = 0;
        }

        //Apply the bias
        output += MatrixXfRM::Constant(output.rows(), output.cols(), bias);

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

    //Zero for Max Layer
    if(poolingType == 0){
        for(int y = 0; (y < input.cols() && input.cols() >= y + kernel.at(1)); y += stride.at(1)){
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
        }
    } 
    //Average pooling
    else if (poolingType == 1) {

    }

    return(output);
}

//Dense Layer operation
MatrixXfRM dense(MatrixXfRM input, MatrixXfRM weights, MatrixXfRM bias, int activation){
    //Print the input shape
    
    //Inputs and Weights arent matching up because of the convolution filter stuff
    if (activation == SOFTMAX) {
        return(Softmax((input * weights) + bias));
    }
    else if (activation == RELU) {
        return(Relu((input * weights) + bias));
    }
    else {
        std::cout << "Error: Invalid Activation Function using relu default!" << std::endl;
    }

    return(Relu((input * weights) + bias));
}


//Calculates the Softmax of a matrix
MatrixXfRM Softmax(MatrixXfRM input) {
    return(input);
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
