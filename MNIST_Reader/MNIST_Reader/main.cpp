#include <iostream>
#include "MNIST_Reader.hpp"

int main(){
	//Read Mnist files and print
	std::vector<std::pair<MatrixXfRM, int>>* image_Labels = read_Mnist_Images("..\\..\\Model_Creation\\samples\\train-images.idx3-ubyte", "..\\..\\Model_Creation\\samples\\train-labels.idx1-ubyte");

	//Print the first digit
	std::cout << "Image- " << std::endl << image_Labels->at(0).first << std::endl << image_Labels->at(0).second;

	return(0);
}

