#include <iostream>
#include "modelRead.hpp"
#include "ConvLayer.hpp"
#include "..\MNIST_Reader\MNIST_Reader\MNIST_Reader.hpp"
#include <vector>


int main() {
	Model* model = createModel("modelHex.lit");
	std::vector<std::pair<MatrixXfRM, int>>* image_Labels = 
		read_Mnist_Images(
			"..\\..\\Model_Creation\\samples\\t10k-images.idx3-ubyte", 
			"..\\..\\Model_Creation\\samples\\t10k-labels.idx1-ubyte"
		);
	
	std::vector<MatrixXfRM>* image;
	float correct = 0;
	int testAmount = 10;
	
	for (int i = 0; i < testAmount; i++) {
		image = new std::vector<MatrixXfRM>;
		image->push_back(image_Labels->at(i).first);
		if (model->calculateOutput(image, image_Labels->at(i).second))
			correct++;
		
		delete image;
	}

	model->outputModel("ModelOutput.bin");

	std::cout << "Accuracy: " << correct / (float)testAmount << std::endl;

    return(0);
}

