#include "FlattenLayer.hpp"

//Constructor
FlattenLayer::FlattenLayer(std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape) : 
	Layer(name, FLATTEN_LAYER, dataType, inputShape, outputShape) {
}

FlattenLayer::FlattenLayer() : Layer(FLATTEN_LAYER) {}

FlattenLayer::~FlattenLayer() {}

//Methods
bool FlattenLayer::calculateOutput() {
	MatrixXfRM output, temp;
	this->outputMatrix->clear();
	
	//Puts all the matrices into one flattened version
	/*for (int i = 0; i < this->inputMatrix->size(); i++) {
		temp = output;
		output.resize(1, output.size() + this->inputMatrix->at(i).size());
		this->inputMatrix->at(i).resize(1, this->inputMatrix->at(i).size());
		output << temp, this->inputMatrix->at(i);
	}
	
	this->outputMatrix->push_back(output);*/

	int size = 0;
	
	for (int i = 0; i < this->getInputMatrix()->size(); i++) {
		size += this->getInputMatrix()->at(i).size();
	}

	output = MatrixXfRM::Zero(1, size);
	size = 0;

	for (int k = 0; k < this->getInputMatrix()->size(); k++) {
		for (int i = 0; i < this->getInputMatrix()->at(0).rows(); i++) {
			for (int j = 0; j < this->getInputMatrix()->at(0).cols(); j++) {
				/*std::cout << k << " " << i << " " << j << std::endl;
				std::cout << size << "- " << this->getInputMatrix()->at(k)(i, j) << std::endl;*/
 				output(0, size) = this->getInputMatrix()->at(k)(i, j);
				size++;
			}
		}
	}

	this->outputMatrix->push_back(output);

	return(true);
}

//Print the layer information
void FlattenLayer::print() {
	Layer::print();
}
