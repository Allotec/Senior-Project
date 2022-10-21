#include "FlattenLayer.hpp"

//Constructor
FlattenLayer::FlattenLayer(std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape) : 
	Layer(name, FLATTEN_LAYER, dataType, inputShape, outputShape) {
}

FlattenLayer::FlattenLayer() : Layer(FLATTEN_LAYER) {}

FlattenLayer::~FlattenLayer() {}

//Methods
bool FlattenLayer::calculateOutput() {
	MatrixXfRM output(1, 0), temp;
	this->outputMatrix->clear();
	
	//Puts all the matrices into one flattened version
	for (int i = 0; i < this->inputMatrix->size(); i++) {
		temp = output;
		output.resize(1, output.size() + this->inputMatrix->at(i).size());
		this->inputMatrix->at(i).resize(1, this->inputMatrix->at(i).size());
		output << temp, this->inputMatrix->at(i);
	}
	
	this->outputMatrix->push_back(output);

	//Check shapes
	
	return(true);
}

//Print the layer information
void FlattenLayer::print() {
	Layer::print();
}
