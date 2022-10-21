#include "DenseLayer.hpp"

//Constructor
DenseLayer::DenseLayer(std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape, int activationFunction) : 
	Layer(name, DENSE_LAYER, dataType, inputShape, outputShape) {
	this->activationFunction = activationFunction;
	this->weights = nullptr;
	this->biases = nullptr;
}

DenseLayer::DenseLayer() : Layer(DENSE_LAYER) {
	this->activationFunction = 0;
	this->weights = nullptr;
	this->biases = nullptr;
}

DenseLayer::~DenseLayer() {
	if (this->weights != nullptr) {
		delete this->weights;
	}
	if (this->biases != nullptr) {
		delete this->biases;
	}
}

//Getters
MatrixXfRM* DenseLayer::getWeights() {
	return this->weights;
}

MatrixXfRM* DenseLayer::getBiases() {
	return this->biases;
}

int DenseLayer::getActivationFunction() {
	return this->activationFunction;
}

//Setters
void DenseLayer::setWeights(MatrixXfRM* weights) {
	if (this->weights != nullptr) {
		delete this->weights;
	}
	this->weights = weights;
}

void DenseLayer::setBiases(MatrixXfRM* biases) {
	if (this->biases != nullptr) {
		delete this->biases;
	}
	this->biases = biases;
}

void DenseLayer::setActivationFunction(int activationFunction) {
	this->activationFunction = activationFunction;
}

//Methods
bool DenseLayer::calculateOutput() {
	this->outputMatrix->clear();
	this->outputMatrix->push_back(
		dense(
			this->inputMatrix->at(0), 
			*this->weights, 
			*this->biases, 
			this->activationFunction)
	);

	//Check shapes

	return(true);
}

//Print
void DenseLayer::printStructure() {
	Layer::print();
	std::cout << "Activation Function: " << this->activationFunction << std::endl;
}

void DenseLayer::printData() {
	std::cout << "Weights: " << std::endl;
	std::cout << *this->weights << std::endl;
	std::cout << "Biases: " << std::endl;
	std::cout << *this->biases << std::endl;
}
