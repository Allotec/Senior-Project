#include "InputLayer.hpp"

//Constructor
InputLayer::InputLayer(bool sparse, bool ragged, std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape) : 
	Layer(name, INPUT_LAYER, dataType, inputShape, outputShape){
	this->sparse = sparse;
	this->ragged = ragged;
}

//Default Constructor
InputLayer::InputLayer() : Layer(INPUT_LAYER) {
	this->name = "";
	this->dataType = -1;
	this->inputShape = std::vector<int>();
	this->outputShape = std::vector<int>();
	this->sparse = false;
	this->ragged = false;
}

//Destructor
InputLayer::~InputLayer() {}

//Getters
bool InputLayer::getSparse() {
	return this->sparse;
}

bool InputLayer::getRagged() {
	return this->ragged;
}

//Setters
void InputLayer::setSparse(bool sparse) {
	this->sparse = sparse;
}

void InputLayer::setRagged(bool ragged) {
	this->ragged = ragged;
}

//Methods
//Calculates the output and returns true if successful false otherwise
bool InputLayer::calculateOutput() {
	this->outputMatrix = this->inputMatrix;
	return(this->inputMatrix != nullptr);
}

//Prints the layer
void InputLayer::print() {
	Layer::print();
	std::cout << "Sparse: " << this->sparse << std::endl;
	std::cout << "Ragged: " << this->ragged << std::endl;
}
