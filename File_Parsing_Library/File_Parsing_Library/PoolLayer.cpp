#include "PoolLayer.hpp"

//Constructors 
PoolLayer::PoolLayer(std::vector<int> poolDimensions, std::vector<int> strides, int padding, int poolType, std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape) : 
	Layer(name, poolType == 0 ? MAXPOOL_LAYER : AVGPOOL_LAYER, dataType, inputShape, outputShape) {
	this->poolDimensions = poolDimensions;
	this->strides = strides;
	this->padding = padding;
	this->poolType = poolType;
}

PoolLayer::PoolLayer() : Layer(MAXPOOL_LAYER) {
	this->poolDimensions = std::vector<int>();
	this->strides = std::vector<int>();
	this->padding = 0;
	this->poolType = 0;
}

//Destructors 
PoolLayer::~PoolLayer() {}

//Getters
std::vector<int> PoolLayer::getPoolDimensions() {
	return this->poolDimensions;
}

std::vector<int> PoolLayer::getStrides() {
	return this->strides;
}

int PoolLayer::getPadding() {
	return this->padding;
}

int PoolLayer::getPoolType() {
	return this->poolType;
}

//Setters
void PoolLayer::setPoolDimensions(std::vector<int> poolDimensions) {
	this->poolDimensions = poolDimensions;
}

void PoolLayer::setStrides(std::vector<int> strides) {
	this->strides = strides;
}

void PoolLayer::setPadding(int padding) {
	this->padding = padding;
}

void PoolLayer::setPoolType(int poolType) {
	this->poolType = poolType;
	this->layerType = this->poolType == 0 ? MAXPOOL_LAYER : AVGPOOL_LAYER;
}

//Methods
bool PoolLayer::calculateOutput() {
	this->outputMatrix->clear();

	for (int i = 0; i < this->inputMatrix->size(); i++) {
		this->outputMatrix->push_back(pooling(this->inputMatrix->at(i), this->padding, this->poolDimensions, this->poolType, this->strides));
	}

	return(true);
}

//Print
void PoolLayer::print() {
	Layer::print();
	std::cout << "Pool Dimensions: ";
	for (int i = 0; i < this->poolDimensions.size(); i++) {
		std::cout << this->poolDimensions[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Strides: ";
	for (int i = 0; i < this->strides.size(); i++) {
		std::cout << this->strides[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Padding: " << this->padding << std::endl;
	std::cout << "Pool Type: " << this->poolType << std::endl;
}
