#include "ConvLayer.hpp"

//Constructor
ConvLayer::ConvLayer(int filters, std::vector<int> kernelDimensions, std::vector<int> strides, int padding, int activationFunction, int groups, std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape) : 
	Layer(name, CONV_LAYER, dataType, inputShape, outputShape) {
	this->filters = filters;
	this->kernelDimensions = kernelDimensions;
	this->strides = strides;
	this->padding = padding;
	this->activationFunction = activationFunction;
	this->groups = groups;
	this->bias = nullptr;
	this->kernels = std::vector<MatrixXfRM*>();
}

//Default Constructor
ConvLayer::ConvLayer() : Layer(CONV_LAYER) {
	this->filters = 0;
	this->kernelDimensions = std::vector<int>();
	this->strides = std::vector<int>();
	this->padding = 0;
	this->activationFunction = 0;
	this->groups = 0;
}

//Destructor
ConvLayer::~ConvLayer() {
	for (auto i : this->kernels) {
		delete i;
	}
	if (this->bias != nullptr) {
		delete this->bias;
	}
}

//Getters
int ConvLayer::getFilters() {
	return this->filters;
}

std::vector<int> ConvLayer::getKernelDimensions() {
	return this->kernelDimensions;
}

std::vector<MatrixXfRM*> ConvLayer::getKernels() {
	return this->kernels;
}

MatrixXfRM* ConvLayer::getBias() {
	return this->bias;
}

std::vector<int> ConvLayer::getStrides() {
	return this->strides;
}

int ConvLayer::getPadding() {
	return this->padding;
}

int ConvLayer::getActivationFunction() {
	return this->activationFunction;
}

int ConvLayer::getGroups() {
	return this->groups;
}

//Setters
void ConvLayer::setFilters(int filters) {
	this->filters = filters;
}

void ConvLayer::setKernelDimensions(std::vector<int> kernelDimensions) {
	this->kernelDimensions = kernelDimensions;
}

void ConvLayer::setStrides(std::vector<int> strides) {
	this->strides = strides;
}

void ConvLayer::setPadding(int padding) {
	this->padding = padding;
}

void ConvLayer::setActivationFunction(int activationFunction) {
	this->activationFunction = activationFunction;
}

void ConvLayer::setGroups(int groups) {
	this->groups = groups;
}

void ConvLayer::addKernel(MatrixXfRM* kernel) {
	this->kernels.push_back(kernel);
}

void ConvLayer::setBias(MatrixXfRM* bias) {
	if (this->bias != nullptr) {
		delete this->bias;
	}
	this->bias = bias;
}

//Methods
bool ConvLayer::calculateOutput() {
	/*std::cout << "Kernel Dimensions in vector: " << this->kernels.at(0)->rows() << ", " << 
		this->kernels.at(0)->cols() << ", " << this->kernels.size() << std::endl;*/
	
	this->outputMatrix->clear();
	int index = 0;

	//std::cout << "Kernel 0- " << std::endl << *kernels.at(0) << std::endl;
	
	for (int j = 0; j < this->filters; j++) {
		MatrixXfRM tempInput = Eigen::MatrixXf::Zero(this->outputShape.at(0), this->outputShape.at(1));
		for (int i = 0; i < this->inputMatrix->size(); i++) {
			tempInput += 
				convolution(
				this->inputMatrix->at(i),
				this->padding,
				*kernels.at(index++),
				this->strides,
				(*this->bias)(0, j),
				this->activationFunction
			);
		}
		
		this->outputMatrix->push_back(tempInput);
	}

	return(true);
}

//Print the layer
void ConvLayer::printStructure() {
	Layer::print();
	std::cout << "Filters: " << this->filters << std::endl;
	
	std::cout << "Kernel Dimensions: ";
	for (int i = 0; i < this->kernelDimensions.size(); i++) {
		std::cout << this->kernelDimensions[i] << " ";
	}
	std::cout << std::endl;
	
	std::cout << "Strides: ";
	for (int i = 0; i < this->strides.size(); i++) {
		std::cout << this->strides[i] << " ";
	}
	std::cout << std::endl;

	if (this->padding == VALID) 
		std::cout << "Padding: VALID" << std::endl;
	else if (this->padding == SAME) 
		std::cout << "Padding: SAME" << std::endl;
	else 
		std::cout << "Padding Invalid: " << this->padding << std::endl;
	
	if (this->activationFunction == RELU)
		std::cout << "Activation Function: RELU" << std::endl;
	else if (this->activationFunction == SIGMOID)
		std::cout << "Activation Function: SIGMOID" << std::endl;
	else if (this->activationFunction == TANH)
		std::cout << "Activation Function: TANH" << std::endl;
	else if (this->activationFunction == SOFTMAX)
		std::cout << "Activation Function: SOFTMAX" << std::endl;
	else
		std::cout << "Activation Function Invalid: " << this->activationFunction << std::endl;
	
	std::cout << "Groups: " << this->groups << std::endl;
}

void ConvLayer::printData() {
	std::cout << "Bias: " << std::endl;
	std::cout << *this->bias << std::endl;

	std::cout << "Kernels: " << std::endl;
	for (int i = 0; i < this->kernels.size(); i++) {
		std::cout << "Kernel " << i << std::endl;
		std::cout << *this->kernels[i] << std::endl;
	}
}

