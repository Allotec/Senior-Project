#include "modelRead.hpp"
#include "LayerConstants.hpp"
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include "FlattenLayer.hpp"
#include "InputLayer.hpp"
#include "PoolLayer.hpp"
#include <fstream>
#include <iostream>
#include <typeinfo>

//Returns a model with all the layers
Model* createModel(std::string path){
	Model* model = new Model();
	std::ifstream file("modelHex.lit", std::ios::binary);
	
	//If the file couldnt open return nothing
	if (!file.is_open()) return(nullptr);

	char val = 0;
	
	while (1) {
		//Checking for start of layer
		file.get(val);

		if (file.eof()) break;

		if ((val & 0xFF) != 0xFF) {
			std::cout << "Error on layer start" << std::endl;
			delete model;
			exit(1);
		}
		
		//Grabbing the layer type information
		file.get(val);

		switch (val) {
			case INPUT_LAYER:
				model->addLayer(readInputLayer(file));
				break;
			
			case CONV_LAYER:
				model->addLayer(readConvLayer(file));
				break;
			
			case MAXPOOL_LAYER:
				model->addLayer(readPoolLayer(file, 0));
				break;
			
			case AVGPOOL_LAYER:
				model->addLayer(readPoolLayer(file, 1));
				break;
				
			case FLATTEN_LAYER:
				model->addLayer(readFlattenLayer(file));
				break;
			
			case DENSE_LAYER:
				model->addLayer(readDenseLayer(file));
				break;
			default:
				std::cout << "Unknown Layer" << std::endl;
				exit(1);
				break;
		}
	}

	file.close();
	return(model);
}

Layer* commonParametersRead(std::ifstream& file, Layer* tempLayer) {
	char c = 1; //Temp byte storage

	//Input Layer Spec
	//The first two bytes were read in by the createModel function
	//Start Structure(1 byte)
	//Type(1 byte)

	//Name(Null terminated string)
	std::string name;

	while (c != '\0') {
		file.get(c);
		name.push_back(c);
	}
	tempLayer->setName(name);

	//Data type(1 byte)
	file.get(c);
	tempLayer->setDataType(c);

	//Input shape number of dimensions(1 byte)
	file.get(c);
	int dimNumber = c;

	//Input dimension size(2 bytes) variable amount according to the dimensions
	std::vector<int> tempVec;
	uint16_t dim;
	for (int i = 0; i < dimNumber; i++) {
		dim = 0;
		for (int j = 0; j < DIM_BYTE_NUMBER; j++) {
			file.get(c);
			dim = (dim << 8) | (c & 0xFF);
		}
		tempVec.push_back(dim);
	}

	tempLayer->setInputShape(tempVec);
	tempVec.clear();

	//Output shape number of dimensions(1 byte)
	file.get(c);
	dimNumber = c;

	//Output dimension size(2 bytes) variable amount according to the dimensions
	for (int i = 0; i < dimNumber; i++) {
		dim = 0;
		for (int j = 0; j < DIM_BYTE_NUMBER; j++) {
			file.get(c);
			dim = (dim << 8) | (c & 0xFF);
		}
		tempVec.push_back(dim);
	}

	tempLayer->setOutputShape(tempVec);
	tempVec.clear();

	return(tempLayer);
}

Layer* readInputLayer(std::ifstream& file) {
	InputLayer* tempLayer = new InputLayer();
	char c;
	
	//Read in the all the common parameters between the layers
	tempLayer = (InputLayer*)commonParametersRead(file, tempLayer);

	//Sparse bool(1 byte)
	file.get(c);
	tempLayer->setSparse(c == 0 ? false : true);

	//Ragged bool(1 byte)
	file.get(c);
	tempLayer->setRagged(c == 0 ? false : true);
	
	//End Structure(1 byte) Check for error
	file.get(c);
	if ((c & 0xFF) != END_STRUCTURE)
		std::cout << "Error reading input layer structure" << std::endl;
	
	//*Doesn't have any associated data
	
	return(tempLayer);
}

Layer* readConvLayer(std::ifstream& file) {
	ConvLayer* tempLayer = new ConvLayer();
	char c;

	//Read the common parameters
	tempLayer = (ConvLayer*)commonParametersRead(file, tempLayer);

	//Kernel shape number of dimensions(1 byte)
	file.get(c);
	int dimNumber = c;

	//Kernel dimension size(2 bytes) variable amount according to the dimensions
	std::vector<int> tempVec;
	uint16_t dim;
	for (int i = 0; i < dimNumber; i++) {
		dim = 0;
		for (int j = 0; j < DIM_BYTE_NUMBER; j++) {
			file.get(c);
			dim = (dim << 8) | (c & 0xFF);
		}
		tempVec.push_back(dim);
	}

	tempLayer->setKernelDimensions(tempVec);
	tempVec.clear();

	//Number of filters(1 bytes)
	file.get(c);
	tempLayer->setFilters(c);
	
	//Input stride number of dimensions(1 byte)
	file.get(c);
	dimNumber = c;
	
	//Input stride size(1 bytes) variable amount according to the dimensions
	for (int i = 0; i < dimNumber; i++) {
		file.get(c);
		tempVec.push_back(c);
	}

	tempLayer->setStrides(tempVec);
	tempVec.clear();
	
	//Padding(1 byte)
	file.get(c);
	tempLayer->setPadding(c);
	
	//Activation function(1 byte)
	file.get(c);
	tempLayer->setActivationFunction(c);

	//Groups(1 byte)
	file.get(c);
	tempLayer->setGroups(c);

	//End Structure(1 byte)
	file.get(c);
	if ((c & 0xFF) != END_STRUCTURE)
		std::cout << "Error reading Convolution layer structure" << std::endl;
	
	//Start Data(1 byte)* Bias Start
	file.get(c);
	if ((c & 0xFF) != START_DATA)
		std::cout << "Error reading Convolution layer bias data start" << std::endl;
	
	//Bias data(Size based on type)
	tempLayer->setBias(readMatrix(file, std::vector<int>{1, tempLayer->getFilters()}, tempLayer->getDataType()));

	//End data(1 byte)* Bias End
	file.get(c);
	if ((c & 0xFF) != END_DATA)
		std::cout << "Error reading Convolution layer bias data end" << std::endl;
	
	//Start Data(1 byte)* Kernel Start
	file.get(c);
	if ((c & 0xFF) != START_DATA)
		std::cout << "Error reading Convolution layer kernel start" << std::endl;

	//Kernel data(Size based on type)
	for(int j = 0; j < tempLayer->getInputShape().at(tempLayer->getInputShape().size() - 1); j++)
		for (int i = 0; i < tempLayer->getFilters(); i++) {
			tempLayer->addKernel(readMatrix(file, tempLayer->getKernelDimensions(), tempLayer->getDataType()));
		}

	//End data(1 byte)* Kernel End
	file.get(c);
	if ((c & 0xFF) != END_DATA)
		std::cout << "Error reading Convolution layer kernel end" << std::endl;
	
	return(tempLayer);
}

Layer* readPoolLayer(std::ifstream& file, int poolType) {
	PoolLayer* tempLayer = new PoolLayer();
	char c;

	//Read the common parameters
	tempLayer = (PoolLayer*)commonParametersRead(file, tempLayer);

	//Pool shape number of dimensions(1 byte)
	file.get(c);
	int dimNumber = c;
	
	//Pool dimension size(2 bytes) variable amount according to the dimensions
	std::vector<int> tempVec;
	uint16_t dim;
	for (int i = 0; i < dimNumber; i++) {
		dim = 0;
		for (int j = 0; j < DIM_BYTE_NUMBER; j++) {
			file.get(c);
			dim = (dim << 8) | (c & 0xFF);
		}
		tempVec.push_back(dim);
	}
	
	tempLayer->setPoolDimensions(tempVec);
	tempVec.clear();
	
	//Input stride number of dimensions(1 byte)
	file.get(c);
	dimNumber = c;
	
	//Input stride size(1 bytes) variable amount according to the dimensions
	for (int i = 0; i < dimNumber; i++) {
		file.get(c);
		tempVec.push_back(c);
	}
	
	tempLayer->setStrides(tempVec);
	tempVec.clear();
	
	//Padding(1 byte)
	file.get(c);
	tempLayer->setPadding(c);
	
	//End Structure(1 byte)
	file.get(c);
	if ((c & 0xFF) != END_STRUCTURE)
		std::cout << "Error reading Pooling layer structure" << std::endl;
	
	//*Doesn't have any associated data

	return(tempLayer);
}

Layer* readFlattenLayer(std::ifstream& file) {
	FlattenLayer* tempLayer = new FlattenLayer();
	char c;

	//Read the common parameters
	tempLayer = (FlattenLayer*)commonParametersRead(file, tempLayer);
	
	//End Structure(1 byte) Check for error
	file.get(c);
	if ((c & 0xFF) != END_STRUCTURE)
		std::cout << "Error reading flatten layer structure" << std::endl;

	//*Doesn't have any associated data

	return(tempLayer);
}

Layer* readDenseLayer(std::ifstream& file) {
	DenseLayer* tempLayer = new DenseLayer();
	char c;

	//Read the common parameters
	tempLayer = (DenseLayer*)commonParametersRead(file, tempLayer);

	//Activation function(1 byte)
	file.get(c);
	tempLayer->setActivationFunction(c);
	
	//End Structure(1 byte)
	file.get(c);
	if ((c & 0xFF) != END_STRUCTURE)
		std::cout << "Error reading dense layer structure" << std::endl;
	
	
	//Start Data(1 byte)* Bias Start
	file.get(c);
	if ((c & 0xFF) != START_DATA)
		std::cout << "Error reading dense layer bias data start" << std::endl;
	
	//Bias data(Size based on type)
	tempLayer->setBiases(readMatrix(file, std::vector<int>{1, tempLayer->getOutputShape().at(0)}, tempLayer->getDataType()));

	//End data(1 byte)* Bias End
	file.get(c);
	if ((c & 0xFF) != END_DATA)
		std::cout << "Error reading dense layer bias data end" << std::endl;
	
	//Start Data(1 byte)* Weights Start
	file.get(c);
	if ((c & 0xFF) != START_DATA)
		std::cout << "Error reading dense layer weight data start" << std::endl;
	
	//Weights data(Size based on type)
	tempLayer->setBiases(readMatrix(file, std::vector<int>{tempLayer->getInputShape().at(0), tempLayer->getOutputShape().at(0)}, tempLayer->getDataType()));

	//End data(1 byte)* Weights End
	file.get(c);
	if ((c & 0xFF) != END_DATA)
		std::cout << "Error reading dense layer weight data end" << std::endl;

	return(tempLayer);
}

Eigen::MatrixXf* readMatrix(std::ifstream& file, std::vector<int> dimensions, int dataType) {
	Eigen::MatrixXf* matrix = new Eigen::MatrixXf();

	matrix->resize(dimensions[0], dimensions[1]);

	//Read in the matrix
	for (int i = 0; i < dimensions[0]; i++) {
		for (int j = 0; j < dimensions[1]; j++) {
			(*matrix)(i, j) = readDecimal(file, dataType == 0 ? sizeof(float) : sizeof(double));
		}
	}

	return(matrix);
}

//Maybe support doubles later if needed
//Just pass union as a pointer and cast to the correct type
float readDecimal(std::ifstream& file, int sizeInBytes) {
	BytesToFloat num;
	char c;

	num.b = 0;
	//Read in bytes
	for (int i = 0; i < sizeInBytes; i++) {
		file.get(c);
		num.b = (num.b << 8) | (c & 0xFF);
	}
	
	return(num.f);
}
