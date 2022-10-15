#pragma once
#include "Model.hpp"

//Returns the full model
Model* createModel(std::string path);

//Reads in the specific layer formats and returns a upcasted parent pointer
Layer* commonParametersRead(std::ifstream& file, Layer* tempLayer);
Layer* readInputLayer(std::ifstream& file);
Layer* readConvLayer(std::ifstream& file);
Layer* readPoolLayer(std::ifstream& file, int poolType);
Layer* readFlattenLayer(std::ifstream& file);
Layer* readDenseLayer(std::ifstream& file);

//Reads in the data and returns a matrix with the size parameters
Eigen::MatrixXf* readMatrix(std::ifstream& file, std::vector<int> dimensions, int dataType);
float readDecimal(std::ifstream& file, int sizeInBytes);

union BytesToFloat {
	double d;
	float f;
	uint64_t b;
};
