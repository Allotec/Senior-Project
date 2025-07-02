#include "Layer.hpp"
#include <iostream>

// Constructor
Layer::Layer(std::string name, int layerType, int dataType,
             std::vector<int> inputShape, std::vector<int> outputShape) {
  this->name = name;
  this->layerType = layerType;
  this->dataType = dataType;
  this->inputShape = inputShape;
  this->outputShape = outputShape;
  this->inputMatrix = nullptr;
  this->outputMatrix = new std::vector<MatrixXfRM>;
}

// Default constructor
Layer::Layer(int layerType) {
  this->name = "";
  this->dataType = -1;
  this->inputShape = std::vector<int>();
  this->outputShape = std::vector<int>();
  this->layerType = layerType;
  this->inputMatrix = nullptr;
  this->outputMatrix = new std::vector<MatrixXfRM>;
}

// Destructor
Layer::~Layer() {
  if (inputMatrix != nullptr) {
    delete inputMatrix;
  }
  if (outputMatrix != nullptr) {
    delete outputMatrix;
  }
}

// Getters
std::string Layer::getName() { return this->name; }

uint8_t Layer::getLayerType() { return this->layerType; }

uint8_t Layer::getDataType() { return this->dataType; }

std::vector<int> Layer::getInputShape() { return this->inputShape; }

std::vector<int> Layer::getOutputShape() { return this->outputShape; }

std::vector<MatrixXfRM> *Layer::getInputMatrix() { return this->inputMatrix; }

std::vector<MatrixXfRM> *Layer::getOutputMatrix() { return this->outputMatrix; }

// Setters
void Layer::setName(std::string name) { this->name = name; }

void Layer::setDataType(uint8_t dataType) { this->dataType = dataType; }

void Layer::setInputShape(std::vector<int> inputShape) {
  this->inputShape = inputShape;
}

void Layer::setOutputShape(std::vector<int> outputShape) {
  this->outputShape = outputShape;
}

// Should make sure that the dimensions are correct before assigning
void Layer::setInputMatrix(std::vector<MatrixXfRM> *inputMatrix) {
  this->inputMatrix = inputMatrix;
}

// Should check if the dimensions are correct before assigning
void Layer::setOutputMatrix(std::vector<MatrixXfRM> *outputMatrix) {
  if (outputMatrix != nullptr) {
    delete this->outputMatrix;
  }
  this->outputMatrix = outputMatrix;
}

// Prints the layer
void Layer::print() {
  std::cout << std::endl << "Name: " << this->name << std::endl;
  std::cout << "Layer Type: " << (int)this->layerType << std::endl;
  std::cout << "Data Type: "
            << (this->dataType == 0   ? "Float32"
                : this->dataType == 1 ? "Float64"
                                      : "Uknown")
            << std::endl;
  std::cout << "Input Shape: ";
  for (int i = 0; i < this->inputShape.size(); i++) {
    std::cout << this->inputShape[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Output Shape: ";
  for (int i = 0; i < this->outputShape.size(); i++) {
    std::cout << this->outputShape[i] << " ";
  }
  std::cout << std::endl;
}
