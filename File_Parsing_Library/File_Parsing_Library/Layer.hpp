#pragma once
#include "LayerConstants.hpp"
#include "networkOps.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

// Parent class of all layers
class Layer {
protected: // Always check for proper layertype when assigning it to a child
  std::string name;                         // Unique name of the layer
  uint8_t layerType, dataType;              // Type of layer
  std::vector<int> inputShape, outputShape; // Matrix shapes
  std::vector<MatrixXfRM> *inputMatrix, *outputMatrix; // Matrices

public:
  // Constructor
  Layer(std::string name, int layerType, int dataType,
        std::vector<int> inputShape, std::vector<int> outputShape);
  Layer(int layerType);
  ~Layer();

  // Getters
  std::string getName();
  uint8_t getLayerType();
  uint8_t getDataType();
  std::vector<int> getInputShape();
  std::vector<int> getOutputShape();
  std::vector<MatrixXfRM> *getInputMatrix();
  std::vector<MatrixXfRM> *getOutputMatrix();

  // Setters
  void setName(std::string name);
  void setDataType(uint8_t dataType);
  void setInputShape(std::vector<int> inputShape);
  void setOutputShape(std::vector<int> outputShape);
  void setInputMatrix(std::vector<MatrixXfRM> *inputMatrix);
  void setOutputMatrix(std::vector<MatrixXfRM> *outputMatrix);

  // Methods
  // Calculates the output and returns true if successful false otherwise
  virtual bool calculateOutput() = 0;
  // Prints the layer
  void print();
};
