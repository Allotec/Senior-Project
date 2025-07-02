#pragma once
#include "Layer.hpp"

// Dense layer class
class DenseLayer : public Layer {
private:
  MatrixXfRM *weights, *biases; // Weights and biases
  int activationFunction;

public:
  // Constructor
  DenseLayer(std::string name, int dataType, std::vector<int> inputShape,
             std::vector<int> outputShape, int activationFunction);
  DenseLayer();
  ~DenseLayer();

  // Getters
  MatrixXfRM *getWeights();
  MatrixXfRM *getBiases();
  int getActivationFunction();

  // Setters
  void setWeights(MatrixXfRM *weights);
  void setBiases(MatrixXfRM *biases);
  void setActivationFunction(int activationFunction);

  // Methods
  bool calculateOutput();
  void printStructure();
  void printData();
};
