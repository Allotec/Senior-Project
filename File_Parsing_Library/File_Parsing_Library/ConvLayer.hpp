#pragma once
#include "Layer.hpp"

// Supports 3D and 2D convolution
class ConvLayer : public Layer {
private:
  int filters;
  std::vector<int> kernelDimensions;
  std::vector<MatrixXfRM *> kernels;
  MatrixXfRM *bias; // Should be a vector
  std::vector<int> strides;
  int padding;
  int activationFunction;
  int groups;

public:
  // Constructor
  ConvLayer(int filters, std::vector<int> kernelDimensions,
            std::vector<int> strides, int padding, int activationFunction,
            int groups, std::string name, int dataType,
            std::vector<int> inputShape, std::vector<int> outputShape);
  ConvLayer();
  ~ConvLayer();

  // Getters
  int getFilters();
  std::vector<int> getKernelDimensions();
  std::vector<MatrixXfRM *> getKernels();
  MatrixXfRM *getBias();
  std::vector<int> getStrides();
  int getPadding();
  int getActivationFunction();
  int getGroups();

  // Setters
  void setFilters(int filters);
  void setKernelDimensions(std::vector<int> kernelDimensions);
  void setStrides(std::vector<int> strides);
  void setPadding(int padding);
  void setActivationFunction(int activationFunction);
  void setGroups(int groups);
  void addKernel(MatrixXfRM *kernel);
  void setBias(MatrixXfRM *bias);

  // Methods
  bool calculateOutput();
  void printStructure();
  void printData();
};
