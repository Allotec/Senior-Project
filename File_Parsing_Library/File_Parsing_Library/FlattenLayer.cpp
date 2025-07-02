#include "FlattenLayer.hpp"

// Constructor
FlattenLayer::FlattenLayer(std::string name, int dataType,
                           std::vector<int> inputShape,
                           std::vector<int> outputShape)
    : Layer(name, FLATTEN_LAYER, dataType, inputShape, outputShape) {}

FlattenLayer::FlattenLayer() : Layer(FLATTEN_LAYER) {}

FlattenLayer::~FlattenLayer() {}

// Methods
bool FlattenLayer::calculateOutput() {
  MatrixXfRM output, temp;
  this->outputMatrix->clear();

  int size = 0;

  for (int i = 0; i < this->getInputMatrix()->size(); i++) {
    size += this->getInputMatrix()->at(i).size();
  }

  output = MatrixXfRM::Zero(1, size);
  size = 0;

  for (int k = 0; k < this->getInputMatrix()->size(); k++) {
    for (int i = 0; i < this->getInputMatrix()->at(0).rows(); i++) {
      for (int j = 0; j < this->getInputMatrix()->at(0).cols(); j++) {
        output(0, size) = this->getInputMatrix()->at(k)(i, j);
        size++;
      }
    }
  }

  this->outputMatrix->push_back(output);

  return (true);
}

// Print the layer information
void FlattenLayer::print() { Layer::print(); }
