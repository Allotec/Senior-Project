#pragma once
#include "Layer.hpp"

class Model {
private:
  std::vector<Layer *> layers;

public:
  Model();
  ~Model();

  // Methods
  void addLayer(Layer *layer); // Adds to the end
  bool calculateOutput(std::vector<MatrixXfRM> *input, int index);
  void print(); // Prints summary
  void
  outputModel(std::string path); // Dumps the output of each layer to a bin file
};
