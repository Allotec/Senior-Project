#include "..\MNIST_Reader\MNIST_Reader\MNIST_Reader.hpp"
#include "ConvLayer.hpp"
#include "modelRead.hpp"
#include <iostream>
#include <vector>
// #include <ostream>

int main() {
  Model *model = createModel("modelHex.lit");
  // Model* model = createModel("ConvTest.lit");
  model->print();
  std::vector<std::pair<MatrixXfRM, int>> *image_Labels = read_Mnist_Images(
      "..\\..\\Model_Creation\\samples\\t10k-images.idx3-ubyte",
      "..\\..\\Model_Creation\\samples\\t10k-labels.idx1-ubyte");

  std::vector<MatrixXfRM> *image;
  float correct = 0;
  int testAmount = 10;

  for (int i = 0; i < testAmount; i++) {
    image = new std::vector<MatrixXfRM>;
    image->push_back(image_Labels->at(i).first);

    /*MatrixXfRM test(5, 5);
    test << 1, 2, 3, 4, 5,
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24;

    MatrixXfRM test2(5, 5);

    test2 << 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            1, 2, 3, 4, 5,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24;

    image->push_back(test);
    image->push_back(test2);*/

    if (model->calculateOutput(image, image_Labels->at(i).second))
      correct++;

    delete image;
  }

  model->outputModel("ModelOutput.bin");

  // std::cout << "Accuracy: " << correct / (float)testAmount << std::endl;

  return (0);
}
