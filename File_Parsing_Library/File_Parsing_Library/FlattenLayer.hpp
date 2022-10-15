#pragma once
#include "Layer.hpp"

class FlattenLayer : public Layer {
private:

public:
	FlattenLayer(std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape);
	FlattenLayer();
	~FlattenLayer();

	bool calculateOutput();
	void print();
};
