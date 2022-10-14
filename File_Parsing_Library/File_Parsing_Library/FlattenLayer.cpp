#include "FlattenLayer.hpp"

//Constructor
FlattenLayer::FlattenLayer(std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape) : 
	Layer(name, FLATTEN_LAYER, dataType, inputShape, outputShape) {
}

FlattenLayer::FlattenLayer() : Layer(FLATTEN_LAYER) {}

FlattenLayer::~FlattenLayer() {}

//Methods
bool FlattenLayer::calculateOutput() {
	return(true);
}

//Print the layer information
void FlattenLayer::print() {
	Layer::print();
}
