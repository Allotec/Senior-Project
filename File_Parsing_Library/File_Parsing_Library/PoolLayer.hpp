#pragma once
#include "Layer.hpp"

class PoolLayer : public Layer {
private:
	std::vector<int> poolDimensions;
	std::vector<int> strides;
	int padding; 
	int poolType; //0 for max, 1 for average
	
public:
	//Constructor
	PoolLayer(std::vector<int> poolDimensions, std::vector<int> strides, int padding, int poolType, std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape);
	PoolLayer();
	~PoolLayer();

	//Getters
	std::vector<int> getPoolDimensions();
	std::vector<int> getStrides();
	int getPadding();
	int getPoolType();

	//Setters
	void setPoolDimensions(std::vector<int> poolDimensions);
	void setStrides(std::vector<int> strides);
	void setPadding(int padding);
	void setPoolType(int poolType);

	//Methods
	bool calculateOutput();
	void print();
};
