#pragma once
#include "Layer.hpp"

//Dense layer class
class DenseLayer : public Layer {
private:
	Eigen::MatrixXf* weights, *biases; //Weights and biases
	int activationFunction;
public:
	//Constructor
	DenseLayer(std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape, int activationFunction);
	DenseLayer();
	~DenseLayer();

	//Getters
	Eigen::MatrixXf* getWeights();
	Eigen::MatrixXf* getBiases();
	int getActivationFunction();

	//Setters
	void setWeights(Eigen::MatrixXf* weights);
	void setBiases(Eigen::MatrixXf* biases);
	void setActivationFunction(int activationFunction);

	//Methods
	bool calculateOutput();
	void printStructure();
	void printData();
};
