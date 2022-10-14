#pragma once
#include "Layer.hpp"

class Model{
private:
	std::vector<Layer*> layers;

public:
	Model();
	~Model();
	
	//Methods
	void addLayer(Layer* layer); //Adds to the end
	void calculateOutput(std::vector<Eigen::MatrixXf*> input);
	void print();
};