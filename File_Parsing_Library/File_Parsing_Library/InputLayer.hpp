#pragma once
#include "Layer.hpp"

class InputLayer : public Layer {
private:
	//No idea what these are for yet
	bool sparse, ragged;

public:
	//Constructor	
	InputLayer(bool sparse, bool ragged, std::string name, int dataType, std::vector<int> inputShape, std::vector<int> outputShape);
	InputLayer();
	~InputLayer();

	//Getters
	bool getSparse();
	bool getRagged();
	
	//Setters
	void setSparse(bool sparse);
	void setRagged(bool ragged);

	//Methods
	bool calculateOutput();
	void print();
};
