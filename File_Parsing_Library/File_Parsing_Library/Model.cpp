#include "Model.hpp"
#include <iostream>
#include <fstream>

Model::Model() {
	this->layers = std::vector<Layer*>();
}

Model::~Model() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->~Layer();
	}
}

void Model::addLayer(Layer* layer) {
	this->layers.push_back(layer);
}

bool Model::calculateOutput(std::vector<MatrixXfRM>* input, int index) {
	this->layers[0]->setInputMatrix(input);

	for (int i = 0; i < this->layers.size(); i++) {
		//Connect the current layer's input to the previous layer's output
		if(i != 0)
			this->layers[i]->setInputMatrix(this->layers[i - 1]->getOutputMatrix());
		
		/*std::cout << "Input Shape of Layer " << this->layers[i]->getName() << ": " << 
			this->layers[i]->getInputMatrix()->at(0).rows() << ", " << this->layers[i]->getInputMatrix()->at(0).cols() <<
			", " << this->layers[i]->getInputMatrix()->size() << std::endl;*/
		
		//Calculate the output
		this->layers[i]->calculateOutput();

		//std::cout << "Layer- " << i << std::endl << this->layers[i]->getOutputMatrix()->at(0) << std::endl;

		/*std::cout << "Output Shape of Layer " << this->layers[i]->getName() << ": " <<
			this->layers[i]->getOutputMatrix()->at(0).rows() << ", " << this->layers[i]->getOutputMatrix()->at(0).cols() <<
			", " << this->layers[i]->getOutputMatrix()->size() << std::endl;*/
	}

	//Find the max index of the output
	int maxIndex = 0;
	for (int i = 0; i < this->layers[this->layers.size() - 1]->getOutputShape().at(0); i++) {
		if (this->layers[this->layers.size() - 1]->getOutputMatrix()->at(0)(0, i) > this->layers[this->layers.size() - 1]->getOutputMatrix()->at(0)(0, maxIndex)) {
			maxIndex = i;
		}
	}

	std::cout << "Predicted: " << maxIndex << ", Actual: " << index << std::endl;

	return(maxIndex == index);
}

void Model::print() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->print();
	}
}

void Model::outputModel(std::string path){
	MatrixXfRM output(1, 0), temp;
	std::ofstream file(path, std::ios::out | std::ios::binary);
	
	//Puts the input matrices into one flattened version and outputs it to a binary file
	for (int i = 0; i < this->layers.size(); i++) {
		output = MatrixXfRM(1, 0);
		for (auto &matrix : *this->layers[i]->getOutputMatrix()) {
			temp = output;
			output.resize(1, output.size() + matrix.size());
			matrix.resize(1, matrix.size());
			output << temp, matrix;
		}

		for (int element = 0; element < output.size(); element++) {
			file.write((char*)&output(0, element), sizeof(float));
		}
	}

	file.close();
}
