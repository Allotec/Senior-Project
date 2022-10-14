#include "Model.hpp"
#include <iostream>

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

void Model::calculateOutput(std::vector<Eigen::MatrixXf*> input) {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->calculateOutput();
	}
}

void Model::print() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->print();
	}
}
