#include "MNIST_Reader.hpp"
#include <iostream>

//Reads in the images and labels from the MNIST data set and return a pointer to a vector of pairs of images and labels
std::vector<std::pair<MatrixXfRM, int>>* read_Mnist_Images(std::string images, std::string labels) {
	std::ifstream imageFile(images, std::ios::binary);
	std::ifstream labelFile(labels, std::ios::binary);

	//Read in the magic number and the number of images
	int magicNumber = 0;
	int numberOfImages = 0;
	int numberOfRows = 0;
	int numberOfColumns = 0;
	imageFile.read((char*)&magicNumber, sizeof(magicNumber));
	imageFile.read((char*)&numberOfImages, sizeof(numberOfImages));
	imageFile.read((char*)&numberOfRows, sizeof(numberOfRows));
	imageFile.read((char*)&numberOfColumns, sizeof(numberOfColumns));

	//Conver to big endian
	magicNumber = reverseInt(magicNumber);
	numberOfImages = reverseInt(numberOfImages);
	numberOfRows = reverseInt(numberOfRows);
	numberOfColumns = reverseInt(numberOfColumns);
	
	//Read in the magic number and the number of labels
	int magicNumber2 = 0;
	int numberOfLabels = 0;
	labelFile.read((char*)&magicNumber2, sizeof(magicNumber2));
	labelFile.read((char*)&numberOfLabels, sizeof(numberOfLabels));

	//Convert to big endian
	magicNumber2 = reverseInt(magicNumber2);
	numberOfLabels = reverseInt(numberOfLabels);
	
	//Create a vector of pairs of images and labels
	std::vector<std::pair<MatrixXfRM, int>>* imageVector = new std::vector<std::pair<MatrixXfRM, int>>();

	//Read in the images and labels and add them to the vector
	for (int i = 0; i < numberOfImages; i++) {
		MatrixXfRM image(numberOfRows, numberOfColumns);
		for (int j = 0; j < numberOfRows; j++) {
			for (int k = 0; k < numberOfColumns; k++) {
				unsigned char temp = 0;
				imageFile.read((char*)&temp, sizeof(temp));
				image(j, k) = (float)temp;
			}
		}
		unsigned char tempLabel = 0;
		labelFile.read((char*)&tempLabel, sizeof(tempLabel));
		imageVector->push_back(std::make_pair(image / 255, (int)tempLabel));
	}

	return imageVector;
}

//Converts big endian to little endian
int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
