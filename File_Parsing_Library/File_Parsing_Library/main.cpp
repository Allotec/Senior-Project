#include <iostream>
#include "modelRead.hpp"

int main() {
	Model* model = createModel("modelHex.lit");

	model->print();

    return(0);
}

