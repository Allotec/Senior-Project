#pragma once
#include "stdint.h"
//Constants for encoding layers into a binary file
//General layer constants
const uint8_t START_STRUCTURE = 0xFF;
const uint8_t END_STRUCTURE = 0xFE;
const uint8_t START_DATA = 0xFD;
const uint8_t END_DATA = 0xFC;

//Layer types
const uint8_t INPUT_LAYER = 0x00;
const uint8_t CONV_LAYER = 0x01;
const uint8_t MAXPOOL_LAYER = 0x02;
const uint8_t DENSE_LAYER = 0x03;
const uint8_t FLATTEN_LAYER = 0x04;
const uint8_t AVGPOOL_LAYER = 0x05;

//Layer data types
const uint8_t FLOAT32 = 0x00;
const uint8_t FLOAT64 = 0x01;

//Layer padding
const uint8_t VALID = 0x00;
const uint8_t SAME = 0x01;

//Layer activation functions
const uint8_t RELU = 0x00;
const uint8_t SIGMOID = 0x01;
const uint8_t TANH = 0x02;
const uint8_t SOFTMAX = 0x03;

//Dimension constants
const uint8_t DIM_BYTE_NUMBER = 2;
const uint8_t STRIDE_BYTE_NUMBER = 1;
