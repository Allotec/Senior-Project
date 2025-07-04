# File Format Description

Each layer is specified by a structure then its data in 32 bit floating point numbers.
Each layer will be followed by another layer until the file ends.

## General layer constants 1 byte-

- Start Structure byte- 0xFF
- End Structure byte- 0xFE

- Start data- 0xFD
- End data- 0xFC

## Layer type 1 byte-

- Input layer- 0x00
- Convolution layer- 0x01
- Max Pooling layer- 0x02
- Dense layer- 0x03
- Flatten layer- 0x04
- Average Pooling Layer- 0x05

## Layer data type 1 byte-

- 32 bit float- 0x00
- 64 bit float- 0x01

## Layer padding 1 byte-

- Valid- 0x00
- Same- 0x01

## Layer activation 1 byte-

- ReLU- 0x00
- Sigmoid- 0x01
- Tanh- 0x02
- Softmax- 0x03

## InputLayer-

- Start Structure (1 byte)
- Type (1 byte)
- Name (Null terminated string)
- Data type (1 byte)
- Input shape number of dimensions (1 byte)
- Input dimension size (2 bytes) variable amount according to the dimensions
- Output shape number of dimensions (1 byte)
- Output dimension size (2 bytes) variable amount according to the dimensions
- Sparse bool (1 byte)
- Ragged bool (1 byte)
- End Structure (1 byte)

## Convolution Layer-

Start Structure (1 byte)

- Type (1 byte)
- Name (Null terminated string)
- Data type (1 byte)
- Input shape number of dimensions (1 byte)
- Input dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Output shape number of dimensions (1 byte)
- Output dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Kernel shape number of dimensions (1 byte)
- Kernel dimension size (2 bytes) variable amount according to the dimensions
- Number of filters (1 bytes)
- Input stride number of dimensions (1 byte)
- Input stride size (1 bytes) variable amount according to the dimensions
- Padding (1 byte)
- Activation function (1 byte)
- Groups (1 byte)
- End Structure (1 byte)
- Start Data (1 byte) \*Bias Start
- Bias data (Size based on type)
- End data (1 byte) \*Bias End
- Start Data (1 byte) \*Weights Start
- Weights data (Size based on type)
- End data (1 byte) \*Weights End

## Pooling Layer-

- Start Structure (1 byte)
- Type (1 byte)
- Name (Null terminated string)
- Data type (1 byte)
- Input shape number of dimensions (1 byte)
- Input dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Output shape number of dimensions (1 byte)
- Output dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Pool shape number of dimensions (1 byte)
- Pool dimension size (2 bytes) variable amount according to the dimensions
- Input stride number of dimensions (1 byte)
- Input stride size (1 bytes) variable amount according to the dimensions
- Padding (1 byte)
- End Structure (1 byte)

## Dense Layer-

- Start Structure (1 byte)
- Type (1 byte)
- Name (Null terminated string)
- Data type (1 byte)
- Input shape number of dimensions (1 byte)
- Input dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Output shape number of dimensions (1 byte)
- Output dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Activation function (1 byte)
- End Structure (1 byte)

- Start Data (1 byte) \*Bias Start
- Bias data (Size based on type)
- End data (1 byte) \*Bias End
- Start Data (1 byte) \*Weights Start
- Weights data (Size based on type)
- End data (1 byte) \*Weights End

## Flatten Layer-

- Start Structure (1 byte)
- Type (1 byte)
- Name (Null terminated string)
- Data type (1 byte)
- Input shape number of dimensions (1 byte)
- Input dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- Output shape number of dimensions (1 byte)
- Output dimension size (2 bytes) variable amount according to the dimensions channels_last (height, width, channels)
- End Structure (1 byte)
