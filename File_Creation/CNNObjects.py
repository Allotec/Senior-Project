#Constants for encoding layers into a binary file
#General layer constants
START_STRUCTURE = 0xFF
END_STRUCTURE = 0xFE
START_DATA = 0xFD
END_DATA = 0xFC

#Layer types
INPUT_LAYER = 0x00
CONV_LAYER = 0x01
MAXPOOL_LAYER = 0x02
DENSE_LAYER = 0x03
FLATTEN_LAYER = 0x04
AVGPOOL_LAYER = 0x05

#Layer data types
FLOAT32 = 0x00
FLOAT64 = 0x01

#Layer padding
VALID = 0x00
SAME = 0x01

#Layer activation functions
RELU = 0x00
SIGMOID = 0x01
TANH = 0x02
SOFTMAX = 0x03

#Returns the correct layer type depending on the name
#The second parameter is a dictionary containting the layer parameters
#The dictionary is passed down to the other classes where they get the parameters they need
#The .get function is used so the program doesn't crash if the parameter is not present
#The second parameter is the default value if the parameter is not present

class Layer:
    def __init__(self, layerDict):
        self.name = layerDict['class_name']
        #Determine the layer type and create the object
        if self.name == 'InputLayer':
            self.layer = InputLayer(layerDict)
        elif self.name == 'Conv2D':
            self.layer = ConvLayer(layerDict)
        elif self.name == 'MaxPooling2D' or self.name == 'AveragePooling2D':
            self.layer = PoolingLayer(layerDict)
        elif self.name == 'Flatten':
            self.layer = FlattenLayer(layerDict)
        elif self.name == 'Dense':
            self.layer = DenseLayer(layerDict)

    def toBytes(self):
        return self.layer.toBytes()

    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.layer)

#Create a class for an input layer
class InputLayer:
    def __init__(self, layerDict):
        #Common
        self.name = layerDict['config'].get('name', 'None') #The name of the layer
        self.input_shape = layerDict['config'].get('batch_input_shape', 'None') #The input shape of the layer as a list
        self.dtype = layerDict['config'].get('dtype', "float32") #The data type of the layer
        self.output_shape = layerDict['config'].get('batch_input_shape', 'None') #The output shape of the layer as a list
        #Unique to input layer
        self.sparse = layerDict['config'].get('sparse', False) #Boolean on whether the layer is sparse or not
        self.ragged = layerDict['config'].get('ragged', False) #Boolean on whether the layer is ragged or not
    
    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.input_shape) + " " + str(self.output_shape)  + " " + str(self.dtype) + " " + str(self.sparse) + " " + str(self.ragged)

    #Returns a byte array of the layer
    def toBytes(self):
        print("Encoding Input Layer")
        layerEncoding = bytearray()

        layerEncoding.append(START_STRUCTURE) #Start structure
        layerEncoding.append(INPUT_LAYER) #Layer type encoding
        layerEncoding.extend((self.name + '\0').encode('utf-8')) #Name of the layer
        
        layerEncoding.append(FLOAT32 if self.dtype == 'float32' else FLOAT64) #Data type of the layer

        #Input shape dimension number and dimension values
        layerEncoding.append(len(self.input_shape)) #Input shape of the layer
        for i in self.input_shape:
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) #Dimension value (2 bytes each)
        
        #Output shape dimension number and dimension values
        layerEncoding.append(len(self.output_shape)) #Output number of dimensions
        for i in self.output_shape:
            layerEncoding.extend(i.to_bytes(2, byteorder='big'))
        
        #Sparse and ragged bools
        layerEncoding.append(0x00 if self.sparse == False else 0x01) #Sparse bool
        layerEncoding.append(0x00 if self.ragged == False else 0x01) #Sparse bool

        layerEncoding.append(END_STRUCTURE) #End structure

        return(layerEncoding)

#Create a class for a convolutional neural network layer
class ConvLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None') #Unique string name for the layer
        self.trainable = layerDict['config'].get('trainable', True) #Boolean indicating whether the layer weights will be updated during training
        self.input_shape = list() #Shape of the input to the layer
        self.output_shape = list() #Shape of the output of the layer
        self.dtype = layerDict['config'].get('dtype', "float32") #Data type of the layer
        self.filters = layerDict['config'].get('filters', 0) #Number of filters in the convolution
        self.kernel_size = layerDict['config'].get('kernel_size', 'None') #List of integers specifying the height and width of the 2D convolution kernel
        self.strides = layerDict['config'].get('strides', 'None') #List of integers specifying the strides of the convolution along the height and width
        self.padding = layerDict['config'].get('padding', 'valid') #One of "valid" or "same" valid means no padding, same means pad evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input
        self.data_format = layerDict['config'].get('data_format', 'channels_last') #Changes the ording of the dimensions channels_last (batch_size, height, width, channels) or channels_first (batch_size, channels, height, width)
        self.groups = layerDict['config'].get('groups', 1) #Number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups
        self.activation = layerDict['config'].get('activation', 'relu') #Activation function to use sigmoid, tanh, relu, softmax, etc
        self.use_bias = layerDict['config'].get('use_bias', True) #Boolean, whether the layer uses a bias vector
        self.kernel_initializer = layerDict['config'].get('kernel_initializer', 'none') #Initializer for the kernel weights matrix
        self.bias_initializer = layerDict['config'].get('bias_initializer', 'None') #Initializer for the bias vector
        self.kernel_regularizer = layerDict['config'].get('kernel_regularizer', 'None') #Regularizer function applied to the kernel weights matrix
        self.bias_regularizer = layerDict['config'].get('bias_regularizer', 'None') #Regularizer function applied to the bias vector
        self.kernel_constraint = layerDict['config'].get('kernel_constraint', 'None') #Constraint function applied to the kernel weights matrix
        self.bias_constraint = layerDict['config'].get('bias_constraint', 'None') #Constraint function applied to the bias vector

    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.trainable) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.dtype) + " " + str(self.filters) + " " + str(self.kernel_size) + " " + str(self.strides) + " " + str(self.padding) + " " + str(self.data_format) + " " + str(self.groups) + " " + str(self.activation) + " " + str(self.use_bias) + " " + str(self.kernel_initializer) + " " + str(self.bias_initializer) + " " + str(self.kernel_regularizer) + " " + str(self.bias_regularizer) + " " + str(self.kernel_constraint) + " " + str(self.bias_constraint)

    #Returns a byte array of the layer
    def toBytes(self):
        print("Encoding Convolutional Layer")
        layerEncoding = bytearray()

        layerEncoding.append(START_STRUCTURE) #Start structure
        layerEncoding.append(CONV_LAYER) #Layer type encoding
        layerEncoding.extend((self.name + '\0').encode('utf-8')) #Name of the layer
        
        layerEncoding.append(FLOAT32 if self.dtype == 'float32' else FLOAT64) #Data type of the layer

        #Input shape dimension number and dimension values
        layerEncoding.append(len(self.input_shape)) #Input shape of the layer
        for i in self.input_shape:
            #(height, width, channels)
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) #Dimension value (2 bytes each)
        
        #Output shape dimension number and dimension values
        layerEncoding.append(len(self.output_shape)) #Output number of dimensions
        for i in self.output_shape:
            layerEncoding.extend(i.to_bytes(2, byteorder='big'))
        
        #Kernel size dimension number and dimension values
        layerEncoding.append(len(self.kernel_size)) #Kernel 
        for i in self.kernel_size:
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) 

        layerEncoding.append(self.filters)

        #Strides dimension number and dimension values
        layerEncoding.append(len(self.strides)) #Strides
        for i in self.strides:
            layerEncoding.append(i) #Stride value (1 byte each)

        #Padding
        layerEncoding.append(VALID if self.padding == 'valid' else SAME) #Padding

        #Activation function
        if self.activation == 'relu':
            layerEncoding.append(RELU)
        elif self.activation == 'sigmoid':
            layerEncoding.append(SIGMOID)
        elif self.activation == 'tanh':
            layerEncoding.append(TANH)
        elif self.activation == 'softmax':
            layerEncoding.append(SOFTMAX)

        
        #Number of groups
        layerEncoding.append(self.groups)

        layerEncoding.append(END_STRUCTURE) #End structure

        return(layerEncoding)

#Create a class for a max pooling layer
class PoolingLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None') #Unique string name for the layer
        self.trainable = layerDict['config'].get('trainable', True) #Boolean indicating whether the layer weights will be updated during training
        self.dtype = layerDict['config'].get('dtype', "float32") #Data type of the layer
        self.input_shape = layerDict['config'].get('batch_input_shape', list()) #Shape of the input to the layer (height, width, channels)
        self.output_shape = list() #Shape of the output of the layer as a list (height, width, channels)
        self.pool_size = layerDict['config'].get('pool_size', 'None') #Size of the pooling window
        self.padding = layerDict['config'].get('padding', 'valid') #One of "valid" or "same" valid means no padding, same means keep the shape of the output the same with adding zeros on the border
        self.strides = layerDict['config'].get('strides', 'None') #List of integers specifying how far the pooling window moves for each pooling step
        self.data_format = layerDict['config'].get('data_format', 'channels_last') #Changes the ording of the dimensions channels_last (height, width, channels) or channels_first (channels, height, width)
        self.poolType = layerDict['class_name'] #The type of pooling layer max or average

    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.trainable) + " " + str(self.dtype) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.pool_size) + " " + str(self.padding) + " " + str(self.strides) + " " + str(self.data_format) + " " + str(self.poolType)

    #Returns a byte array of the layer
    def toBytes(self):
        print("Encoding Pooling Layer")
        layerEncoding = bytearray()

        layerEncoding.append(START_STRUCTURE) #Start structure
        layerEncoding.append(MAXPOOL_LAYER if 'Max' in self.poolType else AVGPOOL_LAYER) #Layer type encoding
        layerEncoding.extend((self.name + '\0').encode('utf-8')) #Name of the layer
        
        layerEncoding.append(FLOAT32 if self.dtype == 'float32' else FLOAT64) #Data type of the layer

        #Input shape dimension number and dimension values
        layerEncoding.append(len(self.input_shape)) #Input shape of the layer
        for i in self.input_shape:
            #(height, width, channels)
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) #Dimension value (2 bytes each)
        
        #Output shape dimension number and dimension values
        layerEncoding.append(len(self.output_shape)) #Output number of dimensions
        for i in self.output_shape:
            layerEncoding.extend(i.to_bytes(2, byteorder='big'))
        
        #Pool size dimension number and dimension values
        layerEncoding.append(len(self.pool_size)) #Kernel 
        for i in self.pool_size:
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) 

        #Strides dimension number and dimension values
        layerEncoding.append(len(self.strides)) #Strides
        for i in self.strides:
            layerEncoding.append(i) #Stride value (1 byte each)

        #Padding
        layerEncoding.append(VALID if self.padding == 'valid' else SAME) #Padding


        layerEncoding.append(END_STRUCTURE) #End structure

        return(layerEncoding)

#Create a class for a fully connected layer
class DenseLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None') #Unique string name for the layer
        self.trainable = layerDict['config'].get('trainable', True) #Boolean indicating whether the layer weights will be updated during training
        self.dtype = layerDict['config'].get('dtype', "float32") #Data type of the layer
        self.input_shape = layerDict['config'].get('batch_input_shape', list()) #Shape of the input to the layer (height, width, channels)
        self.output_shape = list() #Shape of the output of the layer as a list (height, width, channels)
        self.units = layerDict['config'].get('units', 0) #Number of neurons in the output layer
        self.activation = layerDict['config'].get('activation', 'relu') #Activation function to use sigmoid, tanh, relu, softmax, etc
        self.use_bias = layerDict['config'].get('use_bias', True) #Boolean, whether the layer uses a bias vector
        self.kernel_initializer = layerDict['config'].get('kernel_initializer', 'none') #Initializer for the kernel weights matrix
        self.bias_initializer = layerDict['config'].get('bias_initializer', 'None') #Initializer for the bias vector
        self.kernel_regularizer = layerDict['config'].get('kernel_regularizer', 'None') #Regularizer function applied to the kernel weights matrix
        self.bias_regularizer = layerDict['config'].get('bias_regularizer', 'None') #Regularizer function applied to the bias vector
        self.activity_regularizer = layerDict['config'].get('activity_regularizer', 'None') #Regularizer function applied to the output of the layer (its "activation")
        self.kernel_constraint = layerDict['config'].get('kernel_constraint', 'None') #Constraint function applied to the kernel weights matrix
        self.bias_constraint = layerDict['config'].get('bias_constraint', 'None') #Constraint function applied to the bias vector

    #converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.trainable) + " " + str(self.dtype) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.units) + " " + str(self.activation) + " " + str(self.use_bias) + " " + str(self.kernel_initializer) + " " + str(self.bias_initializer) + " " + str(self.kernel_regularizer) + " " + str(self.bias_regularizer) + " " + str(self.activity_regularizer) + " " + str(self.kernel_constraint) + " " + str(self.bias_constraint)

    #Returns a byte array of the layer
    def toBytes(self):
        print("Encoding Dense layer")
        layerEncoding = bytearray()

        layerEncoding.append(START_STRUCTURE) #Start structure
        layerEncoding.append(DENSE_LAYER) #Layer type encoding
        layerEncoding.extend((self.name + '\0').encode('utf-8')) #Name of the layer
        
        layerEncoding.append(FLOAT32 if self.dtype == 'float32' else FLOAT64) #Data type of the layer

        #Input shape dimension number and dimension values
        layerEncoding.append(len(self.input_shape)) #Input shape of the layer
        for i in self.input_shape:
            #(height, width, channels)
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) #Dimension value (2 bytes each)

        #Output shape dimension number and dimension values
        layerEncoding.append(len(self.output_shape)) #Output number of dimensions
        for i in self.output_shape:
            layerEncoding.extend(i.to_bytes(2, byteorder='big'))
        
        #Activation function
        if self.activation == 'relu':
            layerEncoding.append(RELU)
        elif self.activation == 'sigmoid':
            layerEncoding.append(SIGMOID)
        elif self.activation == 'tanh':
            layerEncoding.append(TANH)
        elif self.activation == 'softmax':
            layerEncoding.append(SOFTMAX)

        layerEncoding.append(END_STRUCTURE) #End structure

        return(layerEncoding)
        

#Create a class for a flatten layer
class FlattenLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None') #Unique string name for the layer
        self.input_shape = layerDict['config'].get('batch_input_shape', list()) #Shape of the input to the layer (height, width, channels)
        self.output_shape = list() #Shape of the output of the layer as a list (size of the vector)
        self.trainable = layerDict['config'].get('trainable', True) #Boolean indicating whether the layer weights will be updated during training
        self.dtype = layerDict['config'].get('dtype', "float32") #Data type of the layer
        self.data_format = layerDict['config'].get('data_format', 'channels_last') #Not sure what this means in a flatten layer

    #converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.trainable) + " " + str(self.dtype) + " " + str(self.data_format)

    #Returns a byte array of the layer
    def toBytes(self):
        print("Encoding Flattened layer")
        layerEncoding = bytearray()

        layerEncoding.append(START_STRUCTURE) #Start structure
        layerEncoding.append(FLATTEN_LAYER) #Layer type encoding
        layerEncoding.extend((self.name + '\0').encode('utf-8')) #Name of the layer
        layerEncoding.append(FLOAT32 if self.dtype == 'float32' else FLOAT64) #Data type of the layer

        #Input shape dimension number and dimension values
        layerEncoding.append(len(self.input_shape)) #Input shape of the layer
        for i in self.input_shape:
            #(height, width, channels)
            layerEncoding.extend(i.to_bytes(2, byteorder='big')) #Dimension value (2 bytes each)
        
        #Output shape dimension number and dimension values
        layerEncoding.append(len(self.output_shape)) #Output number of dimensions
        for i in self.output_shape:
            layerEncoding.extend(i.to_bytes(2, byteorder='big'))

        layerEncoding.append(END_STRUCTURE) #End structure

        return(layerEncoding)
        
    