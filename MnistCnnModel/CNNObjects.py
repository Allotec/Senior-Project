#Returns the correct layer type depending on the name
class Layer:
    def __init__(self, layerDict):
        self.name = layerDict['class_name']
        
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

    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.layer)

#Create a class for an input layer
class InputLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None')
        self.input_shape = layerDict['config'].get('batch_input_shape', 'None')
        self.dtype = layerDict['config'].get('dtype', "float32")
        self.sparse = layerDict['config'].get('sparse', False)
        self.ragged = layerDict['config'].get('ragged', False)
        self.output_shape = layerDict['config'].get('batch_input_shape', 'None')
    
    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.input_shape) + " " + str(self.output_shape)  + " " + str(self.dtype) + " " + str(self.sparse) + " " + str(self.ragged)

#Create a class for a convolutional neural network layer
class ConvLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None') #Unique string name for the layer
        self.trainable = layerDict['config'].get('trainable', True) #Boolean indicating whether the layer weights will be updated during training
        self.input_shape = list() #Shape of the input to the layer
        self.output_shape = self.input_shape #Shape of the output of the layer
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

#Create a class for a max pooling layer
class PoolingLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None') #Unique string name for the layer
        self.trainable = layerDict['config'].get('trainable', True) #Boolean indicating whether the layer weights will be updated during training
        self.dtype = layerDict['config'].get('dtype', "float32")
        self.input_shape = layerDict['config'].get('batch_input_shape', list())
        self.output_shape = self.input_shape
        self.pool_size = layerDict['config'].get('pool_size', 'None')
        self.padding = layerDict['config'].get('padding', 'valid')
        self.strides = layerDict['config'].get('strides', 'None')
        self.data_format = layerDict['config'].get('data_format', 'channels_last')
        self.poolType = layerDict['class_name']

    #Converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.trainable) + " " + str(self.dtype) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.pool_size) + " " + str(self.padding) + " " + str(self.strides) + " " + str(self.data_format) + " " + str(self.poolType)
        

#Create a class for a fully connected layer
class DenseLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None')
        self.trainable = layerDict['config'].get('trainable', True)
        self.dtype = layerDict['config'].get('dtype', "float32")
        self.input_shape = layerDict['config'].get('batch_input_shape', list())
        self.output_shape = self.input_shape
        self.units = layerDict['config'].get('units', 0)
        self.activation = layerDict['config'].get('activation', 'relu')
        self.use_bias = layerDict['config'].get('use_bias', True)
        self.kernel_initializer = layerDict['config'].get('kernel_initializer', 'none')
        self.bias_initializer = layerDict['config'].get('bias_initializer', 'None')
        self.kernel_regularizer = layerDict['config'].get('kernel_regularizer', 'None')
        self.bias_regularizer = layerDict['config'].get('bias_regularizer', 'None')
        self.activity_regularizer = layerDict['config'].get('activity_regularizer', 'None')
        self.kernel_constraint = layerDict['config'].get('kernel_constraint', 'None')
        self.bias_constraint = layerDict['config'].get('bias_constraint', 'None')

    #converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.trainable) + " " + str(self.dtype) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.units) + " " + str(self.activation) + " " + str(self.use_bias) + " " + str(self.kernel_initializer) + " " + str(self.bias_initializer) + " " + str(self.kernel_regularizer) + " " + str(self.bias_regularizer) + " " + str(self.activity_regularizer) + " " + str(self.kernel_constraint) + " " + str(self.bias_constraint)
        

#Create a class for a flatten layer
class FlattenLayer:
    def __init__(self, layerDict):
        self.name = layerDict['config'].get('name', 'None')
        self.input_shape = layerDict['config'].get('batch_input_shape', list())
        self.output_shape = self.input_shape
        self.trainable = layerDict['config'].get('trainable', True)
        self.dtype = layerDict['config'].get('dtype', "float32")
        self.data_format = layerDict['config'].get('data_format', 'channels_last')

    #converts the layer to a string
    def __str__(self):
        return str(self.name) + " " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.trainable) + " " + str(self.dtype) + " " + str(self.data_format)
        