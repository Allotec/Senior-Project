from encodings import utf_8
from sys import byteorder
import h5py
import numpy as np
import json
from CNNObjects import *

#obtain all the datasets from an .h5 file and store in numpy array
def get_datasets(h5_file):
    datasets = {}
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                datasets[path] = np.array(item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                h5py_dataset_iterator(item, path)

    h5py_dataset_iterator(h5_file)
    return datasets

#Obtain the model structure from the .h5 file
def get_model_structure(h5_file):
    model_structure = h5_file.attrs['model_config']
    return model_structure

#If a string is a substring of a key in a dictionary, return the value
def get_value_from_key(string, dictionary):
    for key in dictionary:
        if string + '/bias' in key:
            biasShape = dictionary[key].shape

    for key in dictionary:
        if string + '/kernel' in key:
            kernelShape = dictionary[key].shape

    return biasShape, kernelShape

def key_to_val(string, dictionary):
    for key in dictionary:
        if string in key:
            return dictionary[key]

#python main function
if __name__ == '__main__':
    #load the .h5 file
    h5_file = h5py.File('../Model_Creation/Model_Files/model.h5', 'r')

    #obtain the model structure as a dictionary from the json
    h5_structure = get_model_structure(h5_file)
    model_structure = json.loads(h5_structure)

    #obtain all the datasets from the .h5 file as a dictionary
    datasets = get_datasets(h5_file)
    
    data = []

    #Get the layers from the model and store in a list as Layer objects
    layers = model_structure['config']['layers']

    layersStruct = []

    #Create the layers list
    for layer in layers:
        layersStruct.append(Layer(layer))

    #Remove None from the input/ouput layer which specifies a variable batch size
    if None in layersStruct[0].layer.input_shape:
        layersStruct[0].layer.input_shape.remove(None)

    if None in layersStruct[0].layer.output_shape:
        layersStruct[0].layer.output_shape.remove(None)

    #Calculate the shapes for each input and output layer
    for i in range(1, len(layersStruct)):
        layersStruct[i].layer.input_shape = layersStruct[i - 1].layer.output_shape
        
        #Calculate the output shape depending on the layer type
        #If the layer is a convolution layer, calculate the output shape
        if layersStruct[i].name == 'Conv2D':
            #Calculate the padding amount
            if layersStruct[i].layer.padding == 'same':
                padding = layersStruct[i].layer.kernel_size[0] // 2 
            elif layersStruct[i].layer.padding == 'valid':
                padding = 0
            else:
                raise ValueError('Invalid padding type')
            
            #Formula for calculating the output shape of a convolution layer for each dimension
            for j in range(len(layersStruct[i].layer.input_shape) - 1):
                layersStruct[i].layer.output_shape.append((layersStruct[i].layer.input_shape[j] - layersStruct[i].layer.kernel_size[j] + 2 * padding) // layersStruct[i].layer.strides[j] + 1)
            
            #Add the number of filters to the output shape as the last dimension
            layersStruct[i].layer.output_shape.append(layersStruct[i].layer.filters)
        #Calculate the output shape for a max pooling or average pooling layer
        elif layersStruct[i].name == 'MaxPooling2D' or layersStruct[i].name == 'AveragePooling2D':
            #Calculating the padding amount
            if layersStruct[i].layer.padding == 'same':
                padding = layersStruct[i].layer.kernel_size[0] // 2 
            elif layersStruct[i].layer.padding == 'valid':
                padding = 0
            else:
                raise ValueError('Invalid padding type')
            
            #Calculating the output shape for each dimension
            for j in range(len(layersStruct[i].layer.input_shape) - 1):
                layersStruct[i].layer.output_shape.append((layersStruct[i].layer.input_shape[j] - layersStruct[i].layer.pool_size[j] + 2 * padding) // layersStruct[i].layer.strides[j] + 1)
            
            #Add the number of filters to the output shape as the last dimension (same as input # of filters)
            layersStruct[i].layer.output_shape.append(layersStruct[i].layer.input_shape[-1])
        #Calculating the output shape for a dense layer
        elif layersStruct[i].name == 'Dense':
            layersStruct[i].layer.output_shape.append(layersStruct[i].layer.units)
        #Calculating the output shape for a flatten layer
        elif layersStruct[i].name == 'Flatten':
            layersStruct[i].layer.output_shape.append(int(np.prod(layersStruct[i].layer.input_shape)))

    #Saving to hex file
    with open('modelHex.lit', 'wb') as f:
        #Write the layers to the file in order
        for layer in layersStruct:
            f.write(layer.toBytes())

            if layer.name == 'Conv2D' or layer.name == 'Dense':
                print("Writing data for " + layer.name)
                #Write the bias to the file
                f.write(START_DATA.to_bytes(1, byteorder='big'))
                f.write(key_to_val(layer.layer.name + '/bias', datasets).flatten().byteswap().tobytes())
                f.write(END_DATA.to_bytes(1, byteorder='big'))

                #Write the kernel to the file
                f.write(START_DATA.to_bytes(1, byteorder='big'))
                f.write(key_to_val(layer.layer.name + '/kernel', datasets).flatten().byteswap().tobytes())
                f.write(END_DATA.to_bytes(1, byteorder='big'))

    
    with open('modelHex.txt', 'w') as f:
        for key, val in datasets.items():
            f.write(key + str(val.shape) + '\n')
            f.write(str(val) + '\n')

    #close the .h5 file
    h5_file.close()
