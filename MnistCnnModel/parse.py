from email.mime import base
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

#Calculate the convolution output layer shape given the input shape, kernel size, stride, and padding
def conv_output_shape(input_shape, filter_size, padding, stride):
    if padding == 'same':
        padding = filter_size // 2
    elif padding == 'valid':
        padding = 0
    else:
        raise ValueError('Invalid padding type')

    return (input_shape[0] - filter_size + 2 * padding) // stride + 1

#String to binary
def str_to_bin(string):
    return ''.join(format(ord(i), 'b') for i in string)

#String to hex with each character represented by 2 hex digits
def str_to_hex(string):
    return ''.join(format(ord(i), '02x') for i in string)

#If a string is a substring of a key in a dictionary, return the value
def get_value_from_key(string, dictionary):
    for key in dictionary:
        if string + '/bias' in key:
            biasShape = dictionary[key].shape

    for key in dictionary:
        if string + '/kernel' in key:
            kernelShape = dictionary[key].shape

    return biasShape, kernelShape

#python main function
if __name__ == '__main__':
    #load the .h5 file
    h5_file = h5py.File('model.h5', 'r')

    h5_structure = get_model_structure(h5_file)

    #Save the model structure to a .json file
    # with open('model_structure.json', 'w') as f:
    #     f.write(h5_structure)

    datasets = get_datasets(h5_file)
    #parsing a json file to a python dictionary
    model_structure = json.loads(h5_structure)

    #Get the layers from the model
    layers = model_structure['config']['layers']

    layersStruct = []

    for layer in layers:
        #Returns the layer type
        layersStruct.append(Layer(layer))

    #Remove None from the input layer
    if None in layersStruct[0].layer.input_shape:
        layersStruct[0].layer.input_shape.remove(None)

    if None in layersStruct[0].layer.output_shape:
        layersStruct[0].layer.output_shape.remove(None)

    print(layersStruct[0].name + str(layersStruct[0].layer.input_shape) + " " + str(layersStruct[0].layer.output_shape))

    #Calculate the shapes for each input and output layer
    for i in range(1, len(layersStruct)):
        print(str(layersStruct[i]))
        layersStruct[i].layer.input_shape = layersStruct[i - 1].layer.output_shape

        #Calculate the output shape depending on the layer type
        #If the layer is a convolution layer, calculate the output shape
        if layersStruct[i].name == 'Conv2D': 
            if layersStruct[i].layer.padding == 'same':
                padding = layersStruct[i].layer.kernel_size[0] // 2
            elif layersStruct[i].layer.padding == 'valid':
                padding = 0
            else:
                raise ValueError('Invalid padding type')
            
            layersStruct[i].layer.output_shape.append((layersStruct[i].layer.input_shape[0] - layersStruct[i].layer.kernel_size[0] + 2 * padding) // layersStruct[i].layer.strides[0] + 1)
            layersStruct[i].layer.output_shape.append((layersStruct[i].layer.input_shape[1] - layersStruct[i].layer.kernel_size[1] + 2 * padding) // layersStruct[i].layer.strides[1] + 1)
            layersStruct[i].layer.output_shape.append(layersStruct[i].layer.filters)
        #elif layersStruct[i].name == 'MaxPooling2D' or layersStruct[i].name == 'AveragePooling2D':
            
        #print(layersStruct[i].name + str(layersStruct[i].layer.input_shape) + " " + str(layersStruct[i].layer.output_shape))

    #Saving to hex file
    with open('modelHex.txt', 'w') as f:
        for key, value in datasets.items():
            f.write('%s %s\n' % (key, value.shape))

            flat = value.flatten()
            for i in range(len(flat)):
                f.write('%s ' % hex(np.float32(flat[i]).view(np.uint32)).replace('0x', ''))

            f.write('\n')

    #close the .h5 file
    h5_file.close()


