import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import backend as K
import tensorflow as tf
import numpy as np
from idx2numpy import convert_from_file
from visualkeras import layered_view
from struct import unpack

#Obtain the model structure from the .h5 file
def get_model_structure(h5_file):
    model_structure = h5_file.attrs['model_config']
    return model_structure

#Loads in the data set
def data_set():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels

#Converts the data set into a format that can be used by the model
#The data comes in as (BatchSize, Height, Width) and needs to be converted to (BatchSize, Height, Width, Channels)
def data_preproc(train_images, train_labels, test_images, test_labels):
    #Reshape and scale to 0-1
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))  # Converting every image to 1d; train_images has a shape of 60000x28x28
    train_images = train_images.astype('float32') / 255  # scaling between 0,1
    
    #Reshape and scale to 0-1
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    
    #Converts the labels from integers to one hot vectors
    #This is done because the model has multiple outputs and you take the argmax of the output to get the predicted label
    train_labels = to_categorical(train_labels)  # Converts list labels to numpy array (one-hot encoding)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    #Loads the dataset
    train_images, train_labels, test_images, test_labels = data_set()
    train_images, train_labels, test_images, test_labels = data_preproc(train_images, train_labels, test_images, test_labels)
    #Loads the model 
    model = tf.keras.models.load_model('../Model_Creation/Model_Files/model.h5')

    #Get the output of each layer for the first test image
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_images[0].reshape(1, 28, 28, 1))

    createdModelOutput = '../File_Parsing_Library/File_Parsing_Library/modelOutput.bin'
    actualModelOutput = 'output.bin'

    #Write the output of each layer of the model to a file
    with open(actualModelOutput, 'wb') as f:
        f.write(test_images[0].astype(np.float32).flatten().tobytes())
        for i in range(len(activations)):
            f.write(activations[i].astype(np.float32).flatten().tobytes())

    file = open('keras.txt', 'w')

    #Compare the output of the created model to the actual model
    with open(createdModelOutput, 'rb') as f:
        tempArray = np.fromfile(f, dtype=np.float32, count=test_images[0].size, sep='', offset=0)
        if np.array_equal(tempArray, test_images[0].astype(np.float32).flatten()):
            print('Input image passed')
        else:
            print('Input image failed')

        count = 0
        for i in activations:
            tempArray = np.fromfile(f, dtype=np.float32, count=i.size, sep='', offset=0)
            if np.array_equal(tempArray, i.astype(np.float32).flatten()):
                print("Layer " + str(count) + " passed")
            else:
                print("Layer " + str(count) + " failed")
            
            np.savetxt('Layer' + str(count) + '.txt', tempArray)
            file.write('Layer- ' + str(count) + '\n')
            file.write(np.array2string(i, threshold=np.inf))
            file.write('\n')

            count += 1