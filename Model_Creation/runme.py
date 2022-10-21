import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
from idx2numpy import convert_from_file
from visualkeras import layered_view

## Overview
# This code runs a CNN model (deep learning) on MNIST dataset and renders model summary, test accuracy and test loss as output
# Please check the README file for helpful notes.
#https://github.com/shivekchhabra/Convolutional-Neural-Network
#https://victorzhou.com/blog/keras-cnn-tutorial/

# Function to visualise the data.
def data_visualisation(data):
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()

# Function to build the model architecture, train/evaluate the model, and save the model.
def modeling(train_images, train_labels, test_images, test_labels):
    #Architecture of the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    #Prints the summary of the model
    model.summary()

    #Compiles and train
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels,
              batch_size=10,
              epochs=1,
              verbose=1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    # serialize weights to HDF5
    model.save("Model_Files/model.h5")
    
    return test_acc, test_loss


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
    #Trains the model, evaluates its performance and saves the model to Model_Files folder
    train_images, train_labels, test_images, test_labels = data_set()
    train_images, train_labels, test_images, test_labels = data_preproc(train_images, train_labels, test_images, test_labels)
    test_acc, test_loss = modeling(train_images, train_labels, test_images, test_labels)
    print('Test Accuracy- ', test_acc)
    print('Test Loss= ', test_loss)

    #Loads the model and saves helpful images to the Model_Image folder
    model = tf.keras.models.load_model('Model_Files/model.h5')

    model_json = model.to_json()
    with open("Model_Files/model.json", "w") as json_file:
        json_file.write(model_json)

    layered_view(model, to_file='Model_Images/layerView.png', legend=True, scale_xy=15, scale_z=10, max_z=100)
    plot_model(model, to_file='Model_Images/ModelSummary.png', show_shapes=True, show_layer_names=True)
