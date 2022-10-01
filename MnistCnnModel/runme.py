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
import os

## Overview
# This code runs a CNN model (deep learning) on MNIST dataset and renders model summary, test accuracy and test loss as output
# Please check the README file for helpful notes.
#https://github.com/shivekchhabra/Convolutional-Neural-Network
#https://victorzhou.com/blog/keras-cnn-tutorial/

# Function to visualise the data.
def data_visualisation(data):
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()


def modeling(train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels,
              batch_size=10,
              epochs=1,
              verbose=1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    # serialize weights to HDF5
    model.save("model.h5")
    
    return test_acc, test_loss


def data_set():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels


def data_preproc(train_images, train_labels, test_images, test_labels):
    train_images = train_images.reshape(
        (60000, 28, 28, 1))  # Converting every image to 1d; train_images has a shape of 60000x28x28
    train_images = train_images.astype('float32') / 255  # scaling between 0,1
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)  # Converts list labels to numpy array (one-hot encoding)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_images, train_labels, test_images, test_labels = data_set()
    train_images, train_labels, test_images, test_labels = data_preproc(train_images, train_labels, test_images, test_labels)
    #test_acc, test_loss = modeling(train_images, train_labels, test_images, test_labels)
    #print('Test Accuracy- ', test_acc)
    #print('Test Loss= ', test_loss)

    X_test, y_test = convert_from_file('./samples/t10k-images.idx3-ubyte'), convert_from_file('./samples/t10k-labels.idx1-ubyte')

    X_test = X_test.reshape((10000, 28, 28, 1))
    X_test = X_test.astype('float32') / 255
    y_test = to_categorical(y_test)

    model = tf.keras.models.load_model('model.h5')
    score = model.evaluate(X_test, y_test)

    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    #layered_view(model, to_file='layerView.png', legend=True, scale_xy=15, scale_z=10, max_z=100)
