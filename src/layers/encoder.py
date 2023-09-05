"""
This module implements the Encoder class, which is used to extract features from images using the Xception model.
For dimensios (299, 299):
The output of the xcpetion model, which is a (batch, 10, 10, 2048) array, is reshaped into a (batch, 100, 2048) array.
We also reshape the importance features, which are (batch, 2048) arrays, into (batch, 1, 2048) arrays.
Both the reshaped xception features and the reshaped importance features are concatenated to form a (batch, 101, 2048) array.
This array is then fed to a Dense layer, which converts it into a (batch, 101, 256) array with a relu activation function.
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception  # type: ignore
from tensorflow.keras.layers import Concatenate, Dense  # type: ignore


class Encoder(tf.keras.Model):

    def __init__(self, img_dims, units):
        """
        Constructor method for the Encoder class.
        :param img_dims: The dimensions of the images
        :param units: The number of units to use in the Dense layer
        """
        super(Encoder, self).__init__()

        # encoder layers
        self.xception = Xception(
            include_top=False, weights='imagenet', input_shape=(*img_dims, 3))
        self.xception.trainable = False
        self.concat = Concatenate(axis=1)
        self.dense = Dense(units=units, activation='relu')

    def call(self, image, importance_features):
        """
        This method performs the forward pass through the model.
        :param x: image input
        :return: The output of the model, which is a (batch_size, max_length, vocab_size) array
        """

        # extract features from the image
        x = self.xception(image)

        # reshape the xception features into a (batch, 100, 2048) array
        shape = tf.shape(x)
        x = tf.reshape(x, shape=(shape[0], shape[1] * shape[2], shape[3]))

        # reshape the importance features into a (batch, 1, 2048) array
        shape = tf.shape(importance_features)
        importance_features = tf.reshape(
            importance_features, shape=(shape[0], 1, shape[1]))

        # concatenate the xception features and the importance features to form a (batch, 101, 2048) array
        x = self.concat([importance_features, x])

        # pass the concatenated features through the dense layer to convert them into a (batch, 101, 256) array
        x = self.dense(x)

        return x
