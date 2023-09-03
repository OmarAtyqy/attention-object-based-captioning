"""
This module implements the Encoder class, which is used to extract features from images using the Xception model.
The output of the xcpetion model, which is a (batch, 10, 10, 2048) array, is reshaped into a (batch, 100, 2048) array.
We also reshape the importance features, which are (batch, 2048) arrays, into (batch, 1, 2048) arrays.
Both the reshaped xception features and the reshaped importance features are concatenated to form a (batch, 101, 2048) array.
This array is then fed to a Dense layer, which converts it into a (batch, 101, 256) array with a relu activation function.
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Concatenate, Dense


class Encoder(tf.keras.Model):

    def __init__(self, units):
        """
        Constructor method for the Encoder class.
        :param units: The number of units to use in the Dense layer
        """
        super(Encoder, self).__init__()

        # encoder layers
        self.concat = Concatenate(axis=1)
        self.dense1 = Dense(units, activation="relu")
        self.xception = Xception(include_top=False, weights='imagenet')

    def call(self, image, importance_features):
        """
        This method performs the forward pass through the model.
        :param x: The input to the model, which is a tuple of (image, importance_features, caption)
        :return: The output of the model, which is a (batch_size, max_length, vocab_size) array
        """

        # get the xcpetion features from the image and reshape them into a (batch_size, 100, 2048) array
        features = self.xception(image)
        features = tf.reshape(
            features, (features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))

        # reshape the importance features
        importance_features = tf.reshape(
            importance_features, (importance_features.shape[0], 1, importance_features.shape[1]))

        # concatenate the xception features and the importance features into a (batch_size, 101, 2048) array
        x = self.concat([features, importance_features])

        # convert the concatenated features into a (batch_size, 101, 256) array
        x = self.dense1(x)

        return x
