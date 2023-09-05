"""
This module implements the Bahdanau Attention layer.
"""

import tensorflow as tf


class BahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        """
        Constructor method for the BahdanauAttention class.
        :param units: The number of units to use in the Dense layers
        """
        super(BahdanauAttention, self).__init__()

        # initialise the Dense layers
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """
        This method performs the forward pass through the layer.
        :param x: The input to the layer, which is a tuple of (features, hidden)
        :return: The context vector
        """

        # expand the dimensions of the hidden state to perform addition
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # calculate the score
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # calculate the attention weights
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # calculate the context vector
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector
