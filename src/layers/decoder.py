"""
This module implements the Decoder class.
"""

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding

from .bahdanau import BahdanauAttention


class Decoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim, units):
        """
        Constructor method for the Decoder class.
        :param units: The number of units to use in the LSTM layer
        :param vocab_size: The size of the vocabulary
        :param max_length: The maximum length of the caption
        """
        super(Decoder, self).__init__()
        self.units = units

        # initialise the Embedding layer
        self.embedding = Embedding(vocab_size, embedding_dim)

        # initialise the LSTM layer
        self.lstm = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

        # dense layers
        self.dense1 = Dense(units, activation="relu")
        self.dense2 = Dense(vocab_size)

        # initialise the BahdanauAttention layer
        self.attention = BahdanauAttention(units)

    def call(self, dec_input, features, hidden):
        """
        This method performs the forward pass through the model.
        :param dec_input: decoder input
        :param features: the encoder output
        :param hidden: hidden state    
        """

        # get the attention weights and context vector
        context_vector, attention_weights = self.attention(features, hidden)

        # embed the decoder input
        x = self.embedding(dec_input)

        # concatenate the context vector and the decoder input
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # pass the concatenated vector to the LSTM layer
        output, state, _ = self.lstm(x)

        # pass the LSTM output through the dense layers
        x = self.dense1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.dense2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        """
        This method resets the LSTM state.
        :param batch_size: The batch size
        """
        return tf.zeros((batch_size, self.units))
