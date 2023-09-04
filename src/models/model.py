"""
This module implements the image captioning model. It is composed of an encoder, a decoder and an attention mechanism.
"""

import tensorflow as tf

from src.layers.decoder import Decoder
from src.layers.encoder import Encoder


class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, image_dimensions, tokenizer, max_length, embedding_dim=256, units=512):
        """
        Intializer method for the ImageCaptioningModel class.
        :param image_dimensions: The dimensions of the images
        :param vocab_size: The size of the vocabulary of the captions
        :param max_length: The maximum length of the captions
        :param embedding_dim: The dimension of the Dense layer in the encoder and the Embedding layer in the decoder
        :param units: The number of units to use in the LTS, Bahdanau attention and Dense layers in the decoder
        """
        super(ImageCaptioningModel, self).__init__()

        # create the loss object
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # set the parameters
        self.image_dimensions = image_dimensions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = len(tokenizer.word_index) + 1

        # create the layers
        self.encoder = Encoder(image_dimensions, embedding_dim)
        self.decoder = Decoder(embedding_dim, units, self.vocab_size)

    @tf.function
    def train_step(self, data):

        # unpack the data
        images = data[0][0]
        importance_features = data[0][1]
        captions = data[1]

        # initialize the loss
        loss = 0.0

        # intialize the hidden state
        hidden = self.decoder.reset_state(batch_size=captions.shape[0])

        # initialize the decoder input
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index['<start>']] * captions.shape[0], 1)

        # use GradientTape to record the gradients
        with tf.GradientTape() as tape:

            # pass the images through the encoder
            features = self.encoder(images, importance_features)

            # iterate over the words of the captions
            for i in tf.range(1, captions.shape[1]):

                # get the predictions and the hidden state
                predictions, hidden, _ = self.decoder(
                    dec_input, features, hidden)

                # calculate the loss
                loss += self.loss_function(captions[:, i], predictions)

                # use teacher forcing
                dec_input = tf.expand_dims(captions[:, i], 1)

        # calculate the gradients
        trainable_variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        # apply the gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # calculate the batch loss
        batch_loss = (loss / int(captions.shape[1]))

        return {
            'loss': batch_loss
        }

    def loss_function(self, real, pred):
        """
        This method calculates the loss.
        """

        # create the mask
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        # calculate the loss
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(
            real, pred, from_logits=True)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_mean(loss_)

        return loss_
