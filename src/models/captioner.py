"""
This module implements the image captioning model. It is composed of an encoder, a decoder and an attention mechanism.
"""

import os

import tensorflow as tf

from src.layers.bahdanau import BahdanauAttention
from src.layers.decoder import Decoder
from src.layers.encoder import Encoder
from src.objects.tokenizer_wrapper import TokenizerWrapper


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
        self.decoder = Decoder(self.vocab_size, embedding_dim, units)
        self.attention = BahdanauAttention(units)

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

                # get the context vector
                context_vector = self.attention(
                    features, hidden)

                # get the predictions and the hidden state
                predictions, hidden = self.decoder(
                    dec_input, context_vector)

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

    def test_step(self, data):
        """
        This method performs a test step. It should only return the loss
        """

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

        # pass the images through the encoder
        features = self.encoder(images, importance_features)

        # iterate over the words of the captions
        for i in tf.range(1, captions.shape[1]):

            # get the context vector
            context_vector = self.attention(
                features, hidden)

            # get the predictions and the hidden state
            predictions, hidden = self.decoder(
                dec_input, context_vector)

            # calculate the loss
            loss += self.loss_function(captions[:, i], predictions)

            # use teacher forcing
            dec_input = tf.expand_dims(captions[:, i], 1)

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
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_mean(loss_)

        return loss_

    def predict_caption(self, image, importance_features):
        """
        Predict the caption for a single given image
        """

        # initialize the hidden state
        hidden = tf.zeros((1, self.units))

        # pass the image through the encoder
        features = self.encoder(image, importance_features)

        # initialize the decoder input
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index['<start>']], 0)

        # initialize the result
        result = []

        # iterate over the words of the captions
        for _ in tf.range(self.max_length):

            # get the context vector
            context_vector = self.attention(
                features, hidden)

            # get the predictions and the hidden state
            predictions, hidden = self.decoder(
                dec_input, context_vector)

            # get the predicted id
            predicted_id = tf.argmax(predictions[0]).numpy()

            # stop if the model predicts the end token
            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            # add the predicted word to the result
            result.append(self.tokenizer.index_word[predicted_id])

            # use teacher forcing
            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def call(self, X):
        """
        Call the model on a batch of inputs.
        """

        # unpack the data
        images = X[0]
        importance_features = X[1]

        # initialize the result
        result = []

        for i in tf.range(images.shape[0]):

            # predict the caption
            caption = self.predict_caption(
                tf.expand_dims(images[i], 0), tf.expand_dims(importance_features[i], 0))

            # add the caption to the result
            result.append(caption)

        return result

    def save(self, path):
        """
        Save each of the submodules of the model.
        """

        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # save the tokenizer and the max_length
        print('Saving the tokenizer and the max_length...')
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer, self.max_length)
        tokenizer_wrapper.save(path)
        print('Tokenizer and max_length saved!')

        # save the models
        print('Saving the models...')
        self.encoder.save(os.path.join(path, 'encoder'), save_format='tf')
        self.decoder.save(os.path.join(path, 'decoder'), save_format='tf')
        self.attention.save(os.path.join(path, 'attention'), save_format='tf')
        print('Models saved!')

    @staticmethod
    def load_model(path, image_dimensions, embedding_dim=256, units=512):
        """
        Load each of the submodules of the model and return the model.
        Make sure that the specified path contains the following files:
        - tokenizer_wrapper.pkl
        - encoder/
        - decoder/
        - attention/
        :param path: The path to the folder containing the model
        :param image_dimensions: The dimensions of the images that the model was trained on
        """

        # load the models
        print('Loading the models...')
        encoder = tf.keras.models.load_model(
            os.path.join(path, 'encoder'))
        decoder = tf.keras.models.load_model(
            os.path.join(path, 'decoder'))
        attention = tf.keras.models.load_model(
            os.path.join(path, 'attention'))
        print('Models loaded!')

        # load the tokenizer and the max_length
        print('Loading the tokenizer and the max_length...')
        tokenizer_wrapper = TokenizerWrapper.load(path)
        tokenizer = tokenizer_wrapper.tokenizer
        max_length = tokenizer_wrapper.max_length

        # create the model
        print('Constructing the captioner...')
        model = ImageCaptioningModel(
            image_dimensions, tokenizer, max_length, embedding_dim, units)

        # set the models
        model.encoder = encoder
        model.decoder = decoder
        model.attention = attention

        # compile and return the model
        model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)
        print('Captioner constructed!')

        return model
