from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense, add
from tensorflow.keras.models import Model
from keras.utils import plot_model


class Decoder:

    def __init__(self, input_shape, vocab_size, max_length):
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.model = self.build()

    def build(self):
        """
        Build the LSTM model.
        """
        # Compress the 2048 input features into a 256 nodes feature vector
        input_layer_1 = Input(shape=self.input_shape)
        fe1 = Dropout(0.5)(input_layer_1)
        fe2 = Dense(1024, activation='relu')(fe1)
        fe3 = Dropout(0.3)(fe2)
        fe4 = Dense(512, activation='relu')(fe3)
        fe5 = Dropout(0.1)(fe4)
        fe6 = Dense(256, activation='relu')(fe5)

        # Define the input layer for the LSTM layer
        input_layer_2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(input_layer_2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        # merge the two input models
        decoder1 = add([fe6, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[input_layer_1, input_layer_2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model

    def summary(self):
        """
        Print the model summary.
        """
        # summarize model
        print(self.model.summary())
        plot_model(self.model, to_file='model.png', show_shapes=True)
    
    def fit(self, train_generator, val_generator, epochs, verbose=1):
        """
        Train the model.
        """
        train_steps = len(train_generator)  # Number of batches in the training data
        val_steps = len(val_generator)  # Number of batches in the validation data
        
        self.model.fit(train_generator, epochs=epochs, verbose=verbose, 
                                 validation_data=val_generator,
                                 steps_per_epoch=train_steps,
                                 validation_steps=val_steps)
    
    def save(self, filename):
        """
        Save the model.
        """
        self.model.save(filename)