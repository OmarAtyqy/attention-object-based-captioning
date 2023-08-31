"""
This module implements the DataGenerator class, which is a subclass of the Sequence class.
It is used to generate batches of data for training the model.
"""

from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    
    def __init__(self, data, batch_size):
        """
        Initializes the DataGenerator object.
        :param data: The data to generate batches from.
        :param batch_size: The batch size.
        """
        
        self.data = data
        self.batch_size = batch_size