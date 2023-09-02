"""
This module implements the DataGenerator class, which is a subclass of the Sequence class.
It is used to generate batches of data for training the model.
"""

from tensorflow.keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    
    def __init__(self, data, batch_size):
        """
        Initializes the DataGenerator object.
        :param data: The data to generate batches from. The data should be in the form of an array of (image, importance_features, caption) pairs.
        :param batch_size: The batch size.
        """
        
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the number of batches per epoch.
        :return: The number of batches per epoch.
        """
        
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generates one batch of data.
        :param index: The index of the batch.
        :return: The batch of data.
        """

        batch = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = np.array([item[0] for item in batch])
        batch_features = np.array([item[1] for item in batch])
        batch_captions = np.array([item[2] for item in batch])
        
        return batch_images, batch_features, batch_captions
    
    def on_epoch_end(self):
        """
        Shuffles the data at the end of each epoch.
        """
        
        np.random.shuffle(self.data) 