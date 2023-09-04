"""
This module implements the DataGenerator class, which inherits from the keras.utils.Sequence class.
It is used to generate the data for the model in batches to avoid loading all the data into memory at once.
"""

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_dic, captions_dic, importance_features_dic, batch_size):
        """
        This function initializes the class.
        :param images_dic: a dictionary containing the images
        :param captions_dic: a dictionary containing the captions
        :param importance_features_dic: a dictionary containing the importance features
        :param batch_size: the batch size
        """

        self.images_dic = images_dic
        self.captions_dic = captions_dic
        self.importance_features_dic = importance_features_dic
        self.batch_size = batch_size
        self.indices = np.arange(len(self.images_dic))
        self.filenames = list(self.images_dic.keys())

    def __len__(self):
        """
        This function returns the number of batches per epoch.
        :return: the number of batches per epoch
        """

        return int(np.floor(len(self.images_dic) / self.batch_size))

    def __getitem__(self, index):
        """
        This function returns the data for the given batch index.
        :param index: the batch index
        :return: the data for the given batch index in the form of X = (images, importance features, captions) and y = captions
        """

        # get the indices of the current batch
        batch_indices = self.indices[index *
                                     self.batch_size:(index+1)*self.batch_size]

        # get the filenames of the current batch
        batch_filenames = [self.filenames[i] for i in batch_indices]

        # get the images of the current batch
        batch_images = np.array([self.images_dic[filename]
                                for filename in batch_filenames])

        # get the captions of the current batch
        batch_captions = np.array([self.captions_dic[filename]
                                  for filename in batch_filenames])

        # get the importance features of the current batch
        batch_importance_features = np.array(
            [self.importance_features_dic[filename] for filename in batch_filenames])

        # return the data for the current batch
        return [batch_images, batch_importance_features], batch_captions

    def on_epoch_end(self):
        """
        This function is called at the end of each epoch to shuffle the indices.
        """

        # shuffle the indices
        np.random.shuffle(self.indices)
