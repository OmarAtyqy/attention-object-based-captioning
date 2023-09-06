"""DataGenerator
This module implements the DataGenerator class, which inherits from the keras.utils.Sequence class.
It is used to generate the data for the model in batches to avoid loading all the data into memory at once.
"""

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_dic, importance_features_dic, batch_size, captions_dic):
        """
        This function initializes the class.
        :param images_dic: a dictionary containing the images
        :param captions_dic: a dictionary containing the captions, If None, the generator will only return the images and their importance features for inference
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
        # the number of batches per epoch
        return len(self.images_dic) // self.batch_size

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

        # initialize the data for training
        images = []
        importance_features = []
        captions = []
        for filename in batch_filenames:
            # for each caption, duplicate the image and importance features
            for caption in self.captions_dic[filename]:
                # add the image, importance features and caption to the data
                images.append(self.images_dic[filename])
                importance_features.append(
                    self.importance_features_dic[filename])
                captions.append(caption)

        # convert the data to numpy arrays
        images = np.array(images)
        importance_features = np.array(importance_features)
        captions = np.array(captions)

        return (images, importance_features), captions

    def on_epoch_end(self):
        """
        This function is called at the end of each epoch to shuffle the indices.
        """

        # shuffle the indices
        np.random.shuffle(self.indices)
