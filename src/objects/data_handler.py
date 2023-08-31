"""
This module implements the DataHandler class, which contains various methods for generating the final dataset, as well as doing train/validation splits.
It returns a generator object that can be used to train the model.
"""

import numpy as np
from tqdm import tqdm

from .caption_handler import CaptionHandler
from .data_generator import DataGenerator
from .image_handler import ImageHandler


class DataHandler:

    def __init__(self, image_handler: ImageHandler, caption_handler: CaptionHandler, val_split=0.2):
        """
        Initializes the DataHandler object.
        :param image_handler: An ImageHandler object.
        :param caption_handler: A CaptionHandler object.
        """

        self.image_handler = image_handler
        self.caption_handler = caption_handler
        
        # check if the filenames in the image handler and the caption handler are the same
        if self.image_handler.filenames != self.caption_handler.filenames:
            raise ValueError("The filenames in the image handler and the caption handler do not match.")
        self.filenames = self.image_handler.filenames

        if val_split < 0 or val_split >= 1:
            raise ValueError("The validation split must be between [0, 1[.")

        # get the training and validation datasets
        train_filenames, val_filenames = self.train_val_split_dics(val_split)
        self.train_dataset = self.create_dataset(train_filenames)
        self.val_dataset = self.create_dataset(val_filenames)
    
    def create_dataset(self, filenames, name=None):
        """
        Create a dataset from a given list of filenames.
        :param filenames: The filenames to create the dataset from.
        :return: an array of (image, caption) pairs.
        """

        if filenames is None:
            return None
        
        if name is not None:
            print("Creating {} dataset...".format(name))

        data = []
        for filename in tqdm(filenames):
            for caption in self.caption_handler.captions_dic[filename]:
                data.append([self.image_handler.images_dic[filename], caption])

        return data
    
    def train_val_split_dics(self, val_split=0.2):
        """
        Splits the filenmaes into training and validation sets.
        :param val_split: The fraction of the data to use for validation.
        :return: The training and validation filenames.
        """

        # get the dics from the image and caption handlers
        images_dic = self.image_handler.images_dic
        captions_dic = self.caption_handler.captions_dic

        # shuffle the filenames
        np.random.shuffle(self.filenames)

        # calculate the number of images to use for validation and training
        num_val_images = int(val_split * len(self.filenames))
        num_train_images = len(self.filenames) - num_val_images
        print(f"Preparing {num_train_images} training images and {num_val_images} validation images...")

        # split the filenames into training and validation sets
        # This should be done before combining the datasets, otherwise the validation set will contain captions for images in the training set.
        if num_val_images > 0:
            train_filenames = self.filenames[:num_train_images]
            val_filenames = self.filenames[num_train_images:]
        else:
            train_filenames = self.filenames
            val_filenames = None

        return train_filenames, val_filenames
    
    def get_generators(self, batch_size=64):
        """
        Creates the generators for the training and validation datasets.
        :return: The training and validation generators.
        """

        train_generator = DataGenerator(self.train_dataset, batch_size)

        if self.val_dataset is None:
            val_generator = None
        else:
            val_generator = DataGenerator(self.val_dataset, batch_size)

        return train_generator, val_generator
