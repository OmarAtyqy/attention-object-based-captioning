"""
This module implements the DataUtils class, which i used to load and preprocess the data.
"""

import torch
from sklearn.model_selection import train_test_split

from src.utils.image_utils import ImageUtils
from src.utils.text_utils import TextUtils


class DataUtils:

    @staticmethod
    def train_test_split(images_dic, importance_features_dic, captions_dic, val_split=0.2):
        """
        This function splits the data into training and validation sets.
        :param images_dic: a dictionary containing the images and their filenames as keys
        :param importance_features_dic: a dictionary containing the importance features and their filenames as keys
        :param captions_dic: a dictionary containing the captions and their filenames as keys
        :param val_split: the validation split
        :return: a dictionnary containing the training and validation sets for the images, importance features and captions
        """

        # get the filenames
        filenames = list(images_dic.keys())

        # split the filenames into training and validation sets
        train_filenames, val_filenames = train_test_split(
            filenames, test_size=val_split)

        # create the training and validation dictionaries
        train_images_dic = {filename: images_dic[filename]
                            for filename in train_filenames}
        train_importance_features_dic = {filename: importance_features_dic[filename]
                                         for filename in train_filenames}
        train_captions_dic = {filename: captions_dic[filename]
                              for filename in train_filenames}
        val_images_dic = {filename: images_dic[filename]
                          for filename in val_filenames}
        val_importance_features_dic = {filename: importance_features_dic[filename]
                                       for filename in val_filenames}
        val_captions_dic = {filename: captions_dic[filename]
                            for filename in val_filenames}

        # return the training and validation dictionaries
        return {
            'train_images_dic': train_images_dic,
            'train_importance_features_dic': train_importance_features_dic,
            'train_captions_dic': train_captions_dic,
            'val_images_dic': val_images_dic,
            'val_importance_features_dic': val_importance_features_dic,
            'val_captions_dic': val_captions_dic
        }

    @staticmethod
    def load_data(images_folder_path, captions_path, image_dimensions=(299, 299), preprocess_function=None):
        """
        This function loads the data from the images and captions folders and returns a dictionary with the data.
        :param images_folder_path: the path to the folder containing the images
        :param captions_path: the path to the file containing the captions
        :param image_dimensions: the dimensions of the images
        :param preprocess_function: the function used to preprocess the images
        :param batch_size: the batch size
        :return: a dictionary containing the images, captions, importance features dictionaries, as well as the tokenizer and max caption length
        """

        # create the object detection model used to extract the importance features (here we use YOLOv5)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # ====================================== IMAGE PREPROCESSING ====================================== #

        # read the images
        images_dic = ImageUtils.read_images(
            folder_path=images_folder_path,
            dimensions=image_dimensions
        )

        # extract the importance features
        # Call this function before preprocessing the images, because the images are modified in the process
        importance_features_dic = ImageUtils.get_importance_features_dic(
            images_dic, model)

        # preprocess the images if a preprocess function is provided
        if preprocess_function is not None:
            images_dic = ImageUtils.preprocess_images(
                images_dic, preprocess_function)

        # ====================================== CAPTION PREPROCESSING ====================================== #

        # read the captions
        captions_dic = TextUtils.read_captions(captions_path)

        # create the tokenizer
        tokenizer = TextUtils.get_tokenizer(captions_dic)

        # get the maximum length of the captions
        # Call this function before tokenizing the captions, because the captions are modified in the process
        max_caption_length = TextUtils.get_max_length(captions_dic)

        # tokenize the captions
        captions_dic = TextUtils.tokenize_captions(
            captions_dic, tokenizer, max_caption_length)

        # ====================================== DATA GENERATION ====================================== #

        # check if the filenames in all three dictionaries are the same
        if not set(images_dic.keys()) == set(captions_dic.keys()) == set(importance_features_dic.keys()):
            raise ValueError(
                "The filenames in the images, captions and importance features dictionaries are not the same.")

        # return the data
        return {
            'images_dic': images_dic,
            'captions_dic': captions_dic,
            'importance_features_dic': importance_features_dic,
            'tokenizer': tokenizer,
            'max_caption_length': max_caption_length
        }
