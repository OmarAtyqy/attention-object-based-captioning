"""
This module implements the DataUtils class, which is mainly used to seperate the data into training and validation sets.
"""

from sklearn.model_selection import train_test_split


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
        