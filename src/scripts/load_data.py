"""
This script is used to load the data.
"""

import torch

from src.utils.image_utils import ImageUtils
from src.utils.text_utils import TextUtils


def load_data(images_folder_path, captions_path, image_dimensions=(299, 299), preprocess_function=None):
    """
    This function loads the data from the images and captions folders and returns a dictionary with the data.
    :param images_folder_path: the path to the folder containing the images
    :param captions_path: the path to the file containing the captions
    :param image_dimensions: the dimensions of the images
    :param preprocess_function: the function used to preprocess the images
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
        images_dic = ImageUtils.preprocess_images(images_dic, preprocess_function)

    # ====================================== CAPTION PREPROCESSING ====================================== #

    # read the captions
    captions_dic = TextUtils.read_captions(captions_path)

    # create the tokenizer
    tokenizer = TextUtils.get_tokenizer(captions_dic)

    # get the maximum length of the captions
    # Call this function before tokenizing the captions, because the captions are modified in the process
    max_caption_length = TextUtils.get_max_length(captions_dic)

    # tokenize the captions
    captions_dic = TextUtils.tokenize_captions(captions_dic, tokenizer, max_caption_length)

    # ====================================== DATA GENERATION ====================================== #

    # check if the filenames in all three dictionaries are the same
    if not set(images_dic.keys()) == set(captions_dic.keys()) == set(importance_features_dic.keys()):
        raise ValueError(
            "The filenames in the images, captions and importance features dictionaries are not the same.")
    
    # return dictionary with all the data
    return {
        "images_dic": images_dic,
        "captions_dic": captions_dic,
        "importance_features_dic": importance_features_dic,
        "tokenizer": tokenizer,
        "max_caption_length": max_caption_length
    }