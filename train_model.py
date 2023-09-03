"""
This script is used to train the model.
"""

import tensorflow as tf
import torch

from src.utils.image_utils import ImageUtils
from src.utils.text_utils import TextUtils


def main():

    # define the paths to the images and captions
    images_folder_path = "./data/test_data/test"
    captions_path = "./data/test_data/test.txt"

    # define the dimensions to which the images should be resized and the preprocess function to use
    image_dimensions = (299, 299)
    preprocess_function = tf.keras.applications.xception.preprocess_input

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

    # preprocess the images
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


# ====================================== MAIN ====================================== #
if __name__ == "__main__":
    main()
