"""
This script is used to train the model.
"""

import tensorflow as tf

from src.scripts.load_data import load_data
from src.utils.data_utils import DataUtils
from src.objects.data_generator import DataGenerator


if __name__ == '__main__':

    # ====================================== PARAMETERS ====================================== #

    # path to the folder containing the images
    images_folder_path = 'data/test_data/test'

    # path to the file containing the captions
    captions_path = 'data/test_data/test.txt'

    # preprocess function to use.
    # We use Xception to extract the features from the images, so we need to preprocess them accordingly using the preprocess_input function from keras.applications.xception
    preprocess_function = tf.keras.applications.xception.preprocess_input

    # batch size
    batch_size = 1

    # image dimensions
    # The Xception model expects images of size 299x299, In order to change the input shape of the model inside the Encoder layer
    image_dimensions = (299, 299)

    # validation split (percentage of the data used for validation)
    val_split = 0.2

    # ====================================== DATA GENERATION ====================================== #

    # load the data and unpack it
    data_dic = load_data(images_folder_path, captions_path,
                         image_dimensions, preprocess_function)

    # unpack it
    images_dic = data_dic['images_dic']
    captions_dic = data_dic['captions_dic']
    importance_features_dic = data_dic['importance_features_dic']
    tokenizer = data_dic['tokenizer']
    max_caption_length = data_dic['max_caption_length']

    # split the data into training and validation sets
    split_dic = DataUtils.train_test_split(images_dic, importance_features_dic, captions_dic, val_split)

    # unpack the split data
    train_images_dic = split_dic['train_images_dic']
    train_importance_features_dic = split_dic['train_importance_features_dic']
    train_captions_dic = split_dic['train_captions_dic']
    val_images_dic = split_dic['val_images_dic']
    val_importance_features_dic = split_dic['val_importance_features_dic']
    val_captions_dic = split_dic['val_captions_dic']

    # create the training and validation data generators
    train_generator = DataGenerator(train_images_dic, train_captions_dic, train_importance_features_dic, batch_size)
    val_generator = DataGenerator(val_images_dic, val_captions_dic, val_importance_features_dic, batch_size)