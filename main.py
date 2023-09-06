"""
This script is used to run inference. It will load the model and the tokenizer and then generate captions in the folder
specified by the user. The user can specify the number of captions to generate for each image.
"""

import os

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.models.captioner import ImageCaptioningModel
from src.utils.data_utils import DataUtils

# ====================================== PARAMETERS ====================================== #

# path to the folder containing the images
images_folder_path = 'data/test/images'

# output folder name
# The captions will be saved in csv format in the folder specified by the user
output_folder_name = 'data'

# number of captions to generate for each image
num_captions_per_image = 5

# dimensions that the images will be resized to before feeding them to the model
# Make sure they match the dimensions used to train the model
# If you're using the pretrained model, leave as is
image_dimensions = (299, 299)

# preprocessing function to use
# Make sure it matches the preprocessing function used to train the model
preprocess_function = tf.keras.applications.xception.preprocess_input

# path to the folder containing the submodels
# Make sure that it contains the following files and folders:
# - tokenizer_wrapper.pkl: Wrapper object for the tokenizer and the max_length
# - encoder/: Folder containing the encoder model
# - decoder/: Folder containing the decoder model
# - attention/: Folder containing the attention model
models_path = 'saved_models'


if __name__ == '__main__':

    # load the model
    model = ImageCaptioningModel.load_model(models_path)

    # load the data
    data_dic = DataUtils.load_inference_data(
        image_dimensions=image_dimensions,
        images_folder_path=images_folder_path,
        preprocess_function=preprocess_function
    )

    # unpack the data
    images_dic = data_dic['images_dic']
    importance_features_dic = data_dic['importance_features_dic']

    # create the output dataframe
    output_df = pd.DataFrame(columns=['image', 'caption'])

    # generate the captions
    print('Generating captions...')
    for filename in tqdm(images_dic.keys()):

        # expand the dimensions of the image and importance features
        image = tf.expand_dims(images_dic[filename], axis=0)
        importance_features = tf.expand_dims(
            importance_features_dic[filename], axis=0)

        for _ in range(num_captions_per_image):

            # generate the captions
            caption = model((image, importance_features))

            caption = " ".join(caption[0])

            # add the caption to the dataframe
            output_df = output_df.append(
                {'image': filename, 'caption': caption}, ignore_index=True)

    # save the dataframe
    output_df.to_csv(os.path.join(output_folder_name, 'captions.csv'),
                     index=False)
    print('Captions saved in {}'.format(
        os.path.join(output_folder_name, 'captions.csv')))
