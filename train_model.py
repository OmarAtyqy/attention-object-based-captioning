"""
This script is used to train the model.
"""

import tensorflow as tf

from src.models.captioner import ImageCaptioningModel
from src.objects.data_generator import DataGenerator
from src.utils.data_utils import DataUtils

# ====================================== PARAMETERS ====================================== #

# path to the folder containing the images
images_folder_path = 'data/images'

# path to the file containing the captions (can either be a csv or a txt file)
# It should be structured as follows: image,caption (include the header)
captions_path = 'data/captions.csv'

# preprocess function to use.
preprocess_function = tf.keras.applications.xception.preprocess_input

# validation split (percentage of the data used for validation)
val_split = 0.2

# batch size
# Make sure that your batch size is < the number of samples in both your training and validation datasets for the generators to work properly
batch_size = 20

# epochs
epochs = 30

# image dimensions
# The Xception model works best with 299x299 images, but you can try other sizes as well if you're having memory issues.
# The dimensios should not be below 71
image_dimensions = (299, 299)

# embedding dimension (dimension of the Dense layer in the encoder and the Embedding layer in the decoder)
embedding_dim = 128

# number of units in the LSTM, Bahdanau attention and Dense layers
units = 256


if __name__ == '__main__':

    # ====================================== DATA GENERATION ====================================== #

    # load the data and unpack it
    data_dic = DataUtils.load_training_data(images_folder_path, captions_path,
                                            image_dimensions, preprocess_function)

    # unpack it
    images_dic = data_dic['images_dic']
    captions_dic = data_dic['captions_dic']
    importance_features_dic = data_dic['importance_features_dic']
    tokenizer = data_dic['tokenizer']
    max_caption_length = data_dic['max_caption_length']

    # split the data into training and validation sets if val_split > 0
    if val_split > 0:
        split_dic = DataUtils.train_test_split(
            images_dic, importance_features_dic, captions_dic, val_split)

        # unpack the split data
        train_images_dic = split_dic['train_images_dic']
        train_importance_features_dic = split_dic['train_importance_features_dic']
        train_captions_dic = split_dic['train_captions_dic']
        val_images_dic = split_dic['val_images_dic']
        val_importance_features_dic = split_dic['val_importance_features_dic']
        val_captions_dic = split_dic['val_captions_dic']

        # create the training and validation data generators
        train_generator = DataGenerator(
            images_dic=train_images_dic,
            captions_dic=train_captions_dic,
            importance_features_dic=train_importance_features_dic,
            batch_size=batch_size
        )
        val_generator = DataGenerator(
            images_dic=val_images_dic,
            captions_dic=val_captions_dic,
            importance_features_dic=val_importance_features_dic,
            batch_size=batch_size
        )
    else:
        # create the training data generator
        train_generator = DataGenerator(
            images_dic=images_dic,
            captions_dic=captions_dic,
            importance_features_dic=importance_features_dic,
            batch_size=batch_size
        )

    # free up memory
    del images_dic
    del captions_dic
    del importance_features_dic

    # ====================================== MODEL ====================================== #

    # create the model
    model = ImageCaptioningModel(
        tokenizer=tokenizer,
        max_length=max_caption_length,
        units=units,
        embedding_dim=embedding_dim,
        image_dimensions=image_dimensions
    )

    # create the callback
    if val_split > 0:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', verbose=1, mode='min', patience=5)
    else:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', verbose=1, mode='min', patience=5)

    # compile the model
    # Leave run_eagerly=True because it doesn't work otherwise AND I DON'T KNOW WHY :)
    model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)

    # ====================================== TRAINING ====================================== #

    # train the model
    if val_split > 0:
        model.fit(
            x=train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[callback]
        )

    else:
        model.fit(
            x=train_generator,
            epochs=epochs,
            callbacks=[callback]
        )

    # ====================================== SAVING ====================================== #

    # save the model
    model.save('saved_models')
