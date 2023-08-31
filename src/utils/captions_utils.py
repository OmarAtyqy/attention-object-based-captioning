import json
import os
import pickle
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def load_raw_captions_data(path, delimiter=","):
    """
    Load raw (unprocessed) captions data from csv file and returns a pandas dataframe
    path: path to csv file
    delimiter: delimiter used in the csv file
    """
    df = pd.read_csv(path, delimiter=delimiter)

    # clean the column names
    df.columns = [col.strip() for col in df.columns]

    # make sure the dataframe has the right columns
    assert "image_name" in df.columns, "image_name column not found"
    assert "comment_number" in df.columns, "comment_number column not found"
    assert "comment" in df.columns, "comment column not found"

    return df


def extract_image_caption(image_name, df):
    """
    For a given image name, extract the corresponding captions from the dataframe. This function is meant to be run in parallel.
    image_name: name of the image
    df: pandas dataframe, should have columns "image_name", "comment_number" and "comment"
    """
    # extract the corresponding captions
    captions = df[df["image_name"] == image_name]["comment"].values

    # return the image name and the captions
    return image_name, captions


def generate_captions_dic(df):
    """
    Generate a dictionary that maps an image_name to its corresponding captions. Use multi-threading to speed up the process.
    df: pandas dataframe, should have columns "image_name", "comment_number" and "comment"
    """
    # create a dictionary that maps an image_name to its corresponding captions
    caption_dic = {}

    # extract the image names using multi-threading
    with Pool() as p:
        for image_name, captions in p.starmap(extract_image_caption, [(image_name, df) for image_name in df["image_name"].unique()]):
            caption_dic[image_name] = captions
    
    return caption_dic


def clean_captions(captions_dic):
    """
    Returns a cleaned version of the captions dictionary. The captions are converted to lower case and special characters are removed.
    """
    # remove special characters and convert to lower case
    for image_name, captions in tqdm(captions_dic.items()):
        captions_dic[image_name] = [re.sub(r"[^a-zA-Z0-9]+", " ", caption).lower().strip() for caption in captions]

    return captions_dic


def save_captions_dic(captions_dic, path):
    """
    Save the captions dictionary to a json file
    """
    filename = os.path.join(path, "processed_captions.json")
    with open(filename, "w") as f:
        json.dump(captions_dic, f)


def load_captions_dic(filepath):
    """
    Load the captions dictionary from a json file
    """
    print("Loading captions dictionary...")
    with open(filepath, "r") as f:
        captions_dic = json.load(f)
    
    return captions_dic


def create_tokenizer(captions_dic):
    """
    Create a tokenizer from the captions dictionary
    """
    # create the tokenizer
    tokenizer = Tokenizer()
    
    # get the captions
    print("Extracting captions...")
    captions = []
    for _, captions_list in tqdm(captions_dic.items()):
        captions.extend(captions_list)
    
    # fit the tokenizer on the captions
    print("Fitting tokenizer...")
    tokenizer.fit_on_texts(captions)
    print(f"Tokenizer fit with vocabulary size: {len(tokenizer.word_index) + 1}")

    return tokenizer


def save_tokenizer(tokenizer, filename):
    """
    Save the tokenizer
    """
    with open(filename, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(filename):
    """
    Load the tokenizer
    """
    print("Loading tokenizer...")
    with open(filename, "rb") as f:
        tokenizer = pickle.load(f)
    
    return tokenizer


def get_max_length(descriptions):
    """
    Return the max length of the captions of all the images
    """
    max_length = 0
    for _, captions_list in descriptions.items():
        for caption in captions_list:
            max_length = max(max_length, len(caption.split()))
    
    return max_length


def word_for_id(integer, tokenizer):
    """
    Returns the word corresponding to the given integer
    """
    for word, index in tokenizer.word_index.items():
        if index == integer: return word
    return None


def generate_desc(model, tokenizer, image, max_length):
    """
    Generate a description for an image
    """
    # get image info
    width = image.shape[0]
    height = image.shape[1]
    n_channels = image.shape[2]

    in_text = 'start'
    image = image.reshape((1, width, height, n_channels))
    for i in tqdm(range(max_length)):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([image, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break

    return in_text
