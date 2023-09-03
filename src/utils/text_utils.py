"""
This module implements the TextUtils class, which is used to load and preprocess captions.
"""

import os
import pickle
import re
import string

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


class TextUtils:

    @staticmethod
    def read_captions_from_txt(filepath):
        """
        Reads the captions from the text file. The file should be structured as follows: image,caption (ignoring the header).
        :param filepath: The path to the text file.
        :return: A dictionary where the keys are the filenames and the values are lists of preprocessed captions.
        """

        # check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        
        # read the file
        print(f"Reading captions from {filepath}...")
        with open(filepath, "r") as f:
            lines = f.readlines()

        # remove the header
        lines = lines[1:]

        # initialise the dictionary
        captions = {}

        # iterate over the lines
        for line in lines:
                
                # get the filename and the caption by splitting the line at the first comma
                filename, caption = line.split(",", 1)
    
                # process the caption
                caption = TextUtils.process_text(caption)
    
                # check if the filename is already in the dictionary, if so, append the caption, otherwise create a new list
                if filename in captions.keys():
                    captions[filename].append(caption)
                else:
                    captions[filename] = [caption]

        return captions
    
    @staticmethod
    def read_captions_from_csv(filepath):
        """
        Reads the captions from the csv file. The file should be structured as follows: image,caption (ignoring the header).
        :param filepath: The path to the csv file.
        :return: A dictionary where the keys are the filenames and the values are lists of preprocessed captions.
        """
        pass

    @staticmethod
    def read_captions(filepath):
        """
        Reads the captions from the filepath depending on the file extension.
        :param filepath: The path to the file.
        :return: A dictionary where the keys are the filenames and the values are lists of preprocessed captions.
        """

        # check if the file is a text file or a csv file
        if filepath.endswith(".txt"):
            return TextUtils.read_captions_from_txt(filepath)
        elif filepath.endswith(".csv"):
            return TextUtils.read_captions_from_csv(filepath)
        else:
            raise ValueError(
                "The file {} is not a text file or a csv file.".format(TextUtils.filepath))
    
    @staticmethod
    def process_text(text):
        """
        Processes a given text by converting it to lowercase, removing punctuation and numbers, and adding start and end tokens.
        :param text: The text to process.
        :return: The processed text.
        """

        # convert to lowercase
        text = text.lower()

        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # remove tabs and line breaks
        text = text.replace('\t', ' ').replace('\n', ' ')

        # remove numbers
        text = re.sub(r"\d+", "", text)

        # remove extra whitespace
        text = text.strip()

        # add start and end tokens
        text = "<start> " + text + " <end>"

        return text
    
    @staticmethod
    def get_tokenizer(captions_dic):
        """
        Returns a tokenizer fitted on the captions.
        :param captions_dic: A dictionary where the keys are the filenames and the values are lists of preprocessed captions.
        :return: A tokenizer fitted on the captions.
        """

        # create a list of all the captions
        captions = []
        for caption_list in captions_dic.values():
            captions.extend(caption_list)

        # create the tokenizer, don't use any filters as they remove the start and end tokens
        # The text should be preprocessed before calling this function.
        tokenizer = Tokenizer(oov_token="<unk>", filters="")

        # fit the tokenizer on the captions
        tokenizer.fit_on_texts(captions)

        return tokenizer
    
    @staticmethod
    def get_max_length(captions_dic):
        """
        Returns the maximum length of the captions.
        :param captions_dic: A dictionary where the keys are the filenames and the values are lists of preprocessed captions.
        :return: The maximum length of the captions.
        """

        # get the maximum length of the captions
        max_length = 0
        for caption_list in captions_dic.values():
            for caption in caption_list:
                max_length = max(max_length, len(caption.split(" ")))
        
        return max_length

    @staticmethod
    def tokenize_captions(captions_dic, tokenizer):
        """
        Tokenizes the captions.
        :param captions_dic: A dictionary where the keys are the filenames and the values are lists of preprocessed captions.
        :return: A dictionary where the keys are the filenames and the values are lists of tokenized captions.
        """

        # tokenize the captions
        tokenized_captions_dic = {}
        for filename, caption_list in tqdm(captions_dic.items()):
            tokenized_captions_dic[filename] = tokenizer.texts_to_sequences(caption_list)

        return tokenized_captions_dic