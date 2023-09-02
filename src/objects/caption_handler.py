"""
This module implements the CaptionHandler class, which contains various methods for reading captions from a given text file, as well
as methods to process, tokenize and pad the captions.
"""

import os
import pickle
import re
import string

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from ..utils.tokenizer_wrapper import TokenizerWrapper


class CaptionHandler:

    def __init__(self, filenames=None, filepath=None, tokenizer_path=None, max_vocab_size=15000):
        """
        Initializes the CaptionHandler object. If filepath is None, then the object is to be used for inference. In this case,
        the tokenizer is loaded from the given path and the object returns the maximum length of the captions for each filename.
        :param filepath: The path to the text file containing the captions.
        :param filenames: A list of filenames to read the captions for.
        :param max_vocab_size: The maximum vocabulary size for the tokenizer.
        """

        # run the handler in training mode
        if filepath is not None and filenames is not None: 
            print("Running caption handler in training mode...")

            # check if the file exists
            if not os.path.exists(filepath):
                raise ValueError("The file {} does not exist.".format(filepath))

            self.max_vocab_size = max_vocab_size
            self.filenames = filenames
            self.captions_dic = self.read_captions(filepath)

            # create the tokenizer and get the maximum length of the captions
            self.tokenizer, self.max_length = self.create_tokenizer()

            # vectorize the captions
            self.captions_dic = self.vectorize_captions()
        # run the handler in inference mode
        elif filepath is None and filenames is None and tokenizer_path is not None:
            print("Running caption handler in inference mode...")

            self.tokenizer, self.max_length = self.load_tokenizer(tokenizer_path)
        else:
            raise ValueError("Provide either a filepath and a list of filenames for training mode or only a tokenizer path for inference mode.")

    def read_captions_from_txt(self, filepath):
        """
        Reads the captions from the text file.
        :return: A dictionary where the keys are the filenames and the values are lists of captions.
        """

        with open(filepath, "r") as f:
            lines = f.readlines()

        # skip the header line and the get the number of lines
        lines = lines[1:]

        # create a dictionary to store the captions
        captions_dic = {k: [] for k in self.filenames}

        print("Reading captions...")
        for line in tqdm(lines):
            # split the line into filename and caption, use the first occurence of "," as the separator
            filename, caption = line.split(",", 1)

            # check if the filename is in the list of filenames
            if filename not in self.filenames:
                raise ValueError(
                    "Filename {} not found in the list of filenames.".format(filename))

            captions_dic[filename].append(self.process_text(caption))

        return captions_dic

    def read_captions_from_csv(self, filepath):
        """
        Reads the captions from the csv file.
        :return: A dictionary where the keys are the filenames and the values are lists of captions.
        """
        pass

    def read_captions(self, filepath):
        """
        Reads the captions from the filepath depending on the file extension.
        :return: A dictionary where the keys are the filenames and the values are lists of captions.
        """

        # check if the file is a text file or a csv file
        if filepath.endswith(".txt"):
            return self.read_captions_from_txt(filepath)
        elif filepath.endswith(".csv"):
            return self.read_captions_from_csv(filepath)
        else:
            raise ValueError(
                "The file {} is not a text file or a csv file.".format(self.filepath))

    def process_text(self, text):
        """
        Processes a given text by converting it to lowercase, removing punctuation and numbers, and adding start and end tokens.
        :param text: The text to process.
        :return: The processed text.
        """

        # convert to lowercase
        text = text.lower()

        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # remove numbers
        text = re.sub(r"\d+", "", text)

        # remove extra whitespace
        text = text.strip()

        # add start and end tokens
        text = "<start> " + text + " <end>"

        return text

    def create_tokenizer(self):
        """
        Creates a tokenizer from the captions.
        :return: The tokenizer and the maximum length of the captions.
        """

        # get the list of texts and the maximum length of the captions
        texts = []
        max_length = 0
        for captions in self.captions_dic.values():
            texts.extend(captions)
            max_length = max(max_length, max(
                [len(caption.split()) for caption in captions]))

        # fit the tokenizer on the texts
        tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<unk>")

        # fit the tokenizer on the texts
        tokenizer.fit_on_texts(texts)

        return tokenizer, max_length

    def vectorize_captions(self):
        """
        Vectorizes the captions.
        :return: A dictionary where the keys are the filenames and the values are lists of vectorized captions.
        """

        # vectorize the captions
        captions_dic = {k: [] for k in self.filenames}
        for filename, captions in self.captions_dic.items():
            captions_dic[filename] = self.tokenizer.texts_to_sequences(
                captions)

        # pad the sequences
        for filename, captions in captions_dic.items():
            captions_dic[filename] = np.array(pad_sequences(
                captions, maxlen=self.max_length, padding="post", truncating="post"))

        return captions_dic
    
    def save_tokenizer(self, filepath):
        """
        Saves the tokenizer, as well as the max_length in a wrapper to the given filepath.
        :param filepath: The filepath to save the tokenizer to.
        """
        
        # create a tokenizer wrapper
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer, self.max_length)

        # save the tokenizer
        print(f"Saving tokenizer to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(tokenizer_wrapper, f)
        
    def load_tokenizer(self, filepath):
        """
        Loads the tokenizer from the given filepath.
        :param filepath: The filepath to load the tokenizer from.
        """
        
        # load the tokenizer
        print("Loading tokenizer from {}...".format(filepath))
        with open(filepath, "rb") as f:
            tokenizer_wrapper = pickle.load(f)
        
        # get the tokenizer and the max_length
        return tokenizer_wrapper.tokenizer, tokenizer_wrapper.max_length