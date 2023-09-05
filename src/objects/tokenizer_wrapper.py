"""
This module implements the tokenizer wrapper class. It is used to save and load both the tokenizer and the max_length.
"""

import os
import pickle


class TokenizerWrapper:

    def __init__(self, tokenizer, max_length):
        """
        Initialize the tokenizer and the max_length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def save(self, path):
        """
        Save the tokenizer and the max_length.
        """

        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # save the wrapper
        with open(os.path.join(path, 'tokenizer_wrapper.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load the tokenizer and the max_length.
        """

        # load the wrapper
        with open(os.path.join(path, 'tokenizer_wrapper.pkl'), 'rb') as f:
            wrapper = pickle.load(f)

        # return the wrapper
        return wrapper
