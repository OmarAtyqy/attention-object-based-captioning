"""
This module contains a wrapper for the tokenizer, as well as the max length of the captions used for padding.
"""

class TokenizerWrapper:

    def __init__(self, tokenizer, max_length):
        """
        Initializes the TokenizerWrapper object.
        :param tokenizer: The tokenizer object.
        :param max_length: The maximum length of the captions.
        """

        self.tokenizer = tokenizer
        self.max_length = max_length