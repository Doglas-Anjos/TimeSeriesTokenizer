"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge
from utils.discretisize import decode_with_float_vocab, encode_with_float_vocab


class BasicTokenizer(Tokenizer):

    def __init__(self, actual_vocab_size, encoding_name="exemple.fvocab"):
        self.encoding_name = encoding_name
        self.actual_vocab_size = actual_vocab_size
        super().__init__(actual_vocab_size)

    def train(self, list_values, target_vocab_size, verbose=False):
        assert target_vocab_size >= self.actual_vocab_size
        num_merges = target_vocab_size - self.actual_vocab_size

        # input text preprocessing
        text_bytes = encode_with_float_vocab(list_values, self.encoding_name) # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: idx for idx in range(self.actual_vocab_size)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = self.actual_vocab_size + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        list_with_tokens = list()
        for id_ in ids:
            if id_ > self.actual_vocab_size:
                list_with_tokens.extend(self.search_recursively_tokens(id_))
            else:
                list_with_tokens.append(id_)
        text, n_edges = decode_with_float_vocab(list_with_tokens, self.encoding_name)
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = encode_with_float_vocab(text, self.encoding_name) # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
