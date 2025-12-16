"""
Token-based Byte Pair Encoding tokenizer.

This tokenizer works directly with pre-discretized tokens (integers) instead of floats.
It's designed to handle sequences where special tokens (PAD, EBOS) are already inserted,
which affects the token pair statistics during BPE training.

Key differences from BasicTokenizer:
- train() accepts a list of integer tokens (not floats)
- No need to call encode_with_float_vocab during training
- Special tokens are preserved during merge operations
- decode() returns tokens (not float values)
"""

from .base import Tokenizer, get_stats, merge
from utils.discretisize import decode_with_float_vocab
import numpy as np


class TokenBasedTokenizer(Tokenizer):
    """
    Tokenizer that trains directly on pre-discretized token sequences.
    
    Use this when you have already discretized your data with simple_discretize()
    or adaptative_bins_discretize() and want to train BPE on those tokens including
    the special tokens (PAD, EBOS) that were inserted.
    """

    def __init__(self, actual_vocab_size, encoding_name="exemple.fvocab", special_tokens=None):
        """
        Initialize TokenBasedTokenizer.
        
        Parameters:
        -----------
        actual_vocab_size : int
            Size of the initial vocabulary (before BPE merges)
        encoding_name : str
            Name of the .fvocab file for float decoding (used in decode_to_floats)
        special_tokens : dict
            Dictionary mapping special token strings to their IDs
            e.g., {'<PAD>': 199, '<EBOS>': 200}
        """
        self.encoding_name = encoding_name
        self.actual_vocab_size = actual_vocab_size
        self.special_tokens = special_tokens if special_tokens else {}
        super().__init__(actual_vocab_size, special_tokens)

    def train(self, token_sequence, target_vocab_size, verbose=False):
        """
        Train BPE on a sequence of pre-discretized tokens.
        
        Parameters:
        -----------
        token_sequence : list or np.ndarray
            Sequence of integer tokens (already discretized, may include special tokens)
        target_vocab_size : int
            Final vocabulary size after BPE merges
        verbose : bool
            Print merge progress
            
        Example:
        --------
        >>> # Get discretized tokens with special tokens
        >>> tokens, edges = simple_discretize(data, N=200, data_st=data_st, special_tokens=special_tokens)
        >>> # Train directly on these tokens
        >>> tokenizer = TokenBasedTokenizer(200, "vocab.fvocab", special_tokens)
        >>> tokenizer.train(tokens, target_vocab_size=1000, verbose=True)
        """
        assert target_vocab_size >= self.actual_vocab_size
        num_merges = target_vocab_size - self.actual_vocab_size

        # Convert to list of integers if numpy array
        if isinstance(token_sequence, np.ndarray):
            ids = token_sequence.tolist()
        else:
            ids = list(token_sequence)
        
        # Validate token range
        unique_tokens = set(ids)
        max_token = max(unique_tokens)
        if max_token > self.actual_vocab_size:
            print(f"Warning: Found token {max_token} which is larger than actual_vocab_size {self.actual_vocab_size}")
            print(f"Adjusting actual_vocab_size to {max_token}")
            self.actual_vocab_size = max_token

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: idx for idx in range(1, self.actual_vocab_size + 1)} # int -> int
        
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            
            if not stats:
                print(f"No more pairs to merge at iteration {i+1}/{num_merges}")
                break
                
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            
            # mint a new token: assign it the next available id
            idx = self.actual_vocab_size + i + 1
            
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            
            # save the merge
            merges[pair] = idx
            vocab[idx] = idx
            
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        
        if verbose:
            print(f"\nTraining complete:")
            print(f"  Initial vocab size: {self.actual_vocab_size}")
            print(f"  Final vocab size: {self.actual_vocab_size + len(self.merges)}")
            print(f"  Number of merges: {len(self.merges)}")

    def encode(self, token_sequence):
        """
        Encode a sequence of base tokens using learned BPE merges.
        
        Parameters:
        -----------
        token_sequence : list or np.ndarray
            Sequence of integer tokens from the base vocabulary
            
        Returns:
        --------
        ids : list
            Encoded sequence with BPE merges applied
            
        Example:
        --------
        >>> # Encode new discretized tokens
        >>> new_tokens, _ = simple_discretize(new_data, N=200, special_tokens=special_tokens)
        >>> encoded = tokenizer.encode(new_tokens)
        """
        # Convert to list of integers if numpy array
        if isinstance(token_sequence, np.ndarray):
            ids = token_sequence.tolist()
        else:
            ids = list(token_sequence)
        
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            # if there are no more merges available, break
            if pair not in self.merges:
                break
                
            # merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids

    def decode(self, ids):
        """
        Decode BPE-encoded sequence back to base tokens.
        
        Parameters:
        -----------
        ids : list
            Sequence of BPE token IDs (may include merged tokens)
            
        Returns:
        --------
        tokens : list
            Sequence of base vocabulary tokens
            
        Example:
        --------
        >>> base_tokens = tokenizer.decode(encoded_ids)
        """
        # Recursively expand merged tokens back to base tokens
        base_tokens = []
        for id_ in ids:
            if id_ > self.actual_vocab_size:
                # This is a merged token, expand it
                base_tokens.extend(self._expand_token(id_))
            else:
                # This is a base token
                base_tokens.append(id_)
        
        return base_tokens
    
    def _expand_token(self, token_id):
        """
        Recursively expand a merged token back to its component base tokens.
        
        Parameters:
        -----------
        token_id : int
            Token ID to expand
            
        Returns:
        --------
        tokens : list
            List of base token IDs
        """
        # Find the pair that created this token
        for (left, right), merged_id in self.merges.items():
            if merged_id == token_id:
                # Recursively expand both sides
                left_tokens = self._expand_token(left) if left > self.actual_vocab_size else [left]
                right_tokens = self._expand_token(right) if right > self.actual_vocab_size else [right]
                return left_tokens + right_tokens
        
        # If not found in merges, it must be a base token
        return [token_id]

    def decode_to_floats(self, ids):
        """
        Decode BPE-encoded sequence to float values using the float vocabulary.
        
        This performs two steps:
        1. Decode BPE merges to base tokens
        2. Convert base tokens to float values using the .fvocab file
        
        Parameters:
        -----------
        ids : list
            Sequence of BPE token IDs
            
        Returns:
        --------
        floats : list
            List of float values (with special token strings for PAD/EBOS)
            
        Example:
        --------
        >>> floats = tokenizer.decode_to_floats(encoded_ids)
        """
        # First decode to base tokens
        base_tokens = self.decode(ids)
        
        # Then convert tokens to floats
        floats, n_edges = decode_with_float_vocab(
            base_tokens, 
            self.encoding_name, 
            special_tokens=self.special_tokens
        )
        
        return floats

    def get_token_statistics(self, token_sequence):
        """
        Get statistics about token pairs in a sequence (useful for analysis).
        
        Parameters:
        -----------
        token_sequence : list or np.ndarray
            Sequence of tokens
            
        Returns:
        --------
        stats : dict
            Dictionary mapping token pairs to their counts
            
        Example:
        --------
        >>> stats = tokenizer.get_token_statistics(tokens)
        >>> for pair, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
        ...     print(f"{pair}: {count}")
        """
        if isinstance(token_sequence, np.ndarray):
            ids = token_sequence.tolist()
        else:
            ids = list(token_sequence)
        
        return get_stats(ids)
    
    def analyze_special_token_context(self, token_sequence, window=5):
        """
        Analyze the context around special tokens to understand their impact.
        
        Parameters:
        -----------
        token_sequence : list or np.ndarray
            Sequence of tokens
        window : int
            Number of tokens before and after special token to show
            
        Returns:
        --------
        contexts : list of dict
            List of contexts around each special token
            
        Example:
        --------
        >>> contexts = tokenizer.analyze_special_token_context(tokens, window=3)
        >>> for ctx in contexts[:5]:
        ...     print(f"Special token {ctx['token']} at position {ctx['position']}")
        ...     print(f"  Before: {ctx['before']}")
        ...     print(f"  After: {ctx['after']}")
        """
        if isinstance(token_sequence, np.ndarray):
            ids = token_sequence.tolist()
        else:
            ids = list(token_sequence)
        
        special_token_ids = set(self.special_tokens.values())
        contexts = []
        
        for i, token in enumerate(ids):
            if token in special_token_ids:
                # Find the special token name
                token_name = [k for k, v in self.special_tokens.items() if v == token][0]
                
                context = {
                    'position': i,
                    'token': token_name,
                    'token_id': token,
                    'before': ids[max(0, i-window):i],
                    'after': ids[i+1:min(len(ids), i+window+1)]
                }
                contexts.append(context)
        
        return contexts
