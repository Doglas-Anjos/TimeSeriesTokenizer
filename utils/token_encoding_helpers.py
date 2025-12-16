"""
Helper functions for integrating TokenBasedTokenizer into main2.py workflow.

These functions provide a clean interface for training and encoding with
pre-discretized token sequences that include special tokens (PAD, EBOS).
"""

import joblib
import os
from utils.token_based import TokenBasedTokenizer


def encode_token_from_discretized(token_sequence, base_name, N_samples, total_vocab_size, special_tokens):
    """
    Encode pre-discretized tokens using TokenBasedTokenizer.
    
    This function trains or loads a TokenBasedTokenizer and encodes the given
    token sequence. It handles both training (if model doesn't exist) and
    inference (if model exists).
    
    Parameters:
    -----------
    token_sequence : list or np.ndarray
        Pre-discretized token sequence (output from simple_discretize or adaptative_bins_discretize)
        May include special tokens like PAD and EBOS
    base_name : str
        Base name for model and vocab files (without extension)
        e.g., "ETTh1_feature_Nsam_100_vocab_1000_column_HUFL_simple_normal_24h"
    N_samples : int
        Number of discretization bins (actual vocab size before BPE)
    total_vocab_size : int
        Target vocabulary size after BPE merges
    special_tokens : dict
        Dictionary mapping special token strings to IDs
        e.g., {'<PAD>': 99, '<EBOS>': 100}
    
    Returns:
    --------
    encoded_ids : list
        Token sequence after BPE encoding
        
    Example:
    --------
    >>> # In main2.py, after discretization:
    >>> y_tokens, edges = simple_discretize(data, N_samples, data_st, special_tokens)
    >>> base_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_simple_normal_24h"
    >>> encoded = encode_token_from_discretized(y_tokens, base_name, N_samples, total_vocab_size, special_tokens)
    """
    model_name = fr"model\{base_name}.model"
    file_vocab_name = fr"{base_name}.fvocab"
    
    if not joblib.os.path.isfile(model_name):
        # Train new tokenizer on token sequence
        print(f"  Training new TokenBasedTokenizer...")
        objtok = TokenBasedTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
        objtok.train(token_sequence, total_vocab_size, verbose=False)
        objtok.save(fr"{base_name}", file_vocab_name)
        print(f"  ✓ Saved model: {model_name}")
    else:
        # Load existing tokenizer
        objtok = TokenBasedTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
        objtok.load(model_name)
        print(f"  ✓ Loaded existing model: {model_name}")
    
    # Encode the token sequence
    encoded_ids = objtok.encode(token_sequence)
    return encoded_ids


def decode_token_sequence(encoded_ids, base_name, N_samples, special_tokens, to_floats=True):
    """
    Decode BPE-encoded tokens back to base tokens or float values.
    
    Parameters:
    -----------
    encoded_ids : list
        BPE-encoded token sequence
    base_name : str
        Base name for model and vocab files (without extension)
    N_samples : int
        Number of discretization bins
    special_tokens : dict
        Dictionary mapping special token strings to IDs
    to_floats : bool
        If True, decode to float values; if False, decode to base tokens
    
    Returns:
    --------
    decoded : list
        Either float values (if to_floats=True) or base token IDs (if to_floats=False)
        
    Example:
    --------
    >>> # Decode to float values
    >>> floats = decode_token_sequence(encoded, base_name, N_samples, special_tokens, to_floats=True)
    >>> # Or decode to base tokens
    >>> tokens = decode_token_sequence(encoded, base_name, N_samples, special_tokens, to_floats=False)
    """
    model_name = fr"model\{base_name}.model"
    file_vocab_name = fr"{base_name}.fvocab"
    
    # Load tokenizer
    objtok = TokenBasedTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
    objtok.load(model_name)
    
    if to_floats:
        # Decode to float values
        return objtok.decode_to_floats(encoded_ids)
    else:
        # Decode to base tokens
        return objtok.decode(encoded_ids)


def compare_encoding_methods(data, data_st, N_samples, total_vocab_size, special_tokens, base_name):
    """
    Compare traditional float-based encoding vs token-based encoding.
    
    This function demonstrates the difference between:
    1. Training on float values (BasicTokenizer)
    2. Training on pre-discretized tokens with special tokens (TokenBasedTokenizer)
    
    Parameters:
    -----------
    data : array-like
        Original float data
    data_st : array-like
        Data with special token placeholders (strings like '<EBOS>')
    N_samples : int
        Number of discretization bins
    total_vocab_size : int
        Target vocabulary size after BPE
    special_tokens : dict
        Special token mapping
    base_name : str
        Base name for model files
        
    Returns:
    --------
    comparison : dict
        Dictionary with statistics comparing both methods
    """
    from utils.basic import BasicTokenizer
    from utils.discretisize import simple_discretize, save_float_vocab
    
    # Method 1: Traditional (float-based)
    print("\n📊 Comparing encoding methods...")
    print("\nMethod 1: BasicTokenizer (trains on floats)")
    tokens_no_st, edges = simple_discretize(data, N_samples, None, special_tokens)
    save_float_vocab(edges.tolist(), f"{base_name}_basic.fvocab")
    
    basic_tok = BasicTokenizer(N_samples, f"{base_name}_basic.fvocab", special_tokens)
    basic_tok.train(data, total_vocab_size, verbose=False)
    basic_encoded = basic_tok.encode(data)
    
    print(f"  Sequence length: {len(basic_encoded)}")
    print(f"  Unique tokens: {len(set(basic_encoded))}")
    print(f"  Merges: {len(basic_tok.merges)}")
    
    # Method 2: Token-based
    print("\nMethod 2: TokenBasedTokenizer (trains on tokens with special tokens)")
    tokens_with_st, edges = simple_discretize(data, N_samples, data_st, special_tokens)
    save_float_vocab(edges.tolist(), f"{base_name}_token.fvocab")
    
    token_tok = TokenBasedTokenizer(N_samples, f"{base_name}_token.fvocab", special_tokens)
    token_tok.train(tokens_with_st, total_vocab_size, verbose=False)
    token_encoded = token_tok.encode(tokens_with_st)
    
    print(f"  Sequence length: {len(token_encoded)}")
    print(f"  Unique tokens: {len(set(token_encoded))}")
    print(f"  Merges: {len(token_tok.merges)}")
    print(f"  Special tokens in sequence: {sum(1 for t in tokens_with_st if t in special_tokens.values())}")
    
    # Compare token pair statistics
    basic_stats = basic_tok.get_token_statistics(tokens_no_st) if hasattr(basic_tok, 'get_token_statistics') else {}
    token_stats = token_tok.get_token_statistics(tokens_with_st)
    
    return {
        'basic': {
            'encoded_length': len(basic_encoded),
            'unique_tokens': len(set(basic_encoded)),
            'num_merges': len(basic_tok.merges),
            'special_tokens': 0
        },
        'token_based': {
            'encoded_length': len(token_encoded),
            'unique_tokens': len(set(token_encoded)),
            'num_merges': len(token_tok.merges),
            'special_tokens': sum(1 for t in tokens_with_st if t in special_tokens.values())
        }
    }


# Convenience function for main2.py integration
def should_use_token_based_encoding(data_st):
    """
    Helper to determine if token-based encoding should be used.
    
    Returns True if data_st contains special tokens that should affect BPE training.
    
    Parameters:
    -----------
    data_st : array-like or None
        Data with special token placeholders
        
    Returns:
    --------
    use_token_based : bool
    """
    if data_st is None:
        return False
    
    # Check if data_st contains special token strings
    special_token_markers = ['<PAD>', '<EBOS>', '<EOS>', '<BOS>']
    for marker in special_token_markers:
        if marker in data_st:
            return True
    
    return False
