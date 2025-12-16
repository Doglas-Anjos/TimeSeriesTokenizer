"""
Example demonstrating the difference between BasicTokenizer and TokenBasedTokenizer.

Shows how special tokens (PAD, EBOS) affect BPE token pair statistics when
training on pre-discretized token sequences vs. training on float values.
"""

import numpy as np
import pandas as pd
from utils.basic import BasicTokenizer
from utils.token_based import TokenBasedTokenizer
from utils.discretisize import simple_discretize, adaptative_bins_discretize, save_float_vocab
import os

# Create directories if needed
os.makedirs('float_vocab', exist_ok=True)
os.makedirs('model', exist_ok=True)


def example_basic_workflow():
    """
    Traditional workflow: train on floats, discretization happens inside BasicTokenizer
    """
    print("="*80)
    print("EXAMPLE 1: BasicTokenizer (Traditional Workflow)")
    print("="*80)
    
    # Setup
    N_samples = 100
    special_tokens = {'<PAD>': 99, '<EBOS>': 100}
    vocab_size = 200
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(500) * 10 + 50  # Some sample time series data
    
    print(f"\n1. Original data: {len(data)} float values")
    print(f"   Sample: {data[:10]}")
    
    # Traditional approach: train directly on float values
    # The tokenizer will discretize internally using encode_with_float_vocab
    print(f"\n2. Training BasicTokenizer on float values...")
    
    base_name = "example_basic"
    vocab_file = f"{base_name}.fvocab"
    
    # First, create the float vocab (discretization bins)
    data_simple, edges = simple_discretize(data, N_samples, None, special_tokens)
    save_float_vocab(edges.tolist(), vocab_file)
    
    # Train tokenizer on float values
    tokenizer = BasicTokenizer(N_samples, vocab_file, special_tokens)
    tokenizer.train(data, vocab_size, verbose=False)
    
    print(f"   ✓ Trained with {len(tokenizer.merges)} merges")
    
    # Encode and decode
    encoded = tokenizer.encode(data[:20])
    print(f"\n3. Encoding first 20 values:")
    print(f"   Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"\n4. Decoding:")
    print(f"   Decoded: {decoded[:10]}")
    
    return tokenizer, data, special_tokens


def example_token_based_workflow():
    """
    New workflow: train on pre-discretized tokens (includes special tokens)
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: TokenBasedTokenizer (Token-First Workflow)")
    print("="*80)
    
    # Setup
    N_samples = 100
    special_tokens = {'<PAD>': 99, '<EBOS>': 100}
    vocab_size = 200
    
    # Generate sample data with special tokens
    np.random.seed(42)
    data = np.random.randn(500) * 10 + 50
    
    # Create data with special tokens inserted (simulate EBOS every 24 values)
    data_with_st = []
    for i, val in enumerate(data):
        if i % 24 == 0 and i > 0:
            data_with_st.append('<EBOS>')
        data_with_st.append(val)
    
    print(f"\n1. Original data with special tokens: {len(data_with_st)} values")
    print(f"   (includes {data_with_st.count('<EBOS>')} EBOS tokens)")
    
    # Discretize to tokens (this returns integers with special tokens already inserted)
    print(f"\n2. Discretizing to token sequence...")
    tokens, edges = simple_discretize(data, N_samples, data_with_st, special_tokens)
    
    base_name = "example_token_based"
    vocab_file = f"{base_name}.fvocab"
    save_float_vocab(edges.tolist(), vocab_file)
    
    print(f"   ✓ Generated {len(tokens)} tokens")
    print(f"   Sample tokens: {tokens[:20]}")
    print(f"   Special tokens in sequence: {sum(1 for t in tokens if t in special_tokens.values())}")
    
    # NEW: Train directly on token sequence
    print(f"\n3. Training TokenBasedTokenizer on token sequence...")
    tokenizer = TokenBasedTokenizer(N_samples, vocab_file, special_tokens)
    tokenizer.train(tokens, vocab_size, verbose=False)
    
    print(f"   ✓ Trained with {len(tokenizer.merges)} merges")
    
    # Analyze special token context
    print(f"\n4. Analyzing special token context...")
    contexts = tokenizer.analyze_special_token_context(tokens, window=3)
    if contexts:
        for ctx in contexts[:3]:
            print(f"   {ctx['token']} at position {ctx['position']}:")
            print(f"     Before: {ctx['before']}")
            print(f"     After:  {ctx['after']}")
    
    # Encode and decode
    test_tokens, _ = simple_discretize(data[100:120], N_samples, None, special_tokens)
    encoded = tokenizer.encode(test_tokens)
    print(f"\n5. Encoding test tokens:")
    print(f"   Original tokens: {test_tokens}")
    print(f"   Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"\n6. Decoding:")
    print(f"   Decoded tokens: {decoded}")
    
    # Decode to floats
    floats = tokenizer.decode_to_floats(encoded)
    print(f"\n7. Decoding to float values:")
    print(f"   Float values: {floats[:10]}")
    
    return tokenizer, tokens, special_tokens


def compare_token_pair_statistics():
    """
    Compare token pair statistics between the two approaches
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Token Pair Statistics")
    print("="*80)
    
    N_samples = 100
    special_tokens = {'<PAD>': 99, '<EBOS>': 100}
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(200) * 10 + 50
    
    # Create data with EBOS every 24 values
    data_with_st = []
    for i, val in enumerate(data):
        if i % 24 == 0 and i > 0:
            data_with_st.append('<EBOS>')
        data_with_st.append(val)
    
    print(f"\nData: {len(data)} values + {data_with_st.count('<EBOS>')} EBOS tokens")
    
    # Method 1: Discretize without special tokens
    tokens_no_st, edges1 = simple_discretize(data, N_samples, None, special_tokens)
    save_float_vocab(edges1.tolist(), "compare_no_st.fvocab")
    
    # Method 2: Discretize with special tokens
    tokens_with_st, edges2 = simple_discretize(data, N_samples, data_with_st, special_tokens)
    save_float_vocab(edges2.tolist(), "compare_with_st.fvocab")
    
    print(f"\nTokens without EBOS: {len(tokens_no_st)}")
    print(f"Tokens with EBOS: {len(tokens_with_st)}")
    
    # Get token pair statistics
    tokenizer_no_st = TokenBasedTokenizer(N_samples, "compare_no_st.fvocab", special_tokens)
    tokenizer_with_st = TokenBasedTokenizer(N_samples, "compare_with_st.fvocab", special_tokens)
    
    stats_no_st = tokenizer_no_st.get_token_statistics(tokens_no_st)
    stats_with_st = tokenizer_with_st.get_token_statistics(tokens_with_st)
    
    print(f"\n📊 Token pair statistics:")
    print(f"\nWithout EBOS - Top 10 pairs:")
    for pair, count in sorted(stats_no_st.items(), key=lambda x: -x[1])[:10]:
        print(f"   {pair}: {count} occurrences")
    
    print(f"\nWith EBOS - Top 10 pairs:")
    for pair, count in sorted(stats_with_st.items(), key=lambda x: -x[1])[:10]:
        pair_str = f"({pair[0]}, {pair[1]})"
        # Check if pair involves special token
        if pair[0] in special_tokens.values() or pair[1] in special_tokens.values():
            st_names = []
            if pair[0] in special_tokens.values():
                st_names.append([k for k, v in special_tokens.items() if v == pair[0]][0])
            if pair[1] in special_tokens.values():
                st_names.append([k for k, v in special_tokens.items() if v == pair[1]][0])
            pair_str += f" [involves {', '.join(st_names)}]"
        print(f"   {pair_str}: {count} occurrences")
    
    # Show pairs that only exist with special tokens
    print(f"\n⚠️  Pairs that only appear with EBOS tokens:")
    unique_to_st = set(stats_with_st.keys()) - set(stats_no_st.keys())
    for pair in sorted(unique_to_st, key=lambda p: -stats_with_st[p])[:10]:
        if pair[0] in special_tokens.values() or pair[1] in special_tokens.values():
            print(f"   {pair}: {stats_with_st[pair]} occurrences")


def create_training_function_example():
    """
    Show how to integrate TokenBasedTokenizer into main2.py workflow
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Integration with main2.py")
    print("="*80)
    
    print("\nSuggested code structure for main2.py:")
    print("""
def encode_token_from_discretized(token_sequence, base_name):
    '''
    Encode pre-discretized tokens using TokenBasedTokenizer.
    Use this instead of encode_token() when you already have discretized tokens.
    '''
    from utils.token_based import TokenBasedTokenizer
    
    model_name = fr"model\\{base_name}.model"
    file_vocab_name = fr"{base_name}.fvocab"
    
    if not joblib.os.path.isfile(model_name):
        # Train on token sequence
        objtok = TokenBasedTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
        objtok.train(token_sequence, total_vocab_size, verbose=True)
        objtok.save(fr"{base_name}", file_vocab_name)
    else:
        # Load existing model
        objtok = TokenBasedTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
        objtok.load(model_name)
    
    # Encode the token sequence
    encoded_ids = objtok.encode(token_sequence)
    return encoded_ids

# Usage in process_files():
for data, data_st, norm_type, ebos_suffix in configs:
    print(f"{norm_type} discretization simple {ebos_suffix}")
    
    # Get discretized tokens (with special tokens)
    y_tokens, bin_edges = simple_discretize(data, N_samples, data_st, special_tokens)
    
    base_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_simple}_{norm_type}_{ebos_suffix}"
    save_float_vocab(bin_edges.tolist(), f"{base_name}.fvocab")
    
    # NEW: Use token-based encoding
    encoded_tokens = encode_token_from_discretized(y_tokens, base_name)
    
    tokenized_dfs[f"{norm_type}_simp_{ebos_suffix}"][column] = pd.Series(encoded_tokens)
    """)


if __name__ == "__main__":
    # Run all examples
    print("\n🚀 Starting Token-Based Tokenizer Examples\n")
    
    # Example 1: Traditional workflow
    tokenizer1, data1, st1 = example_basic_workflow()
    
    # Example 2: Token-based workflow
    tokenizer2, tokens2, st2 = example_token_based_workflow()
    
    # Example 3: Compare statistics
    compare_token_pair_statistics()
    
    # Example 4: Integration guide
    create_training_function_example()
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. BasicTokenizer trains on float values (discretizes internally)")
    print("2. TokenBasedTokenizer trains on pre-discretized tokens")
    print("3. Special tokens (EBOS, PAD) affect token pair statistics")
    print("4. Use TokenBasedTokenizer when special tokens are already in sequence")
    print("5. Both can decode back to float values using .fvocab files")
