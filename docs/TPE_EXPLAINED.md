# Token Pair Encoding (TPE) for Time Series

## Overview

**Token Pair Encoding (TPE)** is a compression algorithm adapted from Byte Pair Encoding (BPE) for time series tokenization. Unlike traditional BPE which operates on bytes or characters, TPE works directly on **pre-discretized tokens** (integers representing time series bins).

This implementation is based on [Andrej Karpathy's minGPT BPE](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py), adapted specifically for time series data.

---

## Why TPE Instead of BPE?

While the algorithm is essentially BPE, we call it **Token Pair Encoding** to emphasize:

1. **Domain difference**: Works on time series tokens, not text bytes
2. **Pre-discretized input**: Operates on discrete bins from continuous values
3. **Special tokens**: Handles temporal markers (`<EBOS>`) and padding (`<PAD>`)
4. **Time series context**: Merges represent temporal patterns, not linguistic patterns

---

## From Continuous Values to Compressed Tokens

### The Complete Pipeline

```
Continuous Time Series
        ↓
[1] Standardization (StandardScaler)
        ↓
[2] Discretization (binning)
        ↓
[3] TPE Compression (this algorithm)
        ↓
Compressed Token Sequence
```

### Example: Step-by-Step

**Original time series** (continuous values):
```
[25.3, 25.1, 24.9, 25.0, 25.2, 25.1, 24.8, 24.7]
```

**After standardization** (mean=0, std=1):
```
[0.89, 0.45, -0.12, 0.11, 0.67, 0.45, -0.45, -0.78]
```

**After discretization** (N=200 bins):
```
[156, 134, 89, 101, 145, 134, 67, 45]
```

**After TPE compression**:
```
[201, 89, 101, 201, 67, 45]
```
*(Where token 201 = merge of (156, 134) the pair more frequently)*

---

## The TPE Algorithm

### Core Concept

TPE iteratively finds and merges the **most frequent adjacent token pairs** into new tokens.

### Algorithm Steps

```python
# Input: sequence of base tokens [t1, t2, t3, ..., tn]
# Goal: compress to fewer tokens while preserving information

1. Count all adjacent token pairs
   Example: [15, 23, 15, 23, 42] → {(15,23): 2, (23,15): 1, (23,42): 1}

2. Find most frequent pair
   → (15, 23) with count 2

3. Create new token (next available ID)
   → token 201 = (15, 23)

4. Replace all occurrences
   [15, 23, 15, 23, 42] → [201, 201, 42]

5. Repeat until target vocabulary size reached
```

### Pseudocode

```python
def train_tpe(tokens, initial_vocab_size, target_vocab_size):
    num_merges = target_vocab_size - initial_vocab_size
    merges = {}  # Store merge rules: (token1, token2) → new_token
    
    for i in range(num_merges):
        # Count all adjacent pairs
        pair_counts = count_pairs(tokens)
        
        if not pair_counts:
            break  # No more pairs to merge
        
        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # Create new token
        new_token_id = initial_vocab_size + i + 1
        
        # Replace all occurrences
        tokens = replace_pair(tokens, best_pair, new_token_id)
        
        # Save merge rule
        merges[best_pair] = new_token_id
    
    return merges
```

---

## Detailed Example

Let's walk through a complete example:

### Initial Setup

```python
# Discretized sequence (after binning)
tokens = [10, 15, 10, 15, 10, 15, 20, 25, 20, 25]

initial_vocab_size = 200  # Bins 1-200
target_vocab_size = 400   # Add 200 merges
```

### Iteration 1

**Count pairs:**
```
(10, 15): 3 occurrences  ← Most frequent!
(15, 10): 2 occurrences
(15, 20): 1 occurrence
(20, 25): 2 occurrences
(25, 20): 1 occurrence
```

**Merge:**
```python
pair = (10, 15)
new_token = 201
tokens = [201, 201, 201, 20, 25, 20, 25]
merges[(10, 15)] = 201
```

### Iteration 2

**Count pairs:**
```
(201, 201): 2 occurrences  ← Most frequent!
(201, 20): 1 occurrence
(20, 25): 2 occurrences
(25, 20): 1 occurrence
```

**Merge:**
```python
pair = (201, 201)
new_token = 202
tokens = [202, 201, 20, 25, 20, 25]
merges[(201, 201)] = 202
```

### Iteration 3

**Count pairs:**
```
(202, 201): 1 occurrence
(201, 20): 1 occurrence
(20, 25): 2 occurrences  ← Most frequent!
(25, 20): 1 occurrence
```

**Merge:**
```python
pair = (20, 25)
new_token = 203
tokens = [202, 201, 203, 203]
merges[(20, 25)] = 203
```

### Final Result

**Compression:**
- Original length: 10 tokens
- Compressed length: 4 tokens
- **Compression rate: 2.5x**

**Merge rules learned:**
```python
merges = {
    (10, 15): 201,   # Frequent pattern: values oscillating between bins 10 and 15
    (201, 201): 202, # Even more compressed: repeated oscillations
    (20, 25): 203,   # Another pattern: bins 20 and 25 together
}
```

---

## Encoding and Decoding

### Encoding (Compression)

Apply learned merges to new sequences:

```python
def encode(tokens, merges):
    """Apply BPE merges to compress a token sequence."""
    while len(tokens) >= 2:
        # Find all current pairs
        pairs = get_pairs(tokens)
        
        # Find the pair with lowest merge index (earliest learned)
        best_pair = min(pairs, key=lambda p: merges.get(p, float('inf')))
        
        # If no merge available, stop
        if best_pair not in merges:
            break
        
        # Apply merge
        new_token = merges[best_pair]
        tokens = replace_pair(tokens, best_pair, new_token)
    
    return tokens
```

**Example:**
```python
# New sequence
new_tokens = [10, 15, 10, 15, 20, 25]

# Apply merges
# Step 1: (10,15) → 201
#   [201, 201, 20, 25]
# Step 2: (201,201) → 202
#   [202, 20, 25]
# Step 3: (20,25) → 203
#   [202, 203]

encoded = [202, 203]  # Compressed from 6 to 2 tokens!
```

### Decoding (Decompression)

Reverse the merge operations:

```python
def decode(compressed_tokens, merges, initial_vocab_size):
    """Expand compressed tokens back to base tokens."""
    base_tokens = []
    
    for token in compressed_tokens:
        if token > initial_vocab_size:
            # This is a merged token - expand it
            base_tokens.extend(expand_token(token, merges))
        else:
            # Base token - keep as is
            base_tokens.append(token)
    
    return base_tokens

def expand_token(token, merges):
    """Recursively expand a merged token."""
    # Find the pair that created this token
    for (left, right), merged_id in merges.items():
        if merged_id == token:
            # Recursively expand both sides
            left_tokens = expand_token(left, merges) if left > initial_vocab_size else [left]
            right_tokens = expand_token(right, merges) if right > initial_vocab_size else [right]
            return left_tokens + right_tokens
    
    # Base token
    return [token]
```

**Example:**
```python
# Compressed sequence
compressed = [202, 203]

# Decode
# Token 202: expand → (201, 201)
#   Token 201: expand → (10, 15)
#   Token 201: expand → (10, 15)
#   Result: [10, 15, 10, 15]
# Token 203: expand → (20, 25)
#   Result: [20, 25]

decoded = [10, 15, 10, 15, 20, 25]  # Back to original!
```

---

## Special Tokens in TPE

### Temporal Markers

Time series often include special tokens for temporal structure:

```python
special_tokens = {
    '<PAD>': 199,   # Padding token
    '<EBOS>': 200   # End/Beginning of Sequence (every 12h or 24h)
}
```

**Example sequence with temporal markers:**
```
[15, 23, 34, 45, <EBOS>, 56, 67, 78, 89, <EBOS>, 12, 23]
     ↓
[15, 23, 34, 45, 200, 56, 67, 78, 89, 200, 12, 23]
```

### Handling Special Tokens

**Key principle:** Special tokens participate in pair counting but are preserved.

**Example:**
```python
tokens = [10, 15, 200, 10, 15, 200, 10, 15]
#          ↑   ↑   ↑    ↑   ↑   ↑    ↑   ↑
#          base    EBOS base    EBOS base

# Pair counts:
# (10, 15): 3    ← Can merge
# (15, 200): 3   ← Can merge (but includes special token)
# (200, 10): 2   ← Can merge (but includes special token)

# If (10, 15) is most frequent:
# Result: [201, 200, 201, 200, 201]
```

**Important:** Merging across `<EBOS>` boundaries can lose temporal information!

---

## Vocabulary Size Design

### Fixed Vocabulary vs Fixed Compression

**Design choice in this project:**
- ✅ **Fixed vocabulary size** (e.g., 600 tokens)
- ❌ **Variable compression rate** (depends on data patterns)

**Rationale:**

1. **Inspired by LLMs**: GPT models use fixed vocabulary (50k tokens)
2. **Transformer-friendly**: Models expect consistent vocabulary size
3. **Natural representation**: Different sequences have different compressibility

**Alternative approach:**
- ⚠️ Fixed compression rate (e.g., always 2x)
- ⚠️ Variable vocabulary size (different for each sequence)
- ⚠️ Less common in NLP, more complex for models

### Vocabulary Growth

```python
initial_vocab = 200        # Discretization bins
num_merges = 400          # TPE iterations
final_vocab = 200 + 400   # = 600 total tokens

# Token ID ranges:
# 1-200:   Base tokens (discretization bins)
# 201-600: Merged tokens (learned patterns)
```

---

## Compression Statistics

### Measuring Compression

```python
compression_rate = original_length / compressed_length
```

**Example statistics:**

| Dataset | Column | Bins | Original | Compressed | Rate |
|---------|--------|------|----------|------------|------|
| ETTh1 | HUFL | 200 | 8,640 | 4,320 | 2.00x |
| ETTh1 | HULL | 200 | 8,640 | 4,285 | 2.02x |
| Weather | T_degC | 100 | 52,696 | 26,348 | 2.00x |

**Observation:** Higher repetition → better compression

### What Affects Compression?

1. **Data patterns**: Repetitive patterns compress better
2. **Vocabulary size**: More merges → higher compression
3. **Sequence length**: Longer sequences → more opportunities
4. **Special tokens**: Can fragment patterns, reducing compression

---

## Implementation Details

### Core Functions (from minGPT)

```python
def get_stats(ids):
    """
    Count occurrences of consecutive token pairs.
    
    Returns:
        dict: {(token1, token2): count}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_token):
    """
    Replace all occurrences of a token pair with a new token.
    
    Args:
        ids: Token sequence
        pair: (token1, token2) to replace
        new_token: Replacement token ID
    
    Returns:
        new_ids: Updated sequence
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # Check if current position matches pair
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_token)
            i += 2  # Skip both tokens in pair
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
```

### TokenBasedTokenizer Class

See [utils/token_based.py](../utils/token_based.py) for full implementation.

**Key methods:**

```python
class TokenBasedTokenizer:
    def train(self, token_sequence, target_vocab_size):
        """Learn TPE merges from token sequence."""
        
    def encode(self, token_sequence):
        """Compress using learned merges."""
        
    def decode(self, compressed_tokens):
        """Expand back to base tokens."""
        
    def decode_to_floats(self, compressed_tokens):
        """Expand to base tokens, then to float values."""
```

---

## TPE for Time Series: What It Learns

### Temporal Motifs

TPE discovers recurring temporal patterns:

**Example: Daily temperature cycle**

```
Original bins (hourly data for 3 days):
[Low, Low, Rising, High, High, Falling, Low, Low, Rising, High, High, Falling, ...]
 ↓
Learned pattern: (Low, Low) → token_201
                 (Rising, High) → token_202
                 (High, High) → token_203
 ↓
Compressed:
[201, 202, 203, Falling, 201, 202, 203, Falling, ...]
```

**Interpretation:** Token_201 represents "nighttime", token_202 "morning warming", token_203 "daytime heat"

### Multi-Step Patterns

Higher-level merges capture longer patterns:

```
Merge 1: (bin_15, bin_23) → 201      # 2-step pattern
Merge 2: (201, 201) → 202             # 4-step pattern (two 2-steps)
Merge 3: (202, 202) → 203             # 8-step pattern!
```

**Result:** Single token can represent 8+ time steps

**Consequence:** Loss of precise temporal alignment

---

## Comparison to NLP BPE

| Aspect | NLP BPE | Time Series TPE |
|--------|---------|-----------------|
| **Input** | Bytes/characters (text) | Discrete bins (time series) |
| **Domain** | Linguistic patterns | Temporal patterns |
| **Patterns** | Words, morphemes | Motifs, cycles |
| **Special tokens** | `<BOS>`, `<EOS>`, `<PAD>` | `<EBOS>` (temporal), `<PAD>` |
| **Pre-processing** | Character encoding | Discretization (binning) |
| **Post-processing** | Characters → text | Bins → floats |
| **Typical vocab** | 50,000+ tokens | 600-1,000 tokens |
| **Inspiration** | Compression, subwords | Pattern discovery |

---

## Advantages of TPE for Time Series

### 1. **Motif Discovery**

Automatically learns recurring patterns without manual feature engineering.

### 2. **Variable-Length Encoding**

Different patterns compressed differently:
- Frequent patterns: high compression
- Rare patterns: low compression
- **Automatic importance weighting**

### 3. **Transformer-Friendly**

Fixed vocabulary size works well with Transformer architectures:
- Consistent embedding dimension
- No dynamic vocabulary issues
- Similar to text tokenization (familiar paradigm)

### 4. **Compression**

Reduces sequence length by 2-3x:
- Faster training (shorter sequences)
- Lower memory usage
- Attention over compressed representations

### 5. **Unsupervised Learning**

No labels needed - learns patterns from data itself.

---

## Disadvantages and Limitations

### 1. **Loss of Temporal Precision**

Merged tokens span multiple time steps → ambiguous timestamps.

**Example:**
```
Token 201 = (bin_10, bin_15)
When did bin_10 occur? Time step t or t+1? Unknown.
```

**Impact:** Difficult to align predictions with exact timestamps.

### 2. **Greedy Algorithm**

Merges are greedy (most frequent first) → may not be globally optimal.

**Better alternative:** Neural tokenization (learned embeddings) - future work.

### 3. **Data-Dependent Compression**

Compression rate varies by sequence:
- Repetitive data: high compression
- Noisy data: low compression
- **Inconsistent across datasets**

### 4. **Special Token Interaction**

Temporal markers (`<EBOS>`) can fragment patterns:

```
Without EBOS: [A, B, C, A, B, C] → high compression
With EBOS:    [A, B, <EBOS>, C, A, <EBOS>, B, C] → lower compression
```

### 5. **Irreversible Discretization**

Discretization (before TPE) loses precision:
- Binning rounds continuous values
- Cannot recover exact original values
- **Fundamental limitation, not TPE-specific**

---

## Training Statistics

The implementation saves detailed statistics during training:

### Summary File

`tokenization_stats/{config}_summary.txt`:

```
TOKENIZATION TRAINING SUMMARY
==================================================

Initial vocab size: 200
Final vocab size: 600
Number of merges: 400
Compression Rate: 2.0000x
```

### Merge Statistics File

`tokenization_stats/{config}_merge_stats.txt`:

```
MERGE STATISTICS
==================================================
Merge_Number=Token_Pair=Occurrence_Count

1=15,23=3600
2=23,34=3200
3=201,45=2800
...
```

**Usage:** Analyze which patterns were most frequent.

---

## References

### Original BPE Paper

**Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units"**, ACL 2016

- Introduced BPE for NLP
- Solves out-of-vocabulary problem
- Subword segmentation

### minGPT Implementation

**Andrej Karpathy - [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py)**

- Clean, educational BPE implementation
- Python-based, easy to understand
- **Basis for this TPE implementation**

### Time Series Tokenization

**Ansari et al., "Chronos: Learning the Language of Time Series"**, arXiv 2024

- Applies tokenization to time series forecasting
- Large-scale pre-training on diverse datasets
- Inspiration for universal vocabulary approach

---

## Example Use Cases

### 1. Univariate Forecasting

```python
from utils.token_based import TokenBasedTokenizer
from utils.discretisize import simple_discretize

# Discretize time series
tokens, bin_edges = simple_discretize(data, N=200)

# Train TPE
tokenizer = TokenBasedTokenizer(200, "vocab.fvocab")
tokenizer.train(tokens, target_vocab_size=600, verbose=True)

# Compress new data
new_tokens, _ = simple_discretize(new_data, N=200)
compressed = tokenizer.encode(new_tokens)

# Train forecasting model on compressed sequences
```

### 2. Multivariate Time Series

```python
# Train separate tokenizer per variable
tokenizers = {}
for column in ['HUFL', 'HULL', 'MUFL']:
    tokens, _ = simple_discretize(data[column], N=200)
    tok = TokenBasedTokenizer(200, f"{column}.fvocab")
    tok.train(tokens, 600)
    tokenizers[column] = tok
```

### 3. Cross-Domain Vocabulary (Chronos)

```python
# Train on diverse datasets
all_tokens = []
for dataset in ['energy', 'traffic', 'weather']:
    tokens, _ = simple_discretize(load_data(dataset), N=200)
    all_tokens.extend(tokens)

# Universal tokenizer
universal_tok = TokenBasedTokenizer(200, "universal.fvocab")
universal_tok.train(all_tokens, 600)

# Apply to new domain
new_domain_compressed = universal_tok.encode(new_domain_tokens)
```

---

## Summary

**Token Pair Encoding (TPE)** adapts BPE for time series:

✅ **Automatic pattern discovery** - no manual feature engineering  
✅ **Compression** - reduces sequence length 2-3x  
✅ **Transformer-friendly** - fixed vocabulary size  
✅ **Unsupervised** - learns from data alone  
✅ **Based on proven algorithm** - BPE from NLP  

⚠️ **Loss of temporal precision** - merged tokens span multiple steps  
⚠️ **Greedy merging** - may not be globally optimal  
⚠️ **Data-dependent** - compression varies by patterns  

**Best for:** Discovering recurring temporal motifs in time series for Transformer-based forecasting.

**Implementation:** Based on [Andrej Karpathy's minGPT BPE](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py), adapted for pre-discretized time series tokens.

---

See also:
- [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) - How TPE fits into the overall experiments
- [WORKFLOW.md](WORKFLOW.md) - When TPE is applied in the pipeline
- [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md) - `TokenBasedTokenizer` API reference
