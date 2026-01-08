# Experiment Design

This document details the scientific design, hypotheses, and methodological choices for the WCCI 2026 paper on **tokenization strategies for time series forecasting**.

---

## Research Question

**How do different tokenization strategies affect time series forecasting performance with Transformer-based models?**

---

## Hypotheses

### H1: Column-Wise vs Universal Vocabulary

**Hypothesis**: Column-wise vocabularies may better capture variable-specific structure compared to a universal vocabulary.

**Reasoning**:
- Each time series column has unique dynamics (e.g., temperature vs humidity)
- Learning specialized vocabularies may capture column-specific patterns better
- Analogy: learning patterns within a single language vs universal language

**Alternative**: Universal vocabulary may generalize better across domains

### H2: BPE Compression Effect

**Hypothesis**: BPE (Byte Pair Encoding) compression creates motif-like tokens that capture temporal patterns.

**Reasoning**:
- Merging frequent token pairs creates higher-level patterns
- Similar to how words capture concepts better than characters in NLP
- May help model learn multi-step patterns

**Alternative**: Compression may lose important fine-grained information

### H3: Temporal Tokens

**Hypothesis**: Special tokens marking time boundaries (12h/24h) help models understand temporal structure.

**Reasoning**:
- Explicit temporal markers may improve long-range dependencies
- Similar to positional embeddings in Transformers
- May help with daily/weekly patterns

**Alternative**: Positional embeddings may be sufficient

---

## Experimental Design

### Two Main Experiments

#### Experiment 1: Column-Wise Tokenization

**Approach**:
- Each dataset column treated as independent "language"
- Separate tokenizer learned per column
- Vocabulary specialized to each variable

**Variations**:
| Factor | Levels |
|--------|--------|
| **Discretization Bins** | 50, 100, 200 |
| **BPE** | Enabled, Disabled |
| **Temporal Tokens** | None, 12h, 24h |
| **Models** | Informer, Transformer, Autoformer |
| **Datasets** | ETTh1 (6 columns), Weather (20 columns) |

**Total**: 3 bins × 2 BPE × 3 temporal × 3 models × 2 datasets = **108 configurations per column**

#### Experiment 2: Chronos Universal Vocabulary

**Approach**:
- Single universal vocabulary learned from Chronos dataset
- Same tokenizer applied to all columns and datasets
- Cross-domain generalization

**Variations**:
| Factor | Levels |
|--------|--------|
| **Discretization Bins** | 50, 100, 200 |
| **BPE** | Always Enabled |
| **Temporal Tokens** | None (heterogeneous frequencies) |
| **Models** | Informer, Transformer, Autoformer |
| **Datasets** | ETTh1, Weather |

**Total**: 3 bins × 3 models × 2 datasets = **18 configurations**

#### Baseline: No Tokenization

**Approach**:
- Continuous values (no discretization, no tokenization)
- Direct numerical forecasting
- Isolates effect of tokenization

**Variations**:
- 3 models × 2 datasets × 6 ETTh1 columns + 20 weather columns = **78 configurations**

---

## Datasets

### Target Forecasting Datasets

#### ETTh1 (Electricity Transformer Temperature)
- **Frequency**: Hourly
- **Columns**: HUFL, HULL, MUFL, MULL, LUFL, LULL
- **Meaning**: High/Medium/Low Usage/Load, Forecast/Load
- **Size**: ~17,000 timesteps
- **Split**: 70% train, 10% validation, 20% test (chronological)

#### Weather
- **Frequency**: 10-minute intervals
- **Columns**: 20 meteorological variables
  - Temperature (T), Dew point (Tdew), Potential temperature (Tpot)
  - Pressure (p), Humidity (rh), Vapor pressure (VP)
  - Wind (wv, wd), Rain, Solar radiation (SWDR, PAR)
- **Size**: ~50,000 timesteps
- **Split**: 70% train, 10% validation, 20% test (chronological)

### Chronos Dataset (Experiment 2 Only)

- **Purpose**: Learn universal vocabulary
- **Size**: 100,000 samples (randomly selected from 27M total)
- **Sources**: Mixed time series from multiple domains
  - Energy, traffic, weather, economics, healthcare
- **Frequencies**: Minute, hourly, daily, monthly, yearly, irregular
- **Selection Criteria**:
  - Chunk size: 100-1000 rows
  - Maximum 50% consecutive zeros
  - Random sampling across all domains

**Why Chronos?**
- Large, diverse collection ensures vocabulary generality
- Multiple domains test cross-domain generalization
- Realistic representation of real-world time series diversity

---

## Tokenization Pipeline

### 1. Standardization

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

- Applied independently to each column
- Scaler saved for inverse transformation
- Ensures comparability across columns

### 2. Discretization

**Simple Equal-Width Discretization**:

```python
bins = np.linspace(data.min(), data.max(), N_samples)
discretized = np.digitize(scaled_data, bins)
```

- Number of bins (N): 50, 100, or 200
- Bins computed from **training data only**
- Test data mapped to existing bins (values outside range clipped)

**Why equal-width?**
- Simple, reproducible
- No distributional assumptions
- Fair comparison across datasets

**Adaptive discretization** (tested but not main focus):
- Adjusts bin sizes based on data density
- May improve representation of non-uniform distributions

### 3. Token Pair Encoding (BPE)

**Merge-Based Compression**:

1. Start with base vocabulary (N discrete tokens)
2. Find most frequent adjacent token pair
3. Merge pair into new token
4. Repeat until vocabulary size = 600

**Example**:
```
Original: [10, 15, 10, 15, 10, 15, 20, 25]
Pair (10,15) is most frequent → merge to token 201
Result:   [201, 201, 201, 20, 25]
```

**Properties**:
- **Fixed vocabulary size** (600 tokens)
- **Variable compression rate** (depends on data)
- **Motif discovery**: Merged tokens represent recurring patterns

**Important Consequence**:
- BPE tokens may span multiple time steps
- **Temporal alignment is partially lost**
- Time of a motif-token is ambiguous

**Inspiration**: Similar to BPE in NLP (GPT models use fixed vocab)

### 4. Special Tokens (Optional)

**Temporal Markers**:

```python
# Insert <EBOS> every 24 hours
for i in range(0, len(data), 24):
    insert_token(data, i, EBOS_TOKEN)
```

- `<EBOS>`: End/Beginning of Sequence
- `<PAD>`: Padding (for combined CSV files)

**Frequencies**:
- **12h**: Every 12 hours (2 per day)
- **24h**: Every 24 hours (1 per day)
- **None**: No temporal markers

**Purpose**:
- Explicit temporal boundaries
- May help with daily/weekly patterns
- Test if model needs explicit time markers

---

## Model Configuration

All models use **identical hyperparameters** for fair comparison:

```python
seq_len = 192        # Input sequence length
label_len = 48       # Label length for decoder
pred_len = 15        # Prediction horizon
enc_in = 1           # Univariate (one column)
d_model = 512        # Model dimension
n_heads = 8          # Attention heads
e_layers = 2         # Encoder layers
d_layers = 1         # Decoder layers
d_ff = 2048          # Feed-forward dimension
dropout = 0.1        # Dropout rate
activation = 'gelu'  # Activation function
epochs = 100         # Maximum training epochs
patience = 3         # Early stopping patience
batch_size = 32      # Batch size
learning_rate = 0.0001  # Adam learning rate
```

**Why univariate?**
- Each column forecast independently
- Isolates tokenization effect on individual variables
- Simplifies interpretation

---

## Evaluation Metrics

### Primary Metrics

1. **MAE** (Mean Absolute Error)
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
   
2. **MSE** (Mean Squared Error)
   $$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   
3. **RMSE** (Root Mean Squared Error)
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   
4. **MAPE** (Mean Absolute Percentage Error)
   $$\text{MAPE} = \frac{100}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
   
5. **MSPE** (Mean Squared Percentage Error)
   $$\text{MSPE} = \frac{100}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2$$

### Two Evaluation Levels

#### Level 1: Before Detokenization

- Metrics calculated on **token-space** predictions
- Useful for model comparison
- **Limited physical interpretability**
- File: `metrics_before.csv`

#### Level 2: After Detokenization

- Tokens mapped back to numeric values using vocabulary
- Metrics calculated on **detokenized** predictions
- **Physically meaningful** comparison
- File: `metrics.csv`

**Detokenization Process**:

```python
# 1. Find nearest token for each predicted value
predicted_tokens = find_nearest_tokens(predictions, vocabulary)

# 2. Map tokens back to float values (midpoint of bin)
detokenized_values = tokens_to_floats(predicted_tokens, bin_edges)

# 3. Inverse standardization
original_scale = scaler.inverse_transform(detokenized_values)
```

**Important**: We do NOT compare detokenized values to original continuous values due to temporal misalignment from BPE.

---

## Data Splitting and Leakage Prevention

### Chronological Splits

All experiments use **chronological splits** (no shuffling):

```
|---------- 70% Train ----------|-- 10% Val --|---- 20% Test ----|
```

**Why chronological?**
- Realistic forecasting scenario
- Prevents temporal leakage
- Models can't "see the future"

### Tokenizer Training

**Experiment 1 (Column-Wise)**:
- Tokenizer learned from **training split only**
- Applied to validation and test (frozen)
- No leakage: tokenizer doesn't see future data

**Experiment 2 (Chronos)**:
- Tokenizer learned from **external Chronos dataset**
- Completely independent from target datasets
- No leakage: tokenizer is universal

**Baseline**:
- No tokenizer
- Only standardization (scaler fit on training)

---

## Methodological Considerations

### Loss of Temporal Alignment

**Problem**: BPE creates multi-step tokens

- Original: `[t₁, t₂, t₃, t₄, t₅, t₆]`
- After BPE: `[token_A, token_B]` where token_A = (t₁,t₂,t₃)

**Consequence**:
- Time of token_A is ambiguous (t₁? t₂? t₃?)
- Direct comparison to original continuous values difficult
- Token-space metrics have limited interpretability

**Solution**:
- Focus on **detokenized metrics** (Level 2)
- Use midpoint of bins for mapping
- Compare detokenized predictions to detokenized ground truth

### Fixed Vocabulary vs Fixed Compression

**Design Choice**: Fix vocabulary size (600), let compression vary

**Rationale**:
- Inspired by LLMs (GPT uses fixed vocab)
- More natural for Transformer models
- Different sequences compress differently

**Alternative**: Fix compression rate
- Would require variable vocabulary sizes
- Less common in NLP
- More complex implementation

### Column-Wise Complexity

**Challenge**: Large number of vocabularies

- ETTh1: 6 columns × 3 bins × 2 BPE × 3 temporal = **108 vocabularies**
- Weather: 20 columns × 3 bins × 2 BPE × 3 temporal = **360 vocabularies**

**Consequence**:
- Storage and management overhead
- Combinatorial explosion of experiments
- Computationally expensive

**Justification**:
- Tests hypothesis about specialized vocabularies
- Provides detailed per-column analysis
- Realistic for practical deployment

---

## Limitations

### 1. Univariate Setting
- Does not test multivariate dependencies
- Real-world forecasting often multivariate
- Future work: multivariate tokenization

### 2. Fixed Hyperparameters
- No hyperparameter tuning per tokenization strategy
- Ensures comparability but may not be optimal
- Some tokenization strategies might benefit from different architectures

### 3. Limited Datasets
- Only ETTh1 and Weather tested
- Results may not generalize to other domains
- Future work: more diverse datasets

### 4. Simple Discretization
- Equal-width bins may not be optimal
- Adaptive discretization tested but not main focus
- Other discretization methods possible

### 5. BPE Greedy Merging
- Greedy algorithm may not find optimal merges
- Alternative: learned embeddings
- Future work: neural tokenization

---

## Expected Outcomes

### Prediction 1: Column-Wise Superior for Specialized Variables

- Column-wise vocabularies expected to outperform universal for ETTh1
- Reason: Each column has distinct patterns

### Prediction 2: Universal Better for Cross-Domain Transfer

- Chronos vocabulary may generalize better
- Reason: Learned from diverse domains

### Prediction 3: BPE Improves Long-Range Patterns

- BPE expected to improve forecasting by capturing motifs
- Reason: Multi-step tokens encode patterns

### Prediction 4: Temporal Tokens Help Daily Patterns

- 24h markers expected to improve performance
- Reason: Explicit daily boundaries

**These are empirically tested - not assumed to hold.**

---

## Contribution

1. **Comprehensive comparison** of tokenization strategies for time series
2. **Novel application** of BPE to time series forecasting
3. **Universal vocabulary** approach using Chronos dataset
4. **Detailed analysis** of discretization, BPE, and temporal tokens
5. **Open-source implementation** for reproducibility

---

## Keywords

- Time Series Forecasting
- Tokenization
- Token Pair Encoding (BPE)
- Transformer Models
- Discrete Representations
- Informer
- Autoformer

---

## Paper Status (WCCI 2026)

- ✅ Methods section finalized
- ✅ Abstract finalized
- ✅ Hypotheses clearly stated
- ✅ Limitations explicitly documented
- ⏳ Results analysis ongoing
- ⏳ Discussion and interpretation
- ⏳ Reviewer-facing justification

---

## References

1. **Informer**: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting," AAAI 2021
2. **Autoformer**: Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting," NeurIPS 2021
3. **Chronos**: Ansari et al., "Chronos: Learning the Language of Time Series," arXiv 2024
4. **BPE**: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units," ACL 2016
