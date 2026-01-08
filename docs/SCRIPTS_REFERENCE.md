# Scripts Reference

Detailed documentation for all Python scripts in the project.

---

## Table of Contents

1. [download_chronos_to_csv.py](#download_chronos_to_csvpy)
2. [process_chronos_dataset.py](#process_chronos_datasetpy)
3. [transform_files_into_tokens.py](#transform_files_into_tokenspy)
4. [transform_with_chronos_vocab.py](#transform_with_chronos_vocabpy)
5. [process_all_results.py](#process_all_resultspy)
6. [generate_comparison_tables.py](#generate_comparison_tablespy)
7. [Utility Modules](#utility-modules)

---

## download_chronos_to_csv.py

### Purpose

Downloads the Chronos dataset from Hugging Face and organizes files by column names.

### Requirements

- Hugging Face account (for large downloads)
- ~250GB free disk space
- Good internet connection

### Usage

```bash
python download_chronos_to_csv.py
```

### Process

1. **Connects to Hugging Face**: Accesses Chronos dataset repository
2. **Scans all files**: Iterates through 27 million time series
3. **Groups by column**: Identifies column names and groups same-named columns
4. **Downloads and saves**: Saves grouped files to `chronos_csv/`

### Output

```
chronos_csv/
├── column_Temperature/
│   ├── file1.csv
│   ├── file2.csv
│   └── ...
├── column_Traffic/
├── column_Energy/
└── ...
```

### Configuration

Edit inside script:
```python
OUTPUT_DIR = 'chronos_csv/'        # Output directory
BATCH_SIZE = 1000                   # Files per batch
```

### Notes

- **Very long running**: May take days depending on connection
- **Can be interrupted**: Resumes from last downloaded file
- **Optional**: Not required for Experiment 1 (column-wise)
- **One-time operation**: Only needed once per machine

### Troubleshooting

**Issue**: Download timeout
**Solution**: Script auto-retries, just restart if interrupted

**Issue**: Out of disk space
**Solution**: Increase disk space or reduce `BATCH_SIZE`

---

## process_chronos_dataset.py

### Purpose

Selects high-quality 100k samples from Chronos for universal vocabulary training.

### Prerequisites

- `chronos_csv/` folder (from `download_chronos_to_csv.py`)

### Usage

```bash
python process_chronos_dataset.py
```

### Process

1. **Scan files**: Iterate through all Chronos CSV files
2. **Quality filtering**:
   - Chunk size: 100-1000 rows
   - Maximum 50% consecutive zeros
   - Valid numeric data
3. **Random sampling**: Select until 100k samples reached
4. **Save dataset**: `chronos_processed/chronos_combined_data.csv`
5. **Generate statistics**: Processing and tokenization reports
6. **Train universal vocabulary**: Create Chronos BPE models

### Output Files

1. **chronos_combined_data.csv**: 100k selected samples
   ```csv
   column_name,target,start,freq
   Temperature,array([25.3, 25.1, ...]),2020-01-01,1H
   Traffic,array([1234, 1245, ...]),2020-01-01,15min
   ```

2. **chronos_file_index.json**: Metadata about selected files
   ```json
   {
       "total_files": 27000000,
       "selected_files": 15432,
       "files": [...]
   }
   ```

3. **processing_report.txt**: Selection statistics
   ```
   Total files processed: 27,000,000
   Files used: 15,432 (0.057%)
   Total samples: 100,000
   ```

4. **tokenization_report.txt**: Tokenization statistics

5. **Vocabularies**: `float_vocab/chronos_*.fvocab`

6. **BPE models**: `model/chronos_*.model`

### Configuration

Edit inside script:
```python
TARGET_SAMPLES = 100000             # Number of samples to collect
MIN_CHUNK_SIZE = 100                # Minimum chunk length
MAX_CHUNK_SIZE = 1000               # Maximum chunk length
MAX_ZERO_RATIO = 0.50               # Maximum 50% consecutive zeros
N_SAMPLES = [50, 100, 200]          # Discretization bins
TOTAL_VOCAB_SIZE = 600              # BPE vocabulary size
```

### Notes

- **Runs once**: Generates universal vocabularies for all experiments
- **Reproducible**: Set random seed for consistent sampling
- **Quality control**: Filters ensure meaningful patterns

### Troubleshooting

**Issue**: "Not enough valid samples"
**Solution**: Reduce MIN_CHUNK_SIZE or increase MAX_ZERO_RATIO

**Issue**: Memory error
**Solution**: Reduce TARGET_SAMPLES or process in batches

---

## transform_files_into_tokens.py

### Purpose

Create column-wise tokenized datasets (Experiment 1).

### Prerequisites

- Original datasets in `data/datasets/`

### Usage

```bash
python transform_files_into_tokens.py
```

### Configuration Flags

Edit at top of script:

```python
# BPE toggle
USE_BPE = True                      # True: apply BPE, False: only discretization

# Output format toggle
SAVE_INDIVIDUAL_COLUMNS = True      # True: one file per column
                                     # False: combined CSV with all columns

# Sampling configurations
list_of_samples = [50, 100, 200]    # Discretization bins to test

# Temporal tokens
hour_context_size_24h = 24          # Insert <EBOS> every 24 hours
hour_context_size_12h = 12          # Insert <EBOS> every 12 hours
# (sem_ebos = no temporal tokens)

# Vocabulary size
total_vocab_size = 600              # After BPE compression
```

### Process

For each dataset (ETTh1, weather):
  For each column (HUFL, HULL, ...):
    For each configuration (50/100/200 bins, BPE on/off, 12h/24h/no temporal):
      1. Load column data
      2. Standardize (StandardScaler)
      3. Save scaler → `scalers/{dataset}_column_{column}_standard.pkl`
      4. Discretize (map to bins)
      5. Save vocabulary → `float_vocab/{config}.fvocab`
      6. (Optional) Apply BPE
      7. Save BPE model → `model/{config}.model`
      8. Save tokenized data → `data/outputs/{config}.csv`

### Output Files

**Tokenized datasets**: `data/outputs/`
```
ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE.csv
ETTh1_disc_standard_simp_24h_N_Samp200_SEM_BPE.csv
...
```

**Or individual columns** (if `SAVE_INDIVIDUAL_COLUMNS = True`):
```
ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE_column_HUFL.csv
ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE_column_HULL.csv
...
```

**Vocabularies**: `float_vocab/`
```
ETTh1_feature_Nsam_200_column_HUFL_simple_standard_24h.fvocab
```

**BPE models**: `model/` (if USE_BPE = True)
```
ETTh1_feature_Nsam_200_vocab_600_column_HUFL_simple_standard_24h.model
```

**Scalers**: `scalers/`
```
ETTh1_column_HUFL_standard.pkl
```

### File Naming Convention

**With BPE**:
```
{dataset}_bpe_{norm}_{disc}_{temporal}_N_Samp{N}_vocab_{V}_COM_BPE.csv
```

**Without BPE**:
```
{dataset}_disc_{norm}_{disc}_{temporal}_N_Samp{N}_SEM_BPE.csv
```

Components:
- `{dataset}`: ETTh1, weather, etc.
- `{norm}`: standard, normal
- `{disc}`: simp (simple), adapt (adaptive)
- `{temporal}`: 12h, 24h, sem_ebos
- `{N}`: 50, 100, 200 (bins)
- `{V}`: 600 (vocab size after BPE)
- `COM_BPE`: with BPE
- `SEM_BPE`: without BPE (discretization only)

### Variations Generated

Per dataset:
- 3 discretization bins × 2 BPE options × 3 temporal options = **18 configurations**

Per column:
- 18 configurations × vocabularies/models/scalers

Total for ETTh1 (6 columns):
- 18 CSV files (if combined)
- 108 individual column files (if `SAVE_INDIVIDUAL_COLUMNS = True`)
- 108 vocabularies
- 54 BPE models
- 6 scalers

### Notes

- **Long running**: May take hours for all configurations
- **Disk intensive**: Generates many files
- **Parallelizable**: Can run different datasets separately

### Troubleshooting

**Issue**: "Memory error"
**Solution**: Set `SAVE_INDIVIDUAL_COLUMNS = True` to reduce memory

**Issue**: "File exists" error
**Solution**: Delete existing output files or rename

---

## transform_with_chronos_vocab.py

### Purpose

Create tokenized datasets using universal Chronos vocabulary (Experiment 2).

### Prerequisites

- Original datasets in `data/datasets/`
- Chronos vocabularies in `float_vocab/chronos_*.fvocab`
- Chronos BPE models in `model/chronos_*.model`

### Usage

```bash
python transform_with_chronos_vocab.py
```

### Configuration

```python
# Output format toggle
SAVE_INDIVIDUAL_COLUMNS = False     # True: one file per column
                                     # False: combined CSV

# Datasets to process
list_of_files = ['ETTh1', 'weather']
```

### Process

1. **Scan for Chronos vocabularies**: Find all `chronos_*.fvocab` files
2. **Extract configurations**: Parse N_samples and vocab_size from filenames
3. **For each dataset**:
   - For each column:
     - Standardize data
     - Save scaler → `scalers/chronos_vocab/{dataset}_N{N}_column_{column}.pkl`
     - **Reuse Chronos bins** (no new vocab creation)
     - Apply Chronos BPE model
     - Save tokenized data

### Key Difference from Experiment 1

| Aspect | Experiment 1 | Experiment 2 |
|--------|--------------|--------------|
| Vocabulary | Created per-column | Reused from Chronos |
| BPE model | Trained per-column | Reused from Chronos |
| Scaler | Created per-column | Created per-column (new) |
| Generalization | Dataset-specific | Cross-domain |

### Output Files

**Tokenized datasets**: `data/outputs/chronos_vocab/`
```
ETTh1_chronos_vocab_N50_V1373_simple.csv
ETTh1_chronos_vocab_N100_V1373_simple.csv
ETTh1_chronos_vocab_N200_V1373_simple.csv
```

**Or individual columns**:
```
ETTh1_chronos_vocab_N50_V1373_simple_column_HUFL.csv
ETTh1_chronos_vocab_N50_V1373_simple_column_HULL.csv
...
```

**Scalers**: `scalers/chronos_vocab/`
```
ETTh1_N50_column_HUFL.pkl
wheather_N100_column_T_degC.pkl
```

### File Naming Convention

```
{dataset}_chronos_vocab_N{N}_V{V}_{disc}.csv
```

Components:
- `{dataset}`: ETTh1, weather
- `{N}`: 50, 100, 200 (bins from Chronos vocab)
- `{V}`: Vocab size (extracted from Chronos model)
- `{disc}`: simple, adaptative

### Compression Statistics

Script prints:
```
Processing column: HUFL
  Simple: 8640 → 4320 tokens (compression: 2.00x)
  Adaptative: 8640 → 4100 tokens (compression: 2.11x)
```

### Notes

- **Fast**: No vocabulary training (reuses Chronos)
- **Requires Chronos processing**: Run `process_chronos_dataset.py` first
- **Universal approach**: Same vocab for all columns

### Troubleshooting

**Issue**: "No chronos_*.fvocab files found"
**Solution**: Run `process_chronos_dataset.py` first

**Issue**: "Model file not found"
**Solution**: Ensure Chronos models in `model/chronos_*.model`

---

## process_all_results.py

### Purpose

Process raw experiment results: detokenize predictions and calculate metrics.

### Prerequisites

- Raw results in `results/`, `results_chronos/`, `results_trues/`
- Vocabularies in `float_vocab/`
- BPE models in `model/`
- Scalers in `scalers/`

### Usage

```bash
# Process all results
python process_all_results.py --all

# Process specific directory
python process_all_results.py \
    --results-dir results \
    --output-dir final_results

# Process single experiment
python process_all_results.py \
    --experiment-path results/ETTh1_HUFL_192_15_Informer_...
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--all` | Process all three result directories | - |
| `--results-dir` | Input directory | `results` |
| `--output-dir` | Output directory | `final_results` |
| `--experiment-path` | Single experiment folder | - |

### Process

For each experiment folder:

1. **Identify experiment**:
   - Parse folder name
   - Extract: dataset, column, model, tokenization config

2. **Load files**:
   - `metrics.csv` (original metrics)
   - `preds_results.csv` (predictions)
   - `trues_results.csv` (true values)

3. **Load artifacts**:
   - Vocabulary: `float_vocab/{config}.fvocab`
   - BPE model: `model/{config}.model` (if BPE used)
   - Scaler: `scalers/{dataset}_column_{column}_standard.pkl`

4. **Detokenization**:
   ```python
   # Map predictions to nearest tokens
   pred_tokens = find_nearest_tokens(predictions, vocabulary)
   
   # Map tokens to float values (bin midpoints)
   detokenized_preds = tokens_to_floats(pred_tokens, bin_edges)
   
   # Inverse standardization
   original_scale_preds = scaler.inverse_transform(detokenized_preds)
   ```

5. **Calculate new metrics**:
   - MAE, MSE, RMSE, MAPE, MSPE
   - On detokenized predictions vs true values

6. **Save results**:
   - `final_results/{experiment}/metrics.csv` - New metrics
   - `final_results/{experiment}/metrics_before.csv` - Original metrics (copy)
   - `final_results/{experiment}/preds_detokenized.csv` - Detokenized preds
   - `final_results/{experiment}/trues_detokenized.csv` - Detokenized trues

### Output Structure

```
final_results/
├── {experiment_folder}/
│   ├── metrics.csv                # NEW: metrics after detokenization
│   ├── metrics_before.csv         # COPY: original metrics
│   ├── preds_detokenized.csv      # Detokenized predictions
│   └── trues_detokenized.csv      # Detokenized true values
└── ...
```

**metrics.csv format**:
```csv
column_name,MAE,MSE,RMSE,MAPE,MSPE
HUFL,0.487,0.365,0.604,14.52,11.23
```

### Important Notes

- **Detokenization is irreversible**: Discretization loses precision
- **Baseline experiments**: Skip detokenization (already continuous)
- **Preserves originals**: Original `metrics.csv` copied to `metrics_before.csv`

### Troubleshooting

**Issue**: "Vocabulary file not found"
**Solution**: Ensure `float_vocab/` contains matching .fvocab files

**Issue**: "Scaler file not found"
**Solution**: Run tokenization scripts first to generate scalers

**Issue**: "All values are constant (-inf error)"
**Solution**: Normal for some columns - indicates true values are constant

---

## generate_comparison_tables.py

### Purpose

Generate comprehensive comparison tables across all experiments.

### Prerequisites

- Processed results in `final_results/`, `final_results_chronos/`, `final_results_trues/`

### Usage

```bash
# Generate all tables
python generate_comparison_tables.py

# With custom paths
python generate_comparison_tables.py \
    --final-results-dir final_results \
    --output-dir comparison_tables
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--final-results-dir` | Input directory | `final_results` |
| `--output-dir` | Output directory | `comparison_tables` |

### Process

1. **Define experiment grid**:
   - 22 experiment configurations
   - 2 datasets (ETTh1, weather)
   - 3 models (Informer, Transformer, Autoformer)
   - 5 metrics (MAE, MSE, RMSE, MAPE, MSPE)
   - 2 evaluation types (metrics, detokenized)

2. **For each combination**:
   - Find matching experiment folders
   - Extract metric values for each column
   - Handle multi-variate (all columns in one folder) and single-variate (one folder per column)
   - Create comparison table

3. **Save tables**: `comparison_tables/{dataset}_{model}_{metric}_{eval_type}.csv`

### Output Files

**60 total files**:
```
comparison_tables/
├── ETTh1_Informer_MAE_metrics.csv
├── ETTh1_Informer_MAE_detokenized.csv
├── ETTh1_Informer_MSE_metrics.csv
├── ETTh1_Informer_MSE_detokenized.csv
├── ETTh1_Informer_RMSE_metrics.csv
├── ...
└── weather_Autoformer_MSPE_detokenized.csv
```

### Table Structure

**Rows**: 22 experiment variations
**Columns**: Dataset columns

**22 Experiments**:
1. BPE_12h_N100
2. BPE_12h_N200
3. BPE_12h_N50
4. BPE_24h_N100
5. BPE_24h_N200
6. BPE_24h_N50
7. BPE_NoTemp_N100
8. BPE_NoTemp_N200
9. BPE_NoTemp_N50
10. NoBPE_12h_N100
11. NoBPE_12h_N200
12. NoBPE_12h_N50
13. NoBPE_24h_N100
14. NoBPE_24h_N200
15. NoBPE_24h_N50
16. NoBPE_NoTemp_N100
17. NoBPE_NoTemp_N200
18. NoBPE_NoTemp_N50
19. Chronos_N100
20. Chronos_N200
21. Chronos_N50
22. Baseline_NoToken

### Example Table

`ETTh1_Informer_MAE_detokenized.csv`:

| Experiment | HUFL | HULL | MUFL | MULL | LUFL | LULL |
|------------|------|------|------|------|------|------|
| BPE_12h_N100 | 0.487 | 0.453 | 0.576 | 0.501 | 0.467 | 0.523 |
| Chronos_N100 | 0.534 | 0.498 | 0.623 | 0.548 | 0.512 | 0.569 |
| Baseline_NoToken | 0.456 | 0.422 | 0.545 | 0.470 | 0.436 | 0.492 |

### Special Values

- **NaN**: Experiment or file not found
- **-inf**: True values are constant (metric undefined)

### Notes

- **Handles both formats**: Multi-variate and single-variate experiments
- **Two evaluation types**: Before (`metrics`) and after (`detokenized`) detokenization
- **Ready for analysis**: CSV format for Excel, Python, R

### Troubleshooting

**Issue**: "Blank tables (all NaN)"
**Solution**: Check folder naming matches expected patterns

**Issue**: "Directory does not exist"
**Solution**: Run `process_all_results.py` first

---

## Utility Modules

### utils/discretisize.py

**Functions**:

```python
def simple_discretize(data, N_samples, data_st=None, special_tokens=None):
    """
    Equal-width discretization.
    
    Args:
        data: Continuous values
        N_samples: Number of bins
        data_st: Special token positions
        special_tokens: Special token mapping
    
    Returns:
        discretized: Discrete tokens
        bin_edges: Bin boundaries
    """

def adaptative_bins_discretize(data, N, K, data_st=None, special_tokens=None):
    """
    Adaptive discretization based on data density.
    
    Args:
        data: Continuous values
        N: Number of bins
        K: Division factor
        data_st: Special token positions
        special_tokens: Special token mapping
    
    Returns:
        bin_edges: Bin boundaries
        discretized: Discrete tokens
        allocation: Bin allocation info
    """

def save_float_vocab(bin_edges, filename):
    """Save bin edges to .fvocab file."""

def load_float_vocab(filename):
    """Load bin edges from .fvocab file."""
```

### utils/token_based.py

**Class**: `TokenBasedTokenizer`

```python
class TokenBasedTokenizer:
    """Byte Pair Encoding (BPE) tokenizer for pre-discretized sequences."""
    
    def __init__(self, N_samples, vocab_file, special_tokens):
        """
        Args:
            N_samples: Base vocabulary size
            vocab_file: Path to .fvocab file
            special_tokens: Dict of special tokens
        """
    
    def train(self, data_list, total_vocab_size, verbose=False):
        """
        Learn BPE merges.
        
        Args:
            data_list: List of discrete token sequences
            total_vocab_size: Target vocabulary size
            verbose: Print progress
        """
    
    def encode(self, tokens):
        """
        Compress token sequence using learned merges.
        
        Args:
            tokens: List of base tokens
        
        Returns:
            compressed: List of compressed tokens
        """
    
    def decode(self, compressed_tokens):
        """
        Decompress token sequence.
        
        Args:
            compressed_tokens: List of compressed tokens
        
        Returns:
            tokens: List of base tokens
        """
    
    def save(self, model_name, vocab_file):
        """Save model to .model file."""
    
    def load(self, model_path):
        """Load model from .model file."""
```

### utils/tokenize.py

**Functions**:

```python
def mark_special_tokens(df, special_tokens, hour_toks, data_freq):
    """
    Mark temporal boundaries with special tokens.
    
    Args:
        df: DataFrame with time series
        special_tokens: Dict of special tokens
        hour_toks: Hours per marker (12 or 24)
        data_freq: Data frequency ('1H', '10min', etc.)
    
    Returns:
        df_marked: DataFrame with special token markers
    """
```

---

## Script Execution Order

**Typical Workflow**:

1. `download_chronos_to_csv.py` (optional, one-time)
2. `process_chronos_dataset.py` (optional, for Experiment 2)
3. `transform_files_into_tokens.py` (Experiment 1)
4. `transform_with_chronos_vocab.py` (Experiment 2, optional)
5. **Training** (external)
6. **Move results** (manual)
7. `process_all_results.py --all`
8. `generate_comparison_tables.py`

---

## Common Workflows

### Experiment 1 Only (Column-Wise)

```bash
# 1. Tokenize datasets
python transform_files_into_tokens.py

# 2. Train models (external)
# Copy data/outputs/*.csv to training directory

# 3. Process results
python process_all_results.py --all

# 4. Generate tables
python generate_comparison_tables.py
```

### Experiment 2 Only (Chronos)

```bash
# 1. Download Chronos (one-time)
python download_chronos_to_csv.py

# 2. Process Chronos
python process_chronos_dataset.py

# 3. Tokenize datasets with Chronos vocab
python transform_with_chronos_vocab.py

# 4. Train models (external)
# Copy data/outputs/chronos_vocab/*.csv to training directory

# 5. Process results
python process_all_results.py --all

# 6. Generate tables
python generate_comparison_tables.py
```

### Both Experiments

```bash
# 1. Chronos setup (one-time)
python download_chronos_to_csv.py
python process_chronos_dataset.py

# 2. Tokenize with both approaches
python transform_files_into_tokens.py
python transform_with_chronos_vocab.py

# 3. Train all models (external)

# 4. Process all results
python process_all_results.py --all

# 5. Generate comparison tables
python generate_comparison_tables.py
```

---

See [WORKFLOW.md](WORKFLOW.md) for complete pipeline visualization.
