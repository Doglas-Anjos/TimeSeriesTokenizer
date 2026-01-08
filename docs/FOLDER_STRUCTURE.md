# Folder Structure

This document explains every folder and file in the project, their purpose, and relationship to the pipeline.

---

## Project Root

```
Project-tokenizer/
├── docs/                          # Documentation (you are here)
├── data/                          # Input and output datasets
├── utils/                         # Utility modules
├── chronos_csv/                   # Downloaded Chronos dataset
├── chronos_processed/             # Processed Chronos samples  
├── float_vocab/                   # Discretization vocabularies
├── model/                         # BPE models
├── scalers/                       # StandardScaler objects
├── results/                       # Raw experiment results (Exp 1)
├── results_chronos/               # Raw experiment results (Exp 2)
├── results_trues/                 # Raw baseline results
├── final_results/                 # Processed results (Exp 1)
├── final_results_chronos/         # Processed results (Exp 2)
├── final_results_trues/           # Processed baseline results
├── comparison_tables/             # Final comparison tables
├── tokenization_stats/            # Tokenization statistics
├── Images/                        # Visualizations and plots
├── vocab/                         # Legacy vocabulary files
└── [Python scripts]               # Processing scripts
```

---

## Detailed Folder Descriptions

### `data/`

**Purpose**: Input datasets and tokenized outputs

```
data/
├── datasets/                      # Original CSV files
│   ├── ETTh1.csv                  # Electricity Transformer (hourly)
│   ├── ETTh2.csv                  # ETT variant 2
│   ├── ETTm1.csv                  # ETT (15-minute)
│   ├── ETTm2.csv                  # ETT variant 2 (15-minute)
│   ├── weather.csv                # Weather measurements (10-min)
│   └── exchange_rate.csv          # Exchange rates
│
└── outputs/                       # Tokenized datasets (Experiment 1)
    ├── ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE.csv
    ├── ETTh1_disc_standard_simp_24h_N_Samp200_SEM_BPE.csv
    ├── ...
    │
    └── chronos_vocab/             # Tokenized datasets (Experiment 2)
        ├── ETTh1_chronos_vocab_N50_V1373_simple.csv
        ├── ETTh1_chronos_vocab_N50_V1373_simple_column_HUFL.csv
        └── ...
```

**File Naming Conventions**:

**Experiment 1** (column-wise):
```
{dataset}_{bpe}_{norm}_{disc}_{temporal}_N_Samp{N}_vocab_{V}_{bpe_status}.csv
└──┬───┘  └┬┘  └──┬─┘ └──┬┘  └───┬──┘        └┬┘        └┬┘  └────┬────┘
Dataset   BPE  Norm  Disc  Temporal         Bins      Vocab     BPE?
```

Examples:
- `ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE.csv` (with BPE)
- `ETTh1_disc_standard_simp_24h_N_Samp200_SEM_BPE.csv` (without BPE)

**Experiment 2** (Chronos):
```
{dataset}_chronos_vocab_N{N}_V{V}_{disc}.csv
```

Examples:
- `ETTh1_chronos_vocab_N50_V1373_simple.csv`

**Individual column files** (when `SAVE_INDIVIDUAL_COLUMNS = True`):
```
{base_filename}_column_{column_name}.csv
```

Examples:
- `ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE_column_HUFL.csv`

---

### `chronos_csv/`

**Purpose**: Downloaded Chronos dataset organized by column names

**Created by**: `download_chronos_to_csv.py`

```
chronos_csv/
├── column_Temperature/
│   ├── dataset1_chunk1.csv
│   ├── dataset2_chunk1.csv
│   └── ...
├── column_Traffic/
├── column_Energy/
└── ...
```

**Size**: ~200GB (27 million time series)

**Format**: Each folder contains all time series with that column name

**Note**: This folder can be deleted after processing to save space

---

### `chronos_processed/`

**Purpose**: Processed and sampled Chronos data for vocabulary training

**Created by**: `process_chronos_dataset.py`

```
chronos_processed/
├── chronos_combined_data.csv      # 100k selected samples
├── processing_report.txt          # Selection statistics
└── tokenization_report.txt        # Tokenization statistics
```

**chronos_combined_data.csv format**:
```csv
column_name,target,start,freq
Temperature,array([25.3, 25.1, ...]),2020-01-01,1H
Traffic,array([1234, 1245, ...]),2020-01-01,15min
```

**processing_report.txt example**:
```
Total files processed: 27,000,000
Files used: 15,432 (0.057%)
Total samples: 100,000
Average chunk size: 245 rows
Zero ratio: 12.3%
```

---

### `chronos_file_index.json`

**Purpose**: Index of which Chronos files were selected

**Created by**: `process_chronos_dataset.py`

**Format**:
```json
{
    "total_files": 27000000,
    "selected_files": 15432,
    "selection_rate": 0.00057,
    "files": [
        {
            "path": "chronos_csv/column_Temperature/file1.csv",
            "chunk_size": 234,
            "zero_ratio": 0.12
        },
        ...
    ]
}
```

**Purpose**: Allows reproducing exact dataset selection

---

### `float_vocab/`

**Purpose**: Discretization vocabularies (bin edges)

**Created by**: 
- `transform_files_into_tokens.py` (Experiment 1)
- `process_chronos_dataset.py` (Experiment 2 - chronos_*.fvocab)

```
float_vocab/
├── ETTh1_feature_Nsam_200_column_HUFL_simple_standard_24h.fvocab
├── chronos_N50_vocab1373_target_simple.fvocab
└── ...
```

**File format** (.fvocab):
```
-2.534
-2.123
-1.876
...
2.345
2.678
```

**Each line**: One bin edge (continuous float value)

**Usage**: 
1. Discretization: Map floats to bins during tokenization
2. Detokenization: Map tokens back to floats (bin midpoints)

**Naming conventions**:

**Experiment 1**:
```
{dataset}_feature_Nsam_{N}_column_{column}_{disc}_{norm}_{temporal}.fvocab
```

**Experiment 2** (Chronos):
```
chronos_N{N}_vocab{V}_target_{disc}.fvocab
```

---

### `model/`

**Purpose**: BPE (Byte Pair Encoding) models

**Created by**:
- `transform_files_into_tokens.py` (Experiment 1)
- `process_chronos_dataset.py` (Experiment 2)

```
model/
├── ETTh1_feature_Nsam_200_vocab_600_column_HUFL_simple_standard_24h.model
├── chronos_N50_vocab1373_target_simple.model
└── ...
```

**File format** (.model - pickled Python object):
Contains:
- Merge rules (which token pairs to merge)
- Vocabulary mapping
- Special tokens

**Usage**:
1. Encoding: Compress discretized sequences
2. Decoding: Expand compressed sequences back to base tokens

**Size**: Small (few KB per model)

---

### `scalers/`

**Purpose**: StandardScaler objects for normalization/denormalization

**Created by**:
- `transform_files_into_tokens.py` → `scalers/`
- `transform_with_chronos_vocab.py` → `scalers/chronos_vocab/`

```
scalers/
├── ETTh1_column_HUFL_standard.pkl
├── ETTh1_column_HULL_standard.pkl
├── ...
│
└── chronos_vocab/
    ├── ETTh1_N50_column_HUFL.pkl
    ├── wheather_N100_column_T_degC.pkl
    └── ...
```

**File format** (.pkl - pickled sklearn StandardScaler):
Contains:
- Mean (μ)
- Standard deviation (σ)

**Usage**:
```python
import joblib
scaler = joblib.load('scalers/ETTh1_column_HUFL_standard.pkl')

# Transform
scaled = scaler.transform(data)

# Inverse transform
original = scaler.inverse_transform(scaled)
```

**Critical for**: Detokenization process

---

### `results/` (and variants)

**Purpose**: Raw experiment results from model training

**Created by**: External training process (manual placement)

**Three variants**:
- `results/` - Experiment 1 (column-wise tokenization)
- `results_chronos/` - Experiment 2 (Chronos universal vocab)
- `results_trues/` - Baseline (no tokenization)

```
results/
├── ETTh1_HUFL_192_15_Informer_ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE/
│   ├── metrics.csv                # Errors before detokenization
│   ├── preds_results.csv          # Predictions (after inverse scaling)
│   └── trues_results.csv          # True values (after inverse scaling)
│
├── ETTh1_HUFL_192_15_Transformer_ETTh1_disc_standard_simp_12h_N_Samp100_SEM_BPE/
│   └── ...
└── ...
```

**Folder naming pattern**:
```
{dataset}_{column}_{seq}_{pred}_{model}_{tokenization_config}/
└───┬──┘ └──┬─┘  └┬┘  └┬┘  └──┬─┘  └──────────┬──────────┘
 Dataset  Column Seq Pred  Model         Config
```

Examples:
- `ETTh1_HUFL_192_15_Informer_ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE/`
- `weather_T_degC_192_15_Transformer_weather_chronos_vocab_N50_V1373_simple/`

**File contents**:

**metrics.csv** (before detokenization):
```csv
mae,mse,rmse,mape,mspe
0.523,0.412,0.642,15.23,12.45
```

**preds_results.csv** (predictions):
```csv
column_name,0,1,2,3,...
HUFL,5.234,5.187,5.312,5.401,...
```

**trues_results.csv** (ground truth):
```csv
column_name,0,1,2,3,...
HUFL,5.827,5.693,5.812,5.934,...
```

---

### `final_results/` (and variants)

**Purpose**: Processed results after detokenization

**Created by**: `process_all_results.py`

**Three variants**:
- `final_results/` - Experiment 1 processed
- `final_results_chronos/` - Experiment 2 processed
- `final_results_trues/` - Baseline processed

```
final_results/
├── ETTh1_HUFL_192_15_Informer_ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE/
│   ├── metrics.csv                    # NEW metrics after detokenization
│   ├── metrics_before.csv             # Original metrics (copy)
│   ├── preds_detokenized.csv          # Detokenized predictions
│   └── trues_detokenized.csv          # Detokenized true values
└── ...
```

**Key difference from `results/`**:
- Contains **detokenized** values
- New metrics calculated on detokenized data
- Physically meaningful comparison

**metrics.csv** (after detokenization):
```csv
column_name,MAE,MSE,RMSE,MAPE,MSPE
HUFL,0.487,0.365,0.604,14.52,11.23
```

**Why separate folders?**
- Preserves original results
- Allows re-detokenization with different parameters
- Clear separation of processing stages

---

### `comparison_tables/`

**Purpose**: Final comparison tables for analysis

**Created by**: `generate_comparison_tables.py`

```
comparison_tables/
├── ETTh1_Informer_MAE_metrics.csv         # Before detokenization
├── ETTh1_Informer_MAE_detokenized.csv     # After detokenization
├── ETTh1_Informer_MSE_metrics.csv
├── ETTh1_Informer_MSE_detokenized.csv
├── ETTh1_Informer_RMSE_metrics.csv
├── ETTh1_Informer_RMSE_detokenized.csv
├── ETTh1_Informer_MAPE_metrics.csv
├── ETTh1_Informer_MAPE_detokenized.csv
├── ETTh1_Informer_MSPE_metrics.csv
├── ETTh1_Informer_MSPE_detokenized.csv
├── ...
└── [60 total files: 2 datasets × 3 models × 5 metrics × 2 eval types]
```

**File structure**:

Rows: 22 experiment variations
Columns: Dataset columns (HUFL, HULL, MUFL, MULL, LUFL, LULL for ETTh1)

**Example** (`ETTh1_Informer_MAE_detokenized.csv`):

| Experiment | HUFL | HULL | MUFL | MULL | LUFL | LULL |
|------------|------|------|------|------|------|------|
| BPE_12h_N100 | 0.487 | 0.453 | 0.576 | 0.501 | 0.467 | 0.523 |
| BPE_12h_N200 | 0.465 | 0.431 | 0.554 | 0.479 | 0.445 | 0.501 |
| ... | ... | ... | ... | ... | ... | ... |
| Chronos_N100 | 0.534 | 0.498 | 0.623 | 0.548 | 0.512 | 0.569 |
| Baseline_NoToken | 0.456 | 0.422 | 0.545 | 0.470 | 0.436 | 0.492 |

**Usage**: Statistical analysis, visualization, paper figures

---

### `tokenization_stats/`

**Purpose**: Statistics about tokenization process

**Created by**: Various tokenization scripts

```
tokenization_stats/
├── compression_rates.csv          # BPE compression statistics
├── vocab_sizes.csv                # Vocabulary size distributions
└── ...
```

**Example** (`compression_rates.csv`):
```csv
dataset,column,n_samples,original_length,compressed_length,compression_rate
ETTh1,HUFL,200,8640,4320,2.00
ETTh1,HULL,200,8640,4285,2.02
```

---

### `Images/`

**Purpose**: Visualizations and plots

```
Images/
├── compression_analysis.png
├── error_comparison.png
├── tokenization_examples.png
└── ...
```

**Usage**: Paper figures, presentations

---

### `utils/`

**Purpose**: Utility modules used by main scripts

```
utils/
├── discretisize.py                # Discretization functions
├── tokenize.py                    # Tokenization utilities
├── basic.py                       # BasicTokenizer class
└── token_based.py                 # TokenBasedTokenizer (BPE)
```

**Key functions**:

`discretisize.py`:
- `simple_discretize()` - Equal-width discretization
- `adaptative_bins_discretize()` - Adaptive discretization
- `save_float_vocab()` - Save bin edges
- `load_float_vocab()` - Load bin edges

`token_based.py`:
- `TokenBasedTokenizer` class - BPE implementation
  - `.train()` - Learn BPE merges
  - `.encode()` - Compress sequence
  - `.decode()` - Decompress sequence
  - `.save()` / `.load()` - Persistence

---

### `vocab/` (Legacy)

**Purpose**: Old vocabulary files (may be deprecated)

**Note**: Replaced by `float_vocab/` - consider removing if unused

---

### `.git/` and `.idea/`

**Purpose**: Version control and IDE configuration

- `.git/` - Git repository
- `.idea/` - PyCharm/IntelliJ project files
- `.gitignore` - Git ignore rules

---

## File Size Estimates

| Folder | Typical Size | Notes |
|--------|--------------|-------|
| `data/datasets/` | ~100 MB | Original CSVs |
| `data/outputs/` | ~500 MB | Tokenized datasets |
| `chronos_csv/` | ~200 GB | Can delete after processing |
| `chronos_processed/` | ~50 MB | Compressed sample |
| `float_vocab/` | ~10 MB | Text files (bin edges) |
| `model/` | ~50 MB | Pickled BPE models |
| `scalers/` | ~5 MB | Pickled scalers |
| `results/` | ~1 GB | Depends on experiments |
| `final_results/` | ~1.5 GB | Includes detokenized |
| `comparison_tables/` | ~5 MB | CSV tables |

**Total** (excluding Chronos download): ~5-10 GB

---

## Cleanup Recommendations

### Can be safely deleted:

1. **`chronos_csv/`** - After processing (saves 200GB)
2. **`results/`** - After processing to `final_results/`
3. **`vocab/`** - If using `float_vocab/` instead
4. **`__pycache__/`** - Python cache (regenerates automatically)
5. **`.idea/`** - IDE files (if not using PyCharm)

### Should keep:

1. **`data/datasets/`** - Original data
2. **`float_vocab/`** - Required for detokenization
3. **`model/`** - Required for detokenization
4. **`scalers/`** - Required for detokenization
5. **`final_results/`** - Processed results
6. **`comparison_tables/`** - Final outputs
7. **`utils/`** - Core functionality

---

## Quick Reference

| Folder | Created By | Used By | Can Delete After |
|--------|------------|---------|------------------|
| `data/datasets/` | Manual | All tokenization scripts | ❌ Keep |
| `data/outputs/` | Tokenization scripts | Training | ❌ Keep |
| `chronos_csv/` | `download_chronos_to_csv.py` | `process_chronos_dataset.py` | ✅ After processing |
| `chronos_processed/` | `process_chronos_dataset.py` | `transform_with_chronos_vocab.py` | ❌ Keep |
| `float_vocab/` | Tokenization scripts | `process_all_results.py` | ❌ Keep (critical) |
| `model/` | Tokenization scripts | `process_all_results.py` | ❌ Keep (critical) |
| `scalers/` | Tokenization scripts | `process_all_results.py` | ❌ Keep (critical) |
| `results/` | Manual (training) | `process_all_results.py` | ✅ After processing |
| `final_results/` | `process_all_results.py` | `generate_comparison_tables.py` | ❌ Keep |
| `comparison_tables/` | `generate_comparison_tables.py` | Analysis | ❌ Keep |

---

See [WORKFLOW.md](WORKFLOW.md) for how these folders relate to the complete pipeline.
