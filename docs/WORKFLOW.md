# Complete Workflow: Dataset Movement Pipeline

This document describes the **complete data flow** through the tokenization experiments, from raw data to final comparison tables.

---

## Overview Diagram

```
RAW DATA
   ↓
TOKENIZATION (2 approaches)
   ↓
TRAINING (external)
   ↓
RESULTS COLLECTION
   ↓
DETOKENIZATION & METRICS
   ↓
COMPARISON TABLES
```

---

## Detailed Pipeline

### Phase 1: Data Preparation

#### 1.1 Original Datasets (Input)

**Location**: `data/datasets/`

Files:
- `ETTh1.csv` - Electricity Transformer Temperature (hourly)
- `weather.csv` - Weather measurements (10-minute)
- Other datasets: `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`, `exchange_rate.csv`

**Format**:
```csv
date,HUFL,HULL,MUFL,MULL,LUFL,LULL
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,4.203,1.340
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,4.142,1.371
...
```

#### 1.2 Chronos Dataset Download (Optional - Experiment 2 Only)

**Script**: `download_chronos_to_csv.py`

**Purpose**: Downloads Chronos dataset from Hugging Face (200GB+)

**Process**:
1. Downloads time series from Chronos repository
2. Organizes files by column names
3. Groups same-named columns into single files

**Output**: `chronos_csv/`

**Structure**:
```
chronos_csv/
├── column_Temperature/
│   ├── file1.csv
│   ├── file2.csv
│   └── ...
├── column_Traffic/
└── column_Energy/
```

**Usage**:
```bash
python download_chronos_to_csv.py
```

**Note**: This step requires significant disk space (~250GB) and bandwidth.

---

### Phase 2: Chronos Processing (Experiment 2 Only)

#### 2.1 Sample Selection

**Script**: `process_chronos_dataset.py`

**Purpose**: Select high-quality 100k samples from Chronos

**Selection Criteria**:
- Chunk size: 100-1000 rows
- Maximum 50% consecutive zeros
- Random sampling across all files

**Output Files**:
1. `chronos_processed/chronos_combined_data.csv` - 100k selected samples
2. `chronos_file_index.json` - Metadata about selected files
3. `chronos_processed/processing_report.txt` - Statistics
4. `chronos_processed/tokenization_report.txt` - Tokenization stats

**Usage**:
```bash
python process_chronos_dataset.py
```

**Sample Output**:
```
Processing Report:
- Total files processed: 27,000,000
- Files used: 15,432 (0.057%)
- Total samples: 100,000
- Average chunk size: 245 rows
```

---

### Phase 3: Tokenization

Two parallel approaches:

#### 3A. Experiment 1: Column-Wise Tokenization

**Script**: `transform_files_into_tokens.py`

**Approach**: Each column gets its own specialized vocabulary

**Configuration** (editable in script):
```python
USE_BPE = True                      # Enable/disable BPE
SAVE_INDIVIDUAL_COLUMNS = True      # Individual vs combined CSV
N_samples = [50, 100, 200]          # Discretization bins
total_vocab_size = 600              # Vocabulary size after BPE
```

**Process** (per dataset, per column):

1. **Load dataset**: Read `data/datasets/ETTh1.csv`
2. **Standardize**: Apply StandardScaler
   - Save scaler: `scalers/ETTh1_column_HUFL_standard.pkl`
3. **Discretize**: Convert to bins
   - Save vocab: `float_vocab/ETTh1_feature_Nsam_200_column_HUFL_simple_standard_24h.fvocab`
4. **Apply BPE** (if enabled): Compress tokens
   - Save model: `model/ETTh1_feature_Nsam_200_vocab_600_column_HUFL_simple_standard_24h.model`
5. **Save tokenized data**: 
   - Combined: `data/outputs/ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE.csv`
   - Individual: `data/outputs/ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE_column_HUFL.csv`

**Output Naming Pattern**:
```
{dataset}_bpe_{norm}_{disc}_{temporal}_N_Samp{N}_vocab_{V}_COM_BPE.csv
└─────┬─────┘  └──┬──┘ └──┬─┘  └───┬───┘      └─┬─┘        └─┬─┘  └──┬──┘
   Dataset     Norm  Disc  Temporal         Bins          Vocab   BPE?

Example: ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE.csv
```

**Variations Generated**:
- Temporal tokens: `12h`, `24h`, `sem_ebos` (none)
- Discretization bins: `50`, `100`, `200`
- BPE: `COM_BPE` (with) or `SEM_BPE` (without)

**Artifacts Created**:
- `data/outputs/` - 18 files per dataset (3 temporal × 3 bins × 2 BPE options)
- `float_vocab/` - Bin edges for each configuration
- `model/` - BPE models (if enabled)
- `scalers/` - StandardScaler objects

**Usage**:
```bash
python transform_files_into_tokens.py
```

---

#### 3B. Experiment 2: Universal Chronos Vocabulary

**Script**: `transform_with_chronos_vocab.py`

**Approach**: Use pre-trained universal vocabulary from Chronos

**Prerequisites**: 
- Chronos vocabularies in `float_vocab/chronos_*.fvocab`
- Created by `process_chronos_dataset.py`

**Configuration**:
```python
SAVE_INDIVIDUAL_COLUMNS = True      # Individual vs combined CSV
```

**Process**:

1. **Find Chronos vocabularies**: Scan `float_vocab/chronos_*.fvocab`
2. **For each target dataset** (ETTh1, weather):
   - Load dataset
   - Standardize each column
   - **Reuse Chronos bin edges** (no new vocab creation)
   - Apply Chronos BPE model
   - Save tokenized data

**Output Naming Pattern**:
```
{dataset}_chronos_vocab_N{N}_V{V}_{disc_type}.csv

Example: ETTh1_chronos_vocab_N50_V1373_simple.csv
```

**Output Location**: `data/outputs/chronos_vocab/`

**Key Difference from Experiment 1**:
- ❌ Does NOT create new vocabularies
- ✅ Reuses Chronos-learned vocabularies
- ✅ Same vocabulary applied to all columns
- ✅ Cross-domain generalization

**Usage**:
```bash
python transform_with_chronos_vocab.py
```

---

### Phase 4: Model Training (External Process)

**Location**: Outside this repository (Informer/Transformer training codebase)

**Input**: Tokenized CSV files from Phase 3
- Copy from `data/outputs/` to training directory
- Copy from `data/outputs/chronos_vocab/` to training directory

**Process**:
1. Load tokenized CSV
2. Train Informer/Transformer/Autoformer
3. Generate predictions on test set
4. Save results

**Output Files** (per experiment):
1. `metrics.csv` - Errors **before** detokenization
   ```csv
   mae,mse,rmse,mape,mspe
   0.523,0.412,0.642,15.23,12.45
   ```

2. `preds_results.csv` - Predictions **after** inverse StandardScaler
   ```csv
   column_name,0,1,2,3,4,...
   HUFL,5.234,5.187,5.312,5.401,5.289,...
   ```

3. `trues_results.csv` - True values **after** inverse StandardScaler
   ```csv
   column_name,0,1,2,3,4,...
   HUFL,5.827,5.693,5.812,5.934,5.723,...
   ```

**Important**: Models see tokenized data, but output continuous values after inverse scaling.

---

### Phase 5: Results Collection (Manual)

**Action**: Move experiment results to appropriate folders

**Folder Structure**:

```
results/                          # Experiment 1 (column-wise)
├── ETTh1_HUFL_192_15_Informer_ETTh1_bpe_standard_simp_24h_N_Samp200_vocab_600_COM_BPE/
│   ├── metrics.csv
│   ├── preds_results.csv
│   └── trues_results.csv
└── ...

results_chronos/                  # Experiment 2 (Chronos universal)
├── ETTh1_HUFL_192_15_Informer_ETTh1_chronos_vocab_N50_V1373_simple/
│   ├── metrics.csv
│   ├── preds_results.csv
│   └── trues_results.csv
└── ...

results_trues/                    # Baseline (no tokenization)
├── ETTh1_HUFL_192_15_Informer_custom_ftS_sl192_ll48_pl15/
│   ├── metrics.csv
│   ├── preds_results.csv
│   └── trues_results.csv
└── ...
```

**Folder Naming Patterns**:

**Experiment 1**:
```
{dataset}_{column}_{seq}_{pred}_{model}_{tokenization_config}/
```

**Experiment 2**:
```
{dataset}_{column}_{seq}_{pred}_{model}_chronos_vocab_N{N}_V{V}_{disc}/
```

**Baseline**:
```
{dataset}_{column}_{seq}_{pred}_{model}_custom_ftS_sl{seq}_ll{label}_pl{pred}/
```

---

### Phase 6: Detokenization & Metrics Calculation

**Script**: `process_all_results.py`

**Purpose**: 
1. Detokenize predictions and true values
2. Calculate metrics on detokenized data
3. Create processed result folders

**Process**:

For each experiment folder in `results/`, `results_chronos/`, `results_trues/`:

1. **Load files**:
   - `metrics.csv` (original errors)
   - `preds_results.csv` (predictions after inverse scaling)
   - `trues_results.csv` (true values after inverse scaling)

2. **Identify tokenization config** from folder name

3. **Load artifacts**:
   - Vocabulary: `float_vocab/{config}.fvocab`
   - BPE model: `model/{config}.model`
   - Scaler: `scalers/{dataset}_column_{column}_standard.pkl`

4. **Detokenization**:
   ```python
   # Predictions are continuous values, but came from tokenized sequences
   # We need to map back to original scale using vocabulary
   
   # Step 1: Find nearest token for each prediction
   tokenized_preds = find_nearest_tokens(preds, vocabulary)
   
   # Step 2: Map tokens back to float values
   detokenized_preds = tokens_to_floats(tokenized_preds, vocabulary)
   
   # Step 3: Inverse standardization
   original_scale_preds = scaler.inverse_transform(detokenized_preds)
   ```

5. **Calculate new metrics** on detokenized data:
   - MAE, MSE, RMSE, MAPE, MSPE
   - Between detokenized predictions and true values

6. **Save processed results**:
   - `final_results/{experiment}/metrics.csv` - New metrics after detokenization
   - `final_results/{experiment}/preds_detokenized.csv` - Detokenized predictions
   - `final_results/{experiment}/trues_detokenized.csv` - Detokenized true values
   - `final_results/{experiment}/metrics_before.csv` - Copy of original metrics

**Output Structure**:
```
final_results/
├── {experiment_1_folder}/
│   ├── metrics.csv              # NEW metrics after detokenization
│   ├── metrics_before.csv       # Original metrics
│   ├── preds_detokenized.csv    # Detokenized predictions
│   └── trues_detokenized.csv    # Detokenized true values
└── ...

final_results_chronos/           # Same structure for Experiment 2
final_results_trues/             # Same structure for baseline
```

**Usage**:
```bash
# Process all results
python process_all_results.py --all

# Process specific directory
python process_all_results.py --results-dir results --output-dir final_results

# Process single experiment
python process_all_results.py --experiment-path results/ETTh1_HUFL_...
```

**Key Outputs**:
- `final_results/` - Experiment 1 processed results
- `final_results_chronos/` - Experiment 2 processed results
- `final_results_trues/` - Baseline processed results

---

### Phase 7: Comparison Tables Generation

**Script**: `generate_comparison_tables.py`

**Purpose**: Create comprehensive comparison tables across all experiments

**Input**: Processed results from `final_results*/`

**Process**:

1. **Define experiment grid**:
   - 22 experiment configurations (21 tokenized + 1 baseline)
   - 2 datasets (ETTh1, weather)
   - 3 models (Informer, Transformer, Autoformer)
   - 5 metrics (MAE, MSE, RMSE, MAPE, MSPE)
   - 2 evaluation types (before/after detokenization)

2. **For each combination**:
   - Find matching experiment folders
   - Extract metric values for each dataset column
   - Create comparison table

3. **Table structure**:
   - **Rows**: 22 experiment variations
   - **Columns**: Dataset columns (HUFL, HULL, etc. for ETTh1)
   - **Values**: Metric values (e.g., MAE)

**Output**: `comparison_tables/`

**Files Generated** (60 total):
```
comparison_tables/
├── ETTh1_Informer_MAE_metrics.csv          # Original metrics
├── ETTh1_Informer_MAE_detokenized.csv      # After detokenization
├── ETTh1_Informer_MSE_metrics.csv
├── ETTh1_Informer_MSE_detokenized.csv
├── ...
├── weather_Transformer_RMSE_metrics.csv
└── weather_Autoformer_MAPE_detokenized.csv
```

**Usage**:
```bash
python generate_comparison_tables.py

# With custom paths
python generate_comparison_tables.py \
    --final-results-dir final_results \
    --output-dir comparison_tables
```

**Example Table** (`ETTh1_Informer_MAE_metrics.csv`):

| Experiment | HUFL | HULL | MUFL | MULL | LUFL | LULL |
|------------|------|------|------|------|------|------|
| BPE_12h_N100 | 0.523 | 0.487 | 0.612 | 0.534 | 0.498 | 0.556 |
| BPE_12h_N200 | 0.501 | 0.465 | 0.589 | 0.512 | 0.476 | 0.534 |
| ... | ... | ... | ... | ... | ... | ... |
| Chronos_N100 | 0.567 | 0.523 | 0.645 | 0.578 | 0.534 | 0.589 |
| Baseline_NoToken | 0.489 | 0.453 | 0.576 | 0.501 | 0.467 | 0.523 |

---

## Complete Dataset Flow Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATASETS                              │
│  data/datasets/ETTh1.csv, weather.csv                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ├─────────────────────────────────────────────────┐
                 │                                                  │
                 ▼                                                  ▼
    ┌────────────────────────────┐              ┌────────────────────────────┐
    │  EXPERIMENT 1: COLUMN-WISE │              │  EXPERIMENT 2: CHRONOS     │
    │  transform_files_into_     │              │  transform_with_chronos_   │
    │  tokens.py                 │              │  vocab.py                  │
    └────────────┬───────────────┘              └────────────┬───────────────┘
                 │                                            │
                 │  Creates:                                  │  Reuses:
                 │  - Per-column vocabularies                │  - Chronos vocabularies
                 │  - Per-column BPE models                  │  - Chronos BPE models
                 │  - Per-column scalers                     │  - New scalers
                 │                                            │
                 ▼                                            ▼
    ┌────────────────────────────┐              ┌────────────────────────────┐
    │  data/outputs/             │              │  data/outputs/             │
    │  - *_COM_BPE.csv           │              │  chronos_vocab/            │
    │  - *_SEM_BPE.csv           │              │  - *_chronos_vocab_*.csv   │
    └────────────┬───────────────┘              └────────────┬───────────────┘
                 │                                            │
                 │ (Manual: copy to training directory)       │
                 │                                            │
                 ▼                                            ▼
    ┌────────────────────────────────────────────────────────────┐
    │              MODEL TRAINING (External)                      │
    │  Informer / Transformer / Autoformer                       │
    └────────────┬───────────────────────────────────────────────┘
                 │
                 │  Generates per experiment:
                 │  - metrics.csv (before detokenization)
                 │  - preds_results.csv
                 │  - trues_results.csv
                 │
                 ▼
    ┌────────────────────────────────────────────────────────────┐
    │              RESULTS COLLECTION (Manual)                    │
    │  results/ | results_chronos/ | results_trues/             │
    └────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────────────────────┐
    │         DETOKENIZATION & METRICS CALCULATION                │
    │         process_all_results.py                             │
    └────────────┬───────────────────────────────────────────────┘
                 │
                 │  Creates:
                 │  - Detokenized predictions
                 │  - Detokenized true values
                 │  - New metrics on detokenized data
                 │
                 ▼
    ┌────────────────────────────────────────────────────────────┐
    │              PROCESSED RESULTS                              │
    │  final_results/ | final_results_chronos/ |                │
    │  final_results_trues/                                      │
    └────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────────────────────┐
    │         COMPARISON TABLES GENERATION                        │
    │         generate_comparison_tables.py                      │
    └────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────────────────────┐
    │              FINAL COMPARISON TABLES                        │
    │  comparison_tables/                                        │
    │  - {dataset}_{model}_{metric}_{eval_type}.csv             │
    │                                                            │
    │  60 CSV files total (2 datasets × 3 models ×              │
    │  5 metrics × 2 eval types)                                │
    └────────────────────────────────────────────────────────────┘
```

---

## Summary

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| 1 | `download_chronos_to_csv.py` | Hugging Face | `chronos_csv/` |
| 2 | `process_chronos_dataset.py` | `chronos_csv/` | `chronos_processed/`, `chronos_file_index.json` |
| 3A | `transform_files_into_tokens.py` | `data/datasets/` | `data/outputs/`, `float_vocab/`, `model/`, `scalers/` |
| 3B | `transform_with_chronos_vocab.py` | `data/datasets/`, `float_vocab/chronos_*` | `data/outputs/chronos_vocab/`, `scalers/chronos_vocab/` |
| 4 | External training | `data/outputs/` | `metrics.csv`, `preds_results.csv`, `trues_results.csv` |
| 5 | Manual collection | Training outputs | `results/`, `results_chronos/`, `results_trues/` |
| 6 | `process_all_results.py` | `results*/` | `final_results*/` |
| 7 | `generate_comparison_tables.py` | `final_results*/` | `comparison_tables/` |

---

## Next Steps

See [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md) for detailed documentation of each script's parameters and options.
