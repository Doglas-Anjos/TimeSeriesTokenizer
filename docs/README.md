# Time Series Tokenization Experiments

## Overview

This project investigates **tokenization strategies for time series forecasting** using Transformer-based models (Informer, Transformer, Autoformer). The work compares two main approaches:

1. **Experiment 1 (Column-Wise)**: Each dataset column has its own specialized vocabulary
2. **Experiment 2 (Chronos Universal)**: A single universal vocabulary learned from the Chronos dataset

Both experiments are compared against a **continuous (non-tokenized) baseline** to isolate the effect of tokenization.

---

## Quick Start

### Prerequisites

- Python 3.8+
- ~250GB disk space (if downloading Chronos dataset)
- Required packages: see `requirements.txt`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Workflow

```bash
# 1. (Optional) Download Chronos dataset for universal vocabulary
python download_chronos_to_csv.py

# 2. (Optional) Process Chronos dataset to create universal vocabulary
python process_chronos_dataset.py

# 3. Generate tokenized datasets - Experiment 1 (column-wise)
python transform_files_into_tokens.py

# 4. Generate tokenized datasets - Experiment 2 (universal Chronos vocab)
python transform_with_chronos_vocab.py

# 5. Train models (external process - see WORKFLOW.md)

# 6. Process results and perform detokenization
python process_all_results.py --all

# 7. Generate final comparison tables
python generate_comparison_tables.py
```

---

## Project Structure

```
Project-tokenizer/
├── docs/                          # Documentation
│   ├── README.md                  # This file
│   ├── WORKFLOW.md                # Detailed pipeline
│   ├── EXPERIMENT_DESIGN.md       # Experiment details
│   ├── FOLDER_STRUCTURE.md        # Folders explained
│   └── SCRIPTS_REFERENCE.md       # Script documentation
│
├── data/                          # Input datasets
│   ├── datasets/                  # Original CSV files (ETTh1, weather)
│   └── outputs/                   # Tokenized datasets
│       └── chronos_vocab/         # Chronos-based tokenized data
│
├── chronos_csv/                   # Downloaded Chronos dataset
├── chronos_processed/             # Processed Chronos samples
│
├── float_vocab/                   # Discretization vocabularies (.fvocab)
├── model/                         # BPE models (.model)
├── scalers/                       # StandardScaler objects (.pkl)
│
├── results/                       # Raw experiment results (Exp 1)
├── results_chronos/               # Raw experiment results (Exp 2)
├── results_trues/                 # Raw baseline results
│
├── final_results/                 # Processed results (Exp 1)
├── final_results_chronos/         # Processed results (Exp 2)
├── final_results_trues/           # Processed baseline results
│
├── comparison_tables/             # Final comparison tables
│
└── utils/                         # Utility modules
```

---

## Key Concepts

### Tokenization Pipeline

1. **Standardization**: Apply StandardScaler to each column
2. **Discretization**: Convert continuous values to discrete bins (50, 100, or 200 bins)
3. **BPE Encoding** (optional): Apply Byte Pair Encoding to compress token sequences
4. **Special Tokens**: Add temporal markers (`<EBOS>`) every 12h or 24h (optional)

### Two Experiments

| Aspect | Experiment 1 (Column-Wise) | Experiment 2 (Chronos Universal) |
|--------|---------------------------|----------------------------------|
| **Vocabulary** | Per-column (HUFL, HULL, etc.) | Universal (all time series) |
| **Training Data** | Target dataset only | Chronos (100k samples) |
| **Generalization** | Dataset-specific | Cross-domain |
| **Files** | `transform_files_into_tokens.py` | `transform_with_chronos_vocab.py` |

### Variations Tested

For each experiment, we test:
- **Discretization bins**: 50, 100, 200
- **BPE**: Enabled / Disabled
- **Temporal tokens**: 12h / 24h / None
- **Models**: Informer, Transformer, Autoformer
- **Datasets**: ETTh1, Weather

**Total configurations**: ~22 per dataset per model = ~132 experiments

---

## Datasets

### Target Forecasting Datasets

- **ETTh1**: Electricity Transformer Temperature (hourly)
  - Columns: HUFL, HULL, MUFL, MULL, LUFL, LULL
  - Location: `data/datasets/ETTh1.csv`

- **Weather**: Meteorological measurements (10-minute intervals)
  - 20 columns: temperature, pressure, humidity, wind, radiation, etc.
  - Location: `data/datasets/weather.csv`

### Chronos Dataset (Experiment 2 Only)

- **Size**: 200GB+ (27 million time series)
- **Source**: [Hugging Face - AutonLab/Chronos](https://huggingface.co/datasets/AutonLab/Chronos)
- **Frequencies**: Minute, hourly, daily, monthly, yearly, irregular
- **Purpose**: Learn universal vocabulary for cross-domain tokenization
- **Sample Size**: 100k sequences randomly selected
- **Location**: `chronos_csv/` → `chronos_processed/`

---

## Models

All experiments use three Transformer-based forecasting models:

1. **Informer** - Efficient attention mechanism
2. **Transformer** - Standard transformer architecture
3. **Autoformer** - Decomposition-based forecasting

### Fixed Hyperparameters

```python
seq_len = 192        # Input sequence length
label_len = 48       # Label length for decoder
pred_len = 15        # Prediction horizon
enc_in = 1           # Univariate (one column at a time)
d_model = 512        # Model dimension
n_heads = 8          # Attention heads
e_layers = 2         # Encoder layers
d_layers = 1         # Decoder layers
d_ff = 2048          # Feed-forward dimension
epochs = 100         # Training epochs
patience = 3         # Early stopping patience
```

---

## Output Files

### After Tokenization

- `data/outputs/*.csv` - Tokenized datasets (Experiment 1)
- `data/outputs/chronos_vocab/*.csv` - Tokenized datasets (Experiment 2)
- `float_vocab/*.fvocab` - Bin edges for discretization
- `model/*.model` - BPE merge rules
- `scalers/*.pkl` - StandardScaler objects for inverse transform

### After Training (Manual Step)

Place model outputs in:
- `results/` - Experiment 1 results
- `results_chronos/` - Experiment 2 results
- `results_trues/` - Baseline results

Each folder should contain:
- `metrics.csv` - Errors before detokenization (MAE, MSE, RMSE, MAPE, MSPE)
- `preds_results.csv` - Predicted values (after inverse transform)
- `trues_results.csv` - True values (after inverse transform)

### After Processing

- `final_results/` - Detokenized metrics (Experiment 1)
- `final_results_chronos/` - Detokenized metrics (Experiment 2)
- `final_results_trues/` - Baseline metrics
- `comparison_tables/*.csv` - Final comparison tables

---

## Documentation

- **[WORKFLOW.md](WORKFLOW.md)** - Complete step-by-step pipeline with dataset flow
- **[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)** - Detailed experimental design and hypotheses
- **[FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)** - Comprehensive folder and file descriptions
- **[SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md)** - Detailed documentation of all Python scripts

---

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{yourname2026tokenization,
  title={Tokenization Strategies for Time Series Forecasting with Transformers},
  author={Your Name},
  booktitle={2026 IEEE World Congress on Computational Intelligence (WCCI)},
  year={2026}
}
```

---

## License

[Specify your license here]

---

## Contact

[Your contact information]
