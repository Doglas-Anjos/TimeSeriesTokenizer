# Time Series Tokenization Experiments

**Investigating tokenization strategies for time series forecasting with Transformer-based models.**

This project compares two main tokenization approaches:
1. **Column-Wise Tokenization**: Specialized vocabularies per variable
2. **Universal Chronos Vocabulary**: Single vocabulary learned from diverse time series

Both are compared against a continuous (non-tokenized) baseline.

---

## 📚 Documentation

Complete documentation is in the [`docs/`](docs/) folder:

- **[Getting Started](docs/README.md)** - Overview, quick start, project structure
- **[Workflow](docs/WORKFLOW.md)** - Complete pipeline with dataset flow visualization
- **[Experiment Design](docs/EXPERIMENT_DESIGN.md)** - Scientific design, hypotheses, methodology
- **[Folder Structure](docs/FOLDER_STRUCTURE.md)** - Detailed explanation of all folders and files
- **[Scripts Reference](docs/SCRIPTS_REFERENCE.md)** - Complete Python scripts documentation

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Tokenize datasets (Experiment 1: column-wise)
python transform_files_into_tokens.py

# Tokenize datasets (Experiment 2: Chronos universal - optional)
python download_chronos_to_csv.py      # One-time, ~200GB
python process_chronos_dataset.py       # Creates universal vocab
python transform_with_chronos_vocab.py  # Applies to datasets

# After training models (external)...

# Process results and detokenize
python process_all_results.py --all

# Generate comparison tables
python generate_comparison_tables.py
```

---

## 📁 Project Structure

```
Project-tokenizer/
├── docs/                          # 📚 Complete documentation
├── data/                          # Input datasets and tokenized outputs
├── utils/                         # Utility modules
├── float_vocab/                   # Discretization vocabularies
├── model/                         # BPE models
├── scalers/                       # StandardScaler objects
├── final_results/                 # Processed experiment results
├── comparison_tables/             # Final comparison tables
└── [Python scripts]               # Processing pipeline
```

---

## 🔬 Experiments

### Experiment 1: Column-Wise Tokenization
- Each column gets its own specialized vocabulary
- Tests: 50/100/200 bins, BPE on/off, temporal tokens 12h/24h/none
- Models: Informer, Transformer, Autoformer

### Experiment 2: Chronos Universal Vocabulary
- Single vocabulary learned from 100k diverse time series
- Cross-domain generalization
- Same models and configurations

### Baseline
- Continuous values (no tokenization)
- Isolates effect of tokenization

---

## 📊 Datasets

- **ETTh1**: Electricity Transformer Temperature (6 columns, hourly)
- **Weather**: Meteorological measurements (20 columns, 10-minute)
- **Chronos**: 100k samples from 27M time series (for universal vocab)

---

## 🛠️ Key Scripts

| Script | Purpose |
|--------|---------|
| `transform_files_into_tokens.py` | Create column-wise tokenized datasets |
| `transform_with_chronos_vocab.py` | Apply Chronos universal vocabulary |
| `process_chronos_dataset.py` | Create universal vocabulary from Chronos |
| `process_all_results.py` | Detokenize and calculate metrics |
| `generate_comparison_tables.py` | Generate final comparison tables |

See [Scripts Reference](docs/SCRIPTS_REFERENCE.md) for detailed documentation.

---

## 📈 Results

Final comparison tables in `comparison_tables/`:
- 60 CSV files total
- 2 datasets × 3 models × 5 metrics × 2 evaluation types
- Rows: 22 experiment variations
- Columns: Dataset-specific features

---

## 🎯 Research Questions

1. Do specialized (column-wise) vocabularies outperform universal vocabularies?
2. Does BPE compression improve forecasting by capturing temporal motifs?
3. Do explicit temporal tokens (12h/24h markers) help model performance?

---

## 📖 Citation

For WCCI 2026 submission.

```bibtex
@inproceedings{yourname2026tokenization,
  title={Tokenization Strategies for Time Series Forecasting with Transformers},
  author={Your Name},
  booktitle={2026 IEEE World Congress on Computational Intelligence (WCCI)},
  year={2026}
}
```

---

## 📝 License

[Specify your license here]

---

## 📧 Contact

[Your contact information]

---

**For complete details, see the [documentation](docs/).**
