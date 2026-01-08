"""
Generate comparison tables for tokenization experiments.

This script creates 60 CSV tables (30 for ETTh1 + 30 for weather):
- 2 datasets × 3 models × 5 metrics × 2 evaluation types = 60 tables

Each table has:
- Rows: 22 experiment variations (21 tokenization + 1 baseline)
- Columns: Dataset-specific columns (HUFL, HULL, MUFL, MULL, LUFL, LULL for ETTh1)
                                      (20 columns for weather)

Output structure:
- comparison_tables/
  - ETTh1_Transformer_MAE_metrics.csv
  - ETTh1_Transformer_MAE_detokenized.csv
  - weather_Informer_RMSE_metrics.csv
  - ...
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Experiment configurations: 21 tokenization experiments + 1 baseline = 22 total
EXPERIMENT_CONFIGS = [
    # BPE + 12h temporal tokens
    {'bpe': True, 'temporal': '12h', 'n_samples': 100, 'vocab': 600, 'name': 'BPE_12h_N100'},
    {'bpe': True, 'temporal': '12h', 'n_samples': 200, 'vocab': 600, 'name': 'BPE_12h_N200'},
    {'bpe': True, 'temporal': '12h', 'n_samples': 50, 'vocab': 600, 'name': 'BPE_12h_N50'},
    
    # BPE + 24h temporal tokens
    {'bpe': True, 'temporal': '24h', 'n_samples': 100, 'vocab': 600, 'name': 'BPE_24h_N100'},
    {'bpe': True, 'temporal': '24h', 'n_samples': 200, 'vocab': 600, 'name': 'BPE_24h_N200'},
    {'bpe': True, 'temporal': '24h', 'n_samples': 50, 'vocab': 600, 'name': 'BPE_24h_N50'},
    
    # BPE + no temporal tokens
    {'bpe': True, 'temporal': 'sem_ebos', 'n_samples': 100, 'vocab': 600, 'name': 'BPE_NoTemp_N100'},
    {'bpe': True, 'temporal': 'sem_ebos', 'n_samples': 200, 'vocab': 600, 'name': 'BPE_NoTemp_N200'},
    {'bpe': True, 'temporal': 'sem_ebos', 'n_samples': 50, 'vocab': 600, 'name': 'BPE_NoTemp_N50'},
    
    # No BPE + 12h temporal tokens
    {'bpe': False, 'temporal': '12h', 'n_samples': 100, 'vocab': None, 'name': 'NoBPE_12h_N100'},
    {'bpe': False, 'temporal': '12h', 'n_samples': 200, 'vocab': None, 'name': 'NoBPE_12h_N200'},
    {'bpe': False, 'temporal': '12h', 'n_samples': 50, 'vocab': None, 'name': 'NoBPE_12h_N50'},
    
    # No BPE + 24h temporal tokens
    {'bpe': False, 'temporal': '24h', 'n_samples': 100, 'vocab': None, 'name': 'NoBPE_24h_N100'},
    {'bpe': False, 'temporal': '24h', 'n_samples': 200, 'vocab': None, 'name': 'NoBPE_24h_N200'},
    {'bpe': False, 'temporal': '24h', 'n_samples': 50, 'vocab': None, 'name': 'NoBPE_24h_N50'},
    
    # No BPE + no temporal tokens
    {'bpe': False, 'temporal': 'sem_ebos', 'n_samples': 100, 'vocab': None, 'name': 'NoBPE_NoTemp_N100'},
    {'bpe': False, 'temporal': 'sem_ebos', 'n_samples': 200, 'vocab': None, 'name': 'NoBPE_NoTemp_N200'},
    {'bpe': False, 'temporal': 'sem_ebos', 'n_samples': 50, 'vocab': None, 'name': 'NoBPE_NoTemp_N50'},
    
    # Chronos
    {'bpe': 'chronos', 'temporal': None, 'n_samples': 100, 'vocab': 600, 'name': 'Chronos_N100'},
    {'bpe': 'chronos', 'temporal': None, 'n_samples': 200, 'vocab': 600, 'name': 'Chronos_N200'},
    {'bpe': 'chronos', 'temporal': None, 'n_samples': 50, 'vocab': 600, 'name': 'Chronos_N50'},
    
    # Baseline (no tokenization)
    {'bpe': None, 'temporal': None, 'n_samples': None, 'vocab': None, 'name': 'Baseline_NoToken'},
]


# Dataset column mappings
DATASET_COLUMNS = {
    'ETTh1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
    'weather': ['p_mbar', 'T_degC', 'Tpot_K', 'Tdew_degC', 'rh_pct', 'VPmax_mbar',
                'VPact_mbar', 'VPdef_mbar', 'sh_gkg', 'H2OC_mmolmol', 'rho_gm3',
                'wv_ms', 'max_wv_ms', 'wd_deg', 'rain_mm', 'raining_s', 'SWDR_Wm2',
                'PAR_umolm2s', 'max_PAR_umolm2s', 'Tlog_degC']
}


# Models and metrics
MODELS = ['Transformer', 'Informer', 'Autoformer']
METRICS = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
EVAL_TYPES = ['metrics', 'detokenized']


def match_folder_to_config(folder_name, dataset, model, config):
    """
    Check if a folder matches a specific experiment configuration.
    
    Args:
        folder_name: Name of the folder to check
        dataset: Dataset name ('ETTh1' or 'weather')
        model: Model name ('Transformer', 'Informer', 'Autoformer')
        config: Experiment configuration dict
        
    Returns:
        bool: True if folder matches configuration
    """
    # Must contain dataset and model
    if dataset not in folder_name or model not in folder_name:
        return False
    
    # Baseline (no tokenization)
    if config['bpe'] is None:
        # Should NOT have BPE indicators
        return ('_bpe_' not in folder_name.lower() and 
                'COM_BPE' not in folder_name and 
                'SEM_BPE' not in folder_name and
                'chronos' not in folder_name.lower())
    
    # Chronos
    if config['bpe'] == 'chronos':
        return ('chronos' in folder_name.lower() and 
                f"_N{config['n_samples']}_" in folder_name)
    
    # BPE experiments
    if config['bpe'] is True:
        if 'COM_BPE' not in folder_name:
            return False
        if f"_N_Samp{config['n_samples']}_" not in folder_name:
            return False
        if config['temporal']:
            return f"_{config['temporal']}_" in folder_name
        return True
    
    # No BPE experiments
    if config['bpe'] is False:
        if 'SEM_BPE' not in folder_name:
            return False
        if f"_N_Samp{config['n_samples']}_" not in folder_name:
            return False
        if config['temporal']:
            return f"_{config['temporal']}_" in folder_name
        return True
    
    return False


def check_constant_values(trues_df, column):
    """
    Check if all values in a column are constant.
    
    Args:
        trues_df: DataFrame with true values
        column: Column name to check
        
    Returns:
        bool: True if all non-NaN values are the same
    """
    if column not in trues_df.columns:
        return True  # Treat missing as constant
    
    values = pd.to_numeric(trues_df[column], errors='coerce').dropna()
    
    if len(values) == 0:
        return True
    
    # Check if all values are the same (within floating point tolerance)
    return values.nunique() == 1 or (values.max() - values.min()) < 1e-10


def extract_metric_from_metrics_csv(metrics_csv_path, column_name, metric_name):
    """
    Extract a specific metric value for a column from metrics.csv.
    
    Args:
        metrics_csv_path: Path to metrics.csv file
        column_name: Name of the column to extract metric for
        metric_name: Metric name ('MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE')
        
    Returns:
        float or np.nan: Metric value
    """
    try:
        df = pd.read_csv(metrics_csv_path)
        
        # Find row for this column
        row = df[df['column_name'] == column_name]
        
        if row.empty:
            return np.nan
        
        if metric_name in row.columns:
            value = row[metric_name].values[0]
            return value if not pd.isna(value) else np.nan
        
        return np.nan
    except Exception as e:
        print(f"    Error reading {metrics_csv_path}: {e}")
        return np.nan





def find_matching_folders(final_results_dir, dataset, model, config, target_column=None):
    """
    Find all folders matching a specific configuration.
    Searches in multiple final_results directories based on experiment type.
    
    Args:
        final_results_dir: Base directory name (will search final_results, final_results_chronos, final_results_trues)
        target_column: Optional target column name for single-variate experiments
    
    Returns:
        list: List of folder paths matching the configuration
    """
    # Determine which directory to search based on experiment type
    search_dirs = []
    
    if config['bpe'] is None:
        # Baseline experiments
        search_dirs = [Path("final_results_trues")]
    elif config['bpe'] == 'chronos':
        # Chronos experiments
        search_dirs = [Path("final_results_chronos")]
    else:
        # Regular tokenized experiments (COM_BPE, SEM_BPE)
        search_dirs = [Path("final_results")]
    
    matching_folders = []
    
    for search_path in search_dirs:
        if not search_path.exists():
            continue
        
        for folder in search_path.iterdir():
            if folder.is_dir() and match_folder_to_config(folder.name, dataset, model, config):
                # For single-variate experiments, also check target column
                if target_column is not None:
                    # For baseline experiments, check pattern: dataset_COLUMNNAME_
                    if config['bpe'] is None:
                        # Baseline pattern: ETTh1_HUFL_192_15_...
                        if f"{dataset}_{target_column}_" in folder.name:
                            matching_folders.append(folder)
                    else:
                        # Tokenized experiments have target_COLUMNNAME_sl pattern
                        if f"target_{target_column}_sl" in folder.name or f"target_{target_column}_192" in folder.name:
                            matching_folders.append(folder)
                else:
                    matching_folders.append(folder)
    
    return matching_folders


def generate_single_table(final_results_dir, dataset, model, metric, eval_type):
    """
    Generate a single comparison table.
    
    Args:
        final_results_dir: Path to final_results directory
        dataset: Dataset name ('ETTh1' or 'weather')
        model: Model name
        metric: Metric name ('MAE', 'MSE', etc.)
        eval_type: 'metrics' or 'detokenized'
        
    Returns:
        pd.DataFrame: Comparison table
    """
    columns = DATASET_COLUMNS[dataset]
    
    # Initialize table with NaN
    table_data = {}
    
    for config in EXPERIMENT_CONFIGS:
        row_values = {}
        
        # For single-variate experiments, find a separate folder for each column
        for col in columns:
            value = np.nan
            
            # Find folder matching this config AND target column
            folders = find_matching_folders(final_results_dir, dataset, model, config, target_column=col)
            
            if folders:
                # Use first matching folder
                folder = folders[0]
                
                if eval_type == 'metrics':
                    # Read from metrics_before.csv (original metrics before destokenization)
                    metrics_file = folder / "metrics_before.csv"
                    if metrics_file.exists():
                        # Read the original metrics file and extract the metric value
                        try:
                            df = pd.read_csv(metrics_file)
                            metric_lower = metric.lower()
                            if metric_lower in df.columns:
                                value = df[metric_lower].values[0]
                                value = value if not pd.isna(value) else np.nan
                            else:
                                value = np.nan
                        except Exception:
                            value = np.nan
                    else:
                        value = np.nan
                
                elif eval_type == 'detokenized':
                    # Read from metrics.csv (new metrics after destokenization)
                    metrics_file = folder / "metrics.csv"
                    if metrics_file.exists():
                        # For single-variate, column name in metrics.csv is the target column
                        value = extract_metric_from_metrics_csv(metrics_file, col, metric)
                    else:
                        value = np.nan
            
            row_values[col] = value
        
        table_data[config['name']] = row_values
    
    # Create DataFrame
    df = pd.DataFrame(table_data).T
    df.index.name = 'Experiment'
    
    return df


def generate_all_tables(final_results_dir="final_results", output_dir="comparison_tables"):
    """
    Generate all 60 comparison tables.
    
    Args:
        final_results_dir: Path to final_results directory
        output_dir: Output directory for tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"GENERATING COMPARISON TABLES")
    print(f"{'='*100}")
    print(f"Reading from:")
    print(f"  - final_results/ (regular tokenized experiments)")
    print(f"  - final_results_chronos/ (chronos experiments)")
    print(f"  - final_results_trues/ (baseline experiments)")
    print(f"Saving to: {output_dir}")
    print(f"Total tables to generate: {len(['ETTh1', 'weather']) * len(MODELS) * len(METRICS) * len(EVAL_TYPES)}")
    
    tables_generated = 0
    
    for dataset in ['ETTh1', 'weather']:
        print(f"\n{'-'*100}")
        print(f"Dataset: {dataset}")
        print(f"{'-'*100}")
        
        for model in MODELS:
            for metric in METRICS:
                for eval_type in EVAL_TYPES:
                    table_name = f"{dataset}_{model}_{metric}_{eval_type}.csv"
                    table_path = output_path / table_name
                    
                    print(f"  Generating {table_name}...", end=" ")
                    
                    try:
                        table_df = generate_single_table(
                            final_results_dir, dataset, model, metric, eval_type
                        )
                        
                        # Save table
                        table_df.to_csv(table_path)
                        tables_generated += 1
                        print("✓")
                        
                    except Exception as e:
                        print(f"✗ ERROR: {e}")
    
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Tables generated: {tables_generated}")
    print(f"Saved in: {output_dir}/")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comparison tables for tokenization experiments"
    )
    parser.add_argument(
        "--final-results-dir",
        type=str,
        default="final_results",
        help="Directory with final results (default: final_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_tables",
        help="Output directory for tables (default: comparison_tables)"
    )
    
    args = parser.parse_args()
    
    generate_all_tables(args.final_results_dir, args.output_dir)
