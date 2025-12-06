"""
Destokenize prediction and true results from tokenized format back to float values.

This script reads the preds_results.csv and trues_results.csv files from the results folder,
identifies the corresponding tokenizer model and vocab for each column, and converts the 
tokenized values back to their original float representation.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from utils.basic import BasicTokenizer
import argparse


def parse_result_folder_name(folder_name):
    """
    Extract dataset name, N_samples, and discretization type from folder name.
    
    Example folder names:
    - ETTh1_token_normal_adapt_N_Samp100_192_5_Transformer_...
    - ETTh1_simp_100_192_15_Transformer_...
    
    Returns: dict with 'dataset', 'n_samples', 'disc_type', 'norm_type'
    """
    info = {}
    
    # Extract dataset name (first part before underscore or pattern)
    dataset_match = re.match(r'^([A-Za-z0-9_]+?)_(?:token_|simp_)', folder_name)
    if dataset_match:
        info['dataset'] = dataset_match.group(1)
    else:
        # Fallback: try to find common dataset names
        for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity']:
            if folder_name.startswith(ds):
                info['dataset'] = ds
                break
    
    # Extract N_samples
    nsamp_match = re.search(r'N_Samp(\d+)', folder_name)
    if nsamp_match:
        info['n_samples'] = int(nsamp_match.group(1))
    else:
        # Alternative pattern for simple naming: dataset_simp_100_...
        nsamp_alt = re.search(r'_(?:simp|token)_(\d+)_', folder_name)
        if nsamp_alt:
            info['n_samples'] = int(nsamp_alt.group(1))
    
    # Determine discretization type and normalization
    if 'token_normal_adapt' in folder_name:
        info['disc_type'] = 'adaptative'
        info['norm_type'] = 'normal'
    elif 'token_standard_adapt' in folder_name or 'token_stand_adapt' in folder_name:
        info['disc_type'] = 'adaptative'
        info['norm_type'] = 'stand'
    elif 'simp' in folder_name:
        info['disc_type'] = 'simple'
        # Try to determine normalization type
        if 'normal' in folder_name:
            info['norm_type'] = 'normal'
        elif 'stand' in folder_name:
            info['norm_type'] = 'stand'
        else:
            info['norm_type'] = 'normal'  # default
    else:
        info['disc_type'] = 'simple'
        info['norm_type'] = 'normal'
    
    return info


def get_vocab_size_for_dataset(dataset, n_samples):
    """
    Determine vocab size based on dataset and n_samples.
    This follows the pattern from main2.py
    """
    if dataset == 'exchange_rate':
        # Special case for exchange_rate
        vocab_size = n_samples + 40
    else:
        # For other datasets
        if n_samples == 50:
            vocab_size = 1000
        elif n_samples == 100:
            vocab_size = 1000
        elif n_samples == 200:
            vocab_size = 1000
        elif n_samples == 202:
            vocab_size = 2200
        elif n_samples == 160:
            vocab_size = 200
        elif n_samples == 60:
            vocab_size = 100
        elif n_samples == 10:
            vocab_size = 50
        else:
            vocab_size = 1000  # default
    
    return vocab_size


def build_model_name(dataset, column, n_samples, vocab_size, disc_type, norm_type):
    """
    Build the model filename based on parameters.
    
    Format: {dataset}_feature_Nsam_{n_samples}_vocab_{vocab_size}_column_{column}_{disc_type}_{norm_type}.model
    """
    base_name = f"{dataset}_feature_Nsam_{n_samples}_vocab_{vocab_size}_column_{column}_{disc_type}_{norm_type}"
    model_file = f"{base_name}.model"
    vocab_file = f"{base_name}.fvocab"
    
    return model_file, vocab_file


def load_tokenizer_for_column(dataset, column, n_samples, vocab_size, disc_type, norm_type):
    """
    Load the tokenizer model and vocab for a specific column.
    """
    model_file, vocab_file = build_model_name(dataset, column, n_samples, vocab_size, disc_type, norm_type)
    
    model_path = os.path.join("model", model_file)
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Initialize tokenizer
    special_tokens = {'<PAD>': n_samples - 1, '<EBOS>': n_samples}
    tokenizer = BasicTokenizer(n_samples, vocab_file, special_tokens=special_tokens)
    
    # Load the model
    tokenizer.load(model_path)
    
    return tokenizer


def destokenize_column(token_values, tokenizer):
    """
    Destokenize a column of token values back to float values.
    
    Args:
        token_values: Series or array of token IDs
        tokenizer: Loaded BasicTokenizer instance
    
    Returns:
        array of destokenized float values
    """
    # Convert to list of integers
    token_ids = [int(val) for val in token_values]
    
    # Decode using the tokenizer
    float_values = tokenizer.decode(token_ids)
    
    return float_values


def get_column_names_for_dataset(dataset):
    """
    Get the actual column names for each dataset.
    """
    column_map = {
        'ETTh1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'ETTh2': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'ETTm1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'ETTm2': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'exchange_rate': ['0', '1', '2', '3', '4', '5', '6', 'OT'],
        'weather': None,  # Will need to check
        'electricity': None,  # Will need to check
    }
    return column_map.get(dataset, None)


def destokenize_results(results_folder, output_folder=None):
    """
    Destokenize all prediction and true results in a results folder.
    
    Args:
        results_folder: Path to folder containing preds_results.csv and trues_results.csv
        output_folder: Optional output folder (defaults to same as results_folder)
    """
    results_path = Path(results_folder)
    
    # Check if folder exists
    if not results_path.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")
    
    # Parse folder name to extract parameters
    folder_name = results_path.name
    info = parse_result_folder_name(folder_name)
    
    print(f"Processing: {folder_name}")
    print(f"  Dataset: {info.get('dataset', 'Unknown')}")
    print(f"  N_samples: {info.get('n_samples', 'Unknown')}")
    print(f"  Discretization: {info.get('disc_type', 'Unknown')}")
    print(f"  Normalization: {info.get('norm_type', 'Unknown')}")
    
    # Determine vocab size
    vocab_size = get_vocab_size_for_dataset(info['dataset'], info['n_samples'])
    print(f"  Vocab size: {vocab_size}")
    
    # Get the actual column names for the dataset
    actual_columns = get_column_names_for_dataset(info['dataset'])
    if actual_columns is None:
        print(f"  Warning: Unknown column mapping for dataset {info['dataset']}")
        actual_columns = None
    
    # Set output folder
    if output_folder is None:
        output_folder = results_path
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process predictions
    preds_file = results_path / "preds_results.csv"
    if preds_file.exists():
        print("\nDestokenizing predictions...")
        preds_df = pd.read_csv(preds_file)
        
        destokenized_preds = pd.DataFrame()
        
        for idx, column in enumerate(preds_df.columns):
            # Map numeric column name to actual column name
            if actual_columns and idx < len(actual_columns):
                actual_col_name = actual_columns[idx]
            else:
                actual_col_name = column
            
            print(f"  Column: {column} -> {actual_col_name}")
            try:
                tokenizer = load_tokenizer_for_column(
                    info['dataset'], 
                    actual_col_name,  # Use the actual column name for the model
                    info['n_samples'], 
                    vocab_size, 
                    info['disc_type'], 
                    info['norm_type']
                )
                
                float_values = destokenize_column(preds_df[column], tokenizer)
                destokenized_preds[column] = float_values
                
            except Exception as e:
                print(f"    ERROR: {e}")
                destokenized_preds[column] = np.nan
        
        # Save destokenized predictions
        output_file = output_folder / "preds_results_detokenized.csv"
        destokenized_preds.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
    else:
        print(f"\nWarning: {preds_file} not found")
    
    # Process true values
    trues_file = results_path / "trues_results.csv"
    if trues_file.exists():
        print("\nDestokenizing true values...")
        trues_df = pd.read_csv(trues_file)
        
        destokenized_trues = pd.DataFrame()
        
        for idx, column in enumerate(trues_df.columns):
            # Map numeric column name to actual column name
            if actual_columns and idx < len(actual_columns):
                actual_col_name = actual_columns[idx]
            else:
                actual_col_name = column
            
            print(f"  Column: {column} -> {actual_col_name}")
            try:
                tokenizer = load_tokenizer_for_column(
                    info['dataset'], 
                    actual_col_name,  # Use the actual column name for the model
                    info['n_samples'], 
                    vocab_size, 
                    info['disc_type'], 
                    info['norm_type']
                )
                
                float_values = destokenize_column(trues_df[column], tokenizer)
                destokenized_trues[column] = float_values
                
            except Exception as e:
                print(f"    ERROR: {e}")
                destokenized_trues[column] = np.nan
        
        # Save destokenized true values
        output_file = output_folder / "trues_results_detokenized.csv"
        destokenized_trues.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
    else:
        print(f"\nWarning: {trues_file} not found")
    
    print("\nDestokenization complete!")


def destokenize_all_results(results_base_folder="results"):
    """
    Destokenize all results in all subfolders of the results directory.
    """
    results_path = Path(results_base_folder)
    
    if not results_path.exists():
        print(f"Results folder not found: {results_base_folder}")
        return
    
    # Find all subdirectories containing preds_results.csv or trues_results.csv
    result_folders = []
    for folder in results_path.iterdir():
        if folder.is_dir():
            if (folder / "preds_results.csv").exists() or (folder / "trues_results.csv").exists():
                result_folders.append(folder)
    
    print(f"Found {len(result_folders)} result folders to process\n")
    
    for i, folder in enumerate(result_folders, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(result_folders)}: {folder.name}")
        print(f"{'='*80}")
        
        try:
            destokenize_results(folder)
        except Exception as e:
            print(f"ERROR processing {folder.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Destokenize prediction and true results from tokenized format"
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        help="Specific results folder to destokenize"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Destokenize all results folders"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output folder for destokenized results (default: same as input)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        destokenize_all_results(args.results_dir)
    elif args.folder:
        destokenize_results(args.folder, args.output)
    else:
        print("Please specify either --folder or --all")
        print("Examples:")
        print("  python destokenize_results.py --folder results/ETTh1_token_normal_adapt_N_Samp100_192_5_Transformer_custom_ftM_sl192_ll48_pl5_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0")
        print("  python destokenize_results.py --all")
