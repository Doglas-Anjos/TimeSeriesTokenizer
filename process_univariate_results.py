"""
Univariate results processing pipeline (for _ftS_ experiments):
1. Read tokenized predictions and true values (both are tokens, single column)
2. Apply ceiling to prediction tokens (convert float tokens to int)
3. Destokenize both using appropriate model for the target column
4. Calculate metrics
5. Save results to final_results_univariate folder
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from utils.basic import BasicTokenizer
from scaler_utils import inverse_transform_series, load_scaler, get_normalization_type_from_folder
import warnings
warnings.filterwarnings('ignore')


# Dataset column name mapping: numeric indices to actual column names
DATASET_COLUMNS = {
    'ETTh1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    'ETTh2': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    'ETTm1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    'ETTm2': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    'exchange_rate': ['0', '1', '2', '3', '4', '5', '6', 'OT'],
    'weather': ['p(mbar)', 'T(degC)', 'Tpot(K)', 'Tdew(degC)', 'rh(%)', 'VPmax(mbar)', 
                'VPact(mbar)', 'VPdef(mbar)', 'sh(g/kg)', 'H2OC(mmol/mol)', 'rho(g/m**3)', 
                'wv(m/s)', 'max_wv(m/s)', 'wd(deg)', 'rain(mm)', 'raining(s)', 'SWDR(W/m**2)', 
                'PAR(umol/m**2/s)', 'max_PAR(umol/m**2/s)', 'Tlog(degC)', 'OT'],
    'electricity': None,  # Has 321 columns, will handle separately
}


def parse_univariate_folder_name(folder_name):
    """Extract dataset name, N_samples, target column, and discretization type from folder name."""
    info = {}
    
    # Check if this is a univariate experiment (_ftS_)
    if '_ftS_' not in folder_name:
        return None
    
    # Extract dataset name
    dataset_match = re.match(r'^([A-Za-z0-9_]+?)_(?:token_|simp_)', folder_name)
    if dataset_match:
        info['dataset'] = dataset_match.group(1)
    else:
        for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity']:
            if folder_name.startswith(ds):
                info['dataset'] = ds
                break
    
    # Extract target column (after _target_)
    target_match = re.search(r'_target_([A-Za-z0-9_]+?)_', folder_name)
    if target_match:
        info['target_column'] = target_match.group(1)
    else:
        # If not found with underscores, try without trailing underscore
        target_match = re.search(r'_target_([A-Za-z0-9_]+)', folder_name)
        if target_match:
            # Extract just the column name, stopping at next underscore pattern
            target_str = target_match.group(1)
            # Split on common patterns like _sl, _ll, _pl, _dm
            for pattern in ['_sl', '_ll', '_pl', '_dm', '_nh', '_el', '_dl', '_df', '_fc', '_eb', '_dt', '_Exp']:
                if pattern in target_str:
                    target_str = target_str.split(pattern)[0]
                    break
            info['target_column'] = target_str
    
    # Extract N_samples
    nsamp_match = re.search(r'N_Samp(\d+)', folder_name)
    if nsamp_match:
        info['n_samples'] = int(nsamp_match.group(1))
    else:
        # Try alternative pattern with column name
        nsamp_alt = re.search(r'_(\d+)_[A-Z]+_\d+_\d+_', folder_name)
        if nsamp_alt:
            info['n_samples'] = int(nsamp_alt.group(1))
    
    # Determine discretization type and normalization
    if 'token_normal' in folder_name and 'adapt' in folder_name:
        info['disc_type'] = 'adaptative'
        info['norm_type'] = 'normal'
    elif 'token_standard' in folder_name and 'adapt' in folder_name:
        info['disc_type'] = 'adaptative'
        info['norm_type'] = 'stand'
    elif 'token_normal' in folder_name and 'simp' in folder_name:
        info['disc_type'] = 'simple'
        info['norm_type'] = 'normal'
    elif 'token_standard' in folder_name and 'simp' in folder_name:
        info['disc_type'] = 'simple'
        info['norm_type'] = 'stand'
    elif 'simp' in folder_name:
        info['disc_type'] = 'simple'
        if 'standard' in folder_name:
            info['norm_type'] = 'stand'
        else:
            info['norm_type'] = 'normal'
    else:
        info['disc_type'] = 'simple'
        info['norm_type'] = 'normal'
    
    return info


def get_vocab_size_for_dataset(dataset, n_samples):
    """Determine vocab size based on dataset and n_samples."""
    if dataset == 'exchange_rate':
        vocab_size = n_samples + 40
    else:
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
            vocab_size = 1000
    
    return vocab_size


def build_model_name(dataset, column, n_samples, vocab_size, disc_type, norm_type):
    """Build the model filename based on parameters."""
    base_name = f"{dataset}_feature_Nsam_{n_samples}_vocab_{vocab_size}_column_{column}_{disc_type}_{norm_type}"
    model_file = f"{base_name}.model"
    vocab_file = f"{base_name}.fvocab"
    
    return model_file, vocab_file


def load_tokenizer_for_column(dataset, column, n_samples, vocab_size, disc_type, norm_type):
    """Load the tokenizer model and vocab for a specific column."""
    model_file, vocab_file = build_model_name(dataset, column, n_samples, vocab_size, disc_type, norm_type)
    
    model_path = os.path.join("model", model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    special_tokens = {'<PAD>': n_samples - 1, '<EBOS>': n_samples}
    tokenizer = BasicTokenizer(n_samples, vocab_file, special_tokens=special_tokens)
    tokenizer.load(model_path)
    
    return tokenizer, model_path, vocab_file


def apply_ceiling_to_tokens(token_series):
    """Apply ceiling operation to token values and convert to int."""
    # Convert to numeric, handling any non-numeric values
    numeric_series = pd.to_numeric(token_series, errors='coerce')
    # Apply ceiling and convert to int
    ceiled_series = np.ceil(numeric_series).astype('Int64')  # Int64 supports NaN
    
    return ceiled_series


def destokenize_column(token_values, tokenizer):
    """Destokenize a column of token values back to float values."""
    # Convert to int, handling NaN
    token_ids = []
    for val in token_values:
        if pd.isna(val):
            token_ids.append(0)  # Use 0 as placeholder for NaN
        else:
            token_ids.append(int(val))
    
    try:
        float_values = tokenizer.decode(token_ids)
        
        # Ensure the length matches by padding with NaN if needed
        if len(float_values) != len(token_values):
            if len(float_values) < len(token_values):
                # Pad with NaN
                float_values = float_values + [np.nan] * (len(token_values) - len(float_values))
            else:
                # Truncate
                float_values = float_values[:len(token_values)]
        
        return float_values
    except Exception as e:
        # If decode fails, return NaN array of same length
        return [np.nan] * len(token_values)


def calculate_metrics(trues, preds, ignore_nan=True):
    """Calculate regression metrics."""
    trues = np.array(trues)
    preds = np.array(preds)
    
    if ignore_nan:
        mask = ~(np.isnan(trues) | np.isnan(preds))
        trues = trues[mask]
        preds = preds[mask]
    
    if len(trues) == 0:
        return {
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan,
            'valid_samples': 0
        }
    
    mse = np.mean((trues - preds) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(trues - preds))
    
    # MAPE (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((trues - preds) / np.where(trues != 0, trues, 1))) * 100
    
    # R-squared
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'valid_samples': len(trues)
    }


def process_single_univariate_result(results_folder, output_base_folder="final_results_univariate"):
    """Process a single univariate results folder with complete pipeline."""
    results_path = Path(results_folder)
    folder_name = results_path.name
    
    print(f"\n{'='*100}")
    print(f"Processing: {folder_name}")
    print(f"{'='*100}")
    
    # Parse folder information
    info = parse_univariate_folder_name(folder_name)
    
    if info is None:
        print(f"  SKIP: Not a univariate experiment (_ftS_ not found)")
        return None
    
    if 'dataset' not in info or 'n_samples' not in info or 'target_column' not in info:
        print(f"  ERROR: Could not parse folder name properly")
        print(f"  Info extracted: {info}")
        return None
    
    print(f"  Dataset: {info['dataset']}")
    print(f"  Target column: {info['target_column']}")
    print(f"  N_samples: {info['n_samples']}")
    print(f"  Discretization: {info['disc_type']}")
    print(f"  Normalization: {info['norm_type']}")
    
    vocab_size = get_vocab_size_for_dataset(info['dataset'], info['n_samples'])
    print(f"  Vocab size: {vocab_size}")
    
    # Check for input files
    preds_file = results_path / "preds_results.csv"
    trues_file = results_path / "trues_results.csv"
    
    if not preds_file.exists():
        print(f"  ERROR: preds_results.csv not found")
        return None
    
    if not trues_file.exists():
        print(f"  ERROR: trues_results.csv not found")
        return None
    
    # Load tokenized data (BOTH are tokenized, single column)
    print(f"\nLoading tokenized data...")
    preds_tokens_df = pd.read_csv(preds_file)
    trues_tokens_df = pd.read_csv(trues_file)
    
    print(f"  Predictions tokens shape: {preds_tokens_df.shape}")
    print(f"  True tokens shape: {trues_tokens_df.shape}")
    
    if preds_tokens_df.shape[1] != 1 or trues_tokens_df.shape[1] != 1:
        print(f"  WARNING: Expected single column, got {preds_tokens_df.shape[1]} and {trues_tokens_df.shape[1]}")
    
    # Get the column (should be '0')
    preds_column = preds_tokens_df.columns[0]
    trues_column = trues_tokens_df.columns[0]
    
    # STEP 1: Apply ceiling to prediction tokens BEFORE destokenization
    print(f"\n[STEP 1] Applying ceiling to prediction tokens (float -> int)...")
    preds_tokens_ceiled = apply_ceiling_to_tokens(preds_tokens_df[preds_column])
    print(f"  Sample before ceiling: {preds_tokens_df[preds_column].iloc[0]}")
    print(f"  Sample after ceiling:  {preds_tokens_ceiled.iloc[0]}")
    
    # STEP 2: Load tokenizer for target column
    print(f"\n[STEP 2] Loading tokenizer for target column '{info['target_column']}'...")
    
    try:
        tokenizer, model_path, vocab_file = load_tokenizer_for_column(
            info['dataset'], 
            info['target_column'],
            info['n_samples'], 
            vocab_size, 
            info['disc_type'], 
            info['norm_type']
        )
        
        print(f"  ✓ Model: {model_path}")
        print(f"  ✓ Vocab: {vocab_file}")
        
        models_used = {
            'target_column': info['target_column'],
            'model_file': model_path,
            'vocab_file': vocab_file
        }
        
    except Exception as e:
        print(f"  ✗ ERROR loading tokenizer: {e}")
        return None
    
    # STEP 3: Destokenize predictions (with ceiling applied)
    print(f"\n[STEP 3] Destokenizing predictions (tokens -> floats)...")
    try:
        preds_detokenized = destokenize_column(preds_tokens_ceiled, tokenizer)
        print(f"  ✓ Destokenized {len(preds_detokenized)} predictions")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return None
    
    # STEP 4: Destokenize true values (no ceiling)
    print(f"\n[STEP 4] Destokenizing true values (tokens -> floats)...")
    try:
        trues_detokenized = destokenize_column(trues_tokens_df[trues_column], tokenizer)
        print(f"  ✓ Destokenized {len(trues_detokenized)} true values")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return None
    
    # Convert to numeric
    preds_numeric = pd.to_numeric(preds_detokenized, errors='coerce')
    trues_numeric = pd.to_numeric(trues_detokenized, errors='coerce')
    
    # STEP 5: Calculate metrics
    print(f"\n[STEP 5] Calculating metrics...")
    total_predictions = len(preds_numeric)
    
    metrics = calculate_metrics(trues_numeric, preds_numeric)
    
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")
    print(f"  Valid samples: {metrics['valid_samples']}/{total_predictions}")
    
    # [STEP 6] Apply inverse transformation if data was scaled
    preds_original = preds_numeric
    trues_original = trues_numeric
    
    norm_type = get_normalization_type_from_folder(folder_name)
    if norm_type == 'standard':
        print(f"\n[STEP 6] Applying inverse StandardScaler transformation...")
        try:
            preds_series = pd.Series(preds_numeric)
            trues_series = pd.Series(trues_numeric)
            
            preds_original = inverse_transform_series(
                preds_series, info['dataset'], info['target_column'], 'standard'
            )
            trues_original = inverse_transform_series(
                trues_series, info['dataset'], info['target_column'], 'standard'
            )
            
            print(f"  ✓ Inverse transformation applied to predictions and true values")
            
            # Calculate metrics on original scale
            metrics_original = calculate_metrics(trues_original.values, preds_original.values)
            print(f"\n  Metrics on original scale:")
            print(f"    MSE:  {metrics_original['MSE']:.6f}")
            print(f"    RMSE: {metrics_original['RMSE']:.6f}")
            print(f"    MAE:  {metrics_original['MAE']:.6f}")
            print(f"    MAPE: {metrics_original['MAPE']:.6f}")
            print(f"    R²:   {metrics_original['R2']:.6f}")
            
        except FileNotFoundError as e:
            print(f"  ⚠ Scaler not found: {e}")
            print(f"  ⚠ Skipping inverse transformation")
    elif norm_type == 'normal':
        print(f"\n[STEP 6] No scaling applied (normal) - skipping inverse transformation")
    
    # Create overall metrics
    overall_metrics = {
        'experiment': folder_name,
        'dataset': info['dataset'],
        'target_column': info['target_column'],
        'n_samples': info['n_samples'],
        'vocab_size': vocab_size,
        'disc_type': info['disc_type'],
        'norm_type': info['norm_type'],
        'MSE': metrics['MSE'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'valid_samples': metrics['valid_samples'],
        'total_predictions': total_predictions
    }
    
    # Create output folder
    output_folder = Path(output_base_folder) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\nSaving results to {output_folder}...")
    
    # Save ceiled tokens
    preds_tokens_ceiled_df = pd.DataFrame({preds_column: preds_tokens_ceiled})
    preds_tokens_ceiled_file = output_folder / "preds_tokens_ceiled.csv"
    preds_tokens_ceiled_df.to_csv(preds_tokens_ceiled_file, index=False)
    print(f"  ✓ {preds_tokens_ceiled_file.name}")
    
    # Save detokenized data
    preds_detokenized_df = pd.DataFrame({info['target_column']: preds_detokenized})
    preds_detokenized_file = output_folder / "preds_detokenized.csv"
    preds_detokenized_df.to_csv(preds_detokenized_file, index=False)
    print(f"  ✓ {preds_detokenized_file.name}")
    
    trues_detokenized_df = pd.DataFrame({info['target_column']: trues_detokenized})
    trues_detokenized_file = output_folder / "trues_detokenized.csv"
    trues_detokenized_df.to_csv(trues_detokenized_file, index=False)
    print(f"  ✓ {trues_detokenized_file.name}")
    
    # Save original scale data if inverse transformation was applied
    if not np.array_equal(preds_original, preds_numeric):
        preds_original_df = pd.DataFrame({info['target_column']: preds_original})
        preds_original_file = output_folder / "preds_original_scale.csv"
        preds_original_df.to_csv(preds_original_file, index=False)
        print(f"  ✓ {preds_original_file.name}")
        
        trues_original_df = pd.DataFrame({info['target_column']: trues_original})
        trues_original_file = output_folder / "trues_original_scale.csv"
        trues_original_df.to_csv(trues_original_file, index=False)
        print(f"  ✓ {trues_original_file.name}")
        
        # Save original scale metrics
        metrics_original_df = pd.DataFrame([metrics_original])
        metrics_original_file = output_folder / "metrics_original_scale.csv"
        metrics_original_df.to_csv(metrics_original_file, index=False)
        print(f"  ✓ {metrics_original_file.name}")
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'target_column': info['target_column'],
        'model_file': model_path,
        'vocab_file': vocab_file,
        'MSE': metrics['MSE'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'valid_samples': metrics['valid_samples'],
        'total_predictions': total_predictions
    }])
    
    metrics_file = output_folder / "metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  ✓ {metrics_file.name}")
    
    # Save model info
    models_info_file = output_folder / "model_used.txt"
    with open(models_info_file, 'w') as f:
        f.write(f"Experiment: {folder_name}\n")
        f.write(f"Dataset: {info['dataset']}\n")
        f.write(f"Target column: {info['target_column']}\n")
        f.write(f"N_samples: {info['n_samples']}\n")
        f.write(f"Vocab size: {vocab_size}\n")
        f.write(f"Discretization: {info['disc_type']}\n")
        f.write(f"Normalization: {info['norm_type']}\n")
        f.write(f"Total predictions: {total_predictions}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Model used for destokenization:\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Target: {info['target_column']}\n")
        f.write(f"  Model: {model_path}\n")
        f.write(f"  Vocab: {vocab_file}\n")
    
    print(f"  ✓ {models_info_file.name}")
    
    print(f"\n✓ Processing complete for {folder_name}\n")
    
    return overall_metrics


def process_all_univariate_results(results_base_folder="results", output_base_folder="final_results_univariate"):
    """Process all univariate result folders (_ftS_ experiments)."""
    results_path = Path(results_base_folder)
    
    if not results_path.exists():
        print(f"ERROR: Results folder not found: {results_base_folder}")
        return
    
    # Find all univariate result folders (_ftS_ with predictions)
    result_folders = []
    for folder in results_path.iterdir():
        if folder.is_dir() and '_ftS_' in folder.name and (folder / "preds_results.csv").exists():
            result_folders.append(folder)
    
    if not result_folders:
        print(f"No univariate result folders (_ftS_) with preds_results.csv found in {results_base_folder}")
        return
    
    print(f"\n{'='*100}")
    print(f"PROCESSING ALL UNIVARIATE RESULTS")
    print(f"{'='*100}")
    print(f"Found {len(result_folders)} univariate result folders to process")
    print(f"Output folder: {output_base_folder}")
    
    all_metrics = []
    successful = 0
    failed = 0
    
    for i, folder in enumerate(result_folders, 1):
        print(f"\n[{i}/{len(result_folders)}]")
        
        try:
            overall_metrics = process_single_univariate_result(folder, output_base_folder)
            if overall_metrics:
                all_metrics.append(overall_metrics)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ ERROR processing {folder.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Create summary report
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Total folders: {len(result_folders)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df = summary_df.sort_values('MSE')
        
        summary_file = Path(output_base_folder) / "summary_all_univariate_experiments.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
        print(f"\nTop 5 experiments by MSE:")
        print(summary_df[['experiment', 'target_column', 'MSE', 'RMSE', 'MAE', 'R2']].head().to_string(index=False))
    
    print(f"\n{'='*100}")
    print(f"ALL UNIVARIATE PROCESSING COMPLETE")
    print(f"{'='*100}")
    print(f"Results saved in: {output_base_folder}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process univariate results (_ftS_): destokenize, apply ceiling, calculate metrics"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Process a single univariate results folder"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all univariate results folders"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="final_results_univariate",
        help="Output directory (default: final_results_univariate)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        process_all_univariate_results(args.results_dir, args.output_dir)
    elif args.folder:
        process_single_univariate_result(args.folder, args.output_dir)
    else:
        print("Please specify either --folder or --all")
        print("\nExamples:")
        print("  python process_univariate_results.py --folder results/ETTh1_token_normal_adapt_N_Samp100_HUFL_192_15_Transformer_custom_ftS_target_HUFL_...")
        print("  python process_univariate_results.py --all")
        print("  python process_univariate_results.py --all --output-dir my_univariate_results")
