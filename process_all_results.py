"""
Complete results processing pipeline:
1. Read tokenized predictions and true values (both are tokens)
2. Apply ceiling to prediction tokens (convert float tokens to int)
3. Destokenize both using appropriate models
4. Calculate metrics
5. Save results to final_results folder
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from utils.basic import BasicTokenizer
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


def parse_result_folder_name(folder_name):
    """Extract dataset name, N_samples, and discretization type from folder name."""
    info = {}
    
    # Check if this is a multivariate experiment (_ftM_)
    if '_ftM_' not in folder_name:
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
    
    # Extract N_samples
    nsamp_match = re.search(r'N_Samp(\d+)', folder_name)
    if nsamp_match:
        info['n_samples'] = int(nsamp_match.group(1))
    else:
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
        if 'normal' in folder_name:
            info['norm_type'] = 'normal'
        elif 'stand' in folder_name:
            info['norm_type'] = 'stand'
        else:
            info['norm_type'] = 'normal'
    else:
        info['disc_type'] = 'simple'
        info['norm_type'] = 'normal'
    
    return info


def get_column_names_for_dataset(dataset, num_columns=None):
    """Get the actual column names for each dataset."""
    columns = DATASET_COLUMNS.get(dataset, None)
    
    # For electricity, generate column indices
    if dataset == 'electricity' and num_columns:
        columns = [str(i) for i in range(num_columns - 1)] + ['OT']
    
    return columns


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


def align_dataframes(df1, df2):
    """Align two dataframes to have the same columns."""
    # Get all unique columns from both dataframes
    all_columns = list(df1.columns) + [col for col in df2.columns if col not in df1.columns]
    
    # Reindex both dataframes to have same columns
    df1_aligned = df1.reindex(columns=all_columns, fill_value=np.nan)
    df2_aligned = df2.reindex(columns=all_columns, fill_value=np.nan)
    
    return df1_aligned, df2_aligned


def apply_ceiling_to_tokens(tokens_df):
    """Apply ceiling operation to token values and convert to int."""
    ceiled_df = tokens_df.copy()
    
    for col in ceiled_df.columns:
        # Convert to numeric, handling any non-numeric values
        numeric_col = pd.to_numeric(ceiled_df[col], errors='coerce')
        # Apply ceiling and convert to int
        ceiled_df[col] = np.ceil(numeric_col).astype('Int64')  # Int64 supports NaN
    
    return ceiled_df


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


def process_single_result_folder(results_folder, output_base_folder="final_results"):
    """Process a single results folder with complete pipeline."""
    results_path = Path(results_folder)
    folder_name = results_path.name
    
    print(f"\n{'='*100}")
    print(f"Processing: {folder_name}")
    print(f"{'='*100}")
    
    # Parse folder information
    info = parse_result_folder_name(folder_name)
    
    if info is None:
        print(f"  SKIP: Not a multivariate experiment (_ftM_ not found)")
        return None
    
    if 'dataset' not in info or 'n_samples' not in info:
        print(f"  ERROR: Could not parse folder name properly")
        return None
    
    print(f"  Dataset: {info['dataset']}")
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
    
    # Load tokenized data (BOTH are tokenized)
    print(f"\nLoading tokenized data...")
    preds_tokens_df = pd.read_csv(preds_file)
    trues_tokens_df = pd.read_csv(trues_file)
    
    print(f"  Predictions tokens shape: {preds_tokens_df.shape}")
    print(f"  True tokens shape: {trues_tokens_df.shape}")
    
    # Get column names
    num_columns = max(len(preds_tokens_df.columns), len(trues_tokens_df.columns))
    actual_columns = get_column_names_for_dataset(info['dataset'], num_columns)
    
    if actual_columns is None:
        print(f"  Warning: Using numeric column names for dataset {info['dataset']}")
        actual_columns = [str(i) for i in range(num_columns)]
    
    print(f"  Column mapping: {list(preds_tokens_df.columns)} -> {actual_columns[:len(preds_tokens_df.columns)]}")
    
    # STEP 1: Apply ceiling to prediction tokens BEFORE destokenization
    print(f"\n[STEP 1] Applying ceiling to prediction tokens (float -> int)...")
    preds_tokens_ceiled = apply_ceiling_to_tokens(preds_tokens_df)
    print(f"  Sample before ceiling: {preds_tokens_df.iloc[0, 0]}")
    print(f"  Sample after ceiling:  {preds_tokens_ceiled.iloc[0, 0]}")
    
    # STEP 2: Destokenize predictions (with ceiling applied)
    print(f"\n[STEP 2] Destokenizing predictions (tokens -> floats)...")
    preds_detokenized = pd.DataFrame()
    models_used = {}
    
    for idx, column in enumerate(preds_tokens_ceiled.columns):
        if idx < len(actual_columns):
            actual_col_name = actual_columns[idx]
        else:
            actual_col_name = str(column)
        
        print(f"  Column [{column}] -> '{actual_col_name}'...", end=" ")
        
        try:
            tokenizer, model_path, vocab_file = load_tokenizer_for_column(
                info['dataset'], 
                actual_col_name,
                info['n_samples'], 
                vocab_size, 
                info['disc_type'], 
                info['norm_type']
            )
            
            float_values = destokenize_column(preds_tokens_ceiled[column], tokenizer)
            preds_detokenized[column] = float_values
            
            models_used[column] = {
                'actual_column': actual_col_name,
                'model_file': model_path,
                'vocab_file': vocab_file
            }
            
            print(f"✓")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            preds_detokenized[column] = np.nan
            models_used[column] = {
                'actual_column': actual_col_name,
                'model_file': 'ERROR',
                'vocab_file': 'ERROR',
                'error': str(e)
            }
    
    # STEP 3: Destokenize true values (no ceiling)
    print(f"\n[STEP 3] Destokenizing true values (tokens -> floats)...")
    trues_detokenized = pd.DataFrame()
    
    for idx, column in enumerate(trues_tokens_df.columns):
        if idx < len(actual_columns):
            actual_col_name = actual_columns[idx]
        else:
            actual_col_name = str(column)
        
        print(f"  Column [{column}] -> '{actual_col_name}'...", end=" ")
        
        try:
            tokenizer, model_path, vocab_file = load_tokenizer_for_column(
                info['dataset'], 
                actual_col_name,
                info['n_samples'], 
                vocab_size, 
                info['disc_type'], 
                info['norm_type']
            )
            
            float_values = destokenize_column(trues_tokens_df[column], tokenizer)
            trues_detokenized[column] = float_values
            
            print(f"✓")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            trues_detokenized[column] = np.nan
    
    # STEP 4: Align dataframes if columns differ
    print(f"\n[STEP 4] Aligning dataframes...")
    if preds_detokenized.shape[1] != trues_detokenized.shape[1]:
        print(f"  Columns differ: preds={preds_detokenized.shape[1]}, trues={trues_detokenized.shape[1]}")
        preds_detokenized, trues_detokenized = align_dataframes(preds_detokenized, trues_detokenized)
        print(f"  After alignment: preds={preds_detokenized.shape[1]}, trues={trues_detokenized.shape[1]}")
    else:
        print(f"  Columns already aligned: {preds_detokenized.shape[1]} columns")
    
    # Convert to numeric
    preds_numeric = preds_detokenized.apply(pd.to_numeric, errors='coerce')
    trues_numeric = trues_detokenized.apply(pd.to_numeric, errors='coerce')
    
    # STEP 5: Calculate metrics
    print(f"\n[STEP 5] Calculating metrics...")
    metrics_data = []
    total_predictions = len(preds_numeric)
    
    for col in preds_numeric.columns:
        if col not in trues_numeric.columns:
            continue
        
        metrics = calculate_metrics(trues_numeric[col], preds_numeric[col])
        
        col_name = models_used.get(col, {}).get('actual_column', col)
        
        metrics_row = {
            'column_index': col,
            'column_name': col_name,
            'model_file': models_used.get(col, {}).get('model_file', 'N/A'),
            'vocab_file': models_used.get(col, {}).get('vocab_file', 'N/A'),
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2'],
            'valid_samples': metrics['valid_samples'],
            'total_predictions': total_predictions
        }
        
        if 'error' in models_used.get(col, {}):
            metrics_row['error'] = models_used[col]['error']
        
        metrics_data.append(metrics_row)
        
        print(f"  [{col}] {col_name}: MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.6f}, "
              f"MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f} (valid={metrics['valid_samples']}/{total_predictions})")
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Calculate overall metrics
    overall_metrics = {
        'experiment': folder_name,
        'dataset': info['dataset'],
        'n_samples': info['n_samples'],
        'vocab_size': vocab_size,
        'disc_type': info['disc_type'],
        'norm_type': info['norm_type'],
        'avg_MSE': metrics_df['MSE'].mean(),
        'avg_RMSE': metrics_df['RMSE'].mean(),
        'avg_MAE': metrics_df['MAE'].mean(),
        'avg_MAPE': metrics_df['MAPE'].mean(),
        'avg_R2': metrics_df['R2'].mean(),
        'total_columns': len(metrics_df),
        'total_predictions': total_predictions
    }
    
    print(f"\n{'='*100}")
    print(f"OVERALL METRICS:")
    print(f"  Average MSE:   {overall_metrics['avg_MSE']:.6f}")
    print(f"  Average RMSE:  {overall_metrics['avg_RMSE']:.6f}")
    print(f"  Average MAE:   {overall_metrics['avg_MAE']:.6f}")
    print(f"  Average MAPE:  {overall_metrics['avg_MAPE']:.6f}")
    print(f"  Average R²:    {overall_metrics['avg_R2']:.6f}")
    print(f"  Total columns: {overall_metrics['total_columns']}")
    print(f"  Total predictions: {overall_metrics['total_predictions']}")
    print(f"{'='*100}")
    
    # Create output folder
    output_folder = Path(output_base_folder) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\nSaving results to {output_folder}...")
    
    # Save ceiled tokens
    preds_tokens_ceiled_file = output_folder / "preds_tokens_ceiled.csv"
    preds_tokens_ceiled.to_csv(preds_tokens_ceiled_file, index=False)
    print(f"  ✓ {preds_tokens_ceiled_file.name}")
    
    # Save detokenized data
    preds_detokenized_file = output_folder / "preds_detokenized.csv"
    preds_detokenized.to_csv(preds_detokenized_file, index=False)
    print(f"  ✓ {preds_detokenized_file.name}")
    
    trues_detokenized_file = output_folder / "trues_detokenized.csv"
    trues_detokenized.to_csv(trues_detokenized_file, index=False)
    print(f"  ✓ {trues_detokenized_file.name}")
    
    # Save metrics
    metrics_file = output_folder / "metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  ✓ {metrics_file.name}")
    
    # Save models used info
    models_info_file = output_folder / "models_used.txt"
    with open(models_info_file, 'w') as f:
        f.write(f"Experiment: {folder_name}\n")
        f.write(f"Dataset: {info['dataset']}\n")
        f.write(f"N_samples: {info['n_samples']}\n")
        f.write(f"Vocab size: {vocab_size}\n")
        f.write(f"Discretization: {info['disc_type']}\n")
        f.write(f"Normalization: {info['norm_type']}\n")
        f.write(f"Total predictions: {total_predictions}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Column Mapping:\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"CSV Indices -> Actual Names:\n")
        for idx, col_name in enumerate(actual_columns[:len(preds_tokens_df.columns)]):
            f.write(f"  [{idx}] -> {col_name}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Models used for destokenization:\n")
        f.write(f"{'='*80}\n\n")
        
        for col, model_info in models_used.items():
            f.write(f"Column [{col}] ({model_info['actual_column']}):\n")
            f.write(f"  Model: {model_info['model_file']}\n")
            f.write(f"  Vocab: {model_info['vocab_file']}\n")
            if 'error' in model_info:
                f.write(f"  Error: {model_info['error']}\n")
            f.write(f"\n")
    
    print(f"  ✓ {models_info_file.name}")
    
    print(f"\n✓ Processing complete for {folder_name}\n")
    
    return overall_metrics


def process_all_results(results_base_folder="results", output_base_folder="final_results"):
    """Process all result folders."""
    results_path = Path(results_base_folder)
    
    if not results_path.exists():
        print(f"ERROR: Results folder not found: {results_base_folder}")
        return
    
    # Find all multivariate result folders (_ftM_ with predictions)
    result_folders = []
    for folder in results_path.iterdir():
        if folder.is_dir() and '_ftM_' in folder.name and (folder / "preds_results.csv").exists():
            result_folders.append(folder)
    
    if not result_folders:
        print(f"No multivariate result folders (_ftM_) with preds_results.csv found in {results_base_folder}")
        return
    
    print(f"\n{'='*100}")
    print(f"PROCESSING ALL RESULTS")
    print(f"{'='*100}")
    print(f"Found {len(result_folders)} multivariate result folders to process")
    print(f"Output folder: {output_base_folder}")
    
    all_metrics = []
    successful = 0
    failed = 0
    
    for i, folder in enumerate(result_folders, 1):
        print(f"\n[{i}/{len(result_folders)}]")
        
        try:
            overall_metrics = process_single_result_folder(folder, output_base_folder)
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
        summary_df = summary_df.sort_values('avg_MSE')
        
        summary_file = Path(output_base_folder) / "summary_all_experiments.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
        print(f"\nTop 5 experiments by average MSE:")
        print(summary_df[['experiment', 'avg_MSE', 'avg_RMSE', 'avg_MAE', 'avg_R2']].head().to_string(index=False))
    
    print(f"\n{'='*100}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"{'='*100}")
    print(f"Results saved in: {output_base_folder}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process multivariate results (_ftM_): destokenize, apply ceiling, calculate metrics"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Process a single multivariate results folder"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all multivariate results folders"
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
        default="final_results",
        help="Output directory (default: final_results)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        process_all_results(args.results_dir, args.output_dir)
    elif args.folder:
        process_single_result_folder(args.folder, args.output_dir)
    else:
        print("Please specify either --folder or --all")
        print("\nExamples:")
        print("  python process_all_results.py --folder results/ETTh1_token_normal_adapt_N_Samp100_192_5_Transformer_...")
        print("  python process_all_results.py --all")
        print("  python process_all_results.py --all --output-dir my_final_results")
