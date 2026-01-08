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
from utils.discretisize import decode_with_float_vocab
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
    
    # Check if this is a single-variate experiment (_ftS_)
    if '_ftS_' not in folder_name:
        return None
    
    # Extract base dataset name (before first underscore)
    for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity']:
        if folder_name.startswith(ds):
            info['dataset'] = ds
            break
    
    # Extract target column from folder name (after 'target_')
    # Match any characters except underscore followed by 'sl' (to avoid matching other parts)
    target_match = re.search(r'target_(.+?)_sl', folder_name)
    if target_match:
        # Clean up the column name (remove any trailing underscores)
        info['target_column'] = target_match.group(1).strip('_')
    else:
        # Fallback for baseline experiments: dataset_COLUMNNAME_sl pattern
        # Example: ETTh1_HUFL_192_15_Autoformer_...
        if 'dataset' in info:
            baseline_pattern = f"{info['dataset']}_(.+?)_\\d+_\\d+_"
            baseline_match = re.search(baseline_pattern, folder_name)
            if baseline_match:
                info['target_column'] = baseline_match.group(1)
    
    # Extract N_samples
    nsamp_match = re.search(r'N_Samp(\d+)', folder_name)
    if nsamp_match:
        info['n_samples'] = int(nsamp_match.group(1))
    else:
        # Try chronos pattern: _N100_V or _N200_V
        nsamp_chronos = re.search(r'_N(\d+)_V', folder_name)
        if nsamp_chronos:
            info['n_samples'] = int(nsamp_chronos.group(1))
        else:
            # Try alternative pattern for older experiments
            nsamp_alt = re.search(r'_(?:simp|token)_(\d+)_', folder_name)
            if nsamp_alt:
                info['n_samples'] = int(nsamp_alt.group(1))
    
    # Determine discretization type and normalization
    if 'token_normal_adapt' in folder_name:
        info['disc_type'] = 'adaptative'
        info['norm_type'] = 'normal'
    elif 'token_standard_adapt' in folder_name or 'token_stand_adapt' in folder_name:
        info['disc_type'] = 'adaptative'
        info['norm_type'] = 'standard'
    elif 'simp' in folder_name:
        info['disc_type'] = 'simple'
        if 'normal' in folder_name:
            info['norm_type'] = 'normal'
        elif 'standard' in folder_name:
            info['norm_type'] = 'standard'
        else:
            info['norm_type'] = 'normal'
    else:
        info['disc_type'] = 'simple'
        info['norm_type'] = 'normal'
    
    # Extract temporal token indicator
    if '_12h_' in folder_name:
        info['temporal'] = '12h'
    elif '_24h_' in folder_name:
        info['temporal'] = '24h'
    elif '_sem_ebos_' in folder_name:
        info['temporal'] = 'sem_ebos'
    else:
        info['temporal'] = None
    
    # Extract BPE indicator
    if 'COM_BPE' in folder_name:
        info['bpe_type'] = 'COM_BPE'
    elif 'SEM_BPE' in folder_name:
        info['bpe_type'] = 'SEM_BPE'
    elif 'chronos' in folder_name.lower():
        info['bpe_type'] = 'chronos'
    else:
        info['bpe_type'] = None
    
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
            vocab_size = 600
        elif n_samples == 100:
            vocab_size = 600
        elif n_samples == 200:
            vocab_size = 600
        elif n_samples == 202:
            vocab_size = 2200
        elif n_samples == 160:
            vocab_size = 200
        elif n_samples == 60:
            vocab_size = 100
        elif n_samples == 10:
            vocab_size = 50
        else:
            vocab_size = 600
    
    return vocab_size


def build_model_name(dataset, column, n_samples, vocab_size, disc_type, norm_type, temporal=None, bpe_type=None):
    """Build the model filename based on parameters."""
    # Chronos uses a different naming pattern (not column-specific)
    if bpe_type == 'chronos':
        # Chronos pattern: chronos_N{n_samples}_vocab{vocab_size}_target_{disc_type}.model
        model_file = f"chronos_N{n_samples}_vocab{vocab_size}_target_{disc_type}.model"
        vocab_file = f"chronos_N{n_samples}_vocab{vocab_size}_target_{disc_type}.fvocab"
        return model_file, vocab_file
    
    # Base pattern: {dataset}_feature_Nsam_{n_samples}_vocab_{vocab_size}_column_{column}_{disc_type}_{norm_type}
    base_name = f"{dataset}_feature_Nsam_{n_samples}_vocab_{vocab_size}_column_{column}_{disc_type}_{norm_type}"
    
    # Add temporal token indicator if present
    if temporal:
        base_name += f"_{temporal}"
    
    # vocab file name does not include BPE type
    vocab_file = f"{base_name}.fvocab"
    
    # Add BPE indicator to model file name only
    if bpe_type:
        base_name += f"_{bpe_type}"
    
    model_file = f"{base_name}.model"
    
    return model_file, vocab_file


def load_tokenizer_for_column(dataset, column, n_samples, vocab_size, disc_type, norm_type, temporal=None, bpe_type=None):
    """Load the tokenizer model and vocab for a specific column."""
    model_file, vocab_file = build_model_name(dataset, column, n_samples, vocab_size, disc_type, norm_type, temporal, bpe_type)
    
    special_tokens = {'<PAD>': n_samples - 1, '<EBOS>': n_samples}
    
    # For BPE experiments (COM_BPE, chronos), use BasicTokenizer
    if bpe_type in ['COM_BPE', 'chronos']:
        model_path = os.path.join("model", model_file)
        
        # For chronos, also check in scalers/chronos_vocab/ directory
        if not os.path.exists(model_path) and bpe_type == 'chronos':
            alt_model_path = os.path.join("scalers", "chronos_vocab", model_file)
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        tokenizer = BasicTokenizer(n_samples, vocab_file, special_tokens=special_tokens)
        tokenizer.load(model_path)
        
        return tokenizer, model_path, vocab_file
    else:
        # For non-BPE experiments (None, SEM_BPE), just return vocab info
        # No .model file needed - destokenization uses .fvocab directly
        return None, None, vocab_file


def destokenize_column(token_values, tokenizer, vocab_file, special_tokens):
    """Destokenize a column of token values back to float values."""
    # Convert to int, handling NaN
    token_ids = []
    for val in token_values:
        if pd.isna(val):
            token_ids.append(0)  # Use 0 as placeholder for NaN
        else:
            token_ids.append(int(val))
    
    try:
        if tokenizer is not None:
            # BPE case: use tokenizer.decode()
            float_values = tokenizer.decode(token_ids)
        else:
            # Non-BPE case: use decode_with_float_vocab()
            float_values, _ = decode_with_float_vocab(token_ids, vocab_file, special_tokens)
        
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
            'MSPE': np.nan,
            'R2': np.nan,
            'valid_samples': 0
        }
    
    mse = np.mean((trues - preds) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(trues - preds))
    
    # MAPE (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((trues - preds) / np.where(trues != 0, trues, 1))) * 100
    
    # MSPE (Mean Squared Percentage Error)
    with np.errstate(divide='ignore', invalid='ignore'):
        mspe = np.mean(((trues - preds) / np.where(trues != 0, trues, 1)) ** 2) * 100
    
    # R-squared
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MSPE': mspe,
        'R2': r2,
        'valid_samples': len(trues)
    }


def process_baseline_experiment(results_path, folder_name, info, output_base_folder="final_results_trues"):
    """Process baseline experiment - no tokenization, just copy files and metrics."""
    print(f"  Processing as baseline experiment (no destokenization needed)")
    
    # Get target column
    if 'target_column' not in info:
        print(f"  ERROR: Could not extract target column from folder name")
        return None
    
    target_column = info['target_column']
    print(f"  Target column: '{target_column}'")
    
    # Create output folder
    output_folder = Path(output_base_folder) / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy preds and trues files directly (already float values)
    preds_file = results_path / "preds_results.csv"
    trues_file = results_path / "trues_results.csv"
    metrics_file = results_path / "metrics.csv"
    
    if not preds_file.exists() or not trues_file.exists():
        print(f"  ERROR: Missing preds_results.csv or trues_results.csv")
        return None
    
    # Read files
    preds_df = pd.read_csv(preds_file)
    trues_df = pd.read_csv(trues_file)
    
    print(f"  Preds shape: {preds_df.shape}")
    print(f"  Trues shape: {trues_df.shape}")
    
    # Save as detokenized (they're already float values)
    preds_detokenized_file = output_folder / "preds_detokenized.csv"
    trues_detokenized_file = output_folder / "trues_detokenized.csv"
    
    preds_df.to_csv(preds_detokenized_file, index=False)
    trues_df.to_csv(trues_detokenized_file, index=False)
    
    print(f"  [OK] {preds_detokenized_file.name}")
    print(f"  [OK] {trues_detokenized_file.name}")
    
    # Calculate metrics on the float values
    preds_numeric = preds_df.apply(pd.to_numeric, errors='coerce')
    trues_numeric = trues_df.apply(pd.to_numeric, errors='coerce')
    
    metrics_data = []
    total_predictions = len(preds_numeric)
    
    for col in preds_numeric.columns:
        if col not in trues_numeric.columns:
            continue
        
        metrics = calculate_metrics(trues_numeric[col], preds_numeric[col])
        
        metrics_row = {
            'column_index': col,
            'column_name': target_column,
            'model_file': 'N/A (baseline)',
            'vocab_file': 'N/A (baseline)',
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'MSPE': metrics['MSPE'],
            'R2': metrics['R2'],
            'valid_samples': metrics['valid_samples'],
            'total_predictions': total_predictions
        }
        
        metrics_data.append(metrics_row)
        
        print(f"  [{col}] {target_column}: MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.6f}, "
              f"MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f} (valid={metrics['valid_samples']}/{total_predictions})")
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save new metrics
    metrics_output_file = output_folder / "metrics.csv"
    metrics_df.to_csv(metrics_output_file, index=False)
    print(f"  [OK] {metrics_output_file.name}")
    
    # Copy original metrics if exists
    if metrics_file.exists():
        metrics_before_file = output_folder / "metrics_before.csv"
        original_metrics_df = pd.read_csv(metrics_file)
        
        columns_to_keep = ['mae', 'mse', 'rmse', 'mape', 'mspe']
        available_columns = [col for col in columns_to_keep if col in original_metrics_df.columns]
        
        if available_columns:
            original_metrics_df = original_metrics_df[available_columns]
        
        original_metrics_df.to_csv(metrics_before_file, index=False)
        print(f"  [OK] {metrics_before_file.name} (copied from original, cleaned)")
    
    # Create info file
    info_file = output_folder / "models_used.txt"
    with open(info_file, 'w') as f:
        f.write(f"Experiment: {folder_name}\n")
        f.write(f"Type: BASELINE (no tokenization)\n")
        f.write(f"Dataset: {info['dataset']}\n")
        f.write(f"Target column: {target_column}\n")
        f.write(f"Total predictions: {total_predictions}\n")
    
    print(f"  [OK] {info_file.name}")
    
    overall_metrics = {
        'experiment': folder_name,
        'dataset': info['dataset'],
        'n_samples': 0,  # N/A for baseline
        'vocab_size': 0,  # N/A for baseline
        'disc_type': 'baseline',
        'norm_type': 'baseline',
        'avg_MSE': metrics_df['MSE'].mean(),
        'avg_RMSE': metrics_df['RMSE'].mean(),
        'avg_MAE': metrics_df['MAE'].mean(),
        'avg_MAPE': metrics_df['MAPE'].mean(),
        'avg_MSPE': metrics_df['MSPE'].mean(),
        'avg_R2': metrics_df['R2'].mean(),
        'total_columns': len(metrics_df),
        'total_predictions': total_predictions
    }
    
    print(f"\n[OK] Baseline processing complete for {folder_name}\n")
    
    return overall_metrics


def process_single_result_folder(results_folder, output_base_folder=None):
    """Process a single results folder with complete pipeline."""
    results_path = Path(results_folder)
    folder_name = results_path.name
    
    print(f"\n{'='*100}")
    print(f"Processing: {folder_name}")
    print(f"{'='*100}")
    
    # Parse folder information
    info = parse_result_folder_name(folder_name)
    
    if info is None:
        print(f"  SKIP: Not a single-variate experiment (_ftS_ not found)")
        return None
    
    if 'dataset' not in info:
        print(f"  ERROR: Could not parse folder name properly")
        return None
    
    # Determine output folder based on experiment type if not specified
    if output_base_folder is None:
        if info.get('bpe_type') is None:
            output_base_folder = "final_results_trues"
        elif info.get('bpe_type') == 'chronos':
            output_base_folder = "final_results_chronos"
        else:
            output_base_folder = "final_results"
    
    # Check if this is a baseline (no tokenization) experiment
    is_baseline = info.get('bpe_type') is None
    
    if is_baseline:
        print(f"  Type: BASELINE (no tokenization)")
        print(f"  Dataset: {info['dataset']}")
        # For baseline, just copy files and metrics directly
        return process_baseline_experiment(results_path, folder_name, info, output_base_folder)
    
    if 'n_samples' not in info:
        print(f"  ERROR: Could not parse n_samples from folder name")
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
    
    # Get column name - for single-variate experiments, MUST come from folder name
    if 'target_column' not in info:
        print(f"  ERROR: Could not extract target column from folder name")
        print(f"  Expected pattern: 'target_COLUMNNAME_sl' in folder name")
        return None
    
    # Single column experiment - use the target column name from folder
    actual_columns = [info['target_column']]
    
    print(f"  Target column: '{actual_columns[0]}'")
    print(f"  Column mapping: CSV[0] -> '{actual_columns[0]}'")
    
    # STEP 1: Apply ceiling to BOTH prediction and true tokens BEFORE destokenization
    print(f"\n[STEP 1] Applying ceiling to tokens (float -> int)...")
    preds_tokens_ceiled = apply_ceiling_to_tokens(preds_tokens_df)
    trues_tokens_ceiled = apply_ceiling_to_tokens(trues_tokens_df)
    print(f"  Preds sample before ceiling: {preds_tokens_df.iloc[0, 0]}")
    print(f"  Preds sample after ceiling:  {preds_tokens_ceiled.iloc[0, 0]}")
    print(f"  Trues sample before ceiling: {trues_tokens_df.iloc[0, 0]}")
    print(f"  Trues sample after ceiling:  {trues_tokens_ceiled.iloc[0, 0]}")
    
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
            special_tokens = {'<PAD>': info['n_samples'] - 1, '<EBOS>': info['n_samples']}
            
            tokenizer, model_path, vocab_file = load_tokenizer_for_column(
                info['dataset'], 
                actual_col_name,
                info['n_samples'], 
                vocab_size, 
                info['disc_type'], 
                info['norm_type'],
                info.get('temporal'),
                info.get('bpe_type')
            )
            
            float_values = destokenize_column(preds_tokens_ceiled[column], tokenizer, vocab_file, special_tokens)
            preds_detokenized[column] = float_values
            
            models_used[column] = {
                'actual_column': actual_col_name,
                'model_file': model_path if model_path else 'N/A (non-BPE)',
                'vocab_file': vocab_file,
                'bpe_type': info.get('bpe_type', 'None')
            }
            
            print(f"OK")
            
        except Exception as e:
            print(f"ERROR: {e}")
            preds_detokenized[column] = np.nan
            models_used[column] = {
                'actual_column': actual_col_name,
                'model_file': 'ERROR',
                'vocab_file': 'ERROR',
                'error': str(e)
            }
    
    # STEP 3: Destokenize true values (with ceiling applied)
    print(f"\n[STEP 3] Destokenizing true values (tokens -> floats)...")
    trues_detokenized = pd.DataFrame()
    
    for idx, column in enumerate(trues_tokens_ceiled.columns):
        if idx < len(actual_columns):
            actual_col_name = actual_columns[idx]
        else:
            actual_col_name = str(column)
        
        print(f"  Column [{column}] -> '{actual_col_name}'...", end=" ")
        
        try:
            special_tokens = {'<PAD>': info['n_samples'] - 1, '<EBOS>': info['n_samples']}
            
            tokenizer, model_path, vocab_file = load_tokenizer_for_column(
                info['dataset'], 
                actual_col_name,
                info['n_samples'], 
                vocab_size, 
                info['disc_type'], 
                info['norm_type'],
                info.get('temporal'),
                info.get('bpe_type')
            )
            
            float_values = destokenize_column(trues_tokens_ceiled[column], tokenizer, vocab_file, special_tokens)
            trues_detokenized[column] = float_values
            
            print(f"OK")
            
        except Exception as e:
            print(f"ERROR: {e}")
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
            'MSPE': metrics['MSPE'],
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
        'avg_MSPE': metrics_df['MSPE'].mean(),
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
    print(f"  Average MSPE:  {overall_metrics['avg_MSPE']:.6f}")
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
    print(f"  [OK] {preds_tokens_ceiled_file.name}")
    
    trues_tokens_ceiled_file = output_folder / "trues_tokens_ceiled.csv"
    trues_tokens_ceiled.to_csv(trues_tokens_ceiled_file, index=False)
    print(f"  [OK] {trues_tokens_ceiled_file.name}")
    
    # Save detokenized data
    preds_detokenized_file = output_folder / "preds_detokenized.csv"
    preds_detokenized.to_csv(preds_detokenized_file, index=False)
    print(f"  [OK] {preds_detokenized_file.name}")
    
    trues_detokenized_file = output_folder / "trues_detokenized.csv"
    trues_detokenized.to_csv(trues_detokenized_file, index=False)
    print(f"  [OK] {trues_detokenized_file.name}")
    
    # Save metrics
    metrics_file = output_folder / "metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  [OK] {metrics_file.name}")
    
    # Copy original metrics from results folder if it exists
    original_metrics_file = results_path / "metrics.csv"
    if original_metrics_file.exists():
        metrics_before_file = output_folder / "metrics_before.csv"
        original_metrics_df = pd.read_csv(original_metrics_file)
        
        # Remove unwanted columns (corr and rse are not needed)
        columns_to_keep = ['mae', 'mse', 'rmse', 'mape', 'mspe']
        available_columns = [col for col in columns_to_keep if col in original_metrics_df.columns]
        
        if available_columns:
            original_metrics_df = original_metrics_df[available_columns]
        
        original_metrics_df.to_csv(metrics_before_file, index=False)
        print(f"  [OK] {metrics_before_file.name} (copied from original, cleaned)")
    
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
    
    print(f"  [OK] {models_info_file.name}")
    
    print(f"\n[OK] Processing complete for {folder_name}\n")
    
    return overall_metrics


def get_source_folder_for_experiment(folder_name, results_base="results"):
    """Determine which source folder to use based on experiment type."""
    if 'chronos' in folder_name.lower():
        return "results_chronos"
    elif ('_bpe_' not in folder_name.lower() and 
          'COM_BPE' not in folder_name and 
          'SEM_BPE' not in folder_name):
        # Baseline (no tokenization)
        return "results_trues"
    else:
        # Regular tokenized experiments
        return results_base


def process_all_results(results_base_folder="results"):
    """Process all result folders from multiple sources."""
    # Collect folders from all three sources
    result_folders = []
    
    # 1. Regular tokenized experiments (results/)
    results_path = Path(results_base_folder)
    if results_path.exists():
        for folder in results_path.iterdir():
            if folder.is_dir() and '_ftS_' in folder.name and (folder / "preds_results.csv").exists():
                # Skip chronos and baseline in main results folder
                if 'chronos' not in folder.name.lower():
                    if ('COM_BPE' in folder.name or 'SEM_BPE' in folder.name):
                        result_folders.append(folder)
    
    # 2. Chronos experiments (results_chronos/)
    chronos_path = Path("results_chronos")
    if chronos_path.exists():
        for folder in chronos_path.iterdir():
            if folder.is_dir() and '_ftS_' in folder.name and (folder / "preds_results.csv").exists():
                result_folders.append(folder)

    # 3. Baseline/no tokenization experiments (results_trues/)
    trues_path = Path("results_trues")
    if trues_path.exists():
        for folder in trues_path.iterdir():
            if folder.is_dir() and '_ftS_' in folder.name and (folder / "preds_results.csv").exists():
                result_folders.append(folder)
    
    if not result_folders:
        print(f"No single-variate result folders (_ftS_) with preds_results.csv found in any source folder")
        return
    
    print(f"\n{'='*100}")
    print(f"PROCESSING ALL RESULTS")
    print(f"{'='*100}")
    print(f"Source folders:")
    print(f"  - {results_base_folder}/ (regular tokenized experiments)")
    print(f"  - results_chronos/ (chronos experiments)")
    print(f"  - results_trues/ (baseline/no tokenization)")
    print(f"Found {len(result_folders)} single-variate result folders to process")
    print(f"Output folders:")
    print(f"  - final_results/ (regular tokenized experiments)")
    print(f"  - final_results_chronos/ (chronos experiments)")
    print(f"  - final_results_trues/ (baseline/no tokenization)")
    
    all_metrics = []
    successful = 0
    failed = 0
    
    for i, folder in enumerate(result_folders, 1):
        print(f"\n[{i}/{len(result_folders)}]")
        
        try:
            # Don't pass output_base_folder - let process_single_result_folder auto-route
            overall_metrics = process_single_result_folder(folder)
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
        
        # Save summary in final_results (main folder)
        summary_file = Path("final_results") / "summary_all_experiments.csv"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n[OK] Summary saved to: {summary_file}")
        
        print(f"\nTop 5 experiments by average MSE:")
        print(summary_df[['experiment', 'avg_MSE', 'avg_RMSE', 'avg_MAE', 'avg_R2']].head().to_string(index=False))
    
    print(f"\n{'='*100}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"{'='*100}")
    print(f"Results saved in:")
    print(f"  - final_results/ (regular tokenized)")
    print(f"  - final_results_chronos/ (chronos)")
    print(f"  - final_results_trues/ (baseline)")


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
    
    args = parser.parse_args()
    
    if args.all:
        process_all_results(args.results_dir)
    elif args.folder:
        # Don't pass output_dir - let auto-routing determine the folder
        process_single_result_folder(args.folder)
    else:
        print("Please specify either --folder or --all")
        print("\nExamples:")
        print("  python process_all_results.py --folder results/ETTh1_token_normal_adapt_N_Samp100_192_5_Transformer_...")
        print("  python process_all_results.py --all")
        print("\nNote: Output folders are automatically determined:")
        print("  - final_results/ (regular tokenized)")
        print("  - final_results_chronos/ (chronos)")
        print("  - final_results_trues/ (baseline)")
