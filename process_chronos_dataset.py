import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
import joblib
from utils.discretisize import simple_discretize, adaptative_bins_discretize, save_float_vocab
from utils.token_based import TokenBasedTokenizer
from collections import defaultdict
import random

# Configuration
CHRONOS_FOLDER = 'chronos_csv'
INDEX_FILE = 'chronos_file_index.json'
OUTPUT_FOLDER = 'chronos_processed'
CHUNK_SIZE = 500
TARGET_TOTAL_ROWS = 100000
MAX_ZERO_RATIO = 0.3  # Max 30% zeros in a chunk to avoid zero-heavy sequences
DIVISION_FACTOR = 10

# Tokenization configurations: (N_samples, total_vocab_size)
TOKENIZATION_CONFIGS = [
    (50, 600),
    (100, 600),
    (200, 600),
]


def create_file_index():
    """Create or load index of CSV files in chronos_csv folder."""
    if os.path.exists(INDEX_FILE):
        print(f"Loading existing file index from {INDEX_FILE}")
        with open(INDEX_FILE, 'r') as f:
            return json.load(f)
    
    print(f"Creating new file index from {CHRONOS_FOLDER}")
    chronos_path = Path(CHRONOS_FOLDER)
    
    if not chronos_path.exists():
        raise FileNotFoundError(f"Folder {CHRONOS_FOLDER} not found!")
    
    # Recursively find all CSV files in subfolders (batch files)
    csv_files = list(chronos_path.rglob("*.csv"))
    
    file_index = {
        'total_files': len(csv_files),
        'files': [str(f.relative_to(chronos_path.parent)) for f in csv_files],
        'created': pd.Timestamp.now().isoformat()
    }
    
    with open(INDEX_FILE, 'w') as f:
        json.dump(file_index, f, indent=2)
    
    print(f"✓ Indexed {len(csv_files)} CSV files")
    return file_index


def calculate_zero_ratio(chunk_data):
    """Calculate the ratio of zeros in a chunk (excluding NaN)."""
    numeric_data = chunk_data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return 1.0
    
    total_values = numeric_data.notna().sum().sum()
    if total_values == 0:
        return 1.0
    
    zero_count = (numeric_data == 0).sum().sum()
    return zero_count / total_values


def extract_chunk_from_file(file_path, chunk_size, used_indices_set, max_attempts=10):
    """
    Extract a chunk from a file avoiding zero-heavy sequences and previously used indices.
    Filters out irrelevant columns (lat/lon/id) and selects single value column.
    Standardizes data before extracting chunk.
    
    Returns: (chunk_df, start_idx, end_idx, folder_name) or (None, None, None, None) if no good chunk found
    """
    try:
        df = pd.read_csv(file_path)
        
        # Get folder name for statistics
        folder_name = Path(file_path).parent.name
        
        # Remove irrelevant columns (latitude, longitude, id-related)
        columns_to_drop = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['lat', 'lon', 'latitude', 'longitude', 'id', 'item_id', 'series_id', 'ts_id', 'unique_id']):
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        # Find the relevant value column (target, capacity_mw, power_mw, etc.)
        value_column = None
        for col_name in ['target', 'value', 'capacity_mw', 'power_mw', 'demand', 'load', 'price']:
            if col_name in df.columns:
                value_column = col_name
                break
        
        if value_column is None:
            # Try to find any numeric column that's not a date/time
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if 'date' not in col.lower() and 'time' not in col.lower():
                    value_column = col
                    break
        
        if value_column is None or value_column not in df.columns:
            return None, None, None, None
        
        # Keep only the value column and rename it to 'target' for consistency
        df = df[[value_column]].rename(columns={value_column: 'target'})
        
        # Standardize the data
        scaler = StandardScaler()
        df_standardized = df.copy()
        non_null_mask = df['target'].notna()
        
        if non_null_mask.sum() > 0:
            df_standardized.loc[non_null_mask, 'target'] = scaler.fit_transform(
                df.loc[non_null_mask, 'target'].values.reshape(-1, 1)
            ).ravel()
        else:
            return None, None, None, None
        
        # If file is smaller than chunk size, return all data if not used yet
        if len(df_standardized) <= chunk_size:
            if 0 not in used_indices_set:
                zero_ratio = calculate_zero_ratio(df_standardized)
                if zero_ratio <= MAX_ZERO_RATIO:
                    used_indices_set.add((0, len(df_standardized)))
                    return df_standardized, 0, len(df_standardized), folder_name
            return None, None, None, None
        
        # Try to find a good chunk
        for attempt in range(max_attempts):
            start_idx = random.randint(0, len(df_standardized) - chunk_size)
            end_idx = start_idx + chunk_size
            
            # Check if this range overlaps with used indices
            overlaps = False
            for used_start, used_end in used_indices_set:
                if not (end_idx <= used_start or start_idx >= used_end):
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            chunk = df_standardized.iloc[start_idx:end_idx]
            zero_ratio = calculate_zero_ratio(chunk)
            
            if zero_ratio <= MAX_ZERO_RATIO:
                used_indices_set.add((start_idx, end_idx))
                return chunk, start_idx, end_idx, folder_name
        
        return None, None, None, None
    
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None, None, None, None


def gather_data_chunks(file_index):
    """Gather chunks from all files to reach target total rows."""
    print(f"\n{'='*60}")
    print(f"GATHERING DATA CHUNKS (Target: {TARGET_TOTAL_ROWS:,} rows)")
    print(f"{'='*60}")
    
    all_chunks = []
    folder_stats = defaultdict(lambda: {'chunks': 0, 'rows': 0})  # Track by folder
    file_stats = defaultdict(lambda: {'chunks': 0, 'rows': 0})    # Still track files for detail
    used_indices = defaultdict(set)  # file -> set of (start, end) tuples
    
    total_rows = 0
    files_list = file_index['files']
    random.shuffle(files_list)  # Randomize file order for diversity
    
    attempt = 0
    max_attempts = len(files_list) * 20  # Allow multiple passes through files
    
    while total_rows < TARGET_TOTAL_ROWS and attempt < max_attempts:
        file_path = files_list[attempt % len(files_list)]
        attempt += 1
        
        chunk, start_idx, end_idx, folder_name = extract_chunk_from_file(
            file_path, CHUNK_SIZE, used_indices[file_path]
        )
        
        if chunk is not None and folder_name is not None:
            chunk_rows = len(chunk)
            all_chunks.append({
                'data': chunk,
                'file': file_path,
                'folder': folder_name,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            
            # Track by folder
            folder_stats[folder_name]['chunks'] += 1
            folder_stats[folder_name]['rows'] += chunk_rows
            
            # Track by file for detailed view
            file_stats[file_path]['chunks'] += 1
            file_stats[file_path]['rows'] += chunk_rows
            
            total_rows += chunk_rows
            
            if total_rows % 10000 < CHUNK_SIZE:  # Progress update
                print(f"  Progress: {total_rows:,} / {TARGET_TOTAL_ROWS:,} rows gathered...")
    
    print(f"\n✓ Gathered {total_rows:,} rows from {len(folder_stats)} folders")
    return all_chunks, folder_stats, file_stats, total_rows


def combine_and_standardize_chunks(chunks):
    """Combine all chunks (already standardized per-file)."""
    print(f"\n{'='*60}")
    print(f"COMBINING DATA (already standardized per-file)")
    print(f"{'='*60}")
    
    # Combine all chunks
    combined_df = pd.concat([c['data'] for c in chunks], ignore_index=True)
    print(f"  Combined shape: {combined_df.shape}")
    
    # Get numeric columns only
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric columns: {len(numeric_columns)}")
    
    if not numeric_columns:
        raise ValueError("No numeric columns found in the data!")
    
    # Save combined data (already standardized)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    combined_csv_path = os.path.join(OUTPUT_FOLDER, 'chronos_combined_data.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"  ✓ Saved combined data to {combined_csv_path}")
    
    return combined_df, numeric_columns


def tokenize_column(data_series, N_samples, total_vocab_size, column_name, disc_type):
    """Tokenize a single column using both simple and adaptative discretization."""
    # Remove NaN values
    clean_data = data_series.dropna()
    
    if len(clean_data) < 10:
        return None, None
    
    special_tokens = {'<PAD>': N_samples - 1, '<EBOS>': N_samples}
    
    results = {}
    
    # Simple discretization
    try:
        y_simple_tok, bin_edges = simple_discretize(clean_data.values, N_samples, None, special_tokens=special_tokens)
        base_name = f"chronos_N{N_samples}_vocab{total_vocab_size}_{column_name}_simple"
        vocab_file = f"{base_name}.fvocab"
        save_float_vocab(bin_edges.tolist(), vocab_file)
        
        # Train tokenizer
        model_path = os.path.join('model', f"{base_name}.model")
        if not os.path.exists(model_path):
            tokenizer = TokenBasedTokenizer(N_samples, vocab_file, special_tokens=special_tokens)
            tokenizer.train(y_simple_tok, total_vocab_size, verbose=False)
            tokenizer.save(base_name, vocab_file)
        else:
            tokenizer = TokenBasedTokenizer(N_samples, vocab_file, special_tokens=special_tokens)
            tokenizer.load(model_path)
        
        encoded = tokenizer.encode(y_simple_tok)
        
        # Calculate compression
        original_size = len(y_simple_tok)
        compressed_size = len(encoded)
        compression_rate = original_size / compressed_size if compressed_size > 0 else 0
        
        results['simple'] = {
            'encoded': encoded,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_rate': compression_rate
        }
    except Exception as e:
        print(f"    Warning: Simple discretization failed for {column_name}: {e}")
    
    # Adaptative discretization
    try:
        edges, y_adapt_tok, alloc = adaptative_bins_discretize(
            clean_data.values, N=N_samples, K=DIVISION_FACTOR, data_st=None, special_tokens=special_tokens
        )
        base_name = f"chronos_N{N_samples}_vocab{total_vocab_size}_{column_name}_adaptative"
        vocab_file = f"{base_name}.fvocab"
        save_float_vocab(edges.tolist(), vocab_file)
        
        # Train tokenizer
        model_path = os.path.join('model', f"{base_name}.model")
        if not os.path.exists(model_path):
            tokenizer = TokenBasedTokenizer(N_samples, vocab_file, special_tokens=special_tokens)
            tokenizer.train(y_adapt_tok, total_vocab_size, verbose=False)
            tokenizer.save(base_name, vocab_file)
        else:
            tokenizer = TokenBasedTokenizer(N_samples, vocab_file, special_tokens=special_tokens)
            tokenizer.load(model_path)
        
        encoded = tokenizer.encode(y_adapt_tok)
        
        # Calculate compression
        original_size = len(y_adapt_tok)
        compressed_size = len(encoded)
        compression_rate = original_size / compressed_size if compressed_size > 0 else 0
        
        results['adaptative'] = {
            'encoded': encoded,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_rate': compression_rate
        }
    except Exception as e:
        print(f"    Warning: Adaptative discretization failed for {column_name}: {e}")
    
    return results


def tokenize_dataset(df, numeric_columns):
    """Tokenize the dataset with multiple configurations."""
    print(f"\n{'='*60}")
    print(f"TOKENIZING DATASET")
    print(f"{'='*60}")
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('float_vocab', exist_ok=True)
    
    tokenization_results = {}
    
    for N_samples, total_vocab_size in TOKENIZATION_CONFIGS:
        print(f"\n  Configuration: N={N_samples}, Vocab={total_vocab_size}")
        config_key = f"N{N_samples}_V{total_vocab_size}"
        tokenization_results[config_key] = {}
        
        for col in numeric_columns[:5]:  # Process first 5 columns to avoid too long execution
            print(f"    Processing column: {col}")
            results = tokenize_column(df[col], N_samples, total_vocab_size, col, 'both')
            
            if results:
                tokenization_results[config_key][col] = results
    
    return tokenization_results


def generate_gathering_report(folder_stats, file_stats, total_rows):
    """Generate data gathering report."""
    print(f"\n{'='*60}")
    print(f"DATA GATHERING REPORT")
    print(f"{'='*60}")
    
    print(f"\n1. DATA GATHERING STATISTICS (BY FOLDER)")
    print(f"   {'─'*56}")
    print(f"   Total rows gathered: {total_rows:,}")
    print(f"   Folders used: {len(folder_stats)}")
    print(f"\n   Per-folder breakdown:")
    print(f"   {'Folder':<50} {'Chunks':<8} {'Rows':<10} {'%':<8}")
    print(f"   {'-'*56}")
    
    sorted_folders = sorted(folder_stats.items(), key=lambda x: x[1]['rows'], reverse=True)
    for folder_name, stats in sorted_folders:
        percentage = (stats['rows'] / total_rows * 100) if total_rows > 0 else 0
        print(f"   {folder_name:<50} {stats['chunks']:<8} {stats['rows']:<10,} {percentage:>6.2f}%")
    
    # Save gathering report
    report_path = os.path.join(OUTPUT_FOLDER, 'gathering_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATA GATHERING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total rows gathered: {total_rows:,}\n")
        f.write(f"Folders used: {len(folder_stats)}\n\n")
        f.write("Folder breakdown:\n")
        for folder_name, stats in sorted_folders:
            percentage = (stats['rows'] / total_rows * 100) if total_rows > 0 else 0
            f.write(f"  {folder_name}: {stats['rows']:,} rows ({percentage:.2f}%)\n")
    
    print(f"\n✓ Gathering report saved to {report_path}")


def generate_tokenization_report(tokenization_results):
    """Generate tokenization results report."""
    print(f"\n{'='*60}")
    print(f"TOKENIZATION REPORT")
    print(f"{'='*60}")
    
    for config_key, columns_results in tokenization_results.items():
        print(f"\n   Configuration: {config_key}")
        print(f"   {'-'*56}")
        
        if not columns_results:
            print(f"   No results available")
            continue
        
        for col, disc_results in columns_results.items():
            print(f"\n   Column: {col}")
            
            if 'simple' in disc_results:
                simple = disc_results['simple']
                print(f"     Simple Discretization:")
                print(f"       Original size: {simple['original_size']:,} tokens")
                print(f"       Compressed size: {simple['compressed_size']:,} tokens")
                print(f"       Compression rate: {simple['compression_rate']:.2f}x")
            
            if 'adaptative' in disc_results:
                adapt = disc_results['adaptative']
                print(f"     Adaptative Discretization:")
                print(f"       Original size: {adapt['original_size']:,} tokens")
                print(f"       Compressed size: {adapt['compressed_size']:,} tokens")
                print(f"       Compression rate: {adapt['compression_rate']:.2f}x")
    
    # Save tokenization report
    report_path = os.path.join(OUTPUT_FOLDER, 'tokenization_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TOKENIZATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        for config_key, columns_results in tokenization_results.items():
            f.write(f"\nConfiguration: {config_key}\n")
            f.write("-"*60 + "\n")
            
            if not columns_results:
                f.write("No results available\n")
                continue
            
            for col, disc_results in columns_results.items():
                f.write(f"\nColumn: {col}\n")
                
                if 'simple' in disc_results:
                    simple = disc_results['simple']
                    f.write(f"  Simple Discretization:\n")
                    f.write(f"    Original size: {simple['original_size']:,} tokens\n")
                    f.write(f"    Compressed size: {simple['compressed_size']:,} tokens\n")
                    f.write(f"    Compression rate: {simple['compression_rate']:.2f}x\n")
                
                if 'adaptative' in disc_results:
                    adapt = disc_results['adaptative']
                    f.write(f"  Adaptative Discretization:\n")
                    f.write(f"    Original size: {adapt['original_size']:,} tokens\n")
                    f.write(f"    Compressed size: {adapt['compressed_size']:,} tokens\n")
                    f.write(f"    Compression rate: {adapt['compression_rate']:.2f}x\n")
        
        f.write(f"\n\nTokenization configurations:\n")
        for N, V in TOKENIZATION_CONFIGS:
            f.write(f"  N={N}, Vocab={V}\n")
    
    print(f"\n✓ Tokenization report saved to {report_path}")


def main():
    """Main execution flow."""
    print(f"{'='*60}")
    print(f"CHRONOS DATASET PROCESSOR")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Chunk size: {CHUNK_SIZE} rows")
    print(f"  Target total: {TARGET_TOTAL_ROWS:,} rows")
    print(f"  Max zero ratio: {MAX_ZERO_RATIO:.1%}")
    print(f"  Tokenization configs: {TOKENIZATION_CONFIGS}")
    
    # Check if combined file already exists
    combined_csv_path = os.path.join(OUTPUT_FOLDER, 'chronos_combined_data.csv')
    
    if os.path.exists(combined_csv_path):
        print(f"\n✓ Found existing combined file: {combined_csv_path}")
        print(f"  Skipping data gathering and combining steps...")
        print(f"  Loading data for tokenization...")
        
        combined_df = pd.read_csv(combined_csv_path)
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"  Loaded shape: {combined_df.shape}")
        print(f"  Numeric columns: {len(numeric_columns)}")
        
        # Skip to tokenization
        tokenization_results = tokenize_dataset(combined_df, numeric_columns)
        generate_tokenization_report(tokenization_results)
        
    else:
        print(f"\n  Combined file not found. Starting full processing...")
        
        # Step 1: Create or load file index
        file_index = create_file_index()
        
        # Step 2: Gather data chunks
        chunks, folder_stats, file_stats, total_rows = gather_data_chunks(file_index)
        
        if total_rows == 0:
            print("\nError: No data gathered. Check your files and configuration.")
            return
        
        # Step 3: Combine data (already standardized)
        combined_df, numeric_columns = combine_and_standardize_chunks(chunks)
        
        # Step 4: Generate gathering report
        generate_gathering_report(folder_stats, file_stats, total_rows)
        
        # Step 5: Tokenize dataset
        tokenization_results = tokenize_dataset(combined_df, numeric_columns)
        
        # Step 6: Generate tokenization report
        generate_tokenization_report(tokenization_results)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
