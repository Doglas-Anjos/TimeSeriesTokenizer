import argparse
from pathlib import Path

import datasets
import pandas as pd
from datasets import get_dataset_config_names, load_dataset_builder, load_dataset


# VERY LARGE configs (you can remove from this set if you really want them)
SKIP_CONFIGS = {
    "weatherbench_daily",
    "weatherbench_hourly_temperature",
    "training_corpus_tsmixup_10m",
    "training_corpus_kernel_synth_1m",
}


def to_pandas_long(ds: datasets.Dataset) -> pd.DataFrame:
    """
    Convert a Chronos dataset to a long-format DataFrame.

    Based on the official dataset card example:
    - Find all Sequence columns (timestamp, target, etc.)
    - Explode them so each row becomes one timestamped observation.
    """
    sequence_columns = [
        col for col in ds.features
        if isinstance(ds.features[col], datasets.Sequence)
    ]
    return ds.to_pandas().explode(sequence_columns).infer_objects()


def to_pandas_long_chunked_streaming(ds: datasets.Dataset, output_folder: Path, chunk_size: int = 10):
    """
    Memory-efficient conversion that saves chunks directly to separate files.
    
    Instead of loading everything into memory, saves each chunk as a separate CSV.
    Returns the number of chunk files created.
    """
    sequence_columns = [
        col for col in ds.features
        if isinstance(ds.features[col], datasets.Sequence)
    ]
    
    total_rows = len(ds)
    chunk_count = 0
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"        Processing rows {start_idx} to {end_idx} of {total_rows}...")
        
        try:
            # Process a small chunk
            chunk_ds = ds.select(range(start_idx, end_idx))
            chunk_df = chunk_ds.to_pandas().explode(sequence_columns).infer_objects()
            
            # Save immediately to disk
            chunk_file = output_folder / f"chunk_{chunk_count:04d}.csv"
            chunk_df.to_csv(chunk_file, index=False)
            chunk_count += 1
            
        except Exception as e:
            print(f"        Warning: Failed to process chunk {start_idx}-{end_idx}: {e}")
            continue
    
    return chunk_count


def to_pandas_long_chunked(ds: datasets.Dataset, chunk_size: int = 100) -> pd.DataFrame:
    """
    Memory-efficient conversion to long-format DataFrame using chunked processing.
    
    Processes the dataset in chunks to avoid loading entire exploded data into memory.
    """
    sequence_columns = [
        col for col in ds.features
        if isinstance(ds.features[col], datasets.Sequence)
    ]
    
    chunks = []
    total_rows = len(ds)
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"        Processing rows {start_idx} to {end_idx} of {total_rows}...")
        
        # Process a small chunk
        chunk_ds = ds.select(range(start_idx, end_idx))
        chunk_df = chunk_ds.to_pandas().explode(sequence_columns).infer_objects()
        chunks.append(chunk_df)
    
    print(f"        Concatenating {len(chunks)} chunks...")
    return pd.concat(chunks, ignore_index=True)


def export_config_to_csv(config_name: str, out_dir: Path, long_format: bool = True):
    if config_name in SKIP_CONFIGS:
        print(f"[SKIP] Config '{config_name}' is in SKIP_CONFIGS (very large).")
        return

    print(f"\n=== Processing config: {config_name} ===")

    # Get available splits (train, validation, test, etc.)
    builder = load_dataset_builder("autogluon/chronos_datasets", config_name)
    splits = list(builder.info.splits.keys())

    if not splits:
        print(f"No splits found for config '{config_name}', skipping.")
        return

    for split in splits:
        # Create a folder for this config_split combination
        split_folder = out_dir / f"chronos_{config_name}_{split}"
        
        # Skip if folder already exists and has files
        if split_folder.exists() and any(split_folder.glob("*.csv")):
            print(f"  -> Split '{split}' already exists at {split_folder}, skipping...")
            continue
        
        split_folder.mkdir(parents=True, exist_ok=True)
            
        print(f"  -> Loading split: {split}")
        ds = load_dataset("autogluon/chronos_datasets", config_name, split=split)

        if long_format:
            print("     Converting to long-format and saving in chunks (streaming mode for memory efficiency)...")
            # Use streaming approach - save chunks directly without loading all into memory
            try:
                chunk_count = to_pandas_long_chunked_streaming(ds, split_folder, chunk_size=10)
                print(f"     ✓ Saved {chunk_count} chunk files to {split_folder}")
                
                # Now we need to process the chunk files and batch them by ID
                print(f"     Batching chunks by ID...")
                all_chunks = []
                for chunk_file in sorted(split_folder.glob("chunk_*.csv")):
                    chunk_df = pd.read_csv(chunk_file)
                    all_chunks.append(chunk_df)
                
                if all_chunks:
                    df = pd.concat(all_chunks, ignore_index=True)
                    
                    # Delete temporary chunk files
                    for chunk_file in split_folder.glob("chunk_*.csv"):
                        chunk_file.unlink()
                else:
                    print(f"     Warning: No chunks were created")
                    continue
                    
            except Exception as e:
                print(f"     Error in streaming mode: {e}")
                continue
        else:
            print("     Converting to wide-format pandas DataFrame...")
            df = ds.to_pandas()

        # Check if there's an 'id' or similar column to split by
        id_column = None
        for col_name in ['id', 'item_id', 'series_id', 'ts_id', 'unique_id']:
            if col_name in df.columns:
                id_column = col_name
                break
        
        if id_column and df[id_column].nunique() > 1:
            print(f"     Splitting by '{id_column}' column into batched files (1000 IDs per file)...")
            unique_ids = df[id_column].unique()
            total_ids = len(unique_ids)
            batch_size = 1000
            
            file_count = 0
            for batch_start in range(0, total_ids, batch_size):
                batch_end = min(batch_start + batch_size, total_ids)
                batch_ids = unique_ids[batch_start:batch_end]
                
                # Filter dataframe for this batch of IDs
                batch_df = df[df[id_column].isin(batch_ids)]
                
                # Save batch file
                out_path = split_folder / f"batch_{file_count:04d}_ids_{batch_start}_to_{batch_end-1}.csv"
                batch_df.to_csv(out_path, index=False)
                file_count += 1
                
                print(f"       Saved batch {file_count}: {len(batch_ids)} IDs ({batch_start} to {batch_end-1})")
             
        else:
            # No id column found or only one unique id, save as single file
            print(f"     No suitable ID column found, saving as single file...")
            out_path = split_folder / "data.csv"
            df.to_csv(out_path, index=False)
            print(f"     ✓ Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download autogluon/chronos_datasets and export all configs/splits to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chronos_csv",
        help="Directory to store CSV files (default: ./chronos_csv)",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Use wide format (one row per time series, sequences as list-like columns) instead of long format.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching config names from 'autogluon/chronos_datasets'...")
    config_names = get_dataset_config_names("autogluon/chronos_datasets")
    print(f"Found {len(config_names)} configs:")
    for name in config_names:
        print(f"  - {name}")

    long_format = not args.wide

    for config_name in config_names:
        export_config_to_csv(config_name, out_dir, long_format=long_format)

    print("\nDone.")


if __name__ == "__main__":
    main()
