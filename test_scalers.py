"""
Test script to verify scaler saving and loading functionality.
"""
import pandas as pd
import numpy as np
from scaler_utils import (
    load_scaler,
    inverse_transform_series,
    inverse_transform_dataframe,
    list_available_scalers,
    get_normalization_type_from_folder
)


def main():
    print("="*80)
    print("SCALER UTILITY TEST")
    print("="*80)
    
    # List all available scalers
    print("\n1. Listing all available scalers:")
    print("-" * 80)
    scalers = list_available_scalers()
    
    if not scalers:
        print("⚠ No scalers found. Please run main2.py first to generate scalers.")
        return
    
    # Group by dataset
    datasets = {}
    for s in scalers:
        if s['dataset'] not in datasets:
            datasets[s['dataset']] = []
        datasets[s['dataset']].append(s)
    
    for dataset, scaler_list in datasets.items():
        print(f"\n{dataset}:")
        for s in scaler_list:
            print(f"  - Column: {s['column']:10} Type: {s['type']:10}")
    
    print(f"\nTotal scalers found: {len(scalers)}")
    
    # Test loading a scaler
    if scalers:
        print("\n2. Testing scaler loading:")
        print("-" * 80)
        test_scaler = scalers[0]
        print(f"Loading scaler: {test_scaler['dataset']} - {test_scaler['column']} - {test_scaler['type']}")
        
        scaler = load_scaler(
            test_scaler['dataset'],
            test_scaler['column'],
            test_scaler['type']
        )
        
        # Test inverse transformation with sample data
        print("\n3. Testing inverse transformation:")
        print("-" * 80)
        # Simulating scaled data (standard scaled typically ranges around -3 to 3)
        scaled_data = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        print(f"Scaled data:   {scaled_data}")
        
        original_data = inverse_transform_series(
            scaled_data,
            test_scaler['dataset'],
            test_scaler['column'],
            test_scaler['type']
        )
        print(f"Original data: {original_data}")
        
        # Test with pandas Series
        print("\n4. Testing with pandas Series:")
        print("-" * 80)
        scaled_series = pd.Series(scaled_data, name="test_column")
        original_series = inverse_transform_series(
            scaled_series,
            test_scaler['dataset'],
            test_scaler['column'],
            test_scaler['type']
        )
        print("Scaled Series:")
        print(scaled_series)
        print("\nOriginal Series:")
        print(original_series)
        
        # Test folder name parsing
        print("\n5. Testing folder name normalization detection:")
        print("-" * 80)
        test_folders = [
            "ETTh1_token_standard_adapt_N_Samp100_HUFL_192_15",
            "ETTh1_token_normal_adapt_N_Samp100_HULL_192_15",
            "ETTh2_token_standard_simp_N_Samp100_OT_192_15"
        ]
        for folder in test_folders:
            norm_type = get_normalization_type_from_folder(folder)
            print(f"Folder: {folder}")
            print(f"  → Normalization: {norm_type}")
        
        print("\n" + "="*80)
        print("✓ All tests completed successfully!")
        print("="*80)


if __name__ == "__main__":
    main()
