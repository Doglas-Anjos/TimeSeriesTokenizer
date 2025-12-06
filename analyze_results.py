"""
Utility script to analyze and compare destokenized results.

This script helps you:
1. Compare destokenized predictions with true values
2. Calculate metrics (MSE, MAE, RMSE, etc.)
3. Visualize the results
4. Export comparison reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def calculate_metrics(trues, preds, ignore_nan=True):
    """
    Calculate common regression metrics.
    
    Args:
        trues: True values (array-like)
        preds: Predicted values (array-like)
        ignore_nan: If True, ignore NaN values in calculations
    
    Returns:
        dict: Dictionary of metrics
    """
    trues = np.array(trues)
    preds = np.array(preds)
    
    if ignore_nan:
        # Remove NaN values
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
            'count': 0
        }
    
    # Mean Squared Error
    mse = np.mean((trues - preds) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(trues - preds))
    
    # Mean Absolute Percentage Error (avoid division by zero)
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
        'count': len(trues)
    }


def analyze_results(results_folder):
    """
    Analyze destokenized results in a folder.
    
    Args:
        results_folder: Path to results folder containing destokenized files
    """
    results_path = Path(results_folder)
    
    print(f"\nAnalyzing: {results_path.name}")
    print("=" * 80)
    
    # Check for destokenized files
    preds_file = results_path / "preds_results_detokenized.csv"
    trues_file = results_path / "trues_results_detokenized.csv"
    
    if not preds_file.exists():
        print(f"Error: {preds_file} not found")
        return
    
    if not trues_file.exists():
        print(f"Error: {trues_file} not found")
        return
    
    # Load data
    preds_df = pd.read_csv(preds_file)
    trues_df = pd.read_csv(trues_file)
    
    print(f"\nData Shape:")
    print(f"  Predictions: {preds_df.shape}")
    print(f"  True Values: {trues_df.shape}")
    
    # Check for special tokens
    special_tokens_found = False
    for col in preds_df.columns:
        if preds_df[col].dtype == 'object':
            special_tokens = preds_df[col].value_counts()
            if len(special_tokens) > 0:
                special_tokens_found = True
                print(f"\n  Warning: Special tokens found in predictions column '{col}':")
                print(f"    {special_tokens.to_dict()}")
    
    for col in trues_df.columns:
        if trues_df[col].dtype == 'object':
            special_tokens = trues_df[col].value_counts()
            if len(special_tokens) > 0:
                special_tokens_found = True
                print(f"\n  Warning: Special tokens found in true values column '{col}':")
                print(f"    {special_tokens.to_dict()}")
    
    # Convert to numeric (will turn special tokens to NaN)
    preds_numeric = preds_df.apply(pd.to_numeric, errors='coerce')
    trues_numeric = trues_df.apply(pd.to_numeric, errors='coerce')
    
    # Calculate metrics per column
    print(f"\nMetrics per Column:")
    print("-" * 80)
    
    all_metrics = {}
    
    for col in preds_numeric.columns:
        if col not in trues_numeric.columns:
            print(f"  Warning: Column '{col}' not found in true values")
            continue
        
        metrics = calculate_metrics(trues_numeric[col], preds_numeric[col])
        all_metrics[col] = metrics
        
        print(f"\n  Column: {col}")
        print(f"    Valid samples: {metrics['count']}")
        print(f"    MSE:  {metrics['MSE']:.6f}")
        print(f"    RMSE: {metrics['RMSE']:.6f}")
        print(f"    MAE:  {metrics['MAE']:.6f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")
        print(f"    R²:   {metrics['R2']:.6f}")
    
    # Overall metrics (average across columns)
    print(f"\n\nOverall Metrics (averaged across columns):")
    print("-" * 80)
    
    avg_mse = np.nanmean([m['MSE'] for m in all_metrics.values()])
    avg_rmse = np.nanmean([m['RMSE'] for m in all_metrics.values()])
    avg_mae = np.nanmean([m['MAE'] for m in all_metrics.values()])
    avg_mape = np.nanmean([m['MAPE'] for m in all_metrics.values()])
    avg_r2 = np.nanmean([m['R2'] for m in all_metrics.values()])
    
    print(f"  Average MSE:  {avg_mse:.6f}")
    print(f"  Average RMSE: {avg_rmse:.6f}")
    print(f"  Average MAE:  {avg_mae:.6f}")
    print(f"  Average MAPE: {avg_mape:.2f}%")
    print(f"  Average R²:   {avg_r2:.6f}")
    
    # NaN statistics
    print(f"\n\nNaN Statistics:")
    print("-" * 80)
    
    for col in preds_numeric.columns:
        preds_nan_count = preds_numeric[col].isna().sum()
        trues_nan_count = trues_numeric[col].isna().sum()
        
        if preds_nan_count > 0 or trues_nan_count > 0:
            preds_nan_pct = (preds_nan_count / len(preds_numeric)) * 100
            trues_nan_pct = (trues_nan_count / len(trues_numeric)) * 100
            
            print(f"  Column '{col}':")
            print(f"    Predictions: {preds_nan_count} NaN values ({preds_nan_pct:.2f}%)")
            print(f"    True values: {trues_nan_count} NaN values ({trues_nan_pct:.2f}%)")
    
    # Export metrics to CSV
    metrics_output = results_path / "destokenized_metrics.csv"
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'column'
    metrics_df.to_csv(metrics_output)
    print(f"\n\nMetrics saved to: {metrics_output}")
    
    return all_metrics


def compare_multiple_results(results_folders):
    """
    Compare metrics across multiple result folders.
    
    Args:
        results_folders: List of paths to results folders
    """
    print("\n" + "=" * 80)
    print("COMPARING MULTIPLE RESULTS")
    print("=" * 80)
    
    all_folder_metrics = {}
    
    for folder in results_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"\nWarning: Folder not found: {folder}")
            continue
        
        metrics = analyze_results(folder)
        if metrics:
            all_folder_metrics[folder_path.name] = metrics
    
    if len(all_folder_metrics) == 0:
        print("\nNo valid results to compare")
        return
    
    # Create comparison table
    print("\n\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    comparison_data = []
    
    for folder_name, metrics in all_folder_metrics.items():
        avg_mse = np.nanmean([m['MSE'] for m in metrics.values()])
        avg_rmse = np.nanmean([m['RMSE'] for m in metrics.values()])
        avg_mae = np.nanmean([m['MAE'] for m in metrics.values()])
        avg_r2 = np.nanmean([m['R2'] for m in metrics.values()])
        
        comparison_data.append({
            'Experiment': folder_name[:60],  # Truncate long names
            'MSE': avg_mse,
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'R²': avg_r2
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MSE')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_output = Path("results_comparison.csv")
    comparison_df.to_csv(comparison_output, index=False)
    print(f"\n\nComparison saved to: {comparison_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and compare destokenized results"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Single results folder to analyze"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Multiple results folders to compare"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all destokenized results in results/ folder"
    )
    
    args = parser.parse_args()
    
    if args.folder:
        analyze_results(args.folder)
    elif args.compare:
        compare_multiple_results(args.compare)
    elif args.all:
        results_dir = Path("results")
        folders = []
        for folder in results_dir.iterdir():
            if folder.is_dir() and (folder / "preds_results_detokenized.csv").exists():
                folders.append(str(folder))
        
        if folders:
            compare_multiple_results(folders)
        else:
            print("No destokenized results found in results/ folder")
    else:
        print("Please specify --folder, --compare, or --all")
        print("\nExamples:")
        print("  python analyze_results.py --folder results/ETTh1_token_normal_adapt_N_Samp100_192_5_Transformer_...")
        print("  python analyze_results.py --compare results/exp1 results/exp2 results/exp3")
        print("  python analyze_results.py --all")
