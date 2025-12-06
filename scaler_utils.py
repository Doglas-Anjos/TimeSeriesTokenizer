"""
Utility functions for loading and applying inverse transformations using saved scalers.
"""
import joblib
import pandas as pd
import numpy as np
import os


def load_scaler(dataset_name, column_name, scaler_type='standard'):
    """
    Load a saved scaler for a specific dataset and column.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'ETTh1', 'ETTh2')
    column_name : str
        Name of the column (e.g., 'HUFL', 'HULL', 'OT')
    scaler_type : str
        Type of scaler: 'standard' or 'minmax'
    
    Returns:
    --------
    scaler : sklearn scaler object
        Loaded scaler ready for inverse_transform
    
    Example:
    --------
    >>> scaler = load_scaler('ETTh1', 'HUFL', 'standard')
    >>> original_values = scaler.inverse_transform(scaled_values.reshape(-1, 1))
    """
    scaler_path = f"scalers/{dataset_name}_column_{column_name}_{scaler_type}.pkl"
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded scaler from: {scaler_path}")
    return scaler


def inverse_transform_series(values, dataset_name, column_name, scaler_type='standard'):
    """
    Apply inverse transformation to a series of scaled values.
    
    Parameters:
    -----------
    values : array-like or pd.Series
        Scaled values to transform back to original scale
    dataset_name : str
        Name of the dataset (e.g., 'ETTh1', 'ETTh2')
    column_name : str
        Name of the column (e.g., 'HUFL', 'HULL', 'OT')
    scaler_type : str
        Type of scaler: 'standard' or 'minmax'
    
    Returns:
    --------
    original_values : np.ndarray or pd.Series
        Values transformed back to original scale
    
    Example:
    --------
    >>> original = inverse_transform_series(scaled_data, 'ETTh1', 'HUFL', 'standard')
    """
    scaler = load_scaler(dataset_name, column_name, scaler_type)
    
    # Convert to numpy array if pandas Series
    if isinstance(values, pd.Series):
        values_array = values.values.reshape(-1, 1)
        result = scaler.inverse_transform(values_array).ravel()
        return pd.Series(result, index=values.index, name=values.name)
    else:
        values_array = np.array(values).reshape(-1, 1)
        return scaler.inverse_transform(values_array).ravel()


def inverse_transform_dataframe(df, dataset_name, scaler_type='standard'):
    """
    Apply inverse transformation to all columns in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with scaled values (columns should match saved scaler names)
    dataset_name : str
        Name of the dataset (e.g., 'ETTh1', 'ETTh2')
    scaler_type : str
        Type of scaler: 'standard' or 'minmax'
    
    Returns:
    --------
    df_original : pd.DataFrame
        DataFrame with values transformed back to original scale
    
    Example:
    --------
    >>> original_df = inverse_transform_dataframe(scaled_df, 'ETTh1', 'standard')
    """
    df_original = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        print(f"Inverse transforming column: {column}")
        df_original[column] = inverse_transform_series(
            df[column], dataset_name, column, scaler_type
        )
    
    return df_original


def list_available_scalers(dataset_name=None):
    """
    List all available scalers, optionally filtered by dataset.
    
    Parameters:
    -----------
    dataset_name : str, optional
        Filter scalers by dataset name
    
    Returns:
    --------
    scalers : list of dict
        List of available scalers with their metadata
    
    Example:
    --------
    >>> scalers = list_available_scalers('ETTh1')
    >>> for s in scalers:
    ...     print(f"{s['dataset']} - {s['column']} - {s['type']}")
    """
    scaler_dir = 'scalers'
    if not os.path.exists(scaler_dir):
        print(f"Scaler directory not found: {scaler_dir}")
        return []
    
    scalers = []
    for filename in os.listdir(scaler_dir):
        if filename.endswith('.pkl'):
            # Parse filename: {dataset}_column_{column}_{type}.pkl
            parts = filename.replace('.pkl', '').split('_column_')
            if len(parts) == 2:
                dataset = parts[0]
                column_and_type = parts[1].rsplit('_', 1)
                if len(column_and_type) == 2:
                    column, scaler_type = column_and_type
                    
                    if dataset_name is None or dataset == dataset_name:
                        scalers.append({
                            'dataset': dataset,
                            'column': column,
                            'type': scaler_type,
                            'path': os.path.join(scaler_dir, filename)
                        })
    
    return scalers


def get_normalization_type_from_folder(folder_name):
    """
    Extract normalization type from result folder name.
    
    Parameters:
    -----------
    folder_name : str
        Name of the result folder
    
    Returns:
    --------
    norm_type : str
        'standard' or 'normal' (no scaling)
    
    Example:
    --------
    >>> norm = get_normalization_type_from_folder('ETTh1_token_standard_adapt_...')
    >>> print(norm)  # 'standard'
    """
    if '_token_standard_' in folder_name:
        return 'standard'
    elif '_token_normal_' in folder_name:
        return 'normal'
    else:
        return None


if __name__ == "__main__":
    # Example usage
    print("Available scalers:")
    scalers = list_available_scalers()
    for s in scalers:
        print(f"  {s['dataset']:15} {s['column']:10} {s['type']:10}")
    
    # Example of loading and using a scaler
    if scalers:
        example = scalers[0]
        print(f"\nExample: Loading scaler for {example['dataset']} - {example['column']} - {example['type']}")
        scaler = load_scaler(example['dataset'], example['column'], example['type'])
        
        # Test with dummy data
        test_data = np.array([0.5, 1.0, -0.5]).reshape(-1, 1)
        original = scaler.inverse_transform(test_data)
        print(f"Test transformation: {test_data.ravel()} -> {original.ravel()}")
