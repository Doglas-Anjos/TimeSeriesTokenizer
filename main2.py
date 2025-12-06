import pandas as pd
from matplotlib import pyplot as plt
from dataloader_torre import *
import numpy as np
import os
from utils.discretisize import *
from utils.tokenize import *
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
import joblib
from utils.basic import BasicTokenizer

base_data_files = 'data/datasets/'
base_data_file_out = 'data/outputs/'
eletricity_data_file = 'electricity'
ETTh1_data_file = 'ETTh1'
ETTh2_data_file = 'ETTh2'
ETTm1_data_file = 'ETTm1'
ETTm2_data_file = 'ETTm2'
weather_data_file = 'weather'
exchange_rate_data_file = 'exchange_rate'
# here we will define special tokens, we could put others if needed
# <PAD>: padding token
# <EBOS>: end or beginning of sequence token
lower_sampling = 50
midium_sampling = 100
high_sampling = 200
list_of_samples = [lower_sampling, midium_sampling, high_sampling]
N_samples = 202
special_tokens = {'<PAD>':N_samples-1, '<EBOS>':N_samples}

total_vocab_size = 1000
division_factor = 10
hour_context_size_24h = 24
hour_context_size_12h = 12
num_merges = total_vocab_size - N_samples

list_of_files = [ETTh1_data_file, ETTh2_data_file, ETTm1_data_file, ETTm2_data_file, weather_data_file, exchange_rate_data_file]

def process_files(path_file, name_file):
    df = pd.read_csv(path_file)
    df_spectial_token_24h = mark_special_tokens(df, special_tokens=special_tokens, hours=hour_context_size_24h)
    df_spectial_token_12h = mark_special_tokens(df, special_tokens=special_tokens, hours=hour_context_size_12h)
    x_time_feature = df.index
    list_columns = list(df.columns)
    list_columns.remove('date')

    disc_type_simple = 'simple'
    disc_type_adaptatice = 'adaptative'
    
    # Store tokenized dataframes by configuration
    tokenized_dfs = {
        'normal_simp_sem_ebos': pd.DataFrame(),
        'normal_adapt_sem_ebos': pd.DataFrame(),
        'standard_simp_sem_ebos': pd.DataFrame(),
        'standard_adapt_sem_ebos': pd.DataFrame(),
        'normal_simp_24h': pd.DataFrame(),
        'normal_adapt_24h': pd.DataFrame(),
        'standard_simp_24h': pd.DataFrame(),
        'standard_adapt_24h': pd.DataFrame(),
        'normal_simp_12h': pd.DataFrame(),
        'normal_adapt_12h': pd.DataFrame(),
        'standard_simp_12h': pd.DataFrame(),
        'standard_adapt_12h': pd.DataFrame(),
    }
    
    # Create directory for scalers if it doesn't exist
    scaler_dir = 'scalers'
    if not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir)
    
    for column in list_columns:
        # if column != 'LUFL':
        #     continue
        # if column != "4":
        #     continue
        print(f"Processing column: {column} in file: {name_file}")
        y_value_feature = df[column]
        y_value_feature_st_24h = df_spectial_token_24h[column]
        y_value_feature_st_12h = df_spectial_token_12h[column]

        # Fit and save scalers
        scaler_standard = StandardScaler()
        scaler_standard_24h = StandardScaler()
        scaler_standard_12h = StandardScaler()

        y_standard_scaled = pd.Series(
            scaler_standard.fit_transform(y_value_feature.to_frame()).ravel(),
            index=y_value_feature.index,
            name=column
        )
        
        # Save scalers for this column
        joblib.dump(scaler_standard, f"scalers/{name_file}_column_{column}_standard.pkl")

        # Define processing configurations: (data, data_st, norm_type, ebos_suffix)
        configs = [
            (y_value_feature_st_24h, y_value_feature_st_24h, 'standard', '24h'),
            (y_standard_scaled, None, 'standard', 'sem_ebos'),

            (y_standard_scaled, y_value_feature_st_12h, 'standard', '12h'),
            (y_value_feature, None, 'normal', 'sem_ebos'),
            (y_value_feature, y_value_feature_st_24h, 'normal', '24h'),
            (y_value_feature, y_value_feature_st_12h, 'normal', '12h'),
        ]
        
        # Process each configuration for simple and adaptative discretization
        for data, data_st, norm_type, ebos_suffix in configs:
            print(f"{norm_type} discretization simple {ebos_suffix.upper().replace('_', ' ')}")
            y_simple_tok, bin_edges = simple_discretize(data, N_samples, data_st, special_tokens=special_tokens)
            base_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_simple}_{norm_type}_{ebos_suffix}"
            save_float_vocab(bin_edges.tolist(), f"{base_name}.fvocab")
            y_tokens = encode_token(data, base_name)
            test = encode_token(y_simple_tok, base_name)
            tokenized_dfs[f"{norm_type}_simp_{ebos_suffix}"][column] = pd.Series(y_tokens)
            
            print(f"{norm_type} discretization adaptative {ebos_suffix.upper().replace('_', ' ')}")
            edges, y_adapt_tok, alloc = adaptative_bins_discretize(data, N=N_samples, K=division_factor, data_st=data_st, special_tokens=special_tokens)
            base_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_adaptatice}_{norm_type}_{ebos_suffix}"
            save_float_vocab(edges.tolist(), f"{base_name}.fvocab")
            y_tokens = encode_token(data, base_name)
            tokenized_dfs[f"{norm_type}_adapt_{ebos_suffix}"][column] = pd.Series(y_tokens)

    # Fill NaN values and save all tokenized dataframes
    output_configs = [
        ('normal_simp_sem_ebos', f"{name_file}_token_normal_simp_sem_ebos_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('normal_adapt_sem_ebos', f"{name_file}_token_normal_adapt_sem_ebos_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('standard_simp_sem_ebos', f"{name_file}_token_standard_simp_sem_ebos_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('standard_adapt_sem_ebos', f"{name_file}_token_standard_adapt_sem_ebos_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('normal_simp_24h', f"{name_file}_token_normal_simp_24h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('normal_adapt_24h', f"{name_file}_token_normal_adapt_24h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('standard_simp_24h', f"{name_file}_token_standard_simp_24h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('standard_adapt_24h', f"{name_file}_token_standard_adapt_24h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('normal_simp_12h', f"{name_file}_token_normal_simp_12h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('normal_adapt_12h', f"{name_file}_token_normal_adapt_12h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('standard_simp_12h', f"{name_file}_token_standard_simp_12h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
        ('standard_adapt_12h', f"{name_file}_token_standard_adapt_12h_N_Samp{N_samples}_vocab_{total_vocab_size}.csv"),
    ]
    
    for config_key, output_filename in output_configs:
        df_tokenized = tokenized_dfs[config_key].fillna(special_tokens['<PAD>'])
        df_tokenized.to_csv(f"{base_data_file_out}{output_filename}", index=False)
        print(f"✓ Saved: {output_filename}")


def encode_token(float_list, base_name):
    model_name = fr"model\{base_name}.model"
    file_vocab_name = fr"{base_name}.fvocab"
    if not joblib.os.path.isfile(model_name):
        objtok = BasicTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
        objtok.train(float_list, total_vocab_size, verbose=True)
        objtok.save(fr"{base_name}", file_vocab_name)
    else:
        objtok = BasicTokenizer(N_samples, file_vocab_name, special_tokens=special_tokens)
        objtok.load(model_name)
    encoded_ids = objtok.encode(float_list)
    return encoded_ids


def process_all_files():
    global special_tokens, N_samples
    global total_vocab_size
    for file in list_of_files:
        for samp in list_of_samples:
            N_samples = samp
            special_tokens = {'<PAD>': N_samples - 1, '<EBOS>': N_samples}
            if file == exchange_rate_data_file:
                N_samples = N_samples - 40
                total_vocab_size = N_samples + 40
                special_tokens = {'<PAD>': N_samples - 1, '<EBOS>': N_samples}
            path_file = f"{base_data_files}{file}.csv"
            process_files(path_file, file)

if __name__ == "__main__":
    process_all_files()