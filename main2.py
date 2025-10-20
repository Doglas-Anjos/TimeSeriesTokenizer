import pandas as pd
from matplotlib import pyplot as plt
from dataloader_torre import *
import numpy as np
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
N_samples = 200
total_vocab_size = 2200
division_factor = 10
num_merges = total_vocab_size - N_samples

list_of_files = [ETTh1_data_file, ETTh2_data_file, ETTm1_data_file, ETTm2_data_file, exchange_rate_data_file]

def process_files(path_file, name_file):
    df = pd.read_csv(path_file)
    epsilon = 1e-6
    x_time_feature = df.index
    list_columns = list(df.columns)
    list_columns.remove('date')
    disc_type_simple = 'simple'
    disc_type_adaptatice = 'adaptative'
    dict_of_scalers = {}
    df_token_normal_simp = pd.DataFrame()
    df_token_minmax_simp = pd.DataFrame()
    df_token_standard_simp = pd.DataFrame()
    df_token_normal_adapt = pd.DataFrame()
    df_token_minmax_adapt = pd.DataFrame()
    df_token_standard_adapt = pd.DataFrame()
    for column in list_columns:
        print(f"Processing column: {column} in file: {name_file}")
        y_value_feature = df[column]
        y_mimmax_scaled = MinMaxScaler().fit_transform(y_value_feature.to_frame())
        y_standard_scaled = StandardScaler().fit_transform(y_value_feature.to_frame())
        print("normal discretization simple")
        y_feature_simp_tok, bin_edges = simple_discretize(y_value_feature, N_samples)
        file_vocab_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_simple}_normal.fvocab"
        save_float_vocab(bin_edges.tolist(), file_vocab_name)
        y_feature_tokens = encode_token(y_value_feature, name_file, column, disc_type=f"{disc_type_simple}_normal")
        df_token_normal_simp[column] = pd.Series(y_feature_tokens)
        print("adaptative discretization")
        edges, y_feature_adapt_tok, alloc = adaptative_bins_discretize(y_value_feature, M=N_samples, K=division_factor)
        file_vocab_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_adaptatice}_normal.fvocab"
        save_float_vocab(edges.tolist(), file_vocab_name)
        y_feature_tokens = encode_token(y_value_feature, name_file, column, disc_type=f"{disc_type_adaptatice}_normal")
        df_token_normal_adapt[column] = pd.Series(y_feature_tokens)
        print("minmax discretization simple")
        y_mimmax_simple_tok, bin_edges = simple_discretize(y_mimmax_scaled, N_samples)
        file_vocab_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_simple}_minmax.fvocab"
        save_float_vocab(bin_edges.tolist(), file_vocab_name)
        y_feature_tokens = encode_token(y_mimmax_scaled, name_file, column, disc_type=f"{disc_type_simple}_minmax")
        df_token_minmax_simp[column] = pd.Series(y_feature_tokens)
        print("minmax discretization adaptative")
        edges, y_mimmax_adapt_tok, alloc = adaptative_bins_discretize(y_mimmax_scaled, M=N_samples, K=division_factor)
        file_vocab_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_adaptatice}_minmax.fvocab"
        save_float_vocab(edges.tolist(), file_vocab_name)
        y_feature_tokens = encode_token(y_mimmax_scaled, name_file, column, disc_type=f"{disc_type_adaptatice}_minmax")
        df_token_minmax_adapt[column] = pd.Series(y_feature_tokens)
        print("standard discretization simple")
        y_standard_simple_tok, bin_edges = simple_discretize(y_standard_scaled, N_samples)
        file_vocab_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_simple}_stand.fvocab"
        save_float_vocab(bin_edges.tolist(), file_vocab_name)
        y_feature_tokens = encode_token(y_standard_scaled, name_file, column, disc_type=f"{disc_type_simple}_stand")
        df_token_standard_simp[column] = pd.Series(y_feature_tokens)
        print("standard discretization adaptative")
        edges, y_standard_adapt_tok, alloc = adaptative_bins_discretize(y_standard_scaled, M=N_samples, K=division_factor)
        file_vocab_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type_adaptatice}_stand.fvocab"
        save_float_vocab(edges.tolist(), file_vocab_name)
        y_feature_tokens = encode_token(y_standard_scaled, name_file, column, disc_type=f"{disc_type_adaptatice}_stand")
        df_token_standard_adapt[column] = pd.Series(y_feature_tokens)

    df_token_normal_simp.to_csv(f"{base_data_file_out}{name_file}_token_normal_simp.csv", index=False)
    df_token_minmax_simp.to_csv(f"{base_data_file_out}{name_file}_token_minmax_simp.csv", index=False)
    df_token_standard_simp.to_csv(f"{base_data_file_out}{name_file}_token_standard_simp.csv", index=False)
    df_token_normal_adapt.to_csv(f"{base_data_file_out}{name_file}_token_normal_adapt.csv", index=False)
    df_token_minmax_adapt.to_csv(f"{base_data_file_out}{name_file}_token_minmax_adapt.csv", index=False)
    df_token_standard_adapt.to_csv(f"{base_data_file_out}{name_file}_token_standard_adapt.csv", index=False)


def encode_token(float_list, name_file, column, disc_type):
    base_name = f"{name_file}_feature_Nsam_{N_samples}_vocab_{total_vocab_size}_column_{column}_{disc_type}"
    model_name = fr"model\{base_name}.model"
    file_vocab_name = fr"{base_name}.fvocab"
    if not joblib.os.path.isfile(model_name):
        objtok = BasicTokenizer(N_samples, file_vocab_name)
        objtok.train(float_list, total_vocab_size, verbose=True)
        objtok.save(fr"{base_name}", file_vocab_name)
    else:
        objtok = BasicTokenizer(N_samples, file_vocab_name)
        objtok.load(model_name)
    encoded_ids = objtok.encode(float_list)
    return encoded_ids


def process_all_files():
    for file in list_of_files:
        path_file = f"{base_data_files}{file}.csv"
        process_files(path_file, file)

if __name__ == "__main__":
    process_all_files()