# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from matplotlib import pyplot as plt
from dataloader_torre import *
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
from utils.basic import BasicTokenizer

tower_number = tower_numer_1
global_file_path = f'data/COMPOSE_Torre0{tower_number}.csv'
df = load_data_tower(global_file_path)
epsilon = 1e-6
X_time_feature = df.index
y_dataframe =  df[column_sensor_temperature_AVG].abs().loc[lambda s: s > epsilon] # to avoid negative and zero values for Box-Cox
# in this casa, we have a problem because in temperature we have negative values, i have to go back here to treat this case
y_value_feature = np.array([y_dataframe.values]).T

y_mimmax_scaled = MinMaxScaler().fit_transform(y_value_feature)
y_standard_scaled = StandardScaler().fit_transform(y_value_feature)
y_abs_max_scaled = MaxAbsScaler().fit_transform(y_value_feature)
y_robust_scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(y_value_feature)
y_power_yeo_johnson = PowerTransformer(method="yeo-johnson").fit_transform(y_value_feature)
y_power_box_cox = PowerTransformer(method="box-cox").fit_transform(y_value_feature)
y_quantile_uniform = QuantileTransformer(output_distribution="uniform", random_state=42).fit_transform(y_value_feature)
y_quantile_gaussian = QuantileTransformer(output_distribution="normal", random_state=42).fit_transform(y_value_feature)
y_normalized = Normalizer().fit_transform(y_value_feature)


def _to_1d_array(data):
    """Coerce DataFrame/Series/array/list to a 1-D NumPy array (drop NaNs)."""
    if pd is not None and isinstance(data, pd.Series):
        x = data.dropna().to_numpy()
    elif pd is not None and isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column.")
        x = data.iloc[:, 0].dropna().to_numpy()
    else:
        x = np.asarray(data).ravel()
        x = x[~np.isnan(x)] if np.issubdtype(x.dtype, np.number) else x
    if x.size == 0:
        raise ValueError("Input data is empty after dropping NaNs.")
    return x


def save_given_distributions(non_norm, norm, bins=50, base_name="distribution", dpi=300):
    """
    Save two plots:
      1) distribution_non_normalized.png  (using your non-normalized data)
      2) distribution_normalized.png      (using your normalized data)

    Parameters
    ----------
    non_norm : array-like / Series / 1-col DataFrame
        Data in original scale.
    norm : array-like / Series / 1-col DataFrame
        Data already normalized (z-scores).
    bins : int
        Histogram bins.
    base_name : str
        Filename prefix.
    dpi : int
        Image resolution.
    """
    x1 = _to_1d_array(non_norm)
    x2 = _to_1d_array(norm)

    # --- Non-normalized ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(x1, bins=bins, density=True, alpha=0.75, edgecolor="none", color="skyblue")
    ax.set_title("Original Scale Distribution")
    ax.yaxis.set_visible(False)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 6))
    fig.tight_layout()
    fig.savefig(f"images/{base_name}_non_normalized.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- Normalized ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(x2, bins=bins, density=True, alpha=0.75, edgecolor="none", color="lightgreen")
    ax.set_title("Normalized (z-score) Distribution")
    ax.yaxis.set_visible(False)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 6))
    fig.tight_layout()
    fig.savefig(f"images/{base_name}_normalized.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("Saved:",
          f"{base_name}_non_normalized.png",
          f"{base_name}_normalized.png")



#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_standard_scaled,bins=180, base_name='standard')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_mimmax_scaled,bins=180, base_name='minmax')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_abs_max_scaled,bins=180, base_name='abs_max')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_robust_scaled,bins=180, base_name='robust')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_power_yeo_johnson,bins=180, base_name='power_yeo_johnson')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_power_box_cox,bins=180, base_name='power_box_cox')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_quantile_uniform,bins=180, base_name='quantile_uniform')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_quantile_gaussian,bins=180, base_name='quantile_gaussian')
#save_given_distributions(df[column_sensor_velocity_30m_1_AVG], y_normalized,bins=180, base_name='normalized')
N = 200 # Number of samples to plot
discretize_data, bin_edges = simple_discretize(df[column_sensor_temperature_AVG], n_bins=20)
discretize_data_2, bin_edges_2 = simple_discretize(y_standard_scaled, n_bins=20)
vocab_size = 2200
# preciso voltar ao paper sobre tokenização para pegar os hiperparametros e verificar principalmente a taxa de compressão comparando com a minha
# há uma boa diferença entre os dados de velocidade do vento e temperatura, a impressão que da é que a temperatura é mais coesa e é "tokenizavel"
#preciso entender quanto tempo e quantos merges são necessários para chegar a um bom resultado
#ainda não rodei a tokenização com o simple discretize
#preciso ver se o load está funcionando corretamente, acho que preciso rever como fazer o encoding e decoding de novos dados ( não sei se está funcionando)

num_merges = vocab_size - N
edges, symbols, alloc = adaptative_bins_discretize(y_standard_scaled, M=N, K=6)
y_scaled = [round(float(element[0]), 5) for element in y_standard_scaled]
save_float_vocab(edges.tolist(), "adaptative_bins.fvocab")
symbols_2 = encode_with_float_vocab(y_scaled, "adaptative_bins.fvocab")
numbers_of_symbols, n_edges = decode_with_float_vocab(symbols_2, "adaptative_bins.fvocab")
ObjTok = BasicTokenizer(N,"adaptative_bins.fvocab")
ObjTok.train(y_scaled, vocab_size, verbose=True)
ObjTok.save("adaptative_bins_tokenizer", "adaptative_bins.fvocab")
Encoded_data = ObjTok.encode(y_scaled[:1000])
Decoded_data = ObjTok.decode(Encoded_data)



df_n = pd.DataFrame([y_scaled, numbers_of_symbols[0]]).T
symb_list = symbols.tolist()
stats = get_stats(symb_list)
top_pair = max(stats, key=stats.get)
print(sorted(((v, k) for k, v in stats.items()), reverse=True))
merge_symb, ids = merge_tokens(num_merges, symbols.tolist(), N=N)
len_old_tokens = len(symb_list)
len_new_tokens = len(ids)

print(f"Compression rate: {len_old_tokens/len_new_tokens:.2f} ({len_old_tokens} -> {len_new_tokens})")
pass