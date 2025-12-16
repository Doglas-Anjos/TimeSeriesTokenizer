import numpy as np
import os
import pandas as pd
global_path = os.getcwd()
vocab_path=global_path + r"\float_vocab"
from datetime import timedelta

def simple_discretize(data, N=10, data_st=None, special_tokens=None):
    """
    Simple discretization of continuous data into bins.
    :param data: array to discretize (if special_token exists x must NOT contain them)
    :param N: is the number of final bins, it will acct like the resolution of the discretization
    :param data_st: array like data but with the special tokens included
    :param special_tokens: dict with the special tokens mapping {token_string: token_id}
    :return: symbols -> array of discretized data (with special tokens if any)
            bin_edges -> array of bin edges
    """
    n_bins = N
    if data_st is not None and special_tokens is not None:
        n_bins = n_bins - len(special_tokens)

    data = np.array(data)

    bin_edges = np.linspace(data.min(), data.max(), n_bins)

    symbols = np.digitize(data, bin_edges[1:], right=True) + 1

    if data_st is not None and special_tokens is not None:
        for index, value in enumerate(data_st):
           if value in special_tokens.keys():
               symbols = np.insert(symbols, index, special_tokens[value])
    return symbols, bin_edges


def quantile_discretize(data, N=10, data_st=None, special_tokens=None):
    """
    quantile discretization of continuous data into bins.
    :param data: array to discretize (if special_token exists x must NOT contain them)
    :param N: is the number of final bins, it will acct like the resolution of the discretization
    :param data_st: array like data but with the special tokens included
    :param special_tokens: dict with the special tokens mapping {token_string: token_id}
    :return: symbols -> array of discretized data (with special tokens if any)
            bin_edges -> array of bin edges
    """
    n_bins = N
    if data_st is not None and special_tokens is not None:
        n_bins = n_bins - len(special_tokens)

    data = np.array(data)

    bin_edges = np.quantile(data, np.linspace(data.min(), data.max(), n_bins + 1))

    symbols = np.digitize(data, bin_edges[1:], right=True) + 1

    if data_st is not None and special_tokens is not None:
        for index, value in enumerate(data_st):
           if value in special_tokens.keys():
               symbols = np.insert(symbols, index, special_tokens[value])
    return symbols, bin_edges

def adaptative_bins_discretize(x, N=10, K=3, data_st=None, special_tokens=None):
    """
    Function to discretize data into adaptative bins, this function creates K pre-division and allocates N bins
    according to the data distribution in each pre-division, data distribution in each pre-division will acct like
    a weight to allocate more bins in high density regions.
    :param x: data to discretize (if special_token exists x must NOT contain them)
    :param N: is the number of final bins, it will acct like the resolution of the discretization
    :param K: is the number of pre-division, it will acct like the coarseness of the discretization
    :param data_st: data like x but with the special tokens included, data_st is used only to insert
                    the special tokens in the output
    :param special_tokens: dict with the special tokens mapping {token_string: token_id}
    :return: edges -> array of bin edges
             symbols -> array of discretized data (with special tokens if any)
            alloc -> array with the number of bins allocated per pre-division
    """
    n_bins = N

    if data_st is not None and special_tokens is not None:
        n_bins = n_bins - len(special_tokens)

    x = np.asarray(x).ravel()
    xmin, xmax = x.min(), x.max()

    # 1) pre-division edges (equal width)
    coarse_edges = np.linspace(xmin, xmax, K + 1)

    # 2) count occurrences in each pre-division
    counts, _ = np.histogram(x, bins=coarse_edges)
    total = counts.sum()
    if total == 0:
        # degenerate
        edges = np.linspace(xmin, xmax, n_bins + 1)[1:-1]
        return edges, np.ones_like(x, int), np.zeros(K, int)

    # 3) allocate final bins by weights (counts / total)
    desired = counts / total * n_bins
    epsilon = 1e-10
    base = np.floor(desired).astype(int)
    base[(desired > epsilon) & (base == 0)] = 1
    rem = n_bins - base.sum()
    # give remaining bins to largest fractional parts, but never to empty pre-bins
    frac = desired - base
    order = np.argsort(-frac)  # descending
    for idx in order:
        if rem == 0:
            break
        if counts[idx] > 0:
            base[idx] += 1
            rem -= 1
    alloc = base  # number of final bins per pre-division (sum == n_bins)

    # 4) build final bin edges
    edges = []

    # inner edges inside each pre-division
    for k in range(K):
        m_k = alloc[k]
        if m_k <= 1:
            continue  # no inner cut if that region is a single bin (or empty)
        a, b = coarse_edges[k], coarse_edges[k + 1]
        inner = np.linspace(a, b, m_k + 1)[1:-1]  # equal width inside region
        edges.extend(inner.tolist())

    # add boundaries between neighboring pre-divisions that both got ≥1 bin
    for k in range(1, K):
        if alloc[k - 1] > 0 and alloc[k] > 0:
            edges.append(coarse_edges[k - 1])

    if len(edges) < n_bins:
        edges.append(coarse_edges[K])

    edges = np.array(sorted(edges))
    # ensure exactly n_bins+1 edges by splitting largest gaps / merging smallest gaps
    target = n_bins

    # helper to get augmented boundaries and gap sizes
    def _aug_and_diffs(ed):
        aug = np.concatenate(([xmin], ed, [xmax]))
        diffs = np.diff(aug)  # gaps between consecutive boundaries
        return aug, diffs

    # If we have too few edges: repeatedly insert the midpoint of the largest gap
    safety = 10_000
    while edges.size < target and safety > 0:
        safety -= 1
        aug, diffs = _aug_and_diffs(edges)
        i = int(np.argmax(diffs))  # gap between aug[i] and aug[i+1]
        a, b = aug[i], aug[i + 1]
        mid = 0.5 * (a + b)
        # guard against degenerate gaps
        if not np.isfinite(mid) or mid <= a or mid >= b:
            break
        edges = np.sort(np.append(edges, mid))

    # If we have too many edges: repeatedly remove the edge with the smallest local gap
    # "local gap" for an interior edge e_j is min(e_j - left, right - e_j)
    safety = 10_000
    while edges.size > target and safety > 0 and edges.size > 0:
        safety -= 1
        # compute local gaps per interior edge
        lefts = np.concatenate(([xmin], edges[:-1]))
        rights = np.concatenate((edges[1:], [xmax]))
        local_min_gap = np.minimum(edges - lefts, rights - edges)

        # pick the edge whose local min gap is the smallest; remove it
        j = int(np.argmin(local_min_gap))
        edges = np.delete(edges, j)

    # final tidy-up
    edges = np.clip(np.unique(np.sort(edges)), xmin, xmax)
    # 5) symbols (q_Ω): map each x_i to 1..n_bins using the thresholds
    symbols = np.digitize(x, edges, right=True) + 1

    if data_st is not None and special_tokens is not None:
        for index, value in enumerate(data_st):
           if value in special_tokens.keys():
               symbols = np.insert(symbols, index, special_tokens[value])

    return edges, symbols, alloc


def save_float_vocab(edges, name_encoding="edges_ex.fvocab"):
    global vocab_path
    file_path = fr"{vocab_path}\{name_encoding}"
    with open(file_path, "w") as f:
        f.write(f"N={len(edges)+1}\n")
        f.write(",".join(f"{x:.5f}" for x in edges))

def load_float_vocab(name_encoding="edges_ex.fvocab"):
    """
    Load bin edges from a .fvocab file.
    
    Parameters:
    -----------
    name_encoding : str
        Path to the .fvocab file (can be relative or absolute)
    
    Returns:
    --------
    edges : np.array
        Array of bin edges
    """
    global vocab_path
    
    # Handle both absolute and relative paths
    if os.path.isabs(name_encoding):
        file_path = name_encoding
    else:
        file_path = fr"{vocab_path}\{name_encoding}"
    
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("N=")
        edges_line = f.readline().strip()
        edges = np.array([float(x) for x in edges_line.split(",")])
    
    return edges

def encode_with_float_vocab(data, name_encoding="edges_ex.fvocab", special_tokens=None):
    global vocab_path
    file_path = fr"{vocab_path}\{name_encoding}"
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("N=")
        n_edges = int(first_line[2:])
        edges_line = f.readline().strip()
        edges = np.array([])
        edges_st = np.array([])
        for x in edges_line.split(","):
            if x in special_tokens.keys():
                    edges_st = np.append(edges, special_tokens[x])
            else:
                edges = np.append(edges, [float(x)])
        assert len(edges) + 1 == n_edges - len(edges_st)
    symbols = np.digitize(data, edges, right=True) + 1
    symbols = [int(s) for s in symbols]

    if special_tokens:
        for index, value in enumerate(data):
           if value in special_tokens.keys():
               symbols = np.insert(symbols, index, special_tokens[value])

    return symbols


def decode_with_float_vocab(symbols, name_encoding="edges_ex.fvocab", special_tokens = None):
    global vocab_path
    file_path = fr"{vocab_path}\{name_encoding}"
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("N=")
        n_edges = int(first_line[2:])
        edges_line = f.readline().strip()
        split_edges = edges_line.split(",")
        edges = np.array([])
        edges_st = np.array([])
        for x in split_edges:
            if x not in special_tokens.keys():
                edges = np.append(edges, [float(x)])
            else:
                edges_st = np.append(edges_st, special_tokens[x])
        assert len(edges)+1 == n_edges
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    decoded = list()
    for s in symbols:
        if special_tokens is not None:
            if s in special_tokens.values():
                decoded.append([key for key, value in special_tokens.items() if value == s][0])
                continue
        if 1 <= s <= len(bin_centers):
            decoded.append(float(bin_centers[s - 1]))
        else:
            if s == 0:
                decoded.append(float(edges[0] - (edges[1] - edges[0]) / 2))
            elif s >= (n_edges -1):
                decoded.append(float(edges[-1] + (edges[-1] - edges[-2]) / 2))
            else:
                decoded.append(np.nan)

    return decoded, n_edges


def mark_special_tokens(df, special_tokens,hour_toks, data_freq='1H'):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    pad_token = special_tokens.get("pad", "<PAD>")
    ebos_token = special_tokens.get("ebos", "<EBOS>")

    # 1) Remove duplicates before reindexing (keep first occurrence)
    df = df.drop_duplicates(subset=["date"], keep="first")
    
    # 2) Reindex to expose gaps (we'll fill them with <PAD>)
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq=data_freq)
    df = df.set_index("date").reindex(full_idx)
    df.index.name = "date"
    df.reset_index(inplace=True)

    # 3) Fill missing rows with <PAD>
    value_cols = [c for c in df.columns if c != "date"]
    for c in value_cols:
        df[c] = df[c].where(df[c].notna(), pad_token)

    # 4) Find the FIRST 00:00 present in the dataframe
    idx = df["date"]
    mid_mask = (idx.dt.hour == 0) & (idx.dt.minute == 0)
    if not mid_mask.any():
        # No midnight in range → nothing to tag as EBOS; return with PADs filled
        return df

    first_midnight = idx.loc[mid_mask].iloc[0]

    # 5) Mark every N hours starting from that first midnight (inclusive)
    elapsed_hours = ((idx - first_midnight) / np.timedelta64(1, "h")).astype("int64")
    ebos_mask = (idx >= first_midnight) & (elapsed_hours % hour_toks == 0)

    # 6) Overwrite those rows with <EBOS>
    for c in value_cols:
        df.loc[ebos_mask, c] = ebos_token

    return df