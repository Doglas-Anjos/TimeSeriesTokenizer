import numpy as np
import os
global_path = os.getcwd()
vocab_path=global_path + r"\float_vocab"

def simple_discretize(data, n_bins=10):
    """
    Discretize continuous data into bins.

    Parameters
    ----------
    data : array-like
        Continuous values to discretize.
    n_bins : int
        Number of bins.
    method : str
        'quantile' = distribution-aware (more bins where density is high),
        'uniform' = equal width bins.
    """
    data = np.array(data)

    bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)

    symbols = np.digitize(data, bin_edges[1:], right=True) + 1
    return symbols, bin_edges


def quantile_discretize(data, n_bins=10):
    """
    Discretize continuous data into bins.

    Parameters
    ----------
    data : array-like
        Continuous values to discretize.
    n_bins : int
        Number of bins.
    method : str
        'quantile' = distribution-aware (more bins where density is high),
        'uniform' = equal width bins.
    """
    data = np.array(data)

    bin_edges = np.quantile(data, np.linspace(data.min(), data.max(), n_bins + 1))

    symbols = np.digitize(data, bin_edges[1:], right=True) + 1
    return symbols, bin_edges

def adaptative_bins_discretize(x, M=10, K=3):
    """
    x : 1D array-like
    M : total final bins (symbols)
    K : coarse pre-divisions

    Returns
    -------
    edges : np.ndarray (length M-1)   # ω_1 .. ω_{M-1}
    symbols : np.ndarray (same len as x) with values in {1..M}
    alloc : np.ndarray (length K)     # how many final bins each pre-division got
    """
    x = np.asarray(x).ravel()
    xmin, xmax = x.min(), x.max()

    # 1) pre-division edges (equal width)
    coarse_edges = np.linspace(xmin, xmax, K + 1)

    # 2) count occurrences in each pre-division
    counts, _ = np.histogram(x, bins=coarse_edges)
    total = counts.sum()
    if total == 0:
        # degenerate
        edges = np.linspace(xmin, xmax, M + 1)[1:-1]
        return edges, np.ones_like(x, int), np.zeros(K, int)

    # 3) allocate final bins by weights (counts / total)
    desired = counts / total * M
    epsilon = 1e-10
    base = np.floor(desired).astype(int)
    base[(desired > epsilon) & (base == 0)] = 1
    rem = M - base.sum()
    # give remaining bins to largest fractional parts, but never to empty pre-bins
    frac = desired - base
    order = np.argsort(-frac)  # descending
    for idx in order:
        if rem == 0:
            break
        if counts[idx] > 0:
            base[idx] += 1
            rem -= 1
    alloc = base  # number of final bins per pre-division (sum == M)

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
            edges.append(coarse_edges[k])

    edges = np.array(sorted(edges))
    # ensure exactly M-1 edges (very rare mismatch → pad/drop)
    # if edges.size < M - 1:
    #     filler = np.linspace(xmin, xmax, (M - 1 - edges.size) + 2)[1:-1]
    #     edges = np.sort(np.unique(np.concatenate([edges, filler])))
    # elif edges.size > M - 1:
    #     edges = np.sort(edges)[: M - 1]

    # 5) symbols (q_Ω): map each x_i to 1..M using the thresholds
    symbols = np.digitize(x, edges, right=True) + 1
    return edges, symbols, alloc


def save_float_vocab(edges, name_encoding="edges_ex.fvocab"):
    global vocab_path
    file_path = fr"{vocab_path}\{name_encoding}"
    with open(file_path, "w") as f:
        f.write(f"N={len(edges)+1}\n")
        f.write(",".join(f"{x:.5f}" for x in edges))

def encode_with_float_vocab(data, name_encoding="edges_ex.fvocab"):
    global vocab_path
    file_path = f"{vocab_path}/{name_encoding}"
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("N=")
        n_edges = int(first_line[2:])
        edges_line = f.readline().strip()
        edges = np.array([float(x) for x in edges_line.split(",")])
        assert len(edges) + 1 == n_edges
    symbols = np.digitize(data, edges, right=True) + 1
    symbols = [int(s) for s in symbols]
    return symbols

def decode_with_float_vocab(symbols, name_encoding="edges_ex.fvocab"):
    global vocab_path
    file_path = f"{vocab_path}/{name_encoding}"
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("N=")
        n_edges = int(first_line[2:])
        edges_line = f.readline().strip()
        edges = np.array([float(x) for x in edges_line.split(",")])
        assert len(edges)+1 == n_edges
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    decoded = list()
    for s in symbols:
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

    # decoded = np.array([bin_centers[s - 1] if 1 <= s <= len(bin_centers) else np.nan for s in symbols])
    # decoded = decoded.tolist()
    # return decoded, n_edges


