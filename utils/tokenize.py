

def get_stats(ids):
    counts = dict()
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def merge_tokens(num_merges, ids, N=256):
    merges = dict()
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = N + i
        print(f"mergin{pair} into a new token {idx} which appeared {stats[pair]} times")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges, ids