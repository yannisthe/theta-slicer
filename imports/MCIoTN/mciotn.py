import networkx as nx

# --- MCIoTN
def mciotn(G, BR1=None, BR2=None, enforce_even=False):
    nodes = sorted(G.nodes())
    n = len(nodes)
    max_cut_weight = -1
    best_partitions = []

    for i in range(1, 2 ** (n - 1)):
        part1 = [nodes[j] for j in range(n) if (i >> j) & 1]
        part2 = [node for node in nodes if node not in part1]

        if not part1 or not part2:
            continue

        if (BR1 != BR2) and (BR1 not in part1 or BR2 not in part2):
            continue

        if nx.is_connected(G.subgraph(part1)) and nx.is_connected(G.subgraph(part2)):
            cut_weight = sum(1 for u in part1 for v in G.neighbors(u) if v in part2)
            cut_weight = cut_weight // 2  # count each edge only once

            if cut_weight > max_cut_weight:
                max_cut_weight = cut_weight
                best_partitions = [(part1, part2)]
            elif cut_weight == max_cut_weight:
                best_partitions.append((part1, part2))

    if enforce_even and best_partitions:
        min_diff = min(abs(len(p1) - len(p2)) for p1, p2 in best_partitions)
        best_partitions = [p for p in best_partitions if abs(len(p[0]) - len(p[1])) == min_diff]

    return best_partitions, max_cut_weight