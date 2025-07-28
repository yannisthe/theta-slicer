import networkx as nx
import random

# --- THETA-SLICER - θ CUTS
def is_valid_multi_partition(G, partitions, BRs):
    for i, part in enumerate(partitions):
        if BRs[i] not in part:
            return False
        if not nx.is_connected(G.subgraph(part)):
            return False
    return True

def cut_weight_multi(G, partitions):
    cut = 0
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            cut += sum(1 for u in partitions[i] for v in G.neighbors(u) if v in partitions[j])
    return cut // 2  # count each edge only once

def theta_slicer(G, BRs, enforce_even=False, max_iter=500):
    k = len(BRs)
    nodes = set(G.nodes())
    partitions = [{br} for br in BRs]
    assigned = set(BRs)
    unassigned = list(nodes - assigned)

    for node in unassigned:
        random.choice(partitions).add(node)

    for _ in range(10):
        if is_valid_multi_partition(G, partitions, BRs):
            break
        part_idx = random.randint(0, k-1)
        non_br_nodes = [n for n in partitions[part_idx] if n not in BRs]
        if non_br_nodes:
            node = random.choice(non_br_nodes)
            partitions[part_idx].remove(node)
            new_idx = random.choice([i for i in range(k) if i != part_idx])
            partitions[new_idx].add(node)

    current_cut = cut_weight_multi(G, partitions)

    for _ in range(max_iter):
        improvement = False
        all_nodes = list(nodes - set(BRs))
        random.shuffle(all_nodes)

        for node in all_nodes:
            current_idx = next(i for i, part in enumerate(partitions) if node in part)
            for new_idx in range(k):
                if new_idx == current_idx:
                    continue

                new_partitions = [set(part) for part in partitions]
                new_partitions[current_idx].remove(node)
                new_partitions[new_idx].add(node)

                if not is_valid_multi_partition(G, new_partitions, BRs):
                    continue

                new_cut = cut_weight_multi(G, new_partitions)
                if new_cut > current_cut:
                    partitions = new_partitions
                    current_cut = new_cut
                    improvement = True
                    break
            if improvement:
                break
        if not improvement:
            break

    if enforce_even:
        all_nodes = list(nodes - set(BRs))
        sizes = [len(p) for p in partitions]
        target_size = sum(sizes) // k
        for _ in range(50):
            max_idx = max(range(k), key=lambda i: len(partitions[i]))
            min_idx = min(range(k), key=lambda i: len(partitions[i]))
            if len(partitions[max_idx]) - len(partitions[min_idx]) <= 1:
                break
            movable = [n for n in partitions[max_idx] if n not in BRs]
            if not movable:
                break
            node = random.choice(movable)
            partitions[max_idx].remove(node)
            partitions[min_idx].add(node)
            if not is_valid_multi_partition(G, partitions, BRs):
                partitions[min_idx].remove(node)
                partitions[max_idx].add(node)

    return [list(p) for p in partitions], current_cut