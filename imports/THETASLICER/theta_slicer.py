# ----------------------------------------
# File: theta_slicer.py
# Author: Ioannis Theodorou
# Created: May 4, 2025
# Last Modified: July 15, 2025
# Description: A novel heuristic algorithm that automates and optimizes the slicing decision process. 
# It generates a topology-aware plan for assigning network nodes to ϑ connected and balanced subgraphs. 
# The algorithm aims to maximize inter-slice edge cuts while preserving intra-slice connectivity and achieving balanced node distribution. 
# Adapted for practical IoT scenarios such as data harvesting, the algorithm accepts border router nodes as input parameters to guide partition anchoring.
# Version: 1.0
#
# Copyright (C) 2025 Ioannis Theodorou
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------

import networkx as nx
import random

# Validation function that ensures the partitions produced are connected
def is_valid_multi_partition(G, partitions, BRs):
    for i, part in enumerate(partitions):
        if BRs[i] not in part:
            return False
        if not nx.is_connected(G.subgraph(part)):
            return False
    return True

# Objective function to determine improvements in the number of cuts made.
def cut_weight_multi(G, partitions):
    cut = 0
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            cut += sum(1 for u in partitions[i] for v in G.neighbors(u) if v in partitions[j])
    return cut // 2  

# --- THETA-SLICER - θ CUTS
def theta_slicer(G, BRs, enforce_even=False, max_iter=500): # G - Network graph, BRs - θ Number of Border Routers, enforce_even - Ensure partitions are even in number where possible, max_iter - Algorithm Iteration Limit
    theta = len(BRs)
    nodes = set(G.nodes())
    partitions = [{br} for br in BRs]
    assigned = set(BRs)
    unassigned = list(nodes - assigned)

    for node in unassigned:
        random.choice(partitions).add(node)

    for _ in range(10):
        if is_valid_multi_partition(G, partitions, BRs):
            break
        part_idx = random.randint(0, theta-1)
        non_br_nodes = [n for n in partitions[part_idx] if n not in BRs]
        if non_br_nodes:
            node = random.choice(non_br_nodes)
            partitions[part_idx].remove(node)
            new_idx = random.choice([i for i in range(theta) if i != part_idx])
            partitions[new_idx].add(node)

    current_cut = cut_weight_multi(G, partitions)

    for _ in range(max_iter):
        improvement = False
        all_nodes = list(nodes - set(BRs))
        random.shuffle(all_nodes)

        for node in all_nodes:
            current_idx = next(i for i, part in enumerate(partitions) if node in part)
            for new_idx in range(theta):
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
        target_size = sum(sizes) // theta
        max_loops = len(G.nodes()) * theta
        for _ in range(max_loops):
            max_idx = max(range(theta), key=lambda i: len(partitions[i]))
            min_idx = min(range(theta), key=lambda i: len(partitions[i]))
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
                continue
            
            new_cut = cut_weight_multi(G, partitions)  # compute cut after the move

            # accept only if new_cut >= current_cut; otherwise revert
            if new_cut < current_cut:
                partitions[min_idx].remove(node)
                partitions[max_idx].add(node) 
                continue # don't update current_cut; try another move / iteration
            else:
                current_cut = new_cut  # move accepted; continue balancing (may further improve)
    return [list(p) for p in partitions], current_cut
