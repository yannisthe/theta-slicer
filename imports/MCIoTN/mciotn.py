# ----------------------------------------
# File: mciotn.py
# Author: Ioannis Theodorou
# Created: April 11, 2025
# Last Modified: July 15, 2025
# Description: A brute-force solution for the graph theory Max-Cut problem, adapted for IoTNs. 
# Ensures that designated border routers are assigned to a bipartition algorithm and only accept cuts
# that yield internally connected subgraphs, guaranteeing operational validity.
# An optional balancing mechanism enforces near-equal partition sizes to support fair
# load distribution in UDIoT scenarios.
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

# --- MCIoTN
def mciotn(G, BR1=None, BR2=None, enforce_even=False): # G - Network graph, BR1 BR2 - Border Routers 1 & 2, enforce_even - Ensures partitions are even where possible
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