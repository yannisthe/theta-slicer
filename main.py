# ----------------------------------------
# File: main.py
# Author: Ioannis Theodorou
# Created: April 10, 2025
# Last Modified: July 15, 2025
# Description: To run experiments between Kernighan-Lin, MCIoTN, and Theta-Slicer
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

from collections import defaultdict
import networkx as nx
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json 
from imports.THETASLICER.theta_slicer import theta_slicer
from imports.MCIoTN.mciotn import mciotn
#------------------------------------------------------------------------------------------
# --- Kernighan–Lin Bisection
def kernighan_lin_cut(G):
    part1, part2 = nx.algorithms.community.kernighan_lin_bisection(G)
    return (part1, part2)

# --- Neighbor Decrease Calculation
def calc_neighbor_decrease_percentage(G, partitions):
    partition_labels = {}
    for idx, part in enumerate(partitions):
        for node in part:
            partition_labels[node] = idx

    percentage_decrease = {}
    neighbour_decrease = {}
    total_original = 0
    total_new = 0
    edge_cut_count = 0

    for node in G.nodes():
        original_neighbors = len(list(G.neighbors(node)))
        intra_partition_neighbors = 0
        inter_partition_neighbors = 0

        for neighbor in G.neighbors(node):
            if partition_labels[node] == partition_labels[neighbor]:
                intra_partition_neighbors += 1
            else:
                inter_partition_neighbors += 1

        decrease = inter_partition_neighbors
        neighbour_decrease[node] = decrease

        if original_neighbors == 0:
            percentage = 0
        else:
            percentage = (decrease / original_neighbors) * 100
        percentage_decrease[node] = round(percentage, 2)

        total_original += original_neighbors
        total_new += intra_partition_neighbors
        edge_cut_count += decrease

    edge_cuts = edge_cut_count // 2

    if total_original == 0:
        overall_decrease = 0
    else:
        overall_decrease = ((total_original - total_new) / total_original) * 100

    num_of_neighbour_decrease = defaultdict(int)
    for decrease in neighbour_decrease.values():
        num_of_neighbour_decrease[decrease] += 1

    return percentage_decrease, round(overall_decrease, 2), neighbour_decrease, dict(sorted(num_of_neighbour_decrease.items())), edge_cuts

#------------------------------------------------------------------------------------------
# --- Output Functions
def print_result_to_screen(run_no, G, R, algorithm_code, exec_time, edge_cuts,
                           overall_decrease, p1_connected, p2_connected,
                           partition1, partition2, pernode_decrease, neighbour_decrease, num_of_neighbour_decrease):
    print(f"\n📌 [{algorithm_code}] Result - Run {run_no}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {(G.number_of_nodes() * (G.number_of_nodes()-1))/2} Max Edges, R = {R}")
    print(f"Network Density     : {G.number_of_edges()/((G.number_of_nodes() * (G.number_of_nodes()-1))/2)*100:.1f}")
    print(f"Execution Time      : {exec_time:.7f} seconds")
    print(f"Edge Cuts           : {edge_cuts}")
    g1 = G.subgraph(partition1).copy()
    g2 = G.subgraph(partition2).copy()
    print(f"Partition Edges     : P1: {g1.number_of_edges()}, P2: {g2.number_of_edges()}")
    print(f"Overall Decrease    : {overall_decrease:.2f} %")
    print(f"Partition 1 Connected: {p1_connected}")
    print(f"Partition 2 Connected: {p2_connected}")
    print(f"Partition 1         : {sorted(partition1)} (size = {len(partition1)})")
    print(f"Partition 2         : {sorted(partition2)} (size = {len(partition2)})")
    print(f"Per-Node Decrease   : {pernode_decrease}")
    print(f"G                   : {G.degree()}")
    print(f"G1                  : {g1.degree()}")
    print(f"G2                  : {g2.degree()}")
    print(f"Neighbour Decrease  : {neighbour_decrease}")
    print(f"Decrease Histogram  : {num_of_neighbour_decrease}, {sum(num_of_neighbour_decrease.values())}")


def save_result(filename, run_no, G, R, algorithm_code, exec_time, edge_cuts,
                overall_decrease, p1_connected, p2_connected, partition1, partition2, pernode_decrease, neighbour_decrease, num_of_neighbour_decrease):
    with open(filename, 'a') as f:
        line = f"{run_no};{G.number_of_nodes()};{G.number_of_edges()};{(G.number_of_nodes() * (G.number_of_nodes()-1))/2};"
        line += f"{R};{G.number_of_edges()/((G.number_of_nodes() * (G.number_of_nodes()-1))/2)*100:.1f};{algorithm_code};"
        line += f"{exec_time:.7f};{edge_cuts};{overall_decrease:.2f};{p1_connected};{p2_connected};"
        line += f"{sorted(partition1)};{len(partition1)};{sorted(partition2)};{len(partition2)};{pernode_decrease};"
        line += f"{neighbour_decrease};{num_of_neighbour_decrease};"
        g1 = G.subgraph(partition1).copy()
        g2 = G.subgraph(partition2).copy()
        line += f"{G.degree()};{G.number_of_edges()};{g1.degree()};{g1.number_of_edges()};{g2.degree()};{g2.number_of_edges()}\n"
        f.write(line)

#------------------------------------------------------------------------------------------
# --- Main Experiment Loop
def run_experiments(runNo, Nodes, R, BRs, KL=False, MC=False, TS=False, TStheta3=False, TStheta5=False):
#    runNo = 5
#    Nodes = 16
#    R = 0.5
#    BRs = [BR1] + [BR2]
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"ExpThetaSlicer_{Nodes}_{R}_{timestamp_str}.txt"
    with open(filename, 'w') as f:
        f.write("RunNo;Nodes;Edges;MaxEdges;R;NetDensity;Algorithm;ExecutionTime;EdgeCuts;OverallDecrease;Partition1Connected;Partition2Connected;Partition1;SizePartition1;Partition2;SizePartition2;PerNodeDecrease;Neighbour Decrease;Decrease in Number of Neighbours;GEdges;NoGEdges;P1Edges;NoP1Edges;P2Edges;NoP2Edges\n")
    f.close()

    for run in range(runNo):
        print(f"\n========== Run {run + 1} ==========")
        G = nx.random_geometric_graph(Nodes, R) # Create Geometric Graph
        for u, v in G.edges():
            G[u][v]['weight'] = 1

        # KERNIGHAN-LIN
        if KL:
            start_time = time.time()
            partitions_kl = kernighan_lin_cut(G)
            exec_time = time.time() - start_time
            pernode_dec, overall_dec, neighbour_decrease, num_of_neighbour_decrease, edge_cuts = calc_neighbor_decrease_percentage(G, partitions_kl)
            p1, p2 = partitions_kl
            c1 = nx.is_connected(G.subgraph(p1))
            c2 = nx.is_connected(G.subgraph(p2))
            print_result_to_screen(run+1, G, R, "KL", exec_time, edge_cuts, overall_dec, c1, c2, p1, p2, pernode_dec, neighbour_decrease, num_of_neighbour_decrease)
            save_result(filename, run+1, G, R, "KL", exec_time, edge_cuts, overall_dec, c1, c2, p1, p2, pernode_dec, neighbour_decrease, num_of_neighbour_decrease)

        # MAX-CUT
        if MC:
            BR1 = BRs[0]
            BR2 = BRs[1]
            start_time = time.time()
            partitions_maxcut, weight_maxcut = mciotn(G, BR1, BR2, enforce_even=True)
            exec_time = time.time() - start_time
            if partitions_maxcut:
                p1, p2 = partitions_maxcut[0]
                pernode_dec, overall_dec, neighbour_decrease, num_of_neighbour_decrease, edge_cuts = calc_neighbor_decrease_percentage(G, [p1, p2])
                c1 = nx.is_connected(G.subgraph(p1))
                c2 = nx.is_connected(G.subgraph(p2))
                print_result_to_screen(run+1, G, R, "MC", exec_time, edge_cuts, overall_dec, c1, c2, p1, p2, pernode_dec, neighbour_decrease, num_of_neighbour_decrease)
                save_result(filename, run+1, G, R, "MC", exec_time, edge_cuts, overall_dec, c1, c2, p1, p2, pernode_dec, neighbour_decrease, num_of_neighbour_decrease)
    
        # THETA-SLICER 2 BRs
        if TS:
            BR1 = BRs[0]
            BR2 = BRs[1]
            start_time = time.time()
            partitions_ts, cut_ts = theta_slicer(G, [BR1]+[BR2], enforce_even=True)
            exec_time = time.time() - start_time
            if len(partitions_ts) == 2:
                p1, p2 = partitions_ts
                pernode_dec, overall_dec, neighbour_decrease, num_of_neighbour_decrease, edge_cuts = calc_neighbor_decrease_percentage(G, [p1, p2])
                c1 = nx.is_connected(G.subgraph(p1))
                c2 = nx.is_connected(G.subgraph(p2))
                print_result_to_screen(run+1, G, R, "TS", exec_time, edge_cuts, overall_dec, c1, c2, p1, p2, pernode_dec, neighbour_decrease, num_of_neighbour_decrease)
                save_result(filename, run+1, G, R, "TS", exec_time, edge_cuts, overall_dec, c1, c2, p1, p2, pernode_dec, neighbour_decrease, num_of_neighbour_decrease)
        
        if TStheta3 and len(BRs) == 3:
            BR1 = BRs[0]
            BR2 = BRs[1]
            BR3 = BRs[2]
            start_time = time.time()
            partitions_ts, cut_ts = theta_slicer(G, [BR1]+[BR2]+[BR3], enforce_even=True)
            exec_time = time.time() - start_time
            if len(partitions_ts) == 3:
                partition_connectivity = [nx.is_connected(G.subgraph(p)) for p in partitions_ts]
                pernode_dec, overall_dec, neighbour_decrease, num_of_neighbour_decrease, edge_cuts = calc_neighbor_decrease_percentage(G, partitions_ts)

                print(f"\n📌 [TStheta3] Result - Run {run+1}")
                print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {(G.number_of_nodes() * (G.number_of_nodes()-1))/2} Max Edges, R = {R}")
                print(f"Network Density     : {G.number_of_edges()/((G.number_of_nodes() * (G.number_of_nodes()-1))/2)*100:.1f}")
                print(f"Execution Time      : {exec_time:.7f} seconds")
                print(f"Edge Cuts           : {edge_cuts}")
                print(f"Overall Decrease    : {overall_dec:.2f} %")

                for idx, part in enumerate(partitions_ts):
                    print(f"Partition {idx+1}         : {sorted(part)} (size = {len(part)})")
                    print(f"Partition {idx+1} Connected: {partition_connectivity[idx]}")

                with open(filename+"3", 'a') as f:
                    if run==0:
                        f.write("RunNo;Nodes;Edges;MaxEdges;R;NetDensity;Algorithm;ExecutionTime;EdgeCuts;OverallDecrease;")

                    # Basic information
                    line = f"{run+1};{G.number_of_nodes()};{G.number_of_edges()};{(G.number_of_nodes() * (G.number_of_nodes()-1))/2};"
                    line += f"{R};{G.number_of_edges()/((G.number_of_nodes() * (G.number_of_nodes()-1))/2)*100:.1f};TStheta3;"
                    line += f"{exec_time:.7f};{edge_cuts};{overall_dec:.2f};"

                    # Partition-specific data
                    for i, part in enumerate(partitions_ts):
                        conn = partition_connectivity[i]
                        subgraph = G.subgraph(part).copy()
                        if run==0:
                            output_str = f"Part{i}Connected;Partition{i};SizePartition{i};P{i}Edges;NoP{i}Edges;"
                            f.write(output_str)
                        line += f"{conn};{json.dumps(sorted(part))};{len(part)};{json.dumps(dict(subgraph.degree()))};{subgraph.number_of_edges()};"

                    # Final metrics
                    if run==0:
                        f.write("PerNodeDecrease;Neighbour Decrease;Decrease in Number of Neighbours;Edges;NoOfEdges\n")
                    line += f"{json.dumps(pernode_dec)};{json.dumps(neighbour_decrease)};{json.dumps(num_of_neighbour_decrease)};"
                    line += f"{json.dumps(dict(G.degree()))};{G.number_of_edges()}\n"
                    f.write(line)

        if TStheta5 and len(BRs) == 5:
            start_time = time.time()
            partitions_ts, cut_ts = theta_slicer(G, BRs, enforce_even=True)
            exec_time = time.time() - start_time
            if len(partitions_ts) == 5:
                partition_connectivity = [nx.is_connected(G.subgraph(p)) for p in partitions_ts]
                pernode_dec, overall_dec, neighbour_decrease, num_of_neighbour_decrease, edge_cuts = calc_neighbor_decrease_percentage(G, partitions_ts)

                print(f"\n📌 [TStheta5] Result - Run {run+1}")
                print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {(G.number_of_nodes() * (G.number_of_nodes()-1))/2} Max Edges, R = {R}")
                print(f"Network Density     : {G.number_of_edges()/((G.number_of_nodes() * (G.number_of_nodes()-1))/2)*100:.1f}")
                print(f"Execution Time      : {exec_time:.7f} seconds")
                print(f"Edge Cuts           : {edge_cuts}")
                print(f"Overall Decrease    : {overall_dec:.2f} %")

                for idx, part in enumerate(partitions_ts):
                    print(f"Partition {idx+1}         : {sorted(part)} (size = {len(part)})")
                    print(f"Partition {idx+1} Connected: {partition_connectivity[idx]}")

                with open(filename+"5", 'a') as f:
                    if run==0:
                        f.write("RunNo;Nodes;Edges;MaxEdges;R;NetDensity;Algorithm;ExecutionTime;EdgeCuts;OverallDecrease;")

                    # Basic information
                    line = f"{run+1};{G.number_of_nodes()};{G.number_of_edges()};{(G.number_of_nodes() * (G.number_of_nodes()-1))/2};"
                    line += f"{R};{G.number_of_edges()/((G.number_of_nodes() * (G.number_of_nodes()-1))/2)*100:.1f};TStheta5;"
                    line += f"{exec_time:.7f};{edge_cuts};{overall_dec:.2f};"

                    # Partition-specific data
                    for i, part in enumerate(partitions_ts):
                        conn = partition_connectivity[i]
                        subgraph = G.subgraph(part).copy()
                        if run==0:
                            output_str = f"Part{i}Connected;Partition{i};SizePartition{i};P{i}Edges;NoP{i}Edges;"
                            f.write(output_str)
                        line += f"{conn};{json.dumps(sorted(part))};{len(part)};{json.dumps(dict(subgraph.degree()))};{subgraph.number_of_edges()};"

                    # Final metrics
                    if run==0:
                        f.write("PerNodeDecrease;Neighbour Decrease;Decrease in Number of Neighbours;Edges;NoOfEdges\n")
                    line += f"{json.dumps(pernode_dec)};{json.dumps(neighbour_decrease)};{json.dumps(num_of_neighbour_decrease)};"
                    line += f"{json.dumps(dict(G.degree()))};{G.number_of_edges()}\n"
                    f.write(line)

        f.close()


# MAIN
if __name__ == "__main__":
    run_experiments(15, 20, 0.7, [1, 3], KL=True, MC=True, TS=True)

    #run_experiments(15, 10, 0.5, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 10, 0.7, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 10, 0.9, [1, 3], KL=True, MC=True, TS=True)

    #run_experiments(15, 15, 0.5, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 15, 0.7, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 15, 0.9, [1, 3], KL=True, MC=True, TS=True)

    #run_experiments(15, 20, 0.5, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 20, 0.7, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 20, 0.9, [1, 3], KL=True, MC=True, TS=True)

    #run_experiments(15, 25, 0.5, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 25, 0.7, [1, 3], KL=True, MC=True, TS=True)
    #run_experiments(15, 25, 0.9, [1, 3], KL=True, MC=True, TS=True)

    #run_experiments(15, 500, 0.5, [1, 3], KL=True, MC=False, TS=True)
    #run_experiments(15, 500, 0.7, [1, 3], KL=True, MC=False, TS=True)
    #run_experiments(15, 500, 0.9, [1, 3], KL=True, MC=False, TS=True)

    #run_experiments(15, 1000, 0.5, [1,3,6,12,200], KL=True, MC=False, TS=True, TStheta3=True, TStheta5=True)
    #run_experiments(15, 1000, 0.7, [1,3,6,12,200], KL=True, MC=False, TS=True, TStheta3=True, TStheta5=True)
    #run_experiments(15, 1000, 0.9, [1,3,6,12,200], KL=True, MC=False, TS=True, TStheta3=True, TStheta5=True)