import os
import csv
import json
import math
import random
from tqdm import tqdm
import networkx as nx
from typing import Dict, Set, Optional, Any

# Percorso del file di salvataggio centralità
CENTRALITY_FILE = "./facebook_betweenness.json"


def ceil_division(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ValueError("Denominator for ceil_division cannot be zero.")
    return -(numerator // -denominator)


def assign_cost_attributes(G: nx.Graph, use_threshold: bool):  # noqa
    cost1: Dict[int, int] = {}
    cost2: Dict[int, int] = {}
    cost3: Dict[int, float] = {}

    for v in tqdm(G.nodes(), desc="Calculating cost1"):
        degree_v = G.degree(v)
        ceil_cost = ceil_division(degree_v, 2) if degree_v > 0 else 0
        cost1[v] = ceil_cost

    random_min_range = min(cost1.values())
    random_max_range = max(cost1.values())

    random.seed(42)

    for v in tqdm(G.nodes(), desc="Assigning random cost2"):
        cost2[v] = random.randint(random_min_range, random_max_range)

    if os.path.exists(CENTRALITY_FILE):
        print("Loading centrality from file...")
        with open(CENTRALITY_FILE, "r") as f:
            centrality = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        print("Computing centrality...")
        centrality = nx.betweenness_centrality(G)
        with open(CENTRALITY_FILE, "w") as f:
            json.dump(centrality, f)
        print("Centrality saved on disk.")
    epsilon = 1e-6

    log_centrality = {
        v: math.log10(centrality[v] + epsilon)
        for v in tqdm(G.nodes(), desc="Computing log centrality")
    }

    min_log = min(log_centrality.values())
    shifted_log_centrality = {
        v: log_centrality[v] - min_log
        for v in tqdm(G.nodes(), desc="Shifting log values")
    }

    scale = random_max_range / max(shifted_log_centrality.values()) if max(shifted_log_centrality.values()) > 0 else 1.0
    for v in tqdm(G.nodes(), desc="Finalizing cost3"):
        cost3[v] = shifted_log_centrality[v] * scale

    nx.set_node_attributes(G, cost1, "cost1")
    nx.set_node_attributes(G, cost2, "cost2")
    nx.set_node_attributes(G, cost3, "cost3")

    if use_threshold:
        threshold = cost1.copy()
        nx.set_node_attributes(G, threshold, "threshold")
        return G, cost1, cost2, cost3, threshold
    else:
        return G, cost1, cost2, cost3



def log_experiment(csv_path: str, algorithm_name: str, cost_function: str, use_threshold: bool, budget: int,
                   seed_set: Set[int], total_cost: int, execution_time: float, G: Optional[nx.Graph] = None,  # noqa
                   additional_info: Optional[Dict[str, Any]] = None) -> None:

    headers = [
        "timestamp",
        "algorithm_name",
        "cost_function",
        "use_threshold",
        "budget",
        "num_nodes",
        "num_edges",
        "seed_set",
        "num_seeds",
        "total_cost",
        "execution_time",
        "additional_info"
    ]

    is_new_file = not os.path.exists(csv_path)
    mode = "a" if not is_new_file else "w"

    row = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "algorithm_name": algorithm_name,
        "cost_function": cost_function,
        "use_threshold": use_threshold,
        "budget": budget,
        "num_nodes": G.number_of_nodes() if G is not None else "",
        "num_edges": G.number_of_edges() if G is not None else "",
        "seed_set": json.dumps(sorted(seed_set)),
        "num_seeds": len(seed_set),
        "total_cost": total_cost,
        "execution_time": execution_time,
        "additional_info": json.dumps(additional_info) if additional_info else ""
    }

    with open(csv_path, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def log_cascade(csv_path: str, algorithm_name: str, seed_set_str: str, seed_size: int, final_influence_size: int,
                final_influence: Set[int], execution_time: float, experiment_result_row: int, round: int,  # noqa
                G: Optional[nx.Graph] = None, additional_info: Optional[Dict[str, Any]] = None) -> None:  # noqa

    headers = [
        "timestamp",
        "algorithm_name",
        "seed_set",
        "seed_size",
        "final_influence",
        "final_influence_size",
        "num_nodes",
        "num_edges",
        "experiment_result_row",
        "execution_time",
        "round",
        "additional_info"
    ]

    is_new_file = not os.path.exists(csv_path)
    mode = "a" if not is_new_file else "w"

    row = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "algorithm_name": algorithm_name,
        "seed_set": seed_set_str,
        "seed_size": seed_size,
        "final_influence":  json.dumps(sorted(final_influence)),
        "final_influence_size": final_influence_size,
        "num_nodes": G.number_of_nodes() if G is not None else "",
        "num_edges": G.number_of_edges() if G is not None else "",
        "execution_time": execution_time,
        "experiment_result_row": experiment_result_row,
        "round": round,
        "additional_info": json.dumps(additional_info) if additional_info else ""
    }

    with open(csv_path, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
