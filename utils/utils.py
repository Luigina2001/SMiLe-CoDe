import random
import os
import csv
import json
from typing import Dict, Set, Optional, Any
import networkx as nx
from networkx import Graph


def ceil_division(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ValueError("Denominator for ceil_division cannot be zero.")
    return -(numerator // -denominator)


def assign_cost_attributes(G: Graph, random_max_range: int, use_threshold: bool ):  # noqa

    if random_max_range < 1:
        raise ValueError("random_max_range must be at least 1.")

    cost1: Dict[int, int] = {}
    cost2: Dict[int, int] = {}
    cost3: Dict[int, float] = {}

    for v in G.nodes():
        degree_v = G.degree(v)
        if degree_v == 0:
            ceil_cost = 0
            inv_cost = 0.0
        else:
            ceil_cost = ceil_division(degree_v, 2)
            inv_cost = 1.0 / degree_v

        cost1[v] = ceil_cost
        cost2[v] = random.randint(1, random_max_range)
        cost3[v] = inv_cost

    nx.set_node_attributes(G, cost1, "cost1")
    nx.set_node_attributes(G, cost2, "cost2")
    nx.set_node_attributes(G, cost3, "cost3")

    if use_threshold:
        # Reuse the same values as cost1 for the threshold attribute
        nx.set_node_attributes(G, cost1, "threshold")

    if use_threshold:
        return G, cost1, cost2, cost3, cost1
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
