from typing import Callable

import networkx as nx
from tqdm import tqdm
import time

from utils.utils import log_experiment, assign_cost_attributes, ceil_division


def sub_function1(S: set, G: nx.Graph):
    """
        Input:
          - S: target set di nodi di G
          - G: grafo non orientato (nx.Graph)
        Output:
          - score: punteggio assegnato dalla funzione sub-modulare 1 al set S
    """

    if not S:
        return 0

    score = 0
    for v in G.nodes():
        degree = G.degree(v)
        half_deg = ceil_division(degree, 2)
        neighbors_in_S = len(S & set(G.neighbors(v)))
        score += min(neighbors_in_S, half_deg)

    return score

def sub_function2(S: set, G: nx.Graph):
    """
        Input:
          - S: target set di nodi di G
          - G: grafo non orientato (nx.Graph)
        Output:
          - score: punteggio assegnato dalla funzione sub-modulare 2 al set S
    """

    if not S:
        return 0

    score = 0
    for v in G.nodes():
        neighbors_in_S = [u for u in G.neighbors(v) if u in S]
        half_deg = ceil_division(G.degree(v), 2)

        for i in range(1, len(neighbors_in_S) + 1):
            score += max(half_deg - i + 1, 0)

    return score


def sub_function3(S: set, G: nx.Graph):
    """
        Input:
          - S: target set di nodi di G
          - G: grafo non orientato (nx.Graph)
        Output:
          - score: punteggio assegnato dalla funzione sub-modulare 3 al set S
    """
    if not S:
        return 0

    score = 0
    for v in G.nodes():
        neighbors_in_S = [u for u in G.neighbors(v) if u in S]
        half_deg = ceil_division(G.degree(v), 2)

        for i in range(1, len(neighbors_in_S) + 1):
            denom = G.degree(v) - i + 1
            if denom > 0:
                score += max((half_deg - i + 1) / denom, 0)

    return score

def cost_seeds_greedy(G: nx.Graph, budget: int, cost_type: str, sub_function: Callable):
    """
        Input:
          - G: grafo non orientato (nx.Graph)
          - budget: somma dei costi del seed set massima totale ammissibile
          - sub_function: the submodular function chosen (can be sub_function1, sub_function2, sub_function3).
        Output:
          - S: target set con costo totale <= budget
    """

    if sub_function not in {sub_function1, sub_function2, sub_function3}:
        raise ValueError("sub_function must be one of the allowed sub_functions")

    S_selected = set()
    total_cost = 0
    remaining_nodes = set(G.nodes())

    pbar = tqdm(total=budget, desc="Cost Seeds Greedy progress")

    # Ciclo aggiunta nodi
    while total_cost < budget and remaining_nodes:
        # Miglior nodo dell'iterazione e il suo score
        best_v = None
        best_score = -float('inf')

        # Valore della funzione submodulare del seed set alla attuale iterazione
        current_value = sub_function(S_selected, G)

        # Ciclo per scegliere il nodo con lo score miglire
        for v in remaining_nodes:
            node_cost = G.nodes[v][cost_type]

            # value rappresenta lo score del nodo da confrontare con gli altri
            gain = sub_function(S_selected | {v}, G) - current_value
            value = gain / node_cost

            if value > best_score:
                best_score = value
                best_v = v

        if best_v is None:
            break

        # Calcolo costo del nodo. Se tale costo fa superare il budget il nodo non viene aggiunto al seed set
        node_cost = G.nodes[best_v][cost_type]
        if total_cost + node_cost > budget:
            break

        total_cost += node_cost
        S_selected.add(best_v)
        remaining_nodes.remove(best_v)
        pbar.update(node_cost)

    pbar.close()
    return S_selected

if __name__ == "__main__":
    G = nx.read_edgelist("../data/facebook_combined.txt", nodetype=int)

    budget_k = 100
    algorithm_name = "CSG"
    cost_function_desc = "cost1: ceiling function of degree(v) / 2"

    G, cost1, cost2, cost3, threshold = assign_cost_attributes(G, budget_k, True)

    start_time = time.time()
    S = cost_seeds_greedy(G, budget_k, "cost1", sub_function1)
    end_time = time.time()

    total_cost = sum(cost1[v] for v in S)
    exec_time = end_time - start_time

    print("Target set S =", S)
    print("Total cost =", total_cost)
    print(f"Execution time: {exec_time:.2f} secondi")

    log_experiment(
        csv_path="./logs/experiment_results.csv",
        algorithm_name=algorithm_name,
        cost_function=cost_function_desc,
        use_threshold=False,
        budget=budget_k,
        seed_set=S,
        total_cost=total_cost,
        execution_time=exec_time,
        G=G,
        additional_info={"note": "Esecuzione su facebook_combined.txt"}
    )
