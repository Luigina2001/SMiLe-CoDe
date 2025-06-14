from typing import Callable, Union, Optional, Set

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

def cost_seeds_greedy(
    G: nx.Graph,
    budget: Union[int, float],
    cost_type: str,
    sub_function: Callable,
    initial_seed_set: Optional[Set] = None,
    current_cost: Union[int, float] = 0
):
    """
        Input:
          - G: grafo non orientato (nx.Graph)
          - budget: somma dei costi del seed set massima totale ammissibile
          - sub_function: the submodular function chosen (can be sub_function1, sub_function2, sub_function3)
          - initial_seed_set: seed set iniziale da cui partire
          - current_cost: costo del seed set iniziale.
        Output:
          - S: target set con costo totale <= budget
    """

    if sub_function not in {sub_function1, sub_function2, sub_function3}:
        raise ValueError("sub_function must be one of the allowed sub_functions")

    remaining_nodes = set(G.nodes())
    total_cost = current_cost
    if initial_seed_set is None:
        S_selected = set()
    else:
        S_selected = initial_seed_set
        remaining_nodes -= initial_seed_set

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

    G, cost1, cost2, cost3 = assign_cost_attributes(G, use_threshold=False)

    # Configurazioni funzioni di costo e relative descrizioni
    cost_functions = {
        "cost1": cost1,
        "cost2": cost2,
        "cost3": cost3
    }

    descriptions = {
        "cost1": "cost1: ceiling function of degree(v) / 2",
        "cost2": "cost2: random int in [min(cost1), max(cost1)]",
        "cost3": "cost3: scaled log10 of betweenness centrality"
    }

    for name, cost in cost_functions.items():
        algorithm_name = "CSG"
        cost_function_desc = descriptions[name]

        # Calcolo range del budget
        min_budget = int(max(cost.values()))
        if int(min(cost.values())) > 0:
            max_budget = int(min(cost.values()) * (len(G.nodes())))
        else:
            max_budget = (int(min(cost.values()) + 1) * (len(G.nodes())))

        if min_budget > max_budget:
            print(f"MinBudget > MaxBudget for {name}")
            min_budget, max_budget = max_budget, min_budget

        if min_budget == max_budget:
            print(f"MinBudget = MaxBudget for {name}")
            continue

        print(f"\n{name} — budget from {min_budget} to {max_budget}")

        current_seed_set = None
        current_cost = 0

        for budget_k in tqdm(
                range(min_budget, max_budget + 1, 100),
                desc=f"Budget loop for {name}",
                unit="budget"
        ):
            start_time = time.time()
            S = cost_seeds_greedy(G, budget_k, name, sub_function1, current_seed_set, current_cost)
            end_time = time.time()

            total_cost = sum(cost[v] for v in S)
            current_seed_set = S
            current_cost = total_cost

            exec_time = end_time - start_time

            print(f"\nFunction: {name} | Budget: {budget_k}")
            print(f"Target set size: {len(S)}")
            print(f"Total cost: {total_cost}")
            print(f"Execution time: {exec_time:.2f} seconds")

            log_experiment(
                csv_path="./logs/prova_centrality.csv",
                algorithm_name=algorithm_name,
                cost_function=cost_function_desc,
                use_threshold=False,
                budget=budget_k,
                seed_set=S,
                total_cost=total_cost,
                execution_time=exec_time,
                G=G,
                additional_info={"note": f"Running on facebook_combined.txt with {name}"}
            )
