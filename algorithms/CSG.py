from typing import Callable

import networkx as nx
from tqdm import tqdm
import random


def ceildiv(a, b):
    return -(a // -b)

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
        half_deg = ceildiv(degree, 2)
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
        half_deg = ceildiv(G.degree(v), 2)

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
        half_deg = ceildiv(G.degree(v), 2)

        for i in range(1, len(neighbors_in_S) + 1):
            denom = G.degree(v) - i + 1
            if denom > 0:
                score += max((half_deg - i + 1) / denom, 0)

    return score

def cost_seeds_greedy(G: nx.Graph, budget: int, sub_function: Callable):
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

    while total_cost < budget and remaining_nodes:
        best_v = None
        best_score = -float('inf')

        current_value = sub_function(S_selected, G)

        for v in remaining_nodes:
            node_cost = G.nodes[v]["cost"]
            gain = sub_function(S_selected | {v}, G) - current_value
            value = gain / node_cost

            if value > best_score:
                best_score = value
                best_v = v

        if best_v is None:
            break

        node_cost = G.nodes[best_v]["cost"]
        if total_cost + node_cost > budget:
            break

        total_cost += node_cost
        S_selected.add(best_v)
        remaining_nodes.remove(best_v)
        pbar.update(node_cost)

    pbar.close()
    return S_selected

if __name__ == "__main__":
    G = nx.read_edgelist("../data/rete_sociale.txt", delimiter=' ', nodetype=int)

    # funzione di costo = grado del nodo / 2
    cost1 = {v: ceildiv(G.degree(v), 2) for v in G.nodes()}
    # funzione di costo randomica
    cost2 = {v: random.randint(1, max(cost1.values())) for v in G.nodes()}
    # funzione di costo inversa grado del nodo
    cost3 = {v: 1 / G.degree(v) if G.degree(v) > 0 else 1 for v in G.nodes()}

    nx.set_node_attributes(G, cost1, "cost")

    budget_k = 100

    S = cost_seeds_greedy(G, budget_k, sub_function3)
    print("Target set S =", S)
    print("Total cost =", sum(G.nodes[v]["cost"] for v in S))

# todo Salvataggio risultati