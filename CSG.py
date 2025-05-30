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
              - score: punteggio assegnato dalla funzione modulare al set S
            """
    if S is None or len(S) == 0:
        return 0

    V = set(G.nodes)
    N = {v: set(G.neighbors(v)) for v in V}
    score = 0

    for v in G.nodes():
        degree = G.degree(v)
        cel = ceildiv(degree, 2)
        intersection_size = len(S.intersection(set(N[v])))
        score += min(intersection_size, cel)

    return score


def sub_function2(S: set, G: nx.Graph):
    """
                Input:
                  - S: target set di nodi di G
                  - G: grafo non orientato (nx.Graph)
                Output:
                  - score: punteggio assegnato dalla funzione modulare al set S
                """
    if S is None or len(S) == 0:
        return 0

    score = 0
    for v in G.nodes():
        neighbors_in_D = [u for u in G.neighbors(v) if u in S]
        half_deg = ceildiv(G.degree(v), 2)

        for i in range(1, len(neighbors_in_D) + 1):
            term = max(half_deg - i + 1, 0)
            score += term
    return score


def sub_function3(S: set, G: nx.Graph):
    """
                Input:
                  - S: target set di nodi di G
                  - G: grafo non orientato (nx.Graph)
                Output:
                  - score: punteggio assegnato dalla funzione modulare al set S
                """
    if S is None or len(S) == 0:
        return 0

    score = 0
    for v in G.nodes():
        neighbors_in_D = [u for u in G.neighbors(v) if u in S]
        half_deg = ceildiv(G.degree(v), 2)

        for i in range(1, len(neighbors_in_D) + 1):
            term = max((half_deg - i + 1) / (G.degree(v) - i + 1), 0)
            score += term
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

    Sp = set()
    Sd = set()
    V = set(G.nodes())
    cost = 0

    pbar = tqdm(total=budget, desc="Cost Seeds Greedy progress")

    def remove_vertex(v):
        """Rimuove v da V"""
        V.remove(v)
        pbar.update(min(budget - cost, G.nodes[v]["cost"]))

    """def remove_vertex2():
        pbar2.update(1)"""

    while cost < budget:
        #pbar2 = tqdm(total=len(V), desc="Nodes viewed")
        def score(v):
            subSd = sub_function(Sd, G)
            Sd.add(v)
            subSd_v = sub_function(Sd, G)
            value = (subSd_v - subSd) / G.nodes[v]["cost"]
            Sd.remove(v)
            #remove_vertex2()
            return value

        v = max(V, key=score)
        #value = (sub_function(Sd.add(v), G) - cost) / G.nodes[v]["cost"]
        add_cost = G.nodes[v]["cost"]
        cost += add_cost
        remove_vertex(v)

        Sp = Sd.copy()
        Sd.add(v)
        #pbar2.close()
    pbar.close()
    return Sp


if __name__ == "__main__":
    G = nx.read_edgelist("data/rete_sociale.txt", delimiter=' ', nodetype=int)

    # funzione di costo = grado del nodo / 2
    cost1 = {v: ceildiv(G.degree(v), 2) for v in G.nodes()}
    # funzione di costo randomica
    cost2 = {v: random.randint(1, max(cost1.values())) for v in G.nodes()}

    cost3 = {v: 1 / G.degree(v) for v in G.nodes()}

    nx.set_node_attributes(G, cost1, "cost")

    budget_k = 100

    """for v in set(G.nodes()):
        print(G.nodes[v]["cost"])"""

    S = cost_seeds_greedy(G, 100, sub_function3)
    print("Target set S =", S)
    print("Total cost =", sum(cost1[v] for v in S))

# todo Salvataggio risultati