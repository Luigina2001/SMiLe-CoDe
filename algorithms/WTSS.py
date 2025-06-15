import networkx as nx
from tqdm import tqdm
import os
import sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import assign_cost_attributes, log_experiment  # noqa


def WTSS(G: nx.Graph, t: dict, c: dict, budget: int):  # noqa
    """
    Input:
      - G: grafo non orientato (nx.Graph)
      - t: dict di thresholds, t[v] = soglia di v
      - c: dict di costs, c[v] = costo di v
      - budget: costo massimo totale ammissibile

    Output:
      - S: target set con costo totale <= budget
    """

    V = set(G.nodes())
    U = set(G.nodes())
    S = set()  # noqa
    total_cost = 0  # noqa

    # delta[v] = grado corrente di v in U (inizialmente grado in G)
    delta = {v: G.degree(v) for v in V}
    # k[v] = t[v] (threshold residua)
    k = dict(t)
    # N[v] = neighbors di v ancora in U
    N = {v: set(G.neighbors(v)) for v in V}

    pbar = tqdm(total=len(V), desc="WTSS progress")

    def remove_vertex(v):
        """Rimuove v da U, aggiorna delta e N dei suoi neighbors."""
        for u in list(N[v]):  # noqa
            # aggiorna grado residuo e lista di vicini
            delta[u] -= 1
            N[u].remove(v)
        U.remove(v)
        N[v].clear()
        pbar.update(1)

    while U:  # While U ≠ 0 do
        zero_thr = [v for v in U if k[v] == 0]
        if zero_thr:  # if there exists v ∈ U s.t. k(v)=0 then
            v = zero_thr[0]
            # Case 1: the selected vertex v is activated by the influence of its neighbors in V − U only;
            # it can then influence its neighbors in U
            for u in N[v]:
                k[u] = max(0, k[u] - 1)  # noqa
            remove_vertex(v)
            continue

        impossible = [v for v in U if delta[v] < k[v]]
        if impossible:  # if there exists v ∈ U s.t. δ(v) < k(v) then
            # Case 2: the vertex v is added to S, since no sufficient neighbors remain in U to activate it;
            # v can then influence its neighbors in U
            v = impossible[0]
            if total_cost + c[v] < budget:
                S.add(v)
                total_cost += c[v]
                # print(f"Total cost = {total_cost}")
            elif total_cost + c[v] == budget:  # possibile in quanto non ci sono nodi con costo uguale a zero
                S.add(v)
                total_cost += c[v]
                # print(f"Total cost = {total_cost}")
                # print(f"BREAK: {total_cost} = {budget}")
                pbar.close()
                return S
            for u in N[v]:
                k[u] = max(0, k[u] - 1)  # riduco la threshold dei vicini
            remove_vertex(v)
            continue

        # Case 3: the selected vertex v will be activated by its neighbors in U
        def score(v):
            return c[v] * k[v] / (delta[v] * (delta[v] + 1))

        v = max(U, key=score)
        remove_vertex(v)

    pbar.close()
    return S


if __name__ == "__main__":
    G = nx.read_edgelist("../data/facebook_combined.txt", nodetype=int)

    G, cost1, cost2, cost3, threshold = assign_cost_attributes(G, use_threshold=True)

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
        algorithm_name = "WTSS"
        cost_function_desc = descriptions[name]

        # Calcolo range del budget
        min_budget = int(max(cost.values()))
        """if int(min(cost.values())) > 0:
            max_budget = int(min(cost.values())*(len(G.nodes())))
        else:
            max_budget = (int(min(cost.values())+1) * (len(G.nodes())))"""
        max_budget = int(sum(cost.values()))
        if min_budget > max_budget:
            print(f"MinBudget > MaxBudget for {name}")
            min_budget, max_budget = max_budget, min_budget

        if min_budget == max_budget:
            print(f"MinBudget = MaxBudget for {name}")
            continue

        print(f"\n{name} — budget from {min_budget} to {max_budget}")

        for budget_k in tqdm(
                range(min_budget, max_budget + 1, 100),
                desc=f"Budget loop for {name}",
                unit="budget"
        ):
            start_time = time.time()
            S = WTSS(G, threshold, cost, budget_k)
            end_time = time.time()

            total_cost = sum(cost[v] for v in S)
            exec_time = end_time - start_time

            print(f"\nFunction: {name} | Budget: {budget_k}")
            print(f"Target set size: {len(S)}")
            print(f"Total cost: {total_cost}")
            print(f"Execution time: {exec_time:.2f} seconds")

            log_experiment(
                csv_path=f"./logs/{name}_WTSS.csv",
                algorithm_name=algorithm_name,
                cost_function=cost_function_desc,
                use_threshold=True,
                budget=budget_k,
                seed_set=S,
                total_cost=total_cost,
                execution_time=exec_time,
                G=G,
                additional_info={"note": f"Running on facebook_combined.txt with {name}"}
            )
