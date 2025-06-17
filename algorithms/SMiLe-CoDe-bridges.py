import os
import sys
import json
import networkx as nx
import community as community_louvain
from tqdm import tqdm
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import assign_cost_attributes, log_experiment  # noqa


def SMiLe_CoDe(G: nx.Graph, cost_attr: str, total_budget: int,  # noqa
               centrality_file: str = "./facebook_betweenness.json"):  # noqa
    """
        Alloca il budget alle comunità in proporzione alla loro dimensione ed esegue una selezione
        greedy (basata sulla centralità) all'interno di ciascuna comunità.

        Args:
            G: Grafo NetworkX
            cost_attr: Attributo di costo dei nodi
            total_budget: Budget totale disponibile
            centrality_file: Percorso al file della betweenness centrality (opzionale)

        Returns:
            Lista dei nodi seed selezionati
    """

    # Caricamento della betweenness centrality se fornita
    if centrality_file:
        with open(centrality_file, "r") as f:
            bc = json.load(f)
        bc = {int(k): float(v) for k, v in bc.items()}
        nx.set_node_attributes(G, bc, "betweenness")

    # Rilevamento delle comunità usando il metodo di Louvain
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # Calcolo dei budget proporzionali per comunità
    n = G.number_of_nodes()
    comm_budgets = {}
    for comm_id, nodes in communities.items():
        comm_size = len(nodes)
        comm_budgets[comm_id] = int(total_budget * comm_size / n)

    # Selezione dei seed in ciascuna comunità
    seeds = []
    remaining_budget = total_budget

    for comm_id, nodes in communities.items():
        # Assegnamento del budget locale, considerando il budget residuo
        local_budget = min(comm_budgets[comm_id], remaining_budget)
        if local_budget <= 0:
            continue

        # Creazione di una vista del sotto grafo per la comunità
        comm_nodes = communities[comm_id]

        # Ordinamento dei nodi per betweenness decrescente (nodi più centrali prima)
        sorted_nodes = sorted(
            comm_nodes,
            key=lambda v: G.nodes[v].get("betweenness", -float("inf")),
            reverse=True
        )

        # Selezione greedy all'interno della comunità
        selected_in_comm = []
        spent_in_comm = 0
        for node in sorted_nodes:
            cost = G.nodes[node].get(cost_attr, float("inf"))  # noqa
            if cost > local_budget - spent_in_comm:  # Se il costo supera il budget rimanente
                continue
            if cost + spent_in_comm <= local_budget:
                selected_in_comm.append(node)
                spent_in_comm += cost
            if spent_in_comm >= local_budget:
                break

        seeds.extend(selected_in_comm)
        remaining_budget -= spent_in_comm
        print(f"Community {comm_id}: selected {len(selected_in_comm)} nodes, spent {spent_in_comm}/{local_budget}")

    # Gestione del budget residuo con selezione globale
    if remaining_budget > 0:
        print(f"Remaining budget: {remaining_budget}, selecting globally...")

        local_bridges = set(nx.local_bridges(G))

        bridge_nodes = set()
        for t in local_bridges:
            n1, n2 = t[0], t[1]
            b1 = G.nodes[n1].get("betweenness", float("inf"))
            b2 = G.nodes[n2].get("betweenness", float("inf"))
            node = n1 if b1 >= b2 else n2
            bridge_nodes.add(node)

        bridge_nodes = sorted(
            bridge_nodes,
            key=lambda v: G.nodes[v].get("betweenness", float("inf"))
        )

        spent_global = 0
        global_selection_count = 0
        for node in bridge_nodes:
            cost = G.nodes[node].get(cost_attr, float("inf"))  # noqa
            if cost <= remaining_budget - spent_global:
                seeds.append(node)
                spent_global += cost
                global_selection_count += 1
            if spent_global >= remaining_budget:
                break

        remaining_budget -= spent_global
        print(f"Added {global_selection_count} global nodes, spent {spent_global}")

        if remaining_budget > 0:
            print(f"Remaining budget: {remaining_budget}, selecting globally...")
            all_nodes = sorted(
                G.nodes(),
                key=lambda v: G.nodes[v].get("betweenness", float("inf"))
            )
            candidates = [n for n in all_nodes if n not in seeds]

            spent_global = 0
            global_selection_count = 0
            for node in candidates:
                cost = G.nodes[node].get(cost_attr, float("inf"))  # noqa
                if cost <= remaining_budget - spent_global:
                    seeds.append(node)
                    spent_global += cost
                    global_selection_count += 1
                if spent_global >= remaining_budget:
                    break
            print(f"Added {global_selection_count} global nodes, spent {spent_global}")

    total_spent = total_budget - remaining_budget + spent_global  # noqa
    print(f"Total seeds: {len(seeds)}, total spent: {total_spent}/{total_budget}")
    return seeds


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
        algorithm_name = "SMiLe-CoDe-bridges"
        cost_function_desc = descriptions[name]
        desc = descriptions[name]

        # Calcolo range del budget
        min_budget = int(max(cost.values()))
        """if int(min(cost.values())) > 0:
            max_budget = int(min(cost.values()) * (len(G.nodes())))
        else:
            max_budget = (int(min(cost.values()) + 1) * (len(G.nodes())))"""
        max_budget = int(sum(cost.values()))

        if min_budget > max_budget:
            print(f"MinBudget > MaxBudget for {name}")
            min_budget, max_budget = max_budget, min_budget

        if min_budget == max_budget:
            print(f"MinBudget = MaxBudget for {name}")
            continue

        tqdm.write(f"\n{name} — budget da {min_budget} a {max_budget}")

        for budget_k in tqdm(
                range(min_budget, max_budget + 1, 100),
                desc=f"Budget loop for {name}",
                unit="budget"
        ):
            start_time = time.time()
            S = SMiLe_CoDe(
                G,
                name,
                budget_k,
                centrality_file="./facebook_betweenness.json"
            )
            end_time = time.time()

            total_cost = sum(cost[v] for v in S)
            exec_time = end_time - start_time

            tqdm.write(f"Function: {name} | Budget: {budget_k}")
            tqdm.write(f"Seed set size: {len(S)}; Total cost: {total_cost}; Time: {exec_time:.2f}s")
            log_experiment(
                csv_path=f"./logs/{name}_SMiLe-CoDe-bridges.csv",
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
