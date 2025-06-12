import csv
import ast
import os
import sys
import time
from tqdm import tqdm
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import log_cascade, ceil_division  # noqa


def leggi_seed_set(csv_path, i):
    with open(csv_path, newline='') as csvfile:  # noqa
        reader = csv.DictReader(csvfile)
        righe = list(reader)

        if i < 0 or i >= len(righe):
            raise IndexError(f"Indice i={i} fuori dal range valido (0-{len(righe) - 1})")

        riga = righe[i]
        seed_set_str = riga['seed_set']  # noqa
        seed_set = set(ast.literal_eval(seed_set_str))  # noqa

    return seed_set


def majority_cascade(G, S):  # noqa
    influenced = set(S)  # Inf[S, 0] = S
    prev_influenced = set()  # Inf[S, r-1]
    r = 0

    pbar = tqdm(total=len(G), desc="Majority Cascade Progress")

    while influenced != prev_influenced:  # il più piccolo t tale che Inf[S,t]=Inf[S,t+1] è l'istante in cui termina
        r += 1
        prev_influenced = influenced.copy()
        newly_influenced = set()

        for v in G.nodes:
            if v not in prev_influenced:
                neighbors = list(G.neighbors(v))
                if not neighbors:
                    continue  # nodo isolato
                active_neighbors = len([u for u in neighbors if u in prev_influenced])
                if active_neighbors >= ceil_division(len(neighbors), 2):
                    newly_influenced.add(v)

        influenced = influenced.union(newly_influenced)  # Inf[S, r] = Inf[S, r-1] ∪ nuovi influenzati
        pbar.set_description(f"Round {r} — Influenced: {len(influenced)}")
        pbar.update()

    pbar.close()
    return influenced, r  # Inf[S,t]=Inf[S,t+1]


if __name__ == "__main__":
    G = nx.read_edgelist("../data/facebook_combined.txt", nodetype=int)

    # Lettura del seed set dal CSV (da riga zero a n-1)
    csv_experiment_row = 35
    seed_set = leggi_seed_set("./logs/experiment_results.csv", csv_experiment_row)

    # Majority cascade
    start_time = time.time()
    final_influence, round = majority_cascade(G, seed_set)  # noqa
    end_time = time.time()

    # print(final_influence.difference(seed_set))
    # print(len(final_influence.difference(seed_set)))

    # Log dei dati
    log_cascade(
        csv_path='./logs/cascade_results.csv',
        algorithm_name="MajorityCascade",
        seed_set_str=str(sorted(seed_set)),
        seed_size=len(seed_set),
        final_influence=final_influence,
        final_influence_size=len(final_influence),
        execution_time=end_time - start_time,
        experiment_result_row=csv_experiment_row+1,
        round=round,
        G=G,
        additional_info={"note": "Esecuzione Majority Cascade su facebook_combined.txt"}
    )

    print(f"Esecuzione completata. Nodi influenzati: {len(final_influence)}")
