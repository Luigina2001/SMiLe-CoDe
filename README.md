<div align="center">
    <img src="./assets/SMiLe-CoDe_logo_noBG.png" alt="SMiLe-CoDe logo" width="300"/>
</div>
<div align="center">
  <h1 style="margin-bottom: 0.2em;">SMiLe-CoDe</h1>
  <p style="font-size: 1.2em; font-style: italic; margin-top: 0;">
    Heuristic seed-set selection for Majority Cascade in social networks
  </p>
  <p style="margin-top: 1em;">
    <a href="https://github.com/Luigina2001" target="_blank">Luigina Costante</a>
    &nbsp;•&nbsp;
    <a href="https://github.com/SalvatoreDL01" target="_blank">Salvatore Michele De Luca</a>
  </p>
</div>

# Introduction

Social influence is the ability of an individual (or a set of individuals) to change others’ behavior through the connections in a network. The **Majority Cascade in Cost Networks** problem asks to select a seed set $S$ of nodes whose total cost is at most $k$, maximizing the final number of influenced nodes under a majority-threshold diffusion process. Because this problem is NP-hard, we rely on heuristics to obtain effective solutions in reasonable time.

**SMiLe-CoDe** proposes a community-based and centrality-driven heuristic, compared experimentally against two other greedy and weighted approaches.

# Methodology

The project implements and compares four seed-selection strategies:

1. **Cost‐Seeds‐Greedy (CSG)**
   A submodular greedy approach that iteratively picks the node with the best $\Delta\text{influence}/\text{cost}$ ratio, using incremental updates and a heap for efficiency.

2. **Weighted Target Set Selection (WTSS)**
   A weighted threshold algorithm that processes nodes under three cases (zero threshold, impossible activation, priority by cost·threshold/degree²), extended to respect a budget constraint.

3. **SMiLe-CoDe**

   * Partition the graph into communities $C_1,\dots,C_m$ using Louvain
   * Allocate budget $k_i = \lfloor k\cdot |C_i|/|V|\rfloor$ to each community
   * Perform a local greedy selection in each community based on betweenness centrality and cost
   * Global phase to exhaust any remaining budget

4. **SMiLe-CoDe\_bridges**
   After the standard phases, picks remaining *local bridges* to extend influence to peripheral nodes, ordered by centrality priority.

All methods run a **Majority Cascade** simulation (threshold $\lceil\deg(v)/2\rceil$) until no new nodes activate.

# Results

Experiments on the **ego-Facebook** network (4,039 nodes, 88,234 edges; average clustering 0.6055) show:

* **WTSS**: near-complete coverage with moderate budgets (∼20 000 for Cost₁), but slower per-iteration time (\~2.5 s).
* **CSG**: consistent performance across cost functions, fast iterations (\~0.25 s), and smoother convergence.
* **SMiLe-CoDe / bridges**: steady growth, slightly below WTSS/CSG, with almost identical results between the base and *bridges* variants (selected local bridges tend to be isolated and of low impact).

Cost functions used:

* **Cost₁**: $\lceil\deg(v)/2\rceil$
* **Cost₂**: random in $[\min(\text{Cost}_1), \max(\text{Cost}_1)]$
* **Cost₃**: normalized betweenness centrality

# Installation Guide

## Requirements

* Python ≥ 3.10
* Git
* (recommended) virtualenv / venv

## Clone the repository

```bash
git clone https://github.com/Luigina2001/SMiLe-CoDe.git
cd SMiLe-CoDe
```

## Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# Usage

Make sure your dataset file is placed in `data/` and named `dataset.txt`.

Example run:

```bash
# 1. Generate seed sets for each cost–algorithm combination
python algorithms/CSG.py
python algorithms/CSG_new.py
python algorithms/WTSS.py
python algorithms/SMiLe-CoDe.py
python algorithms/SMiLe-CoDe-bridges.py

# 2. Simulate Majority Cascade on the computed seeds
python algorithms/cascade.py \
  --experiment_csv_path algorithms/logs/cost1_CSG.csv \
  --graph_path data/dataset.txt \
  --output_csv_path algorithms/logs/cascade/cost1_CSG_results.csv

# Repeat the simulation by varying cost{1,2,3} and algorithm in {CSG, CSG_new, WTSS, SMiLe-CoDe, SMiLe-CoDe-bridges}
```
