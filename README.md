# Counterfactual Regret Minimization for Imperfect Information Games

BSc Computer Science (AI) Final Year Project

## What's This About

This project implements CFR (Counterfactual Regret Minimization) to compute Nash equilibrium strategies for imperfect information games. The focus is on poker variants - specifically Kuhn Poker and Leduc Hold'em - which are commonly used as benchmarks in game theory research.

CFR works by iteratively updating regrets for each decision point, eventually converging to an unexploitable strategy. The same core ideas power systems like Libratus and Pluribus that beat professional poker players.

## Structure

```
code/
├── kuhn_poker.py       # CFR for Kuhn Poker (3-card simplified poker)
├── leduc_holdem.py     # CFR for Leduc Hold'em (6-card poker variant)
├── rps-cfr.py          # Simple Rock-Paper-Scissors CFR example
├── Evaluation/         # Scripts for testing and benchmarking
└── visualisations/     # Strategy convergence plots and analysis
```

## Games Implemented

**Kuhn Poker** - The simplest interesting poker game. 3 cards, 2 players, one betting round. Small enough to solve exactly but captures key poker concepts like bluffing.

**Leduc Hold'em** - A step up in complexity. 6 cards (2 suits × 3 ranks), two betting rounds, one community card. Still tractable but closer to real poker dynamics.

**Rock-Paper-Scissors** - Used as a sanity check for the regret matching algorithm.

## Running

```bash
python code/kuhn_poker.py
python code/leduc_holdem.py
```

The algorithms will run for the specified number of iterations and output the computed strategies.