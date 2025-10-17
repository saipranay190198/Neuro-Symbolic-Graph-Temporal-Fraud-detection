# 🕸️ Graph-Based Model for Fraud Detection

This repository presents an **end-to-end fraud detection framework** built around **graph-based modeling, temporal learning, and explainable AI** principles.  
It integrates **synthetic data simulation** and **PaySim transaction data** to analyze **circular, layered, and coordinated financial fraud networks**.

---

## 🚀 Overview

Financial fraud has evolved into **multi-entity, ring-structured schemes** that often bypass traditional, rule-based detection systems.  
This framework leverages **graph learning** and **temporal representation** to:

- Model relationships between senders and receivers  
- Uncover hidden **fraud rings and mule networks**  
- Assign explainable risk scores to each entity  
- Generate **interpretable symbolic rules** for interpretable reasoning 

---

## 🧩 The Framework Combines

| Component | Purpose |
|------------|----------|
| 🧠 **Self-Supervised Temporal Encoder (SSTE)** | Learns behavioral embeddings from transaction sequences |
| 🔗 **Graph Risk Propagation (GRP)** using **GraphSAGE (PyTorch Geometric)** | Detects layered fraud and coordinated ring patterns |
| 💬 **Symbolic Rule Inducer (SRI)** | Extracts human-readable fraud patterns from deep embeddings |
| 🌐 **PyVis Interactive Network Graphs** | Visualizes circular money movements and anomalies |

---

## ⚙️ Core Idea

Fraud networks often form **closed loops** (rings) or **layered money mule chains**.  
This framework identifies such structures by:

1. Constructing daily or batch-wise **transaction graphs**
2. Detecting **repeated cyclic patterns**
3. Assigning **risk scores** using graph embeddings
4. Using **symbolic reasoning** to extract explainable rules

---

## 🧠 Architecture Pipeline

```python
# GraphSAGE + Symbolic Rule Inducer (Decision Tree) Pipeline
# ----------------------------------------------------------
# Build graph features → Learn embeddings via GraphSAGE → 
# Generate interpretable fraud detection rules (Decision Tree)

import os
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Assumes df_sample (with sender, receiver, amount, is_fraud) is available
# ----------------------------------------------------------
# Build directed graph, engineer node features, and compute embeddings

# [Code truncated here for brevity – see full version in `Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1).py`]

The full working code is included in Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1) .py


---

## 🧮 Outputs Generated

After running the pipeline:

File	Description
model_artifacts/node_embeddings.npy	Learned GraphSAGE node embeddings (latent features)
model_artifacts/node_features_with_embeddings.csv	Combined numeric + relational features per node
model_artifacts/sri_rules.txt	Human-readable fraud rules extracted from the decision tree

Example symbolic rules snippet:

|--- in_deg <= 1.31
|   |--- emb_0 <= 3511.47
|   |   |--- degree <= -1.41: Fraud
|   |--- else: Non-Fraud
|--- else: Non-Fraud


These rules approximate the learned deep patterns in simple threshold-based form — bridging AI interpretability with operational fraud analysis.

## 🧭 How to Run

Install dependencies:

pip install torch torch-geometric networkx pyvis scikit-learn pandas numpy

Run:

python Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1).py


Review results in model_artifacts/.

## 🧠 Explainability & Usage

The GraphSAGE embeddings encode relational patterns (who transacts with whom and how often).

The Decision Tree converts these continuous embeddings into transparent, rule-based insights.

These can support AML analysts, auditors, or fintech risk teams for investigative dashboards or alerts.

## 🌐 Visualization

Use PyVis (or NetworkX drawing utilities) to render fraud rings:

from pyvis.network import Network
net = Network(notebook=True)
# Add nodes/edges based on df_sample
net.show("fraud_network.html")

## 📁 Folder Structure
📂 Graph-Fraud-Detection
├── Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1).py         # Main pipeline script
├── model_artifacts/                  # Saved models, embeddings, and rules
├── data/                             # Sample or PaySim data
└── README.md                         # Documentation (this file)

## 🧩 Future Work

Integration with Temporal GNNs (TGAT, DySAT)

Federated Graph Fraud Learning for cross-bank collaboration

Visualization of layered fund flows

Deployment-ready API for live risk scoring

## ✨ Citation

Rachumalla, S.P. (2025). Graph-Based Model for Fraud Detection: 
An Explainable AI Framework for Transaction Networks. GitHub Repository.
