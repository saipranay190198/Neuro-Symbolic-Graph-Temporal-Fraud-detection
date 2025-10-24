# 🕸️ Graph-Based Model for Fraud Detection  
*A Research-Driven Proof-of-Concept for Explainable Financial Fraud Analysis*  

---

## 🚀 Overview  
This repository presents a **conceptual and methodological framework** for detecting financial fraud using **graph-based learning**, **temporal encoding**, and **explainable AI**.  

Developed as a **research-oriented proof-of-concept**, this framework demonstrates how **graph representation learning** can identify complex fraud topologies such as **circular money flows**, **layered mule chains**, and **coordinated fraud rings**.  

The project employs **PaySim**, a publicly available synthetic dataset that simulates real-world mobile transaction systems.  
Although the data is not sourced from actual financial institutions, it enables **controlled experimentation** to evaluate how advanced AI methods can reveal hidden fraud patterns and provide **interpretable, human-readable insights**.

---

## 🎯 Research Motivation  
Traditional fraud detection systems rely on **rule-based logic or isolated transaction classification**, often missing collaborative and time-evolving fraud behavior.  
To address this limitation, the project integrates:  

- 🧠 **Graph Neural Networks (GraphSAGE)** for relational and topological feature learning  
- ⏳ **Temporal Feature Encoding** to capture evolving transaction patterns  
- 💬 **Symbolic Rule Induction (Decision Tree)** for explainable, rule-based reasoning  

Together, these components form a hybrid system that balances **deep learning power** with **human interpretability**, aligning with the current shift toward **trustworthy and transparent AI** in financial domains.  

--

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
## GraphSAGE + Symbolic Rule Inducer (Decision Tree) Pipeline

 Build graph features → Learn embeddings via GraphSAGE →
 Generate interpretable fraud detection rules (Decision Tree)


### 🧱 Components

| Module | Description |
|---------|-------------|
| **Graph Construction** | Builds a directed graph from sender-receiver relationships in the transaction dataset |
| **Feature Engineering** | Adds node-level metrics (in-degree, out-degree, transaction volume, etc.) |
| **GraphSAGE Embedding** | Learns relational representations of each account/node |
| **Symbolic Rule Inducer (Decision Tree)** | Translates deep embeddings into explainable “if–then” rules for fraud detection |

### 🧩 Implementation Snippet

python
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

Full code Implementation can be found here: Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1).py

## 🧮 Outputs Generated

After running the pipeline, the following artifacts are created:

| File | Description |
|------|--------------|
| `model_artifacts/node_embeddings.npy` | Learned **GraphSAGE node embeddings** (latent relational features) |
| `model_artifacts/node_features_with_embeddings.csv` | Combined **numeric + structural features** for each node |
| `model_artifacts/sri_rules.txt` | **Human-readable fraud rules** extracted from the Decision Tree |

**Example symbolic rules snippet:**

--- in_deg <= 1.31
| |--- emb_0 <= 3511.47
| | |--- degree <= -1.41: Fraud
| |--- else: Non-Fraud
|--- else: Non-Fraud

These rules **approximate the learned deep patterns** in simple threshold-based form — bridging **AI interpretability** with **operational fraud analysis**.

---

## 🧭 How to Run

| Step | Command / Description |
|------|------------------------|
| **1. Install Dependencies** | ```bash pip install torch torch-geometric networkx pyvis scikit-learn pandas numpy ``` |
| **2. Run the Model Script** | ```bash python "Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1).py" ``` |
| **3. Review Outputs** | Artifacts will be generated inside the `model_artifacts/` directory. |

---

## 🧠 Explainability & Usage

| Concept | Description |
|----------|-------------|
| **GraphSAGE Embeddings** | Capture **relational behavior** — who transacts with whom, how often, and how strongly. |
| **Decision Tree (SRI)** | Converts continuous embeddings into **transparent, rule-based insights**. |
| **Analyst Utility** | Supports **AML analysts**, **auditors**, and **fintech risk teams** for investigative dashboards and fraud alerts. |

---

## 🌐 Visualization

Use **PyVis** (or **NetworkX**) to visualize **fraud rings** and money-flow patterns:

python
from pyvis.network import Network
net = Network(notebook=True)
Add nodes and edges based on df_sample
net.show("fraud_network.html")

## 📁 Folder Structure
📂 Graph-Fraud-Detection
├── Graph Based Model to detect fraudulent transactions(Circular Ring Fraud) (1).py         # Main pipeline script
├── model_artifacts/                  # Saved models, embeddings, and rules
├── data/                             # Sample or PaySim data
└── README.md                         # Documentation (this file)

## Research Contribution

This framework contributes toward the explainable graph learning paradigm in fraud analytics by demonstrating:

How graph topology and temporal behavior reveal multi-entity fraud

How symbolic reasoning can make deep models auditable

How synthetic simulations like PaySim can safely advance fraud research without exposing sensitive data

## 🧩 Future Work(Based on limitations/challenges)

 How can large-scale transaction graphs (millions of nodes) be efficiently processed using distributed or sampling-based GNNs?
 How can we design a scalable streaming graph framework that supports real-time graph updates and anomaly detection for continuously evolving financial transaction networks?
 How can temporal GNNs (e.g., TGAT, DySAT) model streaming financial data for *near-real-time* detection?                      
 How can symbolic reasoning or rule extraction make black-box embeddings interpretable for compliance and auditors?            
 How can *federated graph learning* enable cross-institution collaboration without violating privacy?                          

#

## ✨ Citation

Rachumalla, S.P. (2025). *Graph-Based Model for Fraud Detection: 
An Explainable AI Framework for Transaction Networks.* GitHub Repository.

