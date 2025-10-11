# 🕸️ Graph-Based Model for Fraud Detection
🕸️ Graph-Based Model for Fraud Detection

This repository presents an **end-to-end fraud detection framework** built around **graph-based modeling, temporal learning, and explainable AI** principles.  
It includes both **synthetic data simulations** and **real-world PaySim integration** for analyzing circular, layered, and coordinated financial fraud networks.

---

## 🚀 Overview

Financial fraud has evolved into **multi-entity, ring-structured** schemes that evade detection by traditional, rule-based systems.  
This project leverages **graph learning** and **temporal representation** to model relationships between senders and receivers, uncover hidden fraud rings, and visualize suspicious flows of funds interactively.

### The Framework Combines:
- 🧠 **Self-Supervised Temporal Encoders (SSTE)** for modeling transaction sequences and user behavior.
- 🔗 **Graph Risk Propagation (GRP)** using **GraphSAGE (PyTorch Geometric)** to detect layered fraud rings.
- 💬 **Symbolic Rule Inducer (SRI)** for extracting interpretable fraud patterns from deep models.
- 🌐 **PyVis Network Graphs** for visualizing circular money movements and network anomalies.

---

## 🧠 Core Idea

Fraud networks often form **closed loops (circular rings)** or **layered money mule chains**.  
This framework identifies such structures by:
- Constructing daily or batch-wise **transaction graphs**
- Detecting **repeated cyclic patterns**
- Assigning **risk scores** via **graph embeddings**
- Using **symbolic reasoning** to generate human-readable fraud rules

---

## ⚙️ Features

- ✅ Synthetic transaction data generation for experimentation
- 🔄 Circular fraud ring injection for controlled testing
- 🔍 Temporal entity behavior encoding
- 🧩 Graph-based relational modeling (via GraphSAGE)
- 🧠 Risk propagation and fraud scoring
- 💬 Interpretable symbolic rule extraction
- 🌐 Interactive visualization using PyVis

