
# Neuro-Symbolic Graph Temporal (NSGT) Fraud Detection

This repository contains an end-to-end framework for real-time fraud detection using a Neuro-Symbolic Graph Temporal (NSGT) approach.

The system combines:

Self-Supervised Temporal Encoders (SSTE)** for transaction sequence modeling.
Graph Risk Propagation (GRP)*with **GraphSAGE (PyTorch Geometric)** to capture layered fraud rings and relational risk.
Symbolic Rule Inducer (SRI) for **interpretable rules** distilled from the ensemble.

It supports **real-time scoring (<100ms latency)** and generates **human-readable explanations** for fraud alerts.

---

## Project Structure

```
nsgt-fraud-detection/
│
├── data/                  # Store PaySim or real transaction datasets
│   └── paysim.csv
│
├── nsgt/                  # Core NSGT modules
│   ├── __init__.py
│   ├── pyg_graphsage.py   # GraphSAGE training + node embeddings
│   ├── temporal_encoder.py# Sequence encoder (SSTE)
│   ├── ensemble.py        # Supervised head + rule distillation
│
├── scripts/
│   └── load_paysim.py     # Loader & preprocessing for PaySim dataset
│
├── notebooks/             # Example experiments & evaluation
│
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

---

##  Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourname/nsgt-fraud-detection.git
cd nsgt-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

Download the **PaySim dataset** and place it in the `data/` folder:

```
data/paysim.csv
```

Then preprocess:

```bash
python scripts/load_paysim.py
```

This saves a cleaned `transactions.parquet` file.

### 4. Train GraphSAGE embeddings

```bash
python nsgt/pyg_graphsage.py
```

This builds a dynamic transaction graph and learns **node embeddings**.

### 5. Integrate with temporal encoder + ensemble

* Use embeddings from GraphSAGE.
* Concatenate with temporal encoder (SSTE) embeddings + tabular features.
* Train ensemble (Gradient Boosted Trees).
* Distill symbolic rules for **interpretable alerts**.

---

##  Evaluation Metrics

* **PR-AUC** (Precision-Recall)
* **Recall\@FPR** (e.g., 0.5% and 1%)
* **Brier Score** (calibration)
* **Cost Utility** (business-centric impact)
* **Latency per alert (<100ms target)**

---

##  Explainability

Each alert comes with:

* **Top symbolic rules** (from SRI distillation)
* **SHAP values** for local feature attribution
* **Graph motifs** highlighting layered risk

---

##  Roadmap

*  Add Deep Graph Infomax objective for unsupervised GraphSAGE.
*  Build Docker Compose for real-time demo (Kafka + Redis + Model Server).
*  Extend PaySim loader to support streaming simulation.
*  Add benchmark notebooks (PR-AUC, Recall\@FPR).

---

## Contributors

* **Sai Pranay Reddy Rachumalla** – Core design, implementation
* Open to contributions via PRs

---
