#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.getcwd())


# In[ ]:


pip install symbolicai


# In[ ]:


pip install torch_geometric


# In[ ]:


pip install pyreason


# In[ ]:


pip install networkx


# In[ ]:


pip install python-igraph


# In[6]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # To ignore all warnings
warnings.filterwarnings('ignore', category=UserWarning) # To ignore specific category

np.random.seed(42)  # for reproducibility

n_entities = 10000
n_transactions = 50000

# Create synthetic entity IDs
entities = [f"User_{i}" for i in range(n_entities)]

# Randomly generate transactions
df_large = pd.DataFrame({
    'sender': np.random.choice(entities, n_transactions),
    'receiver': np.random.choice(entities, n_transactions),
    'amount': np.round(np.random.exponential(scale=1000, size=n_transactions), 2),
    'timestamp': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 180, size=n_transactions), unit='D')
})

# Remove self-transfers
df_large = df_large[df_large['sender'] != df_large['receiver']].reset_index(drop=True)

print(df_large.head())
print(f"Total transactions: {len(df_large)}")

df_large['day'] = df_large['timestamp'].dt.date
daily_graphs = {day: df_large[df_large['day'] == day] for day in df_large['day'].unique()}

display(daily_graphs)


# In[2]:


# build graph

import networkx as nx

G = nx.from_pandas_edgelist(
    df_large, 
    source='sender', 
    target='receiver', 
    edge_attr=['amount', 'timestamp'], 
    create_using=nx.DiGraph()
)
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


# Note:
# Node	A point or entity in the network
# Edge	A connection or relationship between nodes
# Node: A bank account or customer
# Edge: A wire transfer between accounts
# Node: A user
# Edge: A friendship or message


# In[3]:


# Engineer Features and Train a Model

# Add a circular fraud ring
ring_entities = [f"Fraud_{i}" for i in range(10)]
ring_tx = []

for i in range(len(ring_entities)):
    sender = ring_entities[i]
    receiver = ring_entities[(i + 1) % len(ring_entities)]
    ring_tx.append({
        'sender': sender,
        'receiver': receiver,
        'amount': np.random.randint(5000, 20000),
        'timestamp': pd.to_datetime('2025-03-01')
    })

# Inject into main dataset
df_large = pd.concat([df_large, pd.DataFrame(ring_tx)], ignore_index=True)

display(df_large)


# In[4]:


# Create Pyvis Graph

# Inject synthetic fraud ring
fraud_ring_ids = [f"User_{900 + i}" for i in range(10)]
fraud_transactions = []

for i in range(len(fraud_ring_ids)):
    fraud_transactions.append({
        'sender': fraud_ring_ids[i],
        'receiver': fraud_ring_ids[(i + 1) % len(fraud_ring_ids)],
        'amount': 9999.99,
        'timestamp': pd.Timestamp('2025-03-01')
    })

# Add fraud to the main DataFrame
df_fraud = pd.DataFrame(fraud_transactions)
df_large = pd.concat([df_large, df_fraud], ignore_index=True).reset_index(drop=True)

# Flag fraud
df_large['is_fraud'] = False
df_large.loc[df_large.index[-len(fraud_transactions):], 'is_fraud'] = True

display(df_large.head())


# In[5]:


import pandas as pd
import numpy as np
from pyvis.network import Network

# Step 1: Create large transaction dataset
np.random.seed(42)
n_entities = 10000
n_transactions = 50000
entities = [f"User_{i}" for i in range(n_entities)]

df_large = pd.DataFrame({
    'sender': np.random.choice(entities, n_transactions),
    'receiver': np.random.choice(entities, n_transactions),
    'amount': np.round(np.random.exponential(scale=1000, size=n_transactions), 2),
    'timestamp': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 180, size=n_transactions), unit='D')
})
df_large = df_large[df_large['sender'] != df_large['receiver']].reset_index(drop=True)

# Step 2: Inject synthetic fraud ring
fraud_ring_ids = [f"User_{900 + i}" for i in range(10)]
fraud_transactions = [{
    'sender': fraud_ring_ids[i],
    'receiver': fraud_ring_ids[(i + 1) % len(fraud_ring_ids)],
    'amount': 9999.99,
    'timestamp': pd.Timestamp('2025-03-01')
} for i in range(len(fraud_ring_ids))]

df_fraud = pd.DataFrame(fraud_transactions)
df_large = pd.concat([df_large, df_fraud], ignore_index=True)
df_large['is_fraud'] = False
df_large.loc[df_large.index[-len(fraud_transactions):], 'is_fraud'] = True

# Step 3: Sample for visualization
df_sample = pd.concat([
    df_large[df_large['is_fraud']],
    df_large[~df_large['is_fraud']].sample(200, random_state=42)
])

# Step 4: Create interactive graph
net = Network(height="700px", width="100%", directed=True)
for _, row in df_sample.iterrows():
    edge_color = "red" if row['is_fraud'] else "green"
    net.add_node(row['sender'], label=row['sender'], color='orange')
    net.add_node(row['receiver'], label=row['receiver'], color='orange')
    net.add_edge(row['sender'], row['receiver'], color=edge_color, title=f"${row['amount']}")

# Step 5: Show
net.write_html("fraud_detection_network.html", open_browser=True)


# In[ ]:


pip install pyvis


# In[6]:


import pandas as pd
import numpy as np
from pyvis.network import Network

# Step 1: Create large transaction dataset
np.random.seed(42)
n_entities = 1000
n_transactions = 5000
entities = [f"User_{i}" for i in range(n_entities)]

df_large = pd.DataFrame({
    'sender': np.random.choice(entities, n_transactions),
    'receiver': np.random.choice(entities, n_transactions),
    'amount': np.round(np.random.exponential(scale=1000, size=n_transactions), 2),
    'timestamp': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 180, size=n_transactions), unit='D')
})
df_large = df_large[df_large['sender'] != df_large['receiver']].reset_index(drop=True)

# Step 2: Inject synthetic fraud ring
fraud_ring_ids = [f"User_{900 + i}" for i in range(10)]
fraud_transactions = [{
    'sender': fraud_ring_ids[i],
    'receiver': fraud_ring_ids[(i + 1) % len(fraud_ring_ids)],
    'amount': 9999.99,
    'timestamp': pd.Timestamp('2025-03-01')
} for i in range(len(fraud_ring_ids))]

df_fraud = pd.DataFrame(fraud_transactions)
df_large = pd.concat([df_large, df_fraud], ignore_index=True)
df_large['is_fraud'] = False
df_large.loc[df_large.index[-len(fraud_transactions):], 'is_fraud'] = True

# Step 3: Sample for visualization
df_sample = pd.concat([
    df_large[df_large['is_fraud']],
    df_large[~df_large['is_fraud']].sample(200, random_state=42)
])

# Step 4: Create interactive graph
net = Network(height="700px", width="100%", directed=True)
# Track nodes we've already added
added_nodes = set()

for _, row in df_sample.iterrows():
    edge_color = "red" if row['is_fraud'] else "green"

    for node in [row['sender'], row['receiver']]:
        if node not in added_nodes:
            # Check if this node appears in any fraud transaction
            is_fraud_node = df_sample[
                ((df_sample['sender'] == node) | (df_sample['receiver'] == node)) & 
                (df_sample['is_fraud'])
            ].any().any()

            node_color = "red" if is_fraud_node else "green"
            net.add_node(node, label=node, color=node_color)
            added_nodes.add(node)

    net.add_edge(row['sender'], row['receiver'], color=edge_color, title=f"${row['amount']}")

# Step 5: Show
net.write_html("fraud_detection_network_1.html", open_browser=True)


# In[7]:


# Sum of amounts sent + received per user
node_amounts = (
    df_sample.groupby('sender')['amount'].sum().add(
    df_sample.groupby('receiver')['amount'].sum(), fill_value=0)
)

import pandas as pd
import numpy as np
from pyvis.network import Network

# Step 1: Create large transaction dataset
np.random.seed(42)
n_entities = 10000
n_transactions = 50000
entities = [f"User_{i}" for i in range(n_entities)]

df_large = pd.DataFrame({
    'sender': np.random.choice(entities, n_transactions),
    'receiver': np.random.choice(entities, n_transactions),
    'amount': np.round(np.random.exponential(scale=1000, size=n_transactions), 2),
    'timestamp': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 180, size=n_transactions), unit='D')
})
df_large = df_large[df_large['sender'] != df_large['receiver']].reset_index(drop=True)

# Step 2: Inject synthetic fraud ring
fraud_ring_ids = [f"User_{900 + i}" for i in range(10)]
fraud_transactions = [{
    'sender': fraud_ring_ids[i],
    'receiver': fraud_ring_ids[(i + 1) % len(fraud_ring_ids)],
    'amount': 9999.99,
    'timestamp': pd.Timestamp('2025-03-01')
} for i in range(len(fraud_ring_ids))]

df_fraud = pd.DataFrame(fraud_transactions)
df_large = pd.concat([df_large, df_fraud], ignore_index=True)
df_large['is_fraud'] = False
df_large.loc[df_large.index[-len(fraud_transactions):], 'is_fraud'] = True

# Step 3: Sample for visualization
df_sample = pd.concat([
    df_large[df_large['is_fraud']],
    df_large[~df_large['is_fraud']].sample(200, random_state=42)
])

# Step 4: Create interactive graph
net = Network(height="700px", width="100%", directed=True)
# Track nodes we've already added
added_nodes = set()

added_nodes = set()

for _, row in df_sample.iterrows():
    edge_color = "red" if row['is_fraud'] else "green"

    for node in [row['sender'], row['receiver']]:
        if node not in added_nodes:
            # Check fraud involvement
            is_fraud_node = df_sample[
                ((df_sample['sender'] == node) | (df_sample['receiver'] == node)) &
                (df_sample['is_fraud'])
            ].any().any()

            node_color = "red" if is_fraud_node else "green"

            # Label with user ID and total transaction amount
            total_amt = node_amounts.get(node, 0)
            node_label = f"{node}\n${total_amt:,.2f}"

            net.add_node(node, label=node_label, color=node_color)
            added_nodes.add(node)

    # Add edge
    net.add_edge(row['sender'], row['receiver'], color=edge_color, title=f"${row['amount']}")

# Step 5: Show
net.write_html("fraud_detection_network_send_receiver.html", open_browser=True)


# In[7]:


import pandas as pd
import numpy as np
import os
from pyvis.network import Network
from tqdm import tqdm

# --- Configuration ---
# Updated to use your specific PaySim file name
PAYSIM_FILE = 'PS_20174392719_1491204439457_log.csv'
# Removed CREATE_DUMMY_DATA flag as requested.

# --- Helper Functions ---

def load_paysim_data(filepath):
    """
    Loads PaySim data from the specified filepath. 
    Exits if the file is not found, as dummy creation is disabled.
    """
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        # Use low_memory=False for large files like PaySim
        df = pd.read_csv(filepath, low_memory=False)
        print("Data loaded successfully.")
        return df
    else:
        # Exit if the file is genuinely missing
        print(f"Error: PaySim file not found at {filepath}. Please ensure the file is in the 'data/' directory.")
        return None

# --- Main Analysis Script ---

# Load data (must exist at the specified path)
df_full = load_paysim_data(PAYSIM_FILE)

if df_full is None:
    # Exit gracefully if loading failed
    exit()

# Rename columns to match the general graph analysis terms
df_full.rename(columns={'nameOrig': 'sender', 'nameDest': 'receiver', 'isFraud': 'is_fraud'}, inplace=True)

# Filter out transactions that don't represent a clear transfer of risk (like DEBIT/PAYMENT to Merchant 'M')
# For simplicity, we'll keep all transactions for now, but a sophisticated model would filter.
df_clean = df_full.copy()

# 1. Calculate the total transactional volume (sent + received) per user
# This is a key feature for detecting high-velocity users in a fraud ring.
print("Calculating total transaction volume per user...")
# Note: This operation can be memory intensive on the full PaySim dataset
node_amounts = (
    df_clean.groupby('sender')['amount'].sum().add(
    df_clean.groupby('receiver')['amount'].sum(), fill_value=0)
)

# 2. Sample data for visualization
# We must keep ALL fraudulent transactions (is_fraud == 1) to ensure the ring is captured.
# Then, sample a small number of legitimate transactions for context.
FRAUD_COUNT = df_clean['is_fraud'].sum()
# Reduce this number if the resulting graph is too slow to load in your browser.
LEGIT_SAMPLE_SIZE = 10  # Increased sample size for better context from the full dataset

df_fraud = df_clean[df_clean['is_fraud'] == 1].copy()

# Ensure we don't try to sample more than available legitimate rows
num_legit = len(df_clean[df_clean['is_fraud'] == 0])
sample_size = min(LEGIT_SAMPLE_SIZE, num_legit)

df_legit_sample = df_clean[df_clean['is_fraud'] == 0].sample(
    sample_size, 
    random_state=42
)
df_sample = pd.concat([df_fraud, df_legit_sample]).reset_index(drop=True)

print(f"Visualization Sample Size: {len(df_sample)} (Fraud: {len(df_fraud)}, Legit Sampled: {len(df_legit_sample)})")

# 3. Create interactive graph using pyvis
net = Network(height="800px", width="100%", directed=True, bgcolor="#222222", font_color="white", cdn_resources='local')
net.heading = "PaySim Fraud Detection Network Analysis (Live Data)"
net.toggle_physics(True) # Enable physics for better clustering

added_nodes = set()

print("Building network visualization...")

# First pass: Determine node colors and calculate labels
node_data = {}
# Only iterate over the unique nodes present in the sampled dataset
all_nodes_in_sample = pd.concat([df_sample['sender'], df_sample['receiver']]).unique()

for node in tqdm(all_nodes_in_sample, desc="Processing Nodes"):
    # Check if the node is involved in ANY fraudulent transaction in the sample
    is_fraud_node = df_sample[
        ((df_sample['sender'] == node) | (df_sample['receiver'] == node)) &
        (df_sample['is_fraud'] == 1)
    ].any().any()

    node_color = "#E33E4D" if is_fraud_node else "#4CAF50" # Red for fraud, Green for legit
    
    # Calculate total volume for the node
    total_amt = node_amounts.get(node, 0)
    
    # Create the label for the node
    node_label = f"{node}\n${total_amt:,.2f}"
    
    node_data[node] = {
        'label': node_label,
        'color': node_color,
        'value': total_amt, # Use amount as value for size scaling
        'title': f"Total Volume: ${total_amt:,.2f}"
    }

# Second pass: Add nodes and edges to the network
for node, data in node_data.items():
    net.add_node(
        node, 
        label=data['label'], 
        color=data['color'], 
        value=data['value'], 
        title=data['title'],
        # Use log scale for size to keep node sizes manageable
        size=min(40, 10 + np.log10(data['value'] + 1) * 5) 
    )

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Adding Edges"):
    edge_color = "#E33E4D" if row['is_fraud'] == 1 else "#50C878" # Red for fraud, Emerald for legit
    
    net.add_edge(
        row['sender'], 
        row['receiver'], 
        color=edge_color, 
        # Add transaction type to the hover title for better inspection
        title=f"Amount: ${row['amount']:,.2f} | Type: {row['type']} | Fraud: {row['is_fraud'] == 1}",
        value=row['amount'], # Use amount for line thickness
        arrows='to'
    )

# 4. Save and display
output_file = "paysim_fraud_network.html"
net.write_html(output_file)
print(f"\n--- Visualization Complete ---")
print(f"Open '{output_file}' in your browser to view the interactive network.")
print(f"Fraudulent nodes and edges (potential fraud rings) are marked in RED.")


# In[9]:


import networkx as nx
import pandas as pd
import numpy as np

# --- Convert sample data (df_sample) into a graph ---
# We use NetworkX for graph analytics; PyVis is just for visualization
G = nx.DiGraph()

# Add weighted directed edges (sender → receiver)
for _, row in df_sample.iterrows():
    G.add_edge(
        row['sender'],
        row['receiver'],
        weight=row['amount'],
        is_fraud=row['is_fraud'],
        tx_type=row['type']
    )

print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Compute key centrality measures ---

# Betweenness Centrality – identifies “brokers” or “intermediaries” controlling fund flow
betweenness = nx.betweenness_centrality(G, k=min(1000, len(G)), weight='weight', normalized=True)

# Closeness Centrality – identifies nodes that can reach many others quickly (potential coordinators)
closeness = nx.closeness_centrality(G)

# Degree Centrality – identifies high-activity entities (senders/receivers)
degree = nx.degree_centrality(G)

# Combine into a DataFrame
centrality_df = pd.DataFrame({
    'node': list(G.nodes),
    'betweenness': [betweenness.get(n, 0) for n in G.nodes],
    'closeness': [closeness.get(n, 0) for n in G.nodes],
    'degree': [degree.get(n, 0) for n in G.nodes],
})

# Add fraud involvement info
centrality_df['is_fraud_node'] = centrality_df['node'].isin(
    df_sample.loc[df_sample['is_fraud'] == 1, ['sender', 'receiver']].values.flatten()
)

# Rank by centrality
centrality_df['rank'] = centrality_df[['betweenness', 'closeness', 'degree']].mean(axis=1).rank(ascending=False)

# --- Identify top suspicious nodes (potential ring centers) ---
top_nodes = centrality_df.sort_values(by='rank').head(15)
print("\nTop 15 Central Nodes (Potential Coordinators or Ring Leaders):")
print(top_nodes[['node', 'betweenness', 'closeness', 'degree', 'is_fraud_node']])

# Optional: Save to CSV
centrality_df.to_csv("centrality_analysis.csv", index=False)


# In[8]:


import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain   # Louvain
from collections import Counter

# Optional: For Leiden (requires installation)
# !pip install leidenalg igraph

import igraph as ig
import leidenalg as la

# --- STEP 1: Convert PaySim sample to NetworkX graph ---
G = nx.DiGraph()

for _, row in df_sample.iterrows():
    G.add_edge(
        row['sender'],
        row['receiver'],
        weight=row['amount'],
        is_fraud=row['is_fraud'],
        tx_type=row['type']
    )

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- STEP 2: Convert to undirected for community detection (optional simplification) ---
# Fraud rings often have mutual transfers, so undirected simplifies modularity detection
G_undirected = G.to_undirected()

# --- STEP 3A: Louvain Community Detection ---
print("\nRunning Louvain community detection...")
partition_louvain = community_louvain.best_partition(G_undirected, weight='weight', resolution=1.0)
nx.set_node_attributes(G_undirected, partition_louvain, 'louvain_community')

# --- STEP 3B: Leiden Community Detection (more accurate, handles small communities) ---
print("Running Leiden community detection...")
g_ig = ig.Graph.TupleList(G_undirected.edges(), directed=False, weights=True)
leiden_partition = la.find_partition(g_ig, la.ModularityVertexPartition)

# Leiden returns a list of vertex clusters; map each node to its cluster ID
node_to_leiden = {g_ig.vs[i]['name']: cid for cid, cluster in enumerate(leiden_partition) for i in cluster}
nx.set_node_attributes(G_undirected, node_to_leiden, 'leiden_community')

# --- STEP 4: Build DataFrame of communities ---
community_df = pd.DataFrame({
    'node': list(G_undirected.nodes()),
    'louvain_community': [partition_louvain[n] for n in G_undirected.nodes()],
    'leiden_community': [node_to_leiden[n] for n in G_undirected.nodes()]
})

# Add fraud info
community_df['is_fraud_node'] = community_df['node'].isin(
    df_sample.loc[df_sample['is_fraud'] == 1, ['sender', 'receiver']].values.flatten()
)

# --- STEP 5: Identify suspicious communities ---
fraud_group_summary = (
    community_df.groupby('louvain_community')['is_fraud_node']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'fraud_nodes', 'count': 'total_nodes'})
    .assign(fraud_ratio=lambda x: x['fraud_nodes'] / x['total_nodes'])
    .sort_values(by='fraud_ratio', ascending=False)
)

print("\nTop Suspicious Louvain Communities:")
print(fraud_group_summary.head(10))

# --- STEP 6: Save and visualize ---
community_df.to_csv("fraud_communities.csv", index=False)
print("\nCommunity analysis saved as 'fraud_communities.csv'")

# Optional: Visualize top communities in PyVis
from pyvis.network import Network
net_comm = Network(height="750px", width="100%", directed=True, bgcolor="#222222", font_color="white")
net_comm.heading = "PaySim Fraud Ring Detection via Louvain + Leiden"

top_comms = fraud_group_summary.head(5).index.tolist()
selected_nodes = community_df[community_df['louvain_community'].isin(top_comms)]['node'].tolist()

for node in selected_nodes:
    n_data = community_df.loc[community_df['node'] == node].iloc[0]
    color = "#E33E4D" if n_data['is_fraud_node'] else "#4CAF50"
    net_comm.add_node(node, color=color, title=f"Louvain: {n_data['louvain_community']}, Leiden: {n_data['leiden_community']}")

for u, v, data in G_undirected.edges(data=True):
    if u in selected_nodes and v in selected_nodes:
        color = "#E33E4D" if (data.get('is_fraud') == 1) else "#50C878"
        net_comm.add_edge(u, v, color=color, title=f"Amount: ${data['weight']:.2f}")

net_comm.write_html("paysim_fraud_communities.html")
print("Visualization saved as 'paysim_fraud_communities.html'")


# In[ ]:


pip install python-louvain


# In[ ]:


pip install leidenalg


# In[10]:


df_clean['timestamp'] = pd.to_datetime(df_clean['step'], unit='s', origin='2025-01-01')
df_clean = df_clean.sort_values('timestamp')


# In[11]:


WINDOW_SIZE = 1000  # instead of 100
time_windows = range(0, int(df_clean['step'].max()), WINDOW_SIZE)


# In[ ]:





# In[12]:


import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt

evolution_data = []

for t in time_windows:
    df_window = df_clean[df_clean['step'] <= t]
    G = nx.from_pandas_edgelist(df_window, 'sender', 'receiver', ['amount', 'is_fraud'])
    
    # Compute centrality
    betweenness = nx.betweenness_centrality(G, k=min(500, len(G)))
    
    # Louvain community detection
    partition = community_louvain.best_partition(G)
    
    # Compute fraud density per community
    df_window['community'] = df_window['sender'].map(partition)
    fraud_density = (
        df_window.groupby('community')['is_fraud'].mean().mean()
        if not df_window.empty else 0
    )
    
    avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0
    
    evolution_data.append({
        'time': t,
        'num_nodes': len(G),
        'num_edges': G.number_of_edges(),
        'avg_betweenness': avg_betweenness,
        'fraud_density': fraud_density,
    })


# In[13]:


df_evo = pd.DataFrame(evolution_data)

plt.figure(figsize=(10,5))
plt.plot(df_evo['time'], df_evo['fraud_density'], label='Fraud Density', color='red')
plt.plot(df_evo['time'], df_evo['avg_betweenness'], label='Avg Betweenness', color='blue')
plt.xlabel('Simulation Step (Time)')
plt.ylabel('Metric Value')
plt.title('Temporal Evolution of Fraud Ring Formation')
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


if df_evo['fraud_density'].iloc[-1] > 0.3 and df_evo['avg_betweenness'].iloc[-1] > threshold:
    print(" Fraud ring likely forming! Investigate high-centrality nodes.")


# In[15]:


top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
print("Potential ring leaders:", [n for n, _ in top_nodes])


# In[ ]:


pip install networkit


# In[ ]:


from datetime import datetime
from Levenshtein import distance as levenshtein_distance

print("Engineering behavioral features (temporal + cognitive metrics)...")

# Ensure we have timestamps if not already
if 'step' in df_clean.columns:
    # Each step in PaySim ~ 1 hour; simulate timestamps for temporal analysis
    df_clean['timestamp'] = pd.to_datetime(df_clean['step'], unit='h', origin='2021-01-01')
else:
    df_clean['timestamp'] = pd.to_datetime(np.arange(len(df_clean)), unit='h', origin='2021-01-01')

# --- RISK AVERSION SCORE ---
# Measures how drastically a user deviates from their typical transaction behavior
def compute_risk_aversion(df):
    risk_scores = {}
    for sender, group in df.groupby('sender'):
        mean_amt = group['amount'].mean()
        recent = group.sort_values('timestamp', ascending=False).head(1)['amount'].iloc[0]
        deviation = abs(recent - mean_amt) / (mean_amt + 1e-6)
        risk_scores[sender] = min(deviation, 1.0)  # normalize
    return risk_scores

risk_aversion = compute_risk_aversion(df_clean)

# --- SEQUENTIAL SIMILARITY ---
# Compare transaction sequences to a "fraud grammar" using Levenshtein distance
def compute_seq_similarity(df, fraud_template=['TRANSFER', 'CASH_OUT']):
    seq_scores = {}
    for sender, group in df.groupby('sender'):
        seq = list(group['type'].values)
        dist = levenshtein_distance(''.join(seq), ''.join(fraud_template))
        seq_scores[sender] = 1 / (1 + dist)  # higher → more similar to fraud pattern
    return seq_scores

seq_similarity = compute_seq_similarity(df_clean)

# --- TIME-GAP ANALYSIS ---
# Measures how consistent (bot-like) the transaction timing is
def compute_time_gap_stability(df):
    stability = {}
    for sender, group in df.groupby('sender'):
        times = group['timestamp'].sort_values()
        gaps = times.diff().dt.total_seconds().dropna()
        if len(gaps) > 1:
            coef_var = np.std(gaps) / (np.mean(gaps) + 1e-6)
            stability[sender] = 1 - min(coef_var, 1.0)  # 1 → very regular → suspicious
        else:
            stability[sender] = 0
    return stability

time_gap_stability = compute_time_gap_stability(df_clean)


# In[1]:


pip install swifter


# In[16]:


# Keep all frauds and a small random subset of legits
fraud_df = df_clean[df_clean['is_fraud'] == 1]
legit_df = df_clean[df_clean['is_fraud'] == 0].sample(5000, random_state=42)
df_behav = pd.concat([fraud_df, legit_df])


# In[17]:


df_behav


# In[18]:


print("Computing Risk Aversion (vectorized)...")


# Ensure we have timestamps if not already
if 'step' in df_clean.columns:
    # Each step in PaySim ~ 1 hour; simulate timestamps for temporal analysis
    df_clean['timestamp'] = pd.to_datetime(df_clean['step'], unit='h', origin='2021-01-01')
else:
    df_clean['timestamp'] = pd.to_datetime(np.arange(len(df_clean)), unit='h', origin='2021-01-01')

# Keep all frauds and a small random subset of legits
fraud_df = df_clean[df_clean['is_fraud'] == 1]
legit_df = df_clean[df_clean['is_fraud'] == 0].sample(5000, random_state=42)
df_behav = pd.concat([fraud_df, legit_df])


recent_tx = df_behav.sort_values(['sender', 'timestamp']).groupby('sender').tail(1)
mean_amt = df_behav.groupby('sender')['amount'].mean()

risk_aversion = (
    abs(recent_tx.set_index('sender')['amount'] - mean_amt) / (mean_amt + 1e-6)
).clip(0, 1).to_dict()


# In[19]:


risk_aversion


# In[20]:


print("Computing Sequence Similarity (approx)...")

fraud_template = {'TRANSFER': 1, 'CASH_OUT': 1}

seq_similarity = {}
for sender, group in df_behav.groupby('sender'):
    counts = group['type'].value_counts(normalize=True).to_dict()
    overlap = sum(min(counts.get(k, 0), fraud_template.get(k, 0)) for k in fraud_template)
    seq_similarity[sender] = overlap  # 0–1 score


# In[21]:


seq_similarity


# In[22]:


print("Computing Time Gap Stability (optimized)...")

df_behav['time_diff'] = (
    df_behav.sort_values(['sender', 'timestamp'])
    .groupby('sender')['timestamp']
    .diff().dt.total_seconds()
)

gap_stats = df_behav.groupby('sender')['time_diff'].agg(['mean', 'std']).fillna(0)
gap_stats['stability'] = 1 - (gap_stats['std'] / (gap_stats['mean'] + 1e-6)).clip(0, 1)
time_gap_stability = gap_stats['stability'].to_dict()


# In[23]:


behavioral_features = pd.DataFrame({
    'sender': list(risk_aversion.keys()),
    'risk_aversion': list(risk_aversion.values()),
    'seq_similarity': [seq_similarity.get(u, 0) for u in risk_aversion.keys()],
    'time_stability': [time_gap_stability.get(u, 0) for u in risk_aversion.keys()],
})
behavioral_features.to_csv("behavioral_features_cache.csv", index=False)


# In[24]:


behavioral_features


# In[25]:


for node in tqdm(all_nodes_in_sample, desc="Processing Nodes with Behavioral Features"):
    is_fraud_node = df_sample[
        ((df_sample['sender'] == node) | (df_sample['receiver'] == node)) &
        (df_sample['is_fraud'] == 1)
    ].any().any()

    node_color = "#E33E4D" if is_fraud_node else "#4CAF50"
    total_amt = node_amounts.get(node, 0)

    # Integrate new behavioral metrics
    risk = risk_aversion.get(node, 0)
    seq_sim = seq_similarity.get(node, 0)
    time_stab = time_gap_stability.get(node, 0)

    # Composite Cognitive Risk Index
    cognitive_risk = (0.5 * risk + 0.3 * seq_sim + 0.2 * time_stab)

    node_data[node] = {
        'label': f"{node}\n$ {total_amt:,.2f}",
        'color': node_color,
        'value': total_amt,
        'title': f"Total: ${total_amt:,.2f} | Risk Aversion: {risk:.2f} | SeqSim: {seq_sim:.2f} | TimeGap: {time_stab:.2f} | CognitiveRisk: {cognitive_risk:.2f}"
    }


# In[17]:


node_data


# In[18]:


import networkx as nx

# Build multi-type transaction graph (Heterogeneous)
G = nx.MultiDiGraph()

for _, row in df_sample.iterrows():
    G.add_edge(row['sender'], row['receiver'], 
               tx_type=row['type'], 
               amount=row['amount'], 
               fraud=row['is_fraud'])

# Counterfactual Risk Propagation: simulate if a high-risk node changes behavior
def counterfactual_risk(G, node):
    if node not in G:
        return None
    neighbors = list(G.successors(node))
    simulated_risk = 0
    for n in neighbors:
        edge_data = G.get_edge_data(node, n)
        if edge_data:
            amt = np.mean([d['amount'] for d in edge_data.values()])
            simulated_risk += np.log1p(amt) * 0.1
    return simulated_risk

# Example: compute for top-5 high-value nodes
top_nodes = node_amounts.sort_values(ascending=False).head(5).index
counterfactuals = {n: counterfactual_risk(G, n) for n in top_nodes}
print("\n--- Counterfactual Risk Propagation ---")
print(counterfactuals)


# In[ ]:




