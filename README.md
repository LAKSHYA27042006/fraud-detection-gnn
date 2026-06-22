# Fraud Detection using Graph Neural Networks (GNN)

## Overview

Fraud Detection using Graph Neural Networks (GNN) is an AI-based system developed to detect fraudulent Bitcoin transactions by analyzing relationships between transactions instead of treating each transaction independently.

Traditional machine learning models analyze transactions as individual records and often fail to capture hidden relationships between them. This project leverages Graph Neural Networks (GNNs) to learn from transaction networks and identify suspicious activities, fraud rings, and connected fraudulent transactions.

This project implements two Graph Neural Network architectures:

- GraphSAGE
- Graph Attention Network (GAT)

An interactive Streamlit dashboard was also developed to visualize fraud analytics and compare model performance.

---

## Problem Statement

Traditional machine learning algorithms struggle to detect organized fraud because they cannot understand relationships between transactions.

Fraudsters often operate in connected networks such as:

- Money laundering chains
- Fraud rings
- Coordinated attacks
- Suspicious transaction clusters

The objective of this project is to build a graph-based fraud detection system that learns transaction relationships and identifies fraudulent activities more effectively.

---

## Objectives

- Build a graph representation of Bitcoin transactions
- Detect fraudulent transactions using Graph Neural Networks
- Implement and compare GraphSAGE and GAT models
- Analyze suspicious transaction patterns
- Detect suspicious transaction networks
- Develop an interactive fraud analytics dashboard

---

## Dataset Used

### Elliptic Bitcoin Dataset

The project uses the Elliptic Bitcoin Dataset.

Dataset Files:

### 1. elliptic_txs_features.csv

Contains:

- Transaction ID
- Time step
- 166 transaction features

### 2. elliptic_txs_edgelist.csv

Contains:

- txId1
- txId2

Represents relationships between transactions.

### 3. elliptic_txs_classes.csv

Contains:

- Transaction ID
- Class labels

Labels:

- 1 → Fraudulent Transaction
- 2 → Genuine Transaction
- Unknown → Unlabeled Transaction

Dataset Source:

https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Deep Learning | PyTorch |
| Graph Deep Learning | PyTorch Geometric |
| Graph Analysis | NetworkX |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Evaluation | Scikit-learn |

---

## System Architecture

Elliptic Dataset

↓

Data Loading

↓

Data Preprocessing

↓

Graph Construction

↓

PyTorch Geometric Data Object

↓

GraphSAGE

↓

GAT

↓

Model Evaluation

↓

Fraud Intelligence

↓

Suspicious Network Detection

↓

Streamlit Dashboard

---

## Project Workflow

1. Dataset Collection
2. Data Loading
3. Data Preprocessing
4. Graph Construction
5. GraphSAGE Implementation
6. GAT Implementation
7. Model Training
8. Model Evaluation
9. Fraud Intelligence Generation
10. Suspicious Network Detection
11. Dashboard Development

---

## Features Implemented

### Fraud Detection

- Fraud classification
- Genuine transaction identification

### Graph Analytics

- Node and edge creation
- Relationship learning

### Fraud Intelligence

- Fraud percentage
- Genuine percentage
- Fraud statistics

### Risk Analysis

- Risk score generation
- Risk categorization

### Suspicious Network Detection

- Fraud ring identification
- Connected component analysis

### Dashboard Features

- Dataset Overview
- Graph Statistics
- Fraud Distribution
- Graph Visualization
- Fraud Intelligence
- Risk Distribution
- Top Suspicious Transactions
- Model Comparison

---

## Models Implemented

### GraphSAGE

GraphSAGE (Graph Sample and Aggregate) learns node representations by aggregating information from neighboring nodes.

Advantages:

- Fast
- Scalable
- Efficient for large graphs

### GAT (Graph Attention Network)

GAT uses attention mechanisms to assign different importance weights to neighboring nodes.

Advantages:

- Better relationship learning
- Intelligent neighbor selection
- Improved graph representation

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score

Note:

Since fraud detection datasets are highly imbalanced, Precision, Recall, and F1 Score were prioritized over Accuracy.

---

## Folder Structure

```text
fraud-detection-gnn/

app.py

README.md

requirements.txt

src/

models/

utils/

saved_models/

dataset/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/LAKSHYA27042006/fraud-detection-gnn.git
```

Navigate to project directory:

```bash
cd fraud-detection-gnn
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run Streamlit dashboard:

```bash
streamlit run app.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

## Challenges Faced

- Understanding graph-based learning
- Handling large datasets
- Class imbalance
- GAT performance optimization
- Dashboard integration

---

## Future Enhancements

- Real-time fraud detection
- Explainable AI
- Temporal Graph Neural Networks
- Cloud deployment
- API integration
- Dynamic graph updates
- Advanced fraud ring detection

---

## Conclusion

This project successfully demonstrates how Graph Neural Networks can be used to detect fraudulent transactions by leveraging transaction relationships instead of analyzing transactions independently.

The integration of GraphSAGE, GAT, Fraud Intelligence, and Suspicious Network Detection creates an industry-relevant solution capable of supporting real-world financial fraud investigations.

---

## References

1. Elliptic Bitcoin Dataset – Kaggle
2. PyTorch Documentation
3. PyTorch Geometric Documentation
4. NetworkX Documentation
5. Streamlit Documentation
6. Scikit-learn Documentation
7. Graph Neural Network Research Papers
8. Fraud Detection using GNN Research Papers
