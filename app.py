import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from src.suspicious_network import detect_suspicious_networks
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.create_graph_data import create_graph_data

from src.train import train_model
from src.train_gat import train_gat
from src.evaluate import evaluate_model

from src.fraud_insights import generate_insights
from src.risk_score import calculate_risk_scores


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Fraud Detection using GNN",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Fraud Detection using Graph Neural Networks")

st.markdown(
    """
This application detects fraudulent Bitcoin transactions using:

- GraphSAGE
- Graph Attention Network (GAT)

Dataset: Elliptic Bitcoin Dataset
"""
)

st.divider()


# ==================================================
# LOAD DATA
# ==================================================

features, edges, labels = load_data()
suspicious_networks = detect_suspicious_networks(
    edges
)
insights = generate_insights(
    features,
    edges,
    labels
)

risk_scores = calculate_risk_scores(
    edges
)


# ==================================================
# DATASET OVERVIEW
# ==================================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

total_transactions = len(features)

total_connections = len(edges)

fraud_count = (labels["class"] == "1").sum()

col1.metric(
    "Transactions",
    total_transactions
)

col2.metric(
    "Connections",
    total_connections
)

col3.metric(
    "Fraud Transactions",
    fraud_count
)

st.divider()
st.header("🕸️ Suspicious Networks")

if len(suspicious_networks):

    st.dataframe(

        suspicious_networks,

        use_container_width=True

    )

else:

    st.info(

        "No suspicious networks detected."

    )

# ==================================================
# GRAPH STATISTICS
# ==================================================

st.header("📈 Graph Statistics")

stats = pd.DataFrame({

    "Metric": [

        "Nodes",

        "Edges",

        "Features per Node"

    ],

    "Value": [

        len(features),

        len(edges),

        features.shape[1] - 1

    ]
})

st.dataframe(
    stats,
    use_container_width=True
)

st.divider()


# ==================================================
# FRAUD DISTRIBUTION
# ==================================================

st.header("🥧 Fraud Distribution")

distribution = (

    labels["class"]

    .value_counts()

    .reset_index()

)

distribution.columns = [

    "Class",

    "Count"

]

distribution["Class"] = distribution["Class"].replace({

    "1": "Fraud",

    "2": "Genuine",

    "unknown": "Unknown"

})

fig = px.pie(

    distribution,

    names="Class",

    values="Count",

    title="Transaction Distribution"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()


# ==================================================
# GRAPH VISUALIZATION
# ==================================================

st.header("🕸️ Sample Transaction Network")

sample_edges = edges.head(300)

G = nx.from_pandas_edgelist(

    sample_edges,

    source="txId1",

    target="txId2"

)

fig, ax = plt.subplots(

    figsize=(10, 7)

)

nx.draw(

    G,

    ax=ax,

    node_size=15,

    with_labels=False

)

st.pyplot(fig)

st.divider()


# ==================================================
# FRAUD INTELLIGENCE
# ==================================================

st.header("🚨 Fraud Intelligence")

col1, col2, col3, col4 = st.columns(4)

col1.metric(

    "Total Transactions",

    insights["total"]

)

col2.metric(

    "Fraud",

    insights["fraud"]

)

col3.metric(

    "Genuine",

    insights["genuine"]

)

col4.metric(

    "Fraud %",

    f'{insights["fraud_percentage"]}%'

)

st.divider()


# ==================================================
# TOP SUSPICIOUS TRANSACTIONS
# ==================================================

st.header("⚠️ Top Suspicious Transactions")


def assign_risk(score):

    if score >= 70:

        return "High"

    elif score >= 30:

        return "Medium"

    else:

        return "Low"


risk_scores["Risk Level"] = (

    risk_scores["Risk Score"]

    .apply(assign_risk)

)

top10 = risk_scores.sort_values(

    "Risk Score",

    ascending=False

).head(10)

st.dataframe(

    top10,

    use_container_width=True

)

st.divider()


# ==================================================
# RISK DISTRIBUTION
# ==================================================

st.header("📉 Risk Distribution")

fig = px.histogram(

    risk_scores,

    x="Risk Level",

    title="Risk Categories"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()


# ==================================================
# TRAIN MODELS
# ==================================================

st.header("🤖 Train Models")

if st.button("Train GraphSAGE and GAT"):

    with st.spinner("Preprocessing Data..."):

        x, edge_index, y = preprocess_data(

            features,

            edges,

            labels

        )

        data = create_graph_data(

            x,

            edge_index,

            y

        )

    # ----------------------------------------
    # GraphSAGE
    # ----------------------------------------

    st.subheader("Training GraphSAGE")

    graphsage_model = train_model(

        data

    )

    graphsage_metrics = evaluate_model(

        graphsage_model,

        data

    )

    # ----------------------------------------
    # GAT
    # ----------------------------------------

    st.subheader("Training GAT")

    gat_model = train_gat(

        data

    )

    gat_metrics = evaluate_model(

        gat_model,

        data

    )

    # ----------------------------------------
    # COMPARISON
    # ----------------------------------------

    st.header("🏆 Model Comparison")

    comparison = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ],

        "GraphSAGE": [

            graphsage_metrics["accuracy"],

            graphsage_metrics["precision"],

            graphsage_metrics["recall"],

            graphsage_metrics["f1_score"]

        ],

        "GAT": [

            gat_metrics["accuracy"],

            gat_metrics["precision"],

            gat_metrics["recall"],

            gat_metrics["f1_score"]

        ]

    })

    st.dataframe(

        comparison,

        use_container_width=True

    )

    if graphsage_metrics["f1_score"] > gat_metrics["f1_score"]:

        winner = "GraphSAGE"

    else:

        winner = "GAT"

    st.success(

        f"🏆 Best Model: {winner}"

    )

st.divider()

st.caption("Fraud Detection using Graph Neural Networks")