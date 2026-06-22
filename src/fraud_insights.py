import pandas as pd


def generate_insights(features, edges, labels):

    total_transactions = len(features)

    fraud_transactions = (
        labels["class"] == "1"
    ).sum()

    genuine_transactions = (
        labels["class"] == "2"
    ).sum()

    fraud_percentage = round(

        fraud_transactions

        / total_transactions

        * 100,

        2

    )

    return {

        "total": total_transactions,

        "fraud": fraud_transactions,

        "genuine": genuine_transactions,

        "fraud_percentage": fraud_percentage

    }