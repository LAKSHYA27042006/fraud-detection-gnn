import pandas as pd


def calculate_risk_scores(edges):

    degree = (

        pd.concat([

            edges["txId1"],

            edges["txId2"]

        ])

        .value_counts()

    )

    risk = (

        degree

        / degree.max()

        * 100

    )

    risk = risk.reset_index()

    risk.columns = [

        "Transaction",

        "Risk Score"

    ]

    return risk