# -*- coding: utf-8 -*-
from RFoT import RFoT
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def main():
    #
    # LOAD DATA
    #
    df = pd.read_pickle("../data/ember_df.p")
    df.dropna(inplace=True)
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    df = df.sample(500, random_state=42)
    print(df.info())

    X = df.drop("y", axis=1)
    y_true = np.array(df["y"].tolist())
    y_experiment = y_true.copy()

    #
    # MARK UNKNOWN RANDOMLY
    #
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(y_experiment.shape[0]) < 0.3
    y_experiment[random_unlabeled_points] = -1

    #
    # PREDICT
    #
    model = RFoT(
        bin_scale=1,
        max_dimensions=5,
        component_purity_tol=0.99,
        # cluster_quality_tol=0.55,
        min_rank=11,
        max_rank=21,
        n_estimators=10,
        bin_entry=True,
        clustering="ms",
        max_depth=1,
        n_jobs=1,
    )
    y_pred = model.predict(X, y_experiment)

    #
    # STATS
    #
    unknown_indices = np.argwhere(y_experiment == -1).flatten()
    did_predict_indices = np.argwhere(y_pred[unknown_indices] != -1).flatten()
    abstaining_count = len(np.argwhere(y_pred == -1))
    f1 = f1_score(
        y_true[unknown_indices][did_predict_indices],
        y_pred[unknown_indices][did_predict_indices],
        average="weighted",
    )

    print("\n\n------------------------")
    print("Num. of Abstaining", abstaining_count)
    print("Percent Abstaining", (abstaining_count / len(unknown_indices)) * 100, "%")
    print("F1=", f1)


if __name__ == "__main__":
    main()
