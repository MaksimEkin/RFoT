# Random Forest of Tensors (RFoT) <img align="left" width="50" height="50" src="RFoT/RFoT.png">

<div align="center", style="font-size: 50px">
    <img src="https://github.com/MaksimEkin/RFoT/actions/workflows/ci_tests.yml/badge.svg?branch=main"></img>
    <img src="https://img.shields.io/hexpm/l/plug"></img>
    <img src="https://img.shields.io/badge/python-v3.8.5-blue"></img>
</div>

<br>

<p align="center">
  <img src="RFoT/rfot_demo.png">
</p>

We introduce a novel semi-supervised ensemble classifier named Random Forest of Tensors (RFoT) that is based on generating a tree of tensors that share the same first dimension, and randomly selecting the remaining of the dimensions and entries of each tensor from the features set. This algorithm is capable of performing abstaining predictions, i.e. say "I do not know" when the prediction is uncertain. Each of the randomly configured tensors that are decomposed with a randomly selected rank will obtain a unique bias and discover different information hidden in our data. Because we exploid the philosophy *"wisdom of crowds*, if one tensor configuration yields poor groupings among the classes, the effect of it would be negligible compared to the other tensors that discovered meaningful arrangements among the classes. We can then apply a clustering algorithm to capture the patterns at each of the latent factors for the first dimension within each R component, for each tensor configurations. Resulting C clusters for each component of each random tensor can be filtered using a cluster *purity score* threshold based on only the known samples within each component. This way, we can remove the inferior components that contain noise. The unknown samples within each cluster can be assigned a class vote using these handful of known instances in a semi-supervised way. These class assignments for the samples are the votes from each of the components. The final class prediction can then be obtained by performing a majority vote on each sample. The process mentioned above, forming a tree of tensors, can be repeated multiple times to form a forest. At each following tree, we look at the abstaining predictions from the prior tree to attempt to capture new patterns.

<div align="center", style="font-size: 50px">

### [:orange_book: Example Notebooks](examples/) &emsp; [:bar_chart: Datasets](data/) &emsp; [:page_facing_up: Abstract](https://www.maksimeren.com/abstract/Random_Forest_of_Tensors_RFoT_MTEM.pdf)  &emsp; [:scroll: Poster](https://www.maksimeren.com/poster/Random_Forest_of_Tensors_RFoT_MTEM.pdf)

</div>


## Installation

#### Option 1: Install using *pip*
```shell
pip install git+https://github.com/MaksimEkin/RFoT.git
```

#### Option 2: Install from source
```shell
git clone https://github.com/MaksimEkin/RFoT.git
cd RFoT
conda create --name RFoT python=3.8.5
source activate RFoT
python setup.py install
```

## Example Usage
```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from RFoT import RFoT

# Load/pre-process the dataset
df = pd.read_pickle("data/mini_ember_df.p")
df.dropna(inplace=True)
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print(df.info())

# organize the dataset
X = df.drop("y", axis=1)
y_true = np.array(df["y"].tolist())
y_experiment = y_true.copy()

# randomly label some as unknown (-1)
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(y_experiment.shape[0]) < 0.3
y_experiment[random_unlabeled_points] = -1

# Predict the unknown sample labels
model = RFoT(
        bin_scale=1,
        max_dimensions=5,
        component_purity_tol=0.99,
        min_rank=11,
        max_rank=21,
        n_estimators=100,
        bin_entry=True,
        clustering="ms",
        max_depth=1,
        n_jobs=10,
)
y_pred = model.predict(X, y_experiment)

# Results
unknown_indices = np.argwhere(y_experiment == -1).flatten()
did_predict_indices = np.argwhere(y_pred[unknown_indices] != -1).flatten()
abstaining_count = len(np.argwhere(y_pred == -1))
f1 = f1_score(
    y_true[unknown_indices][did_predict_indices],
    y_pred[unknown_indices][did_predict_indices],
    average="weighted",
)
print(f1)
```

## Prerequisites
- [Anaconda](https://docs.anaconda.com/anaconda/install/)(Optional)
- numpy>=1.19.2
- pandas>=1.0.5
- matplotlib>=3.3.4
- joblib>=1.0.1
- scikit-learn>=0.22.2
- scipy>=1.5.3
- seaborn>=0.11.1
- torch>=1.6.0
- requests>=2.25.1
- tqdm>=4.62.3
- sparse>=0.13.0
