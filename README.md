# Random Forest of Tensors (RFoT) <img align="left" width="50" height="50" src="RFoT/RFoT.png">

<div align="center", style="font-size: 50px">
    <img src="https://github.com/MaksimEkin/RFoT/actions/workflows/ci_tests.yml/badge.svg?branch=main"></img>
    <img src="https://img.shields.io/hexpm/l/plug"></img>
    <img src="https://img.shields.io/badge/python-v3.8.5-blue"></img>
</div>

<br>

<p align="center">
  <img width="500" src="RFoT/rfot_demo.png">
</p>

Tensor decomposition is a powerful unsupervised ML method that enables the modeling of multi-dimensional data, including malware data. This paper introduces a novel ensemble semi-supervised classification algorithm, named Random Forest of Tensors (RFoT), that utilizes tensor decomposition to extract the complex and multi-faceted latent patterns from data. Our hybrid model leverages the strength of multi-dimensional analysis combined with clustering to capture the sample groupings in the latent components whose combinations distinguish malware and benign-ware. The patterns extracted from a given data with tensor decomposition depend on the configuration of the tensor such as dimension, entry, and rank selection. To capture the unique perspectives of different tensor configurations we employ the *"wisdom of crowds"* philosophy, and make use of decisions made by the majority of a randomly generated ensemble of tensors with varying dimensions, entries, and ranks. We show the capabilities of RFoT when classifying malware and benign-ware from the EMBER-2018 dataset.


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
pip install git+https://github.com/lanl/pyCP_APR.git
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
# Use a small version of EMBER-2018 dataset
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
        min_dimensions=3,
        max_dimensions=8,
        component_purity_tol=1.0,
        rank=2,
        n_estimators=200,
        bin_entry=False,
        clustering="ms",
        n_jobs=50,
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
**See the [examples](examples/) for more.**

## Prerequisites
- numpy~=1.19.2
- matplotlib>=3.3.4
- pandas>=1.0.5
- scikit-learn>=0.22.2
- scipy>=1.5.3
- seaborn>=0.11.1
- tqdm>=4.62.3
- sparse>=0.13.0
- joblib>=1.0.1
- numpy-indexed>=0.3.5
- torch>=1.6.0
- requests>=2.25.1
- spacy

## Developer Test Suite
Developer test suites are located under [```tests/```](tests/) directory. Tests can be ran from this folder using ```python -m unittest *```.

## References
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.

[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.

[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.

