# Random Forest of Tensors (RFoT) <img align="left" width="50" height="50" src="RFoT/RFoT.png">

<div align="center", style="font-size: 50px">
    <img src="https://img.shields.io/hexpm/l/plug"></img>
    <img src="https://img.shields.io/badge/python-v3.8.5-blue"></img>
</div>

<br>

<p align="center">
  <img width="500" src="RFoT/rfot_demo.png">
</p>

Tensor decomposition is a powerful unsupervised Machine Learning method that enables the modeling of multi-dimensional data, including malware data. We introduce a novel ensemble semi-supervised classification algorithm, named Random Forest of Tensors (RFoT), that utilizes tensor decomposition to extract the complex and multi-faceted latent patterns from data. Our hybrid model leverages the strength of multi-dimensional analysis combined with clustering to capture the sample groupings in the latent components, whose combinations distinguish malware and benign-ware. The patterns extracted from a given data with tensor decomposition depend upon the configuration of the tensor such as dimension, entry, and rank selection. To capture the unique perspectives of different tensor configurations, we employ the “wisdom of crowds” philosophy and make use of decisions made by the majority of a randomly generated ensemble of tensors with varying dimensions, entries, and ranks.

As the tensor decomposition backend, RFoT offers two CPD algorithms. First, RFoT package includes the Python implementation of **CP-ALS** (**[pyCP_ALS](https://github.com/MaksimEkin/pyCP_ALS)**) algorithm that was originally introduced in the [MATLAB Tensor Toolbox](https://www.tensortoolbox.org/cp.html>) [2,3,4,5]. CP-ALS backend can also be used to **decompose each random tensor in a parallel manner**. RFoT can also be used with the Python implentation of the **CP-APR** algorithm with the **GPU capability** [1]. Use of CP-APR backend allows decomposing each random tensor configuration both in an **embarrassingly parallel fashion in a single GPU**, and in a **multi-GPU parallel execution**.

<div align="center", style="font-size: 50px">

### [:information_source: Documentation](https://maksimekin.github.io/RFoT/index.html) &emsp; [:orange_book: Example Notebooks](examples/) &emsp; [:bar_chart: Datasets](data/) &emsp; [:page_facing_up: Abstract](https://www.maksimeren.com/abstract/Random_Forest_of_Tensors_RFoT_MTEM.pdf)  &emsp; [:scroll: Poster](https://www.maksimeren.com/poster/Random_Forest_of_Tensors_RFoT_MTEM.pdf)

</div>


## Installation

```shell
conda create --name RFoT python=3.8.5
conda activate RFoT
pip install git+https://github.com/MaksimEkin/RFoT.git
```

## Example Usage
```python
import pickle
import numpy as np
from sklearn.metrics import f1_score
from RFoT import RFoT

# load the exmple data
data = pickle.load(open("data/example.p"))
X = data["X"]
y_experiment = data["y_experiment"]
y_true = data["y_true"]

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

print("Num. of Abstaining", abstaining_count)
print("Percent Abstaining", (abstaining_count / len(unknown_indices)) * 100, "%")
print("F1=", f1)
```
**See the [examples](examples/) for more.**

## How to Cite RFoT?
If you use RFoT, please cite it.

```latex
@MISC{Eren2022RFoT,
  author = {M. E. {Eren} and C. {Nicholas} and E. {Raff} and R. {Yus} and J. S. {Moore} and B. S. {Alexandrov}},
  title = {RFoT},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MaksimEkin/RFoT}}
}

@MISC{eren2021RFoT,
      title={Random Forest of Tensors (RFoT)}, 
      author={M. E. {Eren} and C. {Nicholas} and R. {McDonald} and C. {Hamer}},
      year={2021},
      note={Presented at the 12th Annual Malware Technical Exchange Meeting, Online, 2021}
}
```

## Acknowledgments

This work was done as part of Maksim E. Eren's Master's Thesis at the University of Maryland, Baltimore County with the thesis committee members and collaborators Charles Nicholas, Edward Raff, Roberto Yus, Boian S. Alexandrov, and Juston S. Moore.

## Developer Test Suite
Developer test suites are located under [```tests/```](tests/) directory. Tests can be ran from this folder using ```python -m unittest *```.

## References
[1] Eren, M.E., Moore, J.S., Skau, E.W., Bhattarai, M., Moore, E.A, and Alexandrov, B.. 2022. General-Purpose Unsupervised Cyber Anomaly Detection via Non-Negative Tensor Factorization. Digital Threats: Research and Practice, 28 pages. DOI: https://doi.org/10.1145/3519602

[2] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.

[3] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.

[4] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.

[5] M. E. Eren. pyCP_ALS. https://github.com/MaksimEkin/pyCP_ALS, 2022.
