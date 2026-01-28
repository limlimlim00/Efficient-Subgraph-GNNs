# Subgraphormer

This repository contains the official code of the paper
**[A Flexible, Equivariant Framework for Subgraph GNNs via Graph Products and Graph Coarsening](https://openreview.net/pdf?id=9cFyqhjEHC) (NeurIPS 2024)**

<p align="center">
  <img src="./Figures/framework.png" width="100%" height="50%">
</p>

## Table of Contents

- [Installation](#installation)
- [Configuration Guide](#configuration-file-guide)
- [Reproducibility](#reproducibility)
- [Credits](#credits)

## Installation

First create a conda environment
```
conda env create -f Eff_subgraphs_environment.yml
```
and activate it
```
conda activate Eff_subgraphs
```

## Configuration File Guide

This provides a detailed guide on how to use the provided configuration file. Each section explains the purpose, possible values, and usage of the parameters.

### General Settings

```yaml
general:
  seed: 1
  device: 0
```

- **`seed`**: The random seed used for ensuring reproducibility across multiple runs.
- **`device`**: Specifies the computing device to be used (e.g., `0`, `1`): Specifies a particular GPU index if multiple GPUs are available.

#### Data Settings

```yaml
data:
  name: zinc12k 
  bs: 128 
  num_workers: 4
```

- **`name`**: The dataset being used. In this case, `zinc12k`.
- **`bs`**: Batch size for training and evaluation.
- **`num_workers`**: Number of parallel data loading workers.

##### Data Preprocessing

```yaml
  preprocess:
    max_dis: 5 
    inf_value: 1001 
    max_spd_elements: 10  
    pad_value: 1002 
    global_attr_max_val: 10 
    n_cluster: 7
    dim_laplacian: 2 
```

- **`max_dis`**: The maximum value allowed for the Shortest Path Distance (SPD). Distances exceeding this threshold are clamped.
- **`inf_value`**: A special value representing infinite distance for nodes that are unreachable from each other -- i.e., instead of using infinity, we use `inf_value`.
- **`max_spd_elements`**: Controls the maximum number of elements included in SPD feature calculations -- the set `S` is restricted to its first `max_spd_elements` elements, sorted by SPD distance. Must be **less than 100**.
- **`pad_value`**: A padding value assigned to SPD feature sets `S` that contain fewer than `max_spd_elements` elements.
- **`global_attr_max_val`**: Defines the maximum allowable value for edge features encoding equivariant layers (different values correspond to different parameters). Different integers correspond to different parameters, and this limit ensures edge feature values remain within the specified clamping threshold.
- **`n_cluster`**: Defines the number of clusters used in spectral clustering, impacting graph partitioning.
- **`dim_laplacian`**: The number of eigenvectors used in spectral clustering computations.

#### Model Configuration

```yaml
model:
  num_layer: 6 
  dim_embed: 96 
  final_dim: 1
  dropout: 0.0
  base_mpnn: Gin 
  H: 4 
  residual: False
  aggs: ["uL", "vL", "point"] 
  sum_pooling: False 
  point_encoder: MLP 
```

- **`num_layer`**: Defines the number of layers in the model.
- **`dim_embed`**: Specifies the embedding size for node representations.
- **`final_dim`**: The output dimension of the model.
- **`dropout`**: Dropout rate applied to prevent overfitting. Set to `0.0` to disable dropout.
- **`base_mpnn`**: Specifies the type of Message Passing Neural Network (MPNN). Supported options:
  - `GatV2`: Graph Attention Network V2.
  - `Transformer_conv`: Transformer-based graph convolution.
  - `Gat`: Graph Attention Network.
  - `Gin`: Graph Isomorphism Network (default).
- **`H`**: Number of attention heads for `GatV2`, `Transformer_conv`, and `Gat`. Ignored for `Gin`.
- **`residual`**: Enables (`True`) or disables (`False`) residual connections between layers.
- **`aggs`**: Defines the aggregation strategies applied during training. Options include `uL`, `vL`, and `point`. To exclude an aggregation method, remove it from the list.
- **`sum_pooling`**: Determines the pooling strategy:
  - `False`: Uses mean pooling.
  - `True`: Uses sum pooling.
- **`point_encoder`**: The encoder used for the "point" aggregation. Supported options:
  - `RELU`
  - `MLP`
  - `NONE`

##### Atom Encoder

```yaml
  atom_encoder:
    linear: False  
    in_dim: 6  
```

- **`linear`**: If `False`, a lookup table is used for atom embeddings. If `True`, a linear transformation is applied.
- **`in_dim`**: The input dimensionality of the atom encoder. Only relevant when `linear` is `True`.

##### Edge Encoder

```yaml
  edge_encoder:
    linear: False 
    in_dim: 4 
    use_edge_attr_vL: True
    use_edge_attr_uL: True
```

- **`linear`**: If `False`, a lookup table is used for edge features. If `True`, a linear transformation is applied.
- **``in_dim``**: The input dimension for edge attributes. Only applicable when `linear` is `True`.
- **`use_edge_attr_vL`**: Determines whether to include edge attributes for `vL`.
- **`use_edge_attr_uL`**: Determines whether to include edge attributes for `uL`.

##### Layer Encoder

```yaml
  layer_encoder:
    linear: False 
```

- **`linear`**: If `False`, an MLP is used instead of a simple linear transformation.

#### Training Settings

```yaml
training:
  lr: 0.01 
  wd: 0 
  epochs: 100
  patience: 30
  warmup: 10
```

- **`lr`**: The learning rate used for gradient updates.
- **`wd`**: Weight decay (L2 regularization) applied to model parameters.
- **`epochs`**: The number of epochs for training.
- **`patience`**: Early stopping criterion; training stops if there is no improvement after this number of epochs.
- **`warmup`**: The number of initial epochs where the learning rate is gradually increased before stabilizing -- only applicable for certain optimizers.

#### Weights & Biases (wandb) Logging

```yaml
wandb:
  project_name: TEST
```

- **`project_name`**: The name of the project for tracking experiments using Weights & Biases (`wandb`).


**NOTE:** All the datasets of the paper are supported. The datasets are:

- **Zinc12**, usage: `zinc12k`
- **Zincfull**, usage: `zincfull`
- **Molhiv**, usage: `ogbg-molhiv`
- **Molbace**, usage: `ogbg-molbace`
- **Molesole**, usage: `ogbg-molesol`
- **Peptides-func**, usage: `Peptides-func`
- **Peptides-struct**, usage: `Peptides-struct`

When running over a dataset for the first time, it will create the required transformation and save it inside a folder "datasets". The next time, it will load the transformation from the disk. 

Thus, it is **recommended** to:
- Run the model once to create the transformation before running a hyperparameter sweep.
- Alternatively, run a sweep, but ensure the **first run is a single run** to generate the transformation.


## Reproducibility

### 1. **Run our model**
To run our model on a specific dataset, set the right parameters in the configuration file and simply run:

```bash
python main.py
```

### 2. **Run a Hyperparameter Sweep**
To run a hyperparameter sweep, follow the following steps:

1. ***Create the sweep using a YAML file from the folder `yamls:`***

    ```bash
    wandb sweep -p <your project name> <path to the yaml file>
    ```
    For example:

    ```bash
    wandb sweep -p Sweep_zinc_5_clusters ./yamls/Sweep_zinc12k_n_5.yaml 
    ```
    will initialize a sweep on the `zinc12k` dataset in the project `Sweep_zinc_5_clusters`, and will produce a sweep ID.

2. ***Run the sweep:***

    ```bash
    wandb agent <sweep id>
    ```

## Acknowledgements
Our code is motivated by the code of **[Subgraphormer](https://github.com/BarSGuy/Subgraphormer)**.

## Credits

For academic citations, please use the following:

```
@inproceedings{
bar-shalom2024a,
title={A Flexible, Equivariant Framework for Subgraph {GNN}s via Graph Products and Graph Coarsening},
author={Guy Bar-Shalom and Yam Eitan and Fabrizio Frasca and Haggai Maron},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=9cFyqhjEHC}
}
```

## GPU1 Conda

```bash
conda create -n csgnn python=3.8 -y
conda activate csgnn

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 --extra-index-url https://download.pytorch.org/whl/cu101

pip install torch-scatter==2.0.7 torch-sparse==0.6.9 torch-cluster==1.5.9 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.7.1+cu101.html

pip install torch-geometric==2.0.4

pip install numpy==1.19.5
pip install scipy==1.5.4
pip install scikit-learn==0.24.2
pip install pandas==1.3.5
pip install matplotlib==3.5.*

pip install tqdm==4.*
pip install pyyaml==6.0.1
pip install requests==2.*
pip install protobuf==3.20.*
pip install ogb==1.3.6
pip install easydict==1.11
pip install wandb==0.16.2
pip install gitpython==3.1.41
pip install littleutils==0.2.2
pip install outdated==0.2.2
pip install click==8.1.7
pip install fsspec==2023.12.2
pip install sqlalchemy==1.4.*
pip install reportlab==4.0.8
pip install psutil==5.*
pip install jinja2==3.1.*
pip install ipython==8.12.*
pip install jupyter_client==8.*
pip install jupyter_core==5.*
pip install traitlets==5.*
pip install pygments==2.*
pip install python-dateutil==2.8.*
pip install pytz==2023.3.post1
  
pip install rdkit-pypi==2022.9.5
pip install pycairo==1.20.1
pip install rlpycairo==0.2.0

conda install -y cudatoolkit=10.1

conda clean --all -y
```
