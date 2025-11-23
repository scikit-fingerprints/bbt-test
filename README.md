# bbt-test

---

BBT-Test is a Python package for Bayesian Bradley-Terry model along with utilities for multi-algorithm multi-dataset statistical evaluation.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [License](#license)

## Installation

You can install bbt-test via pip:

```bash
pip install bbt-test
```

If needed, you can also install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/scikit-fingerprints/bbt-test
```

## Quickstart

To generate results from BBT model you need to first fit posterior MCMC samples. BBT-Test supports unpaired (1 metric readout per algorithm per dataset) and paired (multiple metric readouts per algorithm per dataset) data.

### Unpaired posterior fitting

Start with single dataframe in shape (n_datasets, n_algorithms), optionally this dataframe can contain a dataset column:

```python
import pandas as pd

df = pd.DataFrame({
    "dataset": ["ds1", "ds2", "ds3"],
    "alg1": [0.8, 0.75, 0.9],
    "alg2": [0.7, 0.8, 0.85],
    "alg3": [0.9, 0.95, 0.88],
})
```

To generate data for BBT model, fit the `PyBBT` model with the dataframe

```python
from bbttest import PyBBT

model = PyBBT(
    local_rope_value=0.01, # Here you can define what is a tie in case of unpaired data, default is None
    # In this case the model will assume that if difference is below 0.01 there's a tie.
).fit(
    df,
    dataset_col="dataset", # If dataset column is present, specify it here
)
```

### Paired posterior fitting

PyBBT model support two variants of input data for paired case, either a single dataframe with multiple rows per algorithm per dataset, or a pair of dataframes, one defining mean performance per algorithm, and the second with standard deviations.

```python
import pandas as pd
from bbttest import PyBBT

df = pd.DataFrame({
    "dataset": ["ds1", "ds1", "ds1", "ds2", "ds2", "ds2", "ds3", "ds3", "ds3"],
    "alg1": [0.8, 0.82, 0.79, 0.75, 0.77, 0.74, 0.9, 0.91, 0.89],
    "alg2": [0.7, 0.72, 0.69, 0.8, 0.78, 0.81, 0.85, 0.86, 0.84],
    "alg3": [0.9, 0.92, 0.91, 0.95, 0.94, 0.96, 0.88, 0.87, 0.89],
})

model = PyBBT(
    local_rope_value=0.1, # In this case ties will be counted if the difference is below square root mean of
    # standard deviations multiplied by local_rope_value
).fit(
    df,
    dataset_col="dataset",
)
```

### Generating BBT posterior statistics and interpretations

Once you obtained a fitted PyBBT model, you can generate statistic dataframe containing information about every hypothesis (i.e. every pair of algorithms). The table includes general statistics in form of mean and delta values, as well as probabilities of one algorithm being better than the other, or being tied. Additionally, by default the table contains weak and strong interpretations of the results based on ROPE values.

```python

stats_df = model.get_stats_dataframe(
    rope_value=(0.45, 0.55), # Defines ROPE of hypothesis for interpretations
    control_model="alg1", # If provided, only hypotheses comparing to control_model will be included
    selected_models=["alg2", "alg3"], # If provided, only hypotheses comparing selected_models will be included
)

print(stats_df)
```

Additionally, you can generate multiple hypothesis interpretations regarding control model for different ROPE values:

```python
from bbttest import multiple_ropes_control_table

stats_df = multiple_ropes_control_table(
    model,
    ropes=[(0.4, 0.6), (0.45, 0.55), (0.48, 0.52)],
    control_model="alg1",
    interpretation_type="weak",
)

print(stats_df)
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
