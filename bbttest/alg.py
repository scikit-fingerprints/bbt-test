import logging as log
from collections.abc import Generator

import arviz as az
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .const import TieSolver

ALG1_COL = 2
ALG2_COL = 3
TIE_COL = 4

logger = log.getLogger(__name__)


def _gen_pairs(no_algs: int) -> Generator[tuple[int, int, int], None, None]:
    k = 0
    for i in range(no_algs):
        for j in range(i + 1, no_algs):
            yield (i, j, k)
            k += 1


def _construct_no_paired(
    data_mean: pd.DataFrame,
    alg_names: list[str],
    lrope_value: float,
    data_sd: pd.DataFrame | None,
    unpaired_rope_value: float | None,
) -> np.ndarray:
    logger.debug("Using unpaired BBT test.")
    if data_sd is not None:
        logger.debug("Using paired ROPE values based on provided standard deviations.")
    elif unpaired_rope_value is not None:
        logger.debug("Using unpaired ROPE value.")
    else:
        logger.debug("No ROPE value provided, no ties will be recorded.")
    no_algs = len(alg_names)
    no_pairs = no_algs * (no_algs - 1) // 2
    out_array = -1 * np.ones(
        (no_pairs, 5),  # alg_1, alg_2, 1_wins, 2_wins, ties
        dtype=np.float32,
    )
    for i, j, k in tqdm(
        _gen_pairs(no_algs),
        total=no_pairs,
        desc="Constructing win table",
        leave=False,
    ):
        i_name = alg_names[i]
        j_name = alg_names[j]
        deltas = data_mean[i_name] - data_mean[j_name]
        if data_sd is not None:
            th = lrope_value * np.sqrt(
                np.power(data_sd[i_name], 2) + np.power(data_sd[j_name], 2)
            )
        elif unpaired_rope_value is not None:
            th = unpaired_rope_value
        else:
            th = 0.0
        w1 = np.sum(deltas > th)
        w2 = np.sum(deltas < -th)
        ties = deltas.shape[0] - w1 - w2
        out_array[k, :] = i, j, w1, w2, ties
    return out_array


def _construct_lrope(
    data: pd.DataFrame,
    alg_names: list[str],
    dataset_col: str | int,
    lrope_value: float,
) -> np.ndarray:
    logger.debug("Using paired BBT test.")
    no_algs = data.shape[1]
    no_pairs = no_algs * (no_algs - 1) // 2
    out_array = -1 * np.ones(
        (no_pairs, 5),  # alg_1, alg_2, 1_wins, 2_wins, ties
        dtype=np.int32,
    )
    for dataset_name in data[dataset_col].unique():
        data_subset = data[data[dataset_col] == dataset_name]
        for i, j, k in tqdm(
            _gen_pairs(no_algs),
            total=no_pairs,
            desc=f"Constructing local ROPE win table for dataset {dataset_name}",
        ):
            i_name = alg_names[i]
            j_name = alg_names[j]
            deltas = data_subset[i_name] - data_subset[j_name]
            mean = np.mean(deltas)
            sd = np.std(deltas)
            win1 = int(mean > lrope_value * sd)
            win2 = int(mean < -lrope_value * sd)
            ties = 1 - win1 - win2
            out_array[k, :] = (
                i,
                j,
                out_array[k, ALG1_COL] + win1,
                out_array[k, ALG2_COL] + win2,
                out_array[k, TIE_COL] + ties,
            )
    return out_array


def _solve_ties(table: np.ndarray, tie_solver: TieSolver) -> np.ndarray:
    if tie_solver == TieSolver.DAVIDSON:
        return table
    if tie_solver == TieSolver.SPREAD:
        tie_val = np.ceil(table[:, TIE_COL] / 2).astype(int)
    elif tie_solver == TieSolver.ADD:
        tie_val = table[:, TIE_COL].astype(int)
    else:
        tie_val = 0
    table[:, ALG1_COL] += tie_val
    table[:, ALG2_COL] += tie_val
    return table


def _construct_win_table(
    data: pd.DataFrame,
    data_sd: pd.DataFrame | None,
    dataset_col: str | int | None,
    local_rope_value: float | None,
    tie_solver: TieSolver,
) -> tuple[np.ndarray, list[str]]:
    # Extract algorithm names
    algorithms_names = set(data.columns.tolist())
    if isinstance(dataset_col, int):
        dataset_col = data.columns[dataset_col]
    if dataset_col is not None:
        algorithms_names.discard(dataset_col)
    algorithms_names = list(algorithms_names)

    if dataset_col is None or data.shape[0] == data[dataset_col].nunique():
        table = _construct_no_paired(
            data_mean=data,
            lrope_value=local_rope_value or 0.0,
            data_sd=data_sd,
            alg_names=algorithms_names,
            unpaired_rope_value=local_rope_value,
        )
    else:
        table = _construct_lrope(
            data=data,
            lrope_value=local_rope_value or 0.0,
            dataset_col=dataset_col,
            alg_names=algorithms_names,
        )
    table = _solve_ties(
        table=table,
        tie_solver=tie_solver,
    )
    return table, algorithms_names


def _get_pwin(
    bbt_result: az.InferenceData,
    alg_names: list[str] | None = None,
    control: str | None = None,
    selected: list[str] | None = None,
):
    def _pairwise_prob(strength_i, strength_j):
        return strength_i / (strength_i + strength_j)

    # Extract beta samples from InferenceData
    # PyMC stores samples in idata.posterior
    beta_samples = bbt_result.posterior["beta"].to_numpy()
    # Flatten chain and draw dimensions: (chains, draws, n_algs) -> (samples, n_algs)
    beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])

    n_algs = beta_samples.shape[1]
    # Order algorithms by mean strength (descending)
    mean_beta = np.mean(beta_samples, axis=0)
    order = np.argsort(-mean_beta)
    ordered_names = np.array(alg_names)[order]

    # Exponentiate to get strengths (exp(beta))
    strengths = np.exp(beta_samples[:, order])

    # Filter by selected algorithms if specified
    if selected is not None:
        selected_set = set(selected)
        indices = [i for i, name in enumerate(ordered_names) if name in selected_set]
        ordered_names = ordered_names[indices]
        strengths = strengths[:, indices]
        n_algs = len(indices)

    comparison_names = []

    # Generate comparisons
    if control is None or control not in ordered_names:
        # All pairwise comparisons
        n_comparisons = n_algs * (n_algs - 1) // 2
        samples = np.empty((strengths.shape[0], n_comparisons))

        for i, j, k in _gen_pairs(n_algs):
            samples[:, k] = _pairwise_prob(strengths[:, i], strengths[:, j])
            comparison_names.append(f"{ordered_names[i]} > {ordered_names[j]}")
    else:
        # Comparisons with control algorithm
        control_idx = np.where(ordered_names == control)[0][0]
        n_comparisons = n_algs - 1
        samples = np.empty((strengths.shape[0], n_comparisons))

        k = 0
        # Comparisons where other algorithm is better than control
        for i in range(control_idx):
            samples[:, k] = _pairwise_prob(strengths[:, i], strengths[:, control_idx])
            comparison_names.append(
                f"{ordered_names[i]} > {ordered_names[control_idx]}"
            )
            k += 1

        # Comparisons where control is better than other algorithm
        for i in range(control_idx + 1, n_algs):
            samples[:, k] = _pairwise_prob(strengths[:, control_idx], strengths[:, i])
            comparison_names.append(
                f"{ordered_names[control_idx]} > {ordered_names[i]}"
            )
            k += 1

    return samples, comparison_names


def _hdi(samples: np.ndarray, hdi_prob: float = 0.89) -> np.ndarray:
    def newhdi(arr):
        x = np.sort(arr)
        n = len(x)
        exclude = int(n - np.floor(n * hdi_prob) - 1)
        low_poss = x[:exclude]
        upp_poss = x[(n - exclude) :]
        best = np.argmin(upp_poss - low_poss)
        return low_poss[best], upp_poss[best]

    return np.apply_along_axis(newhdi, 0, samples)
