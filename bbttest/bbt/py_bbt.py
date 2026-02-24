from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ._types import HyperPriorType, ReportedPropertyColumnType, TieSolverType
from ._utils import _validate_params
from .alg import _construct_win_table, _get_pwin, _hdi
from .model import _mcmcbbt_pymc


class PyBBT:
    """
    BBT model estimator used for multi-dataset multi-model comparison [1]_.
    The model estimates posterior probabilities for each pair of the model.

    Parameters
    ----------
    local_rope_value: float | None, default 0.1
        The value of the local ROPE to be used when constructing win/tie/loss pairs. If the models is unpaired (i.e., only one score per model per dataset),
        this value is used to determine the threshold for ties in the followin manner:

            - score_a - score_b > local_rope_value => model A wins
            - score_b - score_a > local_rope_value => model B wins
            - otherwise => tie

        In case of paired BBT (i.e. multiple readings per model per dataset or data_sd provided), the ties are determined based on the following conditions:

            - sigma = sqrt(sd_a^2 + sd_b^2)
            - score_a - score_b > local_rope_value * sigma => model A wins
            - score_b - score_a > local_rope_value * sigma => model B wins
            - otherwise => tie

        If None, no ties are recorded.

    tie_solver: str, defaults to `spread`
        The strategy to handle ties when sampling the BBT model.

            - `add` - Adds 1 win to both players for each tie.
            - `spread` - Adds 0.5 win to both players for each tie.
            - `forget` - Ignores the ties.
            - `davidson` - Uses Davidson's method to handle ties in the BBT model. See [1]_.

    hyper_prior: str, default `log_normal`
        The hyper prior distribution to be used for the BBT MCMC sampling.

    scale: float, default 1.0
        The scale parameter for the hyper prior distribution.

    maximize: bool, default True
        Whether higher scores indicate better performance (e.g. accuracy/f1). If using a metric where the goal is to
        minimize the score (e.g. RMSE) set this to False.

    Attributes
    ----------
    fitted: bool
        Whether the model has been fitted.

    Examples
    --------
    >>> import pandas as pd
    >>> from bbttest import PyBBT
    >>> data = pd.DataFrame({
    ...     'dataset': ['ds1', 'ds2', 'ds3'],
    ...     'model_a': [0.8, 0.75, 0.9],
    ...     'model_b': [0.7, 0.8, 0.85],
    ...     'model_c': [0.6, 0.65, 0.7]
    ... })
    >>> model = PyBBT(local_rope_value=0.01, tie_solver="spread")
    >>> model.fit(data, dataset_col='dataset')
    >>> model.posterior_table(rope_value=(0.45, 0.55))

    References
    ----------
    .. [1] `Jacques Wainer
        "A Bayesian Bradley-Terry model to compare multiple ML algorithms on multiple data sets"
        Journal of Machine Learning Research 24 (2023): 1-34
        <http://jmlr.org/papers/v24/22-0907.html>`_
    """

    _WEAK_INTERPRETATION_THRESHOLD = 0.95
    _STRONG_INTERPRETATION_BETTER_THRESHOLD = 0.70
    _STRONG_INTERPRETATION_EQUAL_THRESHOLD = 0.55

    @_validate_params
    def __init__(
        self,
        local_rope_value: float | None = None,
        tie_solver: TieSolverType = "spread",
        hyper_prior: HyperPriorType = "log_normal",
        maximize: bool = True,
        scale: float = 1.0,
    ):
        self._local_rope_value = local_rope_value
        self._tie_solver = tie_solver
        self._use_davidson = self._tie_solver == "davidson"
        self._hyper_prior = hyper_prior
        self._maximize = maximize
        self._scale = scale
        self._fitted = False

    def _check_if_fitted(self):
        if not self._fitted:
            raise RuntimeError("The model must be fitted before accessing this method.")

    @property
    def fitted(self):
        """Whether the model has been fitted."""
        return self._fitted

    def fit(
        self,
        data: pd.DataFrame,
        data_sd: pd.DataFrame | None = None,
        dataset_col: str = "dataset",
        **pymc_kwargs,
    ):
        """
        Fits the BBT for a given result dataframes.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing scores for the models on the datasets.
            If data_sd is provided, this dataframe should contain mean scores per model per dataset.
            If multiple scores per model per dataset are provided, data_sd is ignored, and dataset_col is required.
        data_sd : pd.DataFrame | None, optional
            Dataframe containing standard deviations of the scores for the models on the datasets.
        dataset_col : str, optional
            Column name for the dataset identifier. Defaults to "dataset".

        Returns
        -------
        self : PyBBT
            Fitted PyBBT instance
        """
        self._win_table, self._algorithms = _construct_win_table(
            data=data,
            data_sd=data_sd,
            dataset_col=dataset_col,
            local_rope_value=self._local_rope_value,
            tie_solver=self._tie_solver,
            maximize=self._maximize,
        )

        self._fit_posterior = _mcmcbbt_pymc(
            table=self._win_table,
            use_davidson=self._use_davidson,
            hyper_prior=self._hyper_prior,
            scale=self._scale,
            **pymc_kwargs,
        )

        self._fitted = True

        return self

    def posterior_table(
        self,
        rope_value: tuple[float, float] = (0.45, 0.55),
        control_model: str | None = None,
        selected_models: Iterable[str] | None = None,
        columns: Iterable[ReportedPropertyColumnType] = (
            "mean",
            "delta",
            "above_50",
            "in_rope",
            "weak_interpretation",
        ),
        hdi_proba: float = 0.89,
        round_ndigits: int | None = 2,
    ) -> pd.DataFrame:
        """Compute posterior table containing sampling results for the fitted BBT model.

        Parameters
        ----------
        rope_value : tuple[float, float], optional
            Region of Practical Equivalence (ROPE). Defaults to (0.45, 0.55).
        control_model : str | None, optional
            Control model for comparison. Defaults to None.
        selected_models : Iterable[str] | None, optional
            Subset of models to include in the posterior table. Defaults to None.
        columns : Iterable[ReportedPropertyColumnType], optional
            Columns to include in the posterior table. Defaults to minimum set for weak interpretation.
        hdi_proba : float, optional
            Highest Density Interval probability. Defaults to 0.89.
        round_ndigits : int | None, optional
            Number of digits to round the results to. Defaults to 2.

        Returns
        -------
        pd.DataFrame
            Posterior table containing sampling results for the fitted BBT model.
        """
        self._check_if_fitted()

        samples, names = _get_pwin(
            bbt_result=self._fit_posterior,
            alg_names=self._algorithms,
            control=control_model,
            selected=list(selected_models) if selected_models is not None else None,
        )
        out_table = pd.DataFrame({"pair": names})
        out_table["left_model"] = out_table["pair"].str.split(">").str[0].str.strip()
        out_table["right_model"] = out_table["pair"].str.split(">").str[1].str.strip()
        out_table["median"] = np.median(samples, axis=0)
        out_table["mean"] = np.mean(samples, axis=0)
        out_table["above_50"] = np.mean(samples > 0.5, axis=0)
        out_table["in_rope"] = np.mean(
            (samples >= rope_value[0]) & (samples <= rope_value[1]), axis=0
        )
        out_table["weak_interpretation_raw"] = np.where(
            out_table["in_rope"] >= self._WEAK_INTERPRETATION_THRESHOLD,
            "=",
            np.where(
                out_table["above_50"] >= self._WEAK_INTERPRETATION_THRESHOLD,
                ">",
                "?",
            ),
        )
        out_table["weak_interpretation"] = np.where(
            out_table["weak_interpretation_raw"] == ">",
            out_table["left_model"] + " better",
            np.where(
                out_table["weak_interpretation_raw"] == "=",
                "Equivalent",
                "Unknown",
            ),
        )

        out_table["strong_interpretation_raw"] = np.where(
            out_table["mean"] > self._STRONG_INTERPRETATION_BETTER_THRESHOLD,
            out_table["left_model"] + ">",
            np.where(
                out_table["mean"] <= self._STRONG_INTERPRETATION_EQUAL_THRESHOLD,
                "=",
                "?",
            ),
        )
        out_table["strong_interpretation"] = np.where(
            out_table["strong_interpretation_raw"].str.endswith(">"),
            out_table["left_model"] + " better",
            np.where(
                out_table["strong_interpretation_raw"] == "=",
                "Equivalent",
                "Unknown",
            ),
        )

        hdi_values = _hdi(samples, hdi_proba)
        out_table["hdi_low"] = hdi_values[0]
        out_table["hdi_high"] = hdi_values[1]
        out_table["delta"] = out_table["hdi_high"] - out_table["hdi_low"]

        if round_ndigits is not None:
            return out_table.round(round_ndigits)[["pair", *columns]]
        for col in columns:
            if col not in out_table.columns:
                raise ValueError(
                    f"Column {col} is not available in the posterior table."
                )
        return out_table[["pair", *columns]]

    def rope_comparison_control_table(
        self,
        rope_values: Sequence[tuple[float, float]],
        control_model: str,
        selected_models: Sequence[str] | None = None,
        interpretation: Literal["weak", "strong"] = "weak",
        return_as_array: bool = False,
        join_char: str = ", ",
    ) -> pd.DataFrame:
        """
        Construct a table comparing models against predefined control models across multiple ROPEs.
        The output table contains N rows (one per ROPE) and 5 columns
        (rope value, better models, equivalent models, worse models, unknown models).

        Parameters
        ----------
        rope_values : Sequence[tuple[float, float]]
            List of ROPE tuples to evaluate.
        control_model : str
            Control model for comparison.
        selected_models : Sequence[str] | None, optional
            Subset of models to include. Defaults to None.
        interpretation : {"weak", "strong"}, optional
            Type of interpretation to use, see [1]_. Defaults to "weak".
        return_as_array : bool, optional
            Whether the individual cells should contain model names as list or joined into single string.
            Defaults to False.
        join_char : str, optional
            Character(s) used to join multiple model names in a single cell. Defaults to ", ".

        Returns
        -------
        pd.DataFrame
            Table comparing models against control models across multiple ROPEs.
        """
        self._check_if_fitted()
        records = []
        for rope in rope_values:
            posterior_df = self.posterior_table(
                rope_value=rope,
                control_model=control_model,
                selected_models=selected_models,
                columns=(
                    "left_model",
                    "weak_interpretation",
                    "strong_interpretation",
                ),
            )
            better_models: list[str] = []
            equivalent_models: list[str] = []
            worse_models: list[str] = []
            unknown_models: list[str] = []
            for _, row in posterior_df.iterrows():
                interpretation_col = (
                    "weak_interpretation"
                    if interpretation == "weak"
                    else "strong_interpretation"
                )
                if row[interpretation_col] == f"{row['left_model']} better":
                    better_models.append(row["left_model"])
                elif row[interpretation_col] == "Equivalent":
                    equivalent_models.append(row["left_model"])
                elif row[interpretation_col] == "Unknown":
                    unknown_models.append(row["left_model"])
                else:
                    worse_models.append(row["left_model"])
            if not return_as_array:
                better_models_str = join_char.join(better_models)
                equivalent_models_str = join_char.join(equivalent_models)
                worse_models_str = join_char.join(worse_models)
                unknown_models_str = join_char.join(unknown_models)
                records.append(
                    {
                        "rope_value": rope,
                        "better_models": better_models_str,
                        "equivalent_models": equivalent_models_str,
                        "worse_models": worse_models_str,
                        "unknown_models": unknown_models_str,
                    }
                )
            else:
                records.append(
                    {
                        "rope_value": rope,
                        "better_models": better_models,
                        "equivalent_models": equivalent_models,
                        "worse_models": worse_models,
                        "unknown_models": unknown_models,
                    }
                )
        result_df = pd.DataFrame.from_records(records)
        return result_df
