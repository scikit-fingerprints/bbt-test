from collections.abc import Sequence

import numpy as np
import pandas as pd

from .alg import _construct_win_table, _get_pwin, _hdi
from .const import DEFAULT_PROPERTIES, HyperPrior, ReportedProperty, TieSolver
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

    tie_solver: TieSolver, default TieSolver.SPREAD
        The strategy to handle ties when sampling the BBT model.
            - ADD - Adds 1 win to both players for each tie.
            - SPREAD - Adds 0.5 win to both players for each tie.
            - FORGET - Ignores the ties.
            - DAVIDSON - Uses Davidson's method to handle ties in the BBT model. See [1]_.

    hyper_prior: HyperPrior, default HyperPrior.LOG_NORMAL
        The hyper prior distribution to be used for the BBT MCMC sampling.

    scale: float, default 1.0
        The scale parameter for the hyper prior distribution. Ignored if the HyperPrior is LOG_NORMAL.

    Attributes
    ----------
    fitted: bool
        Whether the model has been fitted.

    Examlples
    ---------
    >>> import pandas as pd
    >>> from bbttest import PyBBT, TieSolver
    >>> data = pd.DataFrame({
    ...     'dataset': ['ds1', 'ds2', 'ds3'],
    ...     'model_a': [0.8, 0.75, 0.9],
    ...     'model_b': [0.7, 0.8, 0.85],
    ...     'model_c': [0.6, 0.65, 0.7]
    ... })
    >>> model = PyBBT(local_rope_value=0.01, tie_solver=TieSolver.SPREAD)
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

    def __init__(
        self,
        local_rope_value: float | None = None,
        tie_solver: TieSolver = TieSolver.SPREAD,
        hyper_prior: HyperPrior = HyperPrior.LOG_NORMAL,
        scale: float = 1.0,
    ):
        self._local_rope_value = local_rope_value
        self._tie_solver = tie_solver
        self._use_davidson = tie_solver == TieSolver.DAVIDSON
        self._hyper_prior = hyper_prior
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

        Args:
            data (pd.DataFrame): Dataframe containing scores for the models on the datasets.
                If data_sd is provided, this dataframe should contain mean scores per model per dataset.
                If multiple scores per model per dataset are provided, data_sd is ignored, and dataset_col is required.
            data_sd (pd.DataFrame | None, optional): Dataframe containing standard deviations of the scores for the models on the datasets.
            dataset_col (str, optional): Column name for the dataset identifier. Defaults to "dataset".

        Returns
        -------
            self: fitted PyBBT instance
        """
        self._win_table, self._algorithms = _construct_win_table(
            data=data,
            data_sd=data_sd,
            dataset_col=dataset_col,
            local_rope_value=self._local_rope_value,
            tie_solver=self._tie_solver,
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
        selected_models: list[str] | None = None,
        columns: Sequence[ReportedProperty | str] = DEFAULT_PROPERTIES,
        hdi_proba: float = 0.89,
        round_ndigits: int | None = 2,
    ) -> pd.DataFrame:
        """Compute posterior table containing sampling results for the fitted BBT model.

        Args:
            rope_value (tuple[float, float], optional): Region of Practical Equivalence (ROPE). Defaults to (0.45, 0.55).
            control_model (str | None, optional): Control model for comparison. Defaults to None.
            selected_models (list[str] | None, optional): Subset of models to include in the posterior table. Defaults to None.
            columns (list[ReportedProperty], optional): Columns to include in the posterior table. Defaults to DEFAULT_PROPERTIES.
            hdi_proba (float, optional): Highest Density Interval probability. Defaults to 0.89.
            round_ndigits (int | None, optional): Number of digits to round the results to. Defaults to 2.

        Returns
        -------
            pd.DataFrame: Posterior table containing sampling results for the fitted BBT model.
        """
        self._check_if_fitted()

        samples, names = _get_pwin(
            bbt_result=self._fit_posterior,
            alg_names=self._algorithms,
            control=control_model,
            selected=selected_models,
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
            return out_table.round(round_ndigits)
        for col in columns:
            if col not in out_table.columns:
                raise ValueError(
                    f"Column {col} is not available in the posterior table."
                )
        return out_table[["pair", *columns]]
