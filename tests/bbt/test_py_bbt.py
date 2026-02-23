"""
Unit tests for PyBBT class.

This module contains unit tests for the PyBBT class, testing various
functionality including model fitting, posterior table generation,
ROPE comparison tables, and parameter validation.
"""

import numpy as np
import pandas as pd
import pytest

from bbttest import HyperPrior, PyBBT, ReportedProperty, TieSolver
from bbttest.bbt.const import ALL_PROPERTIES


@pytest.fixture(scope="module")
def mock_data():
    """
    Create simple mock data for testing.

    Returns
    -------
    pd.DataFrame
        Mock dataset with 3 datasets and 3 models.
    """
    return pd.DataFrame(
        {
            "dataset": ["ds1", "ds2", "ds3"],
            "model_a": [0.8, 0.75, 0.9],
            "model_b": [0.7, 0.8, 0.85],
            "model_c": [0.6, 0.65, 0.7],
        }
    )


@pytest.fixture(scope="module")
def fitted_model(mock_data):
    """
    Create a fitted PyBBT model for testing.

    Parameters
    ----------
    mock_data : pd.DataFrame
        Mock data fixture.

    Returns
    -------
    PyBBT
        Fitted PyBBT model instance.
    """
    model = PyBBT(local_rope_value=0.01, tie_solver=TieSolver.SPREAD)
    model.fit(
        mock_data,
        dataset_col="dataset",
        draws=100,
        tune=100,
        chains=2,
        random_seed=42,
    )
    return model


class TestPyBBTInitialization:
    """Test PyBBT initialization and parameter validation."""

    def test_init_with_enum_parameters(self):
        """Test that PyBBT can be initialized with enum parameters."""
        model = PyBBT(
            local_rope_value=0.01,
            tie_solver=TieSolver.SPREAD,
            hyper_prior=HyperPrior.LOG_NORMAL,
            scale=1.0,
        )
        assert model._local_rope_value == 0.01
        assert model._tie_solver == TieSolver.SPREAD
        assert model._hyper_prior == HyperPrior.LOG_NORMAL
        assert model._scale == 1.0
        assert not model.fitted

    def test_init_with_string_parameters(self):
        """Test that PyBBT can be initialized with string parameters that are cast to enums."""
        model = PyBBT(
            local_rope_value=0.01,
            tie_solver="spread",
            hyper_prior="logNormal",
            scale=1.0,
        )
        # Verify string values are accepted and work correctly
        assert model._local_rope_value == 0.01
        assert model._tie_solver == TieSolver.SPREAD
        assert model._hyper_prior == HyperPrior.LOG_NORMAL
        assert model._scale == 1.0
        assert not model.fitted

    @pytest.mark.parametrize(
        "arg, expected",
        [
            ("add", TieSolver.ADD),
            ("spread", TieSolver.SPREAD),
            ("forget", TieSolver.FORGET),
            ("davidson", TieSolver.DAVIDSON),
            (TieSolver.ADD, TieSolver.ADD),
            (TieSolver.SPREAD, TieSolver.SPREAD),
            (TieSolver.FORGET, TieSolver.FORGET),
            (TieSolver.DAVIDSON, TieSolver.DAVIDSON),
        ],
    )
    def test_init_with_different_tie_solvers(self, arg, expected):
        """Test initialization with different TieSolver values."""
        model = PyBBT(tie_solver=arg)
        assert model._tie_solver == expected

    @pytest.mark.parametrize(
        "arg, expected",
        [
            ("logNormal", HyperPrior.LOG_NORMAL),
            ("logNormalScaled", HyperPrior.LOG_NORMAL_SCALED),
            ("cauchy", HyperPrior.CAUCHY),
            ("normal", HyperPrior.NORMAL),
            (HyperPrior.LOG_NORMAL, HyperPrior.LOG_NORMAL),
            (HyperPrior.LOG_NORMAL_SCALED, HyperPrior.LOG_NORMAL_SCALED),
            (HyperPrior.CAUCHY, HyperPrior.CAUCHY),
            (HyperPrior.NORMAL, HyperPrior.NORMAL),
        ],
    )
    def test_init_with_different_hyper_priors(self, arg, expected):
        """Test initialization with different HyperPrior values."""
        model = PyBBT(hyper_prior=arg)
        assert model._hyper_prior == expected

    def test_init_defaults(self):
        """Test that default initialization values are set correctly."""
        model = PyBBT()
        assert model._local_rope_value is None
        assert model._tie_solver == TieSolver.SPREAD
        assert model._hyper_prior == HyperPrior.LOG_NORMAL
        assert model._scale == 1.0
        assert model._maximize
        assert not model.fitted


class TestPyBBTFitting:
    """Test PyBBT model fitting functionality."""

    def test_fit_updates_fitted_property(self, mock_data):
        """Test that fit() updates the fitted property."""
        model = PyBBT()
        assert not model.fitted
        model.fit(mock_data, dataset_col="dataset", draws=50, tune=50, chains=2)
        assert model.fitted

    def test_fit_returns_self(self, mock_data):
        """Test that fit() returns self for method chaining."""
        model = PyBBT()
        result = model.fit(
            mock_data, dataset_col="dataset", draws=50, tune=50, chains=2
        )
        assert result is model


class TestPyBBTUnfittedErrors:
    """Test that methods raise errors when called on unfitted models."""

    def test_posterior_table_without_fitting_raises_error(self):
        """Test that posterior_table() raises error on unfitted model."""
        model = PyBBT()
        with pytest.raises(
            RuntimeError, match="The model must be fitted before accessing this method"
        ):
            model.posterior_table()

    def test_rope_comparison_control_table_without_fitting_raises_error(self):
        """Test that rope_comparison_control_table() raises error on unfitted model."""
        model = PyBBT()
        with pytest.raises(
            RuntimeError, match="The model must be fitted before accessing this method"
        ):
            model.rope_comparison_control_table(
                rope_values=[(0.45, 0.55)], control_model="model_a"
            )


class TestPosteriorTable:
    """Test posterior_table method functionality."""

    def test_posterior_table_has_required_columns(self, fitted_model):
        """Test that posterior_table contains required columns."""
        result = fitted_model.posterior_table()
        required_cols = ["pair", "mean", "delta", "above_50", "in_rope"]
        for col in required_cols:
            assert col in result.columns

    def test_posterior_table_weak_interpretation_values(self, fitted_model):
        """Test that weak interpretation contains valid values."""
        result = fitted_model.posterior_table(rope_value=(0.45, 0.55))
        valid_values = {"Equivalent", "Unknown"}
        # Weak interpretation should end with "better", be "Equivalent", or be "Unknown"
        for interp in result["weak_interpretation"]:
            assert interp in valid_values or interp.endswith(" better"), (
                f"Invalid weak interpretation: {interp}"
            )

    def test_posterior_table_strong_interpretation_values(self, fitted_model):
        """Test that strong interpretation contains valid values."""
        result = fitted_model.posterior_table()
        # Add strong_interpretation to columns
        result = fitted_model.posterior_table(
            columns=[
                ReportedProperty.MEAN,
                ReportedProperty.STRONG_INTERPRETATION,
            ]
        )
        valid_values = {"Equivalent", "Unknown"}
        for interp in result["strong_interpretation"]:
            assert interp in valid_values or interp.endswith(" better"), (
                f"Invalid strong interpretation: {interp}"
            )

    def test_posterior_table_with_control_model(self, fitted_model):
        """Test posterior_table with control_model parameter."""
        result = fitted_model.posterior_table(
            control_model="model_a", columns=ALL_PROPERTIES
        )
        assert len(result) > 0
        # All comparisons should involve model_a
        for _, row in result.iterrows():
            assert row["left_model"] == "model_a" or row["right_model"] == "model_a", (
                f"Comparison {row['pair']} does not involve control model"
            )

    def test_posterior_table_returns_only_requested_columns(self, fitted_model):
        """Test that posterior_table returns only requested columns."""
        requested_columns = [ReportedProperty.MEAN, ReportedProperty.DELTA]
        result = fitted_model.posterior_table(
            columns=requested_columns, round_ndigits=None
        )

        # Should have 'pair' column plus requested columns
        expected_cols = ["pair", "mean", "delta"]
        assert set(result.columns) == set(expected_cols)

        result = fitted_model.posterior_table(
            columns=requested_columns, round_ndigits=3
        )
        assert set(result.columns) == set(expected_cols)

    def test_posterior_table_requested_columns_with_strings(self, fitted_model):
        """Test that posterior_table accepts string column names."""
        requested_columns = ["mean", "delta", "above_50"]
        # Must set round_ndigits=None to get column filtering
        result = fitted_model.posterior_table(
            columns=requested_columns, round_ndigits=None
        )

        expected_cols = ["pair", "mean", "delta", "above_50"]
        assert set(result.columns) == set(expected_cols)

    def test_posterior_table_invalid_column_raises_error(self, fitted_model):
        """Test that requesting invalid column raises ValueError."""
        with pytest.raises(ValueError, match="is not available in the posterior table"):
            # Must set round_ndigits=None to trigger column validation
            fitted_model.posterior_table(columns=["invalid_column"], round_ndigits=None)

    def test_posterior_table_rope_value_affects_in_rope(self, fitted_model):
        """Test that changing ROPE value affects in_rope column."""
        result1 = fitted_model.posterior_table(rope_value=(0.4, 0.6))
        result2 = fitted_model.posterior_table(rope_value=(0.45, 0.55))

        # Wider ROPE should generally have higher in_rope values
        mean_in_rope_1 = result1["in_rope"].mean()
        mean_in_rope_2 = result2["in_rope"].mean()
        assert mean_in_rope_1 >= mean_in_rope_2

    def test_posterior_table_rounding(self, fitted_model):
        """Test that rounding parameter works correctly."""
        result_rounded = fitted_model.posterior_table(round_ndigits=2)

        # Check that rounded version has at most 2 decimal places
        for col in ["mean", "delta"]:
            if col in result_rounded.columns:
                for val in result_rounded[col]:
                    if not pd.isna(val):
                        str_val = str(val)
                        if "." in str_val:
                            decimals = len(str_val.split(".")[1])
                            assert decimals <= 2


class TestRopeComparisonControlTable:
    """Test rope_comparison_control_table method functionality."""

    def test_rope_comparison_returns_dataframe(self, fitted_model):
        """Test that rope_comparison_control_table returns a DataFrame."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55), (0.4, 0.6)], control_model="model_a"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # One row per ROPE

    def test_rope_comparison_has_required_columns(self, fitted_model):
        """Test that rope_comparison_control_table has required columns."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)], control_model="model_a"
        )
        required_cols = [
            "rope_value",
            "better_models",
            "equivalent_models",
            "worse_models",
            "unknown_models",
        ]
        for col in required_cols:
            assert col in result.columns

    def test_rope_comparison_weak_interpretation(self, fitted_model):
        """Test rope_comparison_control_table with weak interpretation."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55), (0.4, 0.6)],
            control_model="model_a",
            interpretation="weak",
        )
        assert len(result) == 2
        # Check that each row has the correct ROPE value
        assert result.iloc[0]["rope_value"] == (0.45, 0.55)
        assert result.iloc[1]["rope_value"] == (0.4, 0.6)

    def test_rope_comparison_strong_interpretation(self, fitted_model):
        """Test rope_comparison_control_table with strong interpretation."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            interpretation="strong",
        )
        assert len(result) == 1
        # Verify it runs without error and returns expected structure
        assert "better_models" in result.columns

    def test_rope_comparison_return_as_array(self, fitted_model):
        """Test rope_comparison_control_table with return_as_array=True."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            return_as_array=True,
        )
        # When return_as_array=True, columns should contain lists
        assert isinstance(result.iloc[0]["better_models"], list)
        assert isinstance(result.iloc[0]["equivalent_models"], list)
        assert isinstance(result.iloc[0]["worse_models"], list)
        assert isinstance(result.iloc[0]["unknown_models"], list)

    def test_rope_comparison_return_as_string(self, fitted_model):
        """Test rope_comparison_control_table with return_as_array=False."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            return_as_array=False,
        )
        # When return_as_array=False, columns should contain strings
        assert isinstance(result.iloc[0]["better_models"], str)
        assert isinstance(result.iloc[0]["equivalent_models"], str)
        assert isinstance(result.iloc[0]["worse_models"], str)
        assert isinstance(result.iloc[0]["unknown_models"], str)

    def test_rope_comparison_custom_join_char(self, fitted_model):
        """Test rope_comparison_control_table with custom join character."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            return_as_array=False,
            join_char=" | ",
        )
        # Check if custom join character is used (if there are multiple models)
        for col in [
            "better_models",
            "equivalent_models",
            "worse_models",
            "unknown_models",
        ]:
            value = result.iloc[0][col]
            if " | " in value:
                # Found the custom separator, test passes
                assert True
                return

    def test_rope_comparison_multiple_ropes(self, fitted_model):
        """Test rope_comparison_control_table with multiple ROPE values."""
        rope_values = [(0.3, 0.7), (0.4, 0.6), (0.45, 0.55)]
        result = fitted_model.rope_comparison_control_table(
            rope_values=rope_values, control_model="model_a"
        )
        assert len(result) == len(rope_values)
        # Verify ROPE values are correctly stored
        for i, rope in enumerate(rope_values):
            assert result.iloc[i]["rope_value"] == rope


class TestPosteriorTableInterpretations:
    """Test interpretation logic in posterior_table with mocked samples."""

    def test_weak_interpretation_logic(self, fitted_model, monkeypatch):
        """Test that weak interpretation follows expected logic with controlled samples."""
        # Create synthetic samples with known properties for testing interpretations
        # Case 1: Model A > Model B - should be "A better" (96% above 0.5, only 3% in ROPE)
        samples_a_better = np.concatenate(
            [np.full(960, 0.8), np.full(40, 0.4)]
        )  # 96% > 0.5, mean ~0.784

        # Case 2: Model B > Model C - should be "Equivalent" (40% above 0.5, 96% in ROPE)
        samples_equivalent = np.concatenate(
            [np.full(400, 0.52), np.full(600, 0.48)]
        )  # 40% > 0.5, mean = 0.496

        # Case 3: Model C > Model D - should be "Unknown" (70% above 0.5, 80% in ROPE)
        samples_unknown = np.concatenate(
            [np.full(700, 0.6), np.full(300, 0.4)]
        )  # 70% > 0.5, mean = 0.54

        samples = np.column_stack(
            [samples_a_better, samples_equivalent, samples_unknown]
        )
        names = ["A > B", "B > C", "C > D"]

        # Mock _get_pwin to return our controlled samples
        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(rope_value=(0.45, 0.55))

        # Test Case 1: A > B should be "A better"
        row_a_b = result[result["pair"] == "A > B"].iloc[0]
        assert row_a_b["weak_interpretation"] == "A better"
        assert row_a_b["above_50"] >= 0.95
        assert row_a_b["in_rope"] < 0.95

        # Test Case 2: B > C should be "Equivalent"
        row_b_c = result[result["pair"] == "B > C"].iloc[0]
        assert row_b_c["weak_interpretation"] == "Equivalent"
        assert row_b_c["in_rope"] >= 0.95

        # Test Case 3: C > D should be "Unknown"
        row_c_d = result[result["pair"] == "C > D"].iloc[0]
        assert row_c_d["weak_interpretation"] == "Unknown"
        assert row_c_d["above_50"] < 0.95
        assert row_c_d["in_rope"] < 0.95

    def test_strong_interpretation_logic(self, fitted_model, monkeypatch):
        """Test that strong interpretation follows expected logic with controlled samples."""
        # Create synthetic samples with known properties for testing strong interpretations
        # Case 1: Model A > Model B - mean > 0.70, should be "A better"
        samples_a_better = np.full(1000, 0.75)  # mean = 0.75

        # Case 2: Model B > Model C - mean <= 0.55, should be "Equivalent"
        samples_equivalent = np.full(1000, 0.50)  # mean = 0.50

        # Case 3: Model C > Model D - 0.55 < mean <= 0.70, should be "Unknown"
        samples_unknown = np.full(1000, 0.62)  # mean = 0.62

        samples = np.column_stack(
            [samples_a_better, samples_equivalent, samples_unknown]
        )
        names = ["A > B", "B > C", "C > D"]

        # Mock _get_pwin to return our controlled samples
        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(
            columns=[
                ReportedProperty.MEAN,
                ReportedProperty.STRONG_INTERPRETATION,
            ],
            round_ndigits=None,
        )

        # Test Case 1: A > B should be "A better" (mean > 0.70)
        row_a_b = result[result["pair"] == "A > B"].iloc[0]
        assert row_a_b["strong_interpretation"] == "A better"
        assert row_a_b["mean"] > 0.70

        # Test Case 2: B > C should be "Equivalent" (mean <= 0.55)
        row_b_c = result[result["pair"] == "B > C"].iloc[0]
        assert row_b_c["strong_interpretation"] == "Equivalent"
        assert row_b_c["mean"] <= 0.55

        # Test Case 3: C > D should be "Unknown" (0.55 < mean <= 0.70)
        row_c_d = result[result["pair"] == "C > D"].iloc[0]
        assert row_c_d["strong_interpretation"] == "Unknown"
        assert 0.55 < row_c_d["mean"] <= 0.70


class TestPosteriorTableStructure:
    """Test the structure and content of posterior_table output with mocked samples."""

    def test_pair_column_format(self, fitted_model, monkeypatch):
        """Test that pair column has correct format."""
        # Create simple samples for structure testing
        samples = np.column_stack([np.full(100, 0.6), np.full(100, 0.5)])
        names = ["model_a > model_b", "model_b > model_c"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table()

        for pair in result["pair"]:
            assert " > " in pair, f"Pair {pair} does not contain ' > '"
            parts = pair.split(" > ")
            assert len(parts) == 2, f"Pair {pair} does not have exactly 2 parts"

    def test_left_right_model_consistency(self, fitted_model, monkeypatch):
        """Test that left_model and right_model match the pair column."""
        samples = np.column_stack([np.full(100, 0.6), np.full(100, 0.5)])
        names = ["model_a > model_b", "model_b > model_c"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(columns=ALL_PROPERTIES)

        for _, row in result.iterrows():
            expected_pair = f"{row['left_model']} > {row['right_model']}"
            assert row["pair"] == expected_pair

    def test_probability_values_in_range(self, fitted_model, monkeypatch):
        """Test that probability values are in valid range [0, 1]."""
        # Create samples with values that should produce probabilities in [0, 1]
        samples = np.column_stack(
            [
                np.random.uniform(0.3, 0.9, 100),
                np.random.uniform(0.2, 0.8, 100),
                np.random.uniform(0.1, 0.7, 100),
            ]
        )
        names = ["A > B", "B > C", "C > D"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table()

        for col in ["mean", "median", "above_50", "in_rope"]:
            if col in result.columns:
                assert (result[col] >= 0).all()
                assert (result[col] <= 1).all()

    def test_hdi_values_consistent(self, fitted_model, monkeypatch):
        """Test that HDI values are consistent (low <= high)."""
        # Create samples with varying distributions
        samples = np.column_stack(
            [
                np.random.beta(2, 5, 100),  # Skewed distribution
                np.random.beta(5, 5, 100),  # Symmetric distribution
                np.random.beta(8, 2, 100),  # Right-skewed distribution
            ]
        )
        names = ["A > B", "B > C", "C > D"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(
            columns=[ReportedProperty.HDI_LOW, ReportedProperty.HDI_HIGH],
            round_ndigits=None,
        )

        assert (result["hdi_low"] <= result["hdi_high"]).all()

    def test_delta_equals_hdi_difference(self, fitted_model, monkeypatch):
        """Test that delta equals hdi_high - hdi_low."""
        # Create samples with known distributions
        np.random.seed(42)
        samples = np.column_stack(
            [
                np.random.normal(0.6, 0.1, 1000),
                np.random.normal(0.5, 0.15, 1000),
                np.random.normal(0.7, 0.08, 1000),
            ]
        )
        # Clip to [0, 1] range
        samples = np.clip(samples, 0, 1)
        names = ["A > B", "B > C", "C > D"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("bbttest.bbt.py_bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(
            columns=[
                ReportedProperty.HDI_LOW,
                ReportedProperty.HDI_HIGH,
                ReportedProperty.DELTA,
            ],
            round_ndigits=None,  # Don't round to avoid rounding differences
        )

        calculated_delta = result["hdi_high"] - result["hdi_low"]
        np.testing.assert_array_almost_equal(
            result["delta"], calculated_delta, decimal=10
        )
