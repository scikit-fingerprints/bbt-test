from io import StringIO

import numpy as np
import pandas as pd
import pytest

from bbttest.tests.bbt.alg import (
    _construct_win_table,
)

SCORES_1 = pd.DataFrame(
    {
        "alg1": [0.705, 0.7, 0.9],
        "alg2": [0.696, 0.7, 0.8],
        "alg3": [0.7, 0.75, 0.9],
    }
)


class TestConstructTable:
    """Test whether the win/tie/loss table is constructed correctly."""

    @pytest.mark.parametrize(
        "data, local_rope_value, maximize, expected_table",
        [
            (
                SCORES_1,
                None,
                True,
                np.array(
                    [
                        [0, 1, 2, 0, 1],  # alg1 vs alg2
                        [0, 2, 1, 1, 1],  # alg1 vs alg3
                        [1, 2, 0, 3, 0],  # alg2 vs alg3
                    ]
                ),
            ),
            (
                SCORES_1,
                0.01,
                True,
                np.array(
                    [
                        [0, 1, 1, 0, 2],  # alg1 vs alg2
                        [0, 2, 0, 1, 2],  # alg1 vs alg3
                        [1, 2, 0, 2, 1],  # alg2 vs alg3
                    ]
                ),
            ),
            (
                SCORES_1,
                0.01,
                False,
                np.array(
                    [
                        [0, 1, 0, 1, 2],  # alg1 vs alg2
                        [0, 2, 1, 0, 2],  # alg1 vs alg3
                        [1, 2, 2, 0, 1],  # alg2 vs alg3
                    ]
                ),
            ),
        ],
    )
    def test_construct_win_table(
        self,
        data: pd.DataFrame,
        local_rope_value: float | None,
        maximize: bool,
        expected_table: np.ndarray,
    ):
        """Test the construction of the win/tie/loss table."""
        # When
        result_table, alg_names = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col=None,
            local_rope_value=local_rope_value,
            tie_solver="davidson",  # Keeps the ties in the table
            maximize=maximize,
        )

        # Then
        np.testing.assert_array_almost_equal(result_table, expected_table)

    def test_construct_win_table_paired_local_rope(self):
        """Test paired-path win/tie/loss construction for repeated datasets."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2", "d2"],
                "alg1": [0.8, 0.9, 0.3, 0.2],
                "alg2": [0.7, 0.8, 0.4, 0.5],
            }
        )

        result_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_value=0.1,
            tie_solver="davidson",
            maximize=True,
        )

        expected_table = np.array(
            [
                [0, 1, 1, 1, 0],
            ]
        )
        np.testing.assert_array_equal(result_table, expected_table)

    def test_construct_win_table_paired_local_rope_three_algorithms(self):
        """Test paired local-ROPE construction with dataset column and 3 algorithms."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2", "d2"],
                "alg1": [0.9, 0.8, 0.4, 0.3],
                "alg2": [0.8, 0.7, 0.3, 0.2],
                "alg3": [0.2, 0.1, 0.5, 0.4],
            }
        )

        result_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_value=0.1,
            tie_solver="davidson",
            maximize=True,
        )

        expected_table = np.array(
            [
                [0, 1, 2, 0, 0],  # alg1 > alg2 on both datasets
                [0, 2, 1, 1, 0],  # split decisions across datasets
                [1, 2, 1, 1, 0],  # split decisions across datasets
            ]
        )
        np.testing.assert_array_equal(result_table, expected_table)


class TestUserWarnings:
    """Test whether the correct warnings are raised."""

    def test_unnamed_columns(self):
        """Test whether a warning is raised when the dataset column is unnamed."""
        # Given - This simulated incorrect reading of a CSV file with an index

        CSV_CONTENT = """,alg1,alg2,alg3
        0,0.705,0.696,0.7
        1,0.7,0.7,0.75
        2,0.9,0.8,0.9
        """

        data = pd.read_csv(StringIO(CSV_CONTENT))
        # When / Then

        with pytest.warns(
            UserWarning,
            match="Some algorithm names are unnamed. This may lead to issues in the win table construction.",
        ):
            _construct_win_table(
                data=data,
                data_sd=None,
                dataset_col=None,  # This column is unnamed
                local_rope_value=None,
                tie_solver="davidson",
                maximize=True,
            )


class TestTieSolvers:
    """Test tie solver semantics for spread/add/forget strategies."""

    def test_add_solver_assigns_full_point_per_tie(self):
        """Each tie contributes 1 win to both algorithms in add mode."""
        data = pd.DataFrame(
            {
                "alg1": [0.7],
                "alg2": [0.7],
            }
        )

        add_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col=None,
            local_rope_value=0.01,
            tie_solver="add",
            maximize=True,
        )

        expected = np.array([[0, 1, 1, 1, 1]])
        np.testing.assert_array_almost_equal(add_table, expected)

    def test_forget_solver_ignores_ties(self):
        """Forget mode should leave tie counts out of win totals."""
        data = pd.DataFrame(
            {
                "alg1": [0.7],
                "alg2": [0.7],
            }
        )

        forget_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col=None,
            local_rope_value=0.01,
            tie_solver="forget",
            maximize=True,
        )

        expected = np.array([[0, 1, 0, 0, 1]])
        np.testing.assert_array_almost_equal(forget_table, expected)
