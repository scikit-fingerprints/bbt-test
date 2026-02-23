from io import StringIO

import numpy as np
import pandas as pd
import pytest

from bbttest.bbt.alg import (
    _construct_win_table,
)
from bbttest.bbt.params import TieSolver

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
            tie_solver=TieSolver.DAVIDSON,  # Keeps the ties in the table
            maximize=maximize,
        )

        # Then
        np.testing.assert_array_almost_equal(result_table, expected_table)


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
                tie_solver=TieSolver.DAVIDSON,
                maximize=True,
            )
