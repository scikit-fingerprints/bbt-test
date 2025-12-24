from typing import Literal

import pandas as pd

from .py_bbt import PyBBT


def multiple_ropes_control_table(
    model: PyBBT,
    ropes: list[tuple[float, float]],
    control_model: str,
    selected_models: list[str] | None = None,
    interpretation: Literal["weak", "strong"] = "weak",
    return_as_array: bool = False,
    join_char: str = ", ",
) -> pd.DataFrame:
    """
    Construct a table comparing models against predefined control models across multiple ROPEs.
    The output table contains N rows (one per ROPE) and 5 columns
    (rope value, better models, equivalent models, worse models, unknown models).

    Args:
        model: Fitted PyBBT model.
        ropes: List of ROPE tuples to evaluate.
        interpretation: Type of interpretation to use ("weak" or "strong"), see [1]_.
        return_as_array: Whether the individual cells should contain model names as list or joined into single string.
        join_char: Character(s) used to join multiple model names in a single cell.

    Returns
    -------
        pd.DataFrame: Table comparing models against control models across multiple ROPEs.

    References
    ----------
    .. [1] `Jacques Wainer
        "A Bayesian Bradley-Terry model to compare multiple ML algorithms on multiple data sets"
        Journal of Machine Learning Research 24 (2023): 1-34
        <http://jmlr.org/papers/v24/22-0907.html>`_
    """
    rows = []
    interpretation_col = f"{interpretation}_interpretation_raw"
    for rope in ropes:
        post_table = model.posterior_table(
            rope_value=rope,
            columns=["left_model", "right_model", interpretation_col],
            control_model=control_model,
            selected_models=selected_models,
        )
        better_models = list()
        equivalent_models = list()
        worse_models = list()
        unknown_models = list()

        for _, row in post_table.iterrows():
            decision = row[interpretation_col]
            if decision == ">":
                if row["left_model"] == control_model:
                    worse_models.append(row["right_model"])
                else:
                    better_models.append(row["left_model"])
            elif decision == "=":
                if row["left_model"] == control_model:
                    equivalent_models.append(row["right_model"])
                else:
                    equivalent_models.append(row["left_model"])
            elif row["left_model"] == control_model:
                unknown_models.append(row["right_model"])
            else:
                unknown_models.append(row["left_model"])

        rows.append({
            "rope": rope,
            "better_models": better_models if return_as_array else join_char.join(better_models),
            "equivalent_models": equivalent_models if return_as_array else join_char.join(equivalent_models),
            "worse_models": worse_models if return_as_array else join_char.join(worse_models),
            "unknown_models": unknown_models if return_as_array else join_char.join(unknown_models),
        })

    return pd.DataFrame(rows)
