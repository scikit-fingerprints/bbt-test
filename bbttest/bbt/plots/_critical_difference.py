import warnings

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

NO_EQUIVALENCE_CLIQUEST_WARNING_TEMPLATE = """No groups of equivalent algorithms were found in the posterior table.
CDD plot will not contain any equivalence bars."""


def get_bars_for_ccd(
    posterior_df: pd.DataFrame,
    models_df: pd.DataFrame,
    interpretation_col: str,
) -> list[tuple[int, int]]:
    """Calculate equivalence bars using the equivalence cliques in the posterior table."""
    # Construct Graph and find the cliques
    g = nx.Graph()

    for _, row in posterior_df.iterrows():
        left = row["left_model"]
        right = row["right_model"]
        equiv = row[interpretation_col] == "="
        if equiv:
            g.add_edge(left, right)

    cliques = list(nx.find_cliques(g))

    # Map cliques to bars
    res = []

    for clique in cliques:
        clique_pos = models_df.loc[models_df["model"].isin(clique), "pos"]
        res.append((clique_pos.min(), clique_pos.max()))

    return res


def assign_bar_position(
    bars: list[tuple[int, int]], min_distance: int = 1
) -> list[int]:
    """Order the bars vertically to minimize the size of the plot."""
    if len(bars) == 0:
        return []

    indexed_bars = [
        (
            i,
            start - min_distance,
            end + min_distance,
        )  # add min distance to the bar sizes
        for i, (start, end) in enumerate(bars)
    ]

    rows: list[tuple[int, int]] = []
    rows_assigments = [0] * len(indexed_bars)

    for task_idx, start, end in indexed_bars:
        assigned = False
        for i, (row_end_value, row_id) in enumerate(rows):
            if row_end_value < start:
                # This row is available
                rows[i] = (end, row_id)
                rows_assigments[task_idx] = row_id
                assigned = True
                break
        if not assigned:
            # No rows are available, create a new one
            new_row_id = len(rows)
            rows.append((end, new_row_id))
            rows_assigments[task_idx] = new_row_id

    return rows_assigments


def _plot_cdd_diagram(
    models_df: pd.DataFrame,
    bars: list[tuple[int, int]],
    bars_positions: list[int],
    bar_y_spacing: float = 0.12,
    ax: plt.Axes | None = None,
    xlabel_spacing: int = 5,
    draw_equivalence_lines_to_axis: bool = True,
) -> plt.Axes:
    """Plot a critical difference diagram."""
    if ax is None:
        _, ax = plt.subplots()

    n_models = len(models_df)

    # Ruler at the top
    ruler_y = 0
    ax.hlines(ruler_y, 0.5, n_models + 0.5, color="black", linewidth=2)

    # Add ticks for each model
    for _, row in models_df.iterrows():
        pos = row["pos"]
        name = row["model"]
        # Invert so rank 1 is on the right
        inv_pos = n_models - pos + 1

        ax.vlines(inv_pos, ruler_y, ruler_y + 0.15, color="black", linewidth=1.2)
        ax.text(
            inv_pos,
            ruler_y + 0.2,
            name,
            ha="left",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    if len(bars) == 0:
        warnings.warn(NO_EQUIVALENCE_CLIQUEST_WARNING_TEMPLATE, UserWarning)
        max_bar_pos = 0
    else:
        max_bar_pos = max(bars_positions)
        # Draw equivalence bars
        for i, (min_pos, max_pos) in enumerate(bars):
            bar_y = ruler_y - 0.4 - bars_positions[i] * bar_y_spacing

            inv_min = n_models - max_pos + 1
            inv_max = n_models - min_pos + 1

            ax.hlines(bar_y, inv_min, inv_max, color="black", linewidth=2.5)

            if draw_equivalence_lines_to_axis:
                ax.vlines(inv_min, bar_y, -0.25, color="black", linewidth=0.5)
                ax.vlines(inv_max, bar_y, -0.25, color="black", linewidth=0.5)
            else:
                ax.vlines(inv_min, bar_y, bar_y + 0.05, color="black", linewidth=1.5)
                ax.vlines(inv_max, bar_y, bar_y + 0.05, color="black", linewidth=1.5)

    # Add rank numbers - first and last manually
    ax.text(
        1,
        ruler_y - 0.1,
        str(n_models),
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        n_models,
        ruler_y - 0.1,
        "1",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    for i in range(xlabel_spacing + 1, n_models, xlabel_spacing):
        inv_pos = n_models - i + 1
        ax.text(inv_pos, ruler_y - 0.1, str(i), ha="center", va="top", fontsize=8)

    # Clip axes
    min_bar_y = ruler_y - 0.4 - max_bar_pos * bar_y_spacing
    ax.set_xlim(0, n_models + 1)
    ax.set_ylim(min_bar_y - 0.3, 2.5)
    ax.axis("off")

    # Legend
    ax.text(
        0.5,
        min_bar_y - 0.1,
        "← worse                                    better →",
        fontsize=8,
        style="italic",
    )

    return ax


def plot_cdd_diagram(
    models_df: pd.DataFrame,
    posterior_df: pd.DataFrame,
    interpretation_col: str,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot a critical difference diagram."""
    bars = get_bars_for_ccd(
        posterior_df=posterior_df,
        models_df=models_df,
        interpretation_col=interpretation_col,
    )
    bars_positions = assign_bar_position(bars)
    return _plot_cdd_diagram(
        models_df=models_df,
        bars=bars,
        bars_positions=bars_positions,
        ax=ax,
        **kwargs,
    )
