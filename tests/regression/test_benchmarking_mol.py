"""
Regression tests for PyBBT model using molecular embeddings benchmarking data.

This test suite validates the PyBBT model's weak interpretation results against
the ECFP baseline using molecular embeddings benchmarking data from the study
of pretrained molecular embedding models.

Notes
-----
The test data is adapted from the benchmarking study:

Praski, Mateusz, Jakub Adamczyk, and Wojciech Czech.
"Benchmarking pretrained molecular embedding models for molecular representation learning."
arXiv preprint arXiv:2508.06199 (2025).
https://arxiv.org/pdf/2508.06199

Data source:
https://github.com/scikit-fingerprints/benchmarking_molecular_models/blob/31779f16c004b3fb8aa555ecceb6b95ca71a1d7d/results/arxiv_preprint_2025_08.csv

The tests validate that the PyBBT model correctly identifies:
- Better performing models (vs ECFP baseline)
- Equivalent performing models (within ROPE)
- Unknown comparisons (insufficient evidence)
- Worse performing models

Test parameters:
- local_rope_value: 0.01
- tie_solver: TieSolver.SPREAD
- MCMC sampling: 2000 draws, 1000 tune, 4 chains
"""

from pathlib import Path

import pandas as pd
import pytest

from bbttest import PyBBT, TieSolver
from bbttest.bbt.const import DEFAULT_PROPERTIES, ReportedProperty


@pytest.fixture(scope="module")
def benchmarking_data():
    """
    Load benchmarking molecular data.

    Returns
    -------
    pd.DataFrame
        Molecular embeddings benchmarking results with columns for dataset
        and various model scores.
    """
    data_path = Path(__file__).parent.parent / "data" / "benchmarking_mol.csv"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def fitted_model(benchmarking_data):
    """
    Fit PyBBT model with local_rope_value=0.01.

    Parameters
    ----------
    benchmarking_data : pd.DataFrame
        Benchmarking molecular data fixture.

    Returns
    -------
    PyBBT
        Fitted PyBBT model instance.
    """
    model = PyBBT(local_rope_value=0.01, tie_solver=TieSolver.SPREAD)
    model.fit(
        benchmarking_data,
        dataset_col="dataset",
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=42,
    )
    return model


def _extract_interpretations(results):
    """
    Extract model interpretations from posterior table results.

    Parameters
    ----------
    results : pd.DataFrame
        Posterior table results with comparisons against ECFP_count.

    Returns
    -------
    dict
        Dictionary mapping model names to their weak interpretations.
    """
    interpretations = {}
    for _, row in results.iterrows():
        if row["left_model"] != "ECFP_count":
            interpretations[row["left_model"]] = row["weak_interpretation"]
        if row["right_model"] != "ECFP_count":
            # Invert interpretation when ECFP is on the left
            if row["weak_interpretation"] == f"{row['left_model']} better":
                interpretations[row["right_model"]] = "ECFP better"
            else:
                interpretations[row["right_model"]] = row["weak_interpretation"]
    return interpretations


class TestWeakInterpretationAgainstECFP:
    """Test weak interpretation results against ECFP baseline for different ROPE values."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "rope,better_models,equivalent_models,unknown_models,worse_models",
        [
            (
                (0.45, 0.55),
                ["CLAMP", "rmat_4M"],
                [],
                [
                    "AtomPair_count",
                    "CDDD",
                    "ChemBERTa-10M-MTR",
                    "mat_masking_2M",
                    "molbert",
                ],
                [
                    "ChemFM-3B",
                    "ChemGPT-4.7M",
                    "GEM",
                    "GNN-GraphCL-sum",
                    "GraphFP-CP",
                    "GraphMVP_CP-max",
                    "MoLFormer-XL-both-10pct",
                    "SELFormer-Lite",
                    "SimSon",
                    "TT",
                    "chemformer_mask",
                    "coati",
                    "grover_large",
                    "mol2vec",
                    "mol_r_tag_1024",
                    "unimolv1",
                    "unimolv2",
                ],
            ),
            (
                (0.4, 0.6),
                ["CLAMP", "rmat_4M"],
                [
                    "CDDD",
                    "ChemBERTa-10M-MTR",
                    "mat_masking_2M",
                    "molbert",
                ],
                ["AtomPair_count"],
                [
                    "ChemFM-3B",
                    "ChemGPT-4.7M",
                    "GEM",
                    "GNN-GraphCL-sum",
                    "GraphFP-CP",
                    "GraphMVP_CP-max",
                    "MoLFormer-XL-both-10pct",
                    "SELFormer-Lite",
                    "SimSon",
                    "TT",
                    "chemformer_mask",
                    "coati",
                    "grover_large",
                    "mol2vec",
                    "mol_r_tag_1024",
                    "unimolv1",
                    "unimolv2",
                ],
            ),
            (
                (0.35, 0.65),
                ["CLAMP"],
                [
                    "AtomPair_count",
                    "CDDD",
                    "ChemBERTa-10M-MTR",
                    "mat_masking_2M",
                    "molbert",
                    "rmat_4M",
                ],
                [],
                [
                    "ChemFM-3B",
                    "ChemGPT-4.7M",
                    "GEM",
                    "GNN-GraphCL-sum",
                    "GraphFP-CP",
                    "GraphMVP_CP-max",
                    "MoLFormer-XL-both-10pct",
                    "SELFormer-Lite",
                    "SimSon",
                    "TT",
                    "chemformer_mask",
                    "coati",
                    "grover_large",
                    "mol2vec",
                    "mol_r_tag_1024",
                    "unimolv1",
                    "unimolv2",
                ],
            ),
            (
                (0.3, 0.7),
                [],
                [
                    "AtomPair_count",
                    "CDDD",
                    "CLAMP",
                    "ChemBERTa-10M-MTR",
                    "MoLFormer-XL-both-10pct",
                    "mat_masking_2M",
                    "mol2vec",
                    "molbert",
                    "rmat_4M",
                ],
                [],
                [
                    "ChemFM-3B",
                    "ChemGPT-4.7M",
                    "GEM",
                    "GNN-GraphCL-sum",
                    "GraphFP-CP",
                    "GraphMVP_CP-max",
                    "SELFormer-Lite",
                    "SimSon",
                    "TT",
                    "chemformer_mask",
                    "coati",
                    "grover_large",
                    "mol_r_tag_1024",
                    "unimolv1",
                    "unimolv2",
                ],
            ),
        ],
        ids=["rope_0.45_0.55", "rope_0.4_0.6", "rope_0.35_0.65", "rope_0.3_0.7"],
    )
    def test_weak_interpretation_for_rope(
        self,
        fitted_model,
        rope,
        better_models,
        equivalent_models,
        unknown_models,
        worse_models,
    ):
        """
        Test weak interpretation results for different ROPE values.

        Parameters
        ----------
        fitted_model : PyBBT
            Fitted PyBBT model fixture.
        rope : tuple of float
            Region of Practical Equivalence (ROPE) bounds.
        better_models : list of str
            Models expected to be better than ECFP.
        equivalent_models : list of str
            Models expected to be equivalent to ECFP.
        unknown_models : list of str
            Models with unknown comparison to ECFP.
        worse_models : list of str
            Models expected to be worse than ECFP.
        """
        results = fitted_model.posterior_table(
            rope_value=rope,
            control_model="ECFP_count",
            columns=list(DEFAULT_PROPERTIES)
            + [ReportedProperty.LEFT_MODEL, ReportedProperty.RIGHT_MODEL],
        )

        interpretations = _extract_interpretations(results)

        # Validate better models
        for model in better_models:
            assert interpretations[model] == f"{model} better", (
                f"Model {model} should be better than ECFP for ROPE {rope}"
            )

        # Validate equivalent models
        for model in equivalent_models:
            assert interpretations[model] == "Equivalent", (
                f"Model {model} should be equivalent to ECFP for ROPE {rope}"
            )

        # Validate unknown models
        for model in unknown_models:
            assert interpretations[model] == "Unknown", (
                f"Model {model} should have unknown comparison with ECFP for ROPE {rope}"
            )

        # Validate worse models
        for model in worse_models:
            assert interpretations[model] == "ECFP better", (
                f"Model {model} should be worse than ECFP for ROPE {rope}"
            )
