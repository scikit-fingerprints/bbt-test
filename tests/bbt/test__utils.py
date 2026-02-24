from typing import Literal

import pytest

from bbttest.bbt._utils import _validate_params

MockLiteralType = Literal["option1", "option2", "option3"]


@_validate_params
def mock_fun(param_lit: MockLiteralType, param_str: str): ...


class TestLiteralValidation:
    """Tests _validate_params decorator for validating parameters."""

    @pytest.mark.parametrize(
        "params, should_raise",
        [
            ({"param_lit": "option1", "param_str": "any string"}, False),
            ({"param_lit": "option4", "param_str": "another string"}, True),
        ],
    )
    def test_literal_validation(self, params, should_raise):
        """Test if _validate_params correctly validates Literal parameters and skips strings."""
        if should_raise:
            with pytest.raises(
                ValueError, match="Invalid value 'option4' for parameter 'param_lit'"
            ):
                mock_fun(**params)
        else:
            mock_fun(**params)
