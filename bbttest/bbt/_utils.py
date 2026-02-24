from __future__ import annotations

import sys
from functools import wraps
from typing import (
    Literal,
    get_args,
    get_origin,
)

from pymc.distributions import Cauchy, LogNormal, Normal

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


def is_literal_value(value: object, typx: object) -> bool:
    if isinstance(typx, TypeAliasType):
        typx = typx.__value__
    if get_origin(typx) is Literal:
        return value in get_args(typx)
    return False


def _validate_params(func):
    from inspect import signature

    sig = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        for kwarg in kwargs:
            if kwarg not in sig.parameters:
                raise ValueError(f"Unexpected keyword argument '{kwarg}'")
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in bound_args.arguments.items():
            param = sig.parameters[name]
            # If type annotation is a Literal, validate the value
            if param.annotation is not param.empty and is_literal_value(
                value, param.annotation
            ):
                continue  # Valid value, continue to next parameter
            elif (
                param.annotation is not param.empty
                and get_origin(param.annotation) is Literal
            ):
                raise ValueError(
                    f"Invalid value '{value}' for parameter '{name}'. Expected one of {get_args(param.annotation)}."
                )
        return func(*args, **kwargs)

    return wrapper


def _get_distribution_for_prior(prior: str, scale: float):
    match prior:
        case "log_normal":
            return LogNormal("sigma", mu=0, sigma=scale)
        case "cauchy":
            return Cauchy("sigma", alpha=0, beta=scale)
        case "normal":
            return Normal("sigma", mu=0, sigma=scale)
        case _:
            raise ValueError(f"Unsupported hyperprior: {prior}")
