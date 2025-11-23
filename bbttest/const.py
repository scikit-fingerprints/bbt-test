from enum import Enum

from pymc.distributions import Cauchy, LogNormal, Normal


class HyperPrior(str, Enum):
    """
    Hyper Prior distributions for BBT MCMC sampling.
    """

    LOG_NORMAL = "logNormal"
    LOG_NORMAL_SCALED = "logNormalScaled"
    CAUCHY = "cauchy"
    NORMAL = "normal"

    def _get_pymc_dist(self, scale, name="sigma"):
        if self == HyperPrior.LOG_NORMAL:
            return LogNormal(name, mu=0, sigma=1)
        elif self == HyperPrior.LOG_NORMAL_SCALED:
            return LogNormal(name, mu=0, sigma=scale)
        elif self == HyperPrior.CAUCHY:
            return Cauchy(name, alpha=0, beta=scale)
        elif self == HyperPrior.NORMAL:
            return Normal(name, mu=0, sigma=scale)
        else:
            raise ValueError(f"Unsupported hyperprior: {self}")


class ReportedProperty(str, Enum):
    """
    Enum containing properties that can be reported from BBT results.
    """

    MEDIAN = "median"
    MEAN = "mean"
    HDI_LOW = "hdi_low"
    HDI_HIGH = "hdi_high"
    DELTA = "delta"
    ABOVE_50 = "above_50"
    IN_ROPE = "in_rope"
    WEAK_INTERPRETATION = "weak_interpretation"
    STRONG_INTERPRETATION = "strong_interpretation"


class TieSolver(str, Enum):
    """
    Enum containing tie solving strategies.

    ADD - Add 1 win to both players.
    SPREAD - Add 1/2 win to both players.
    FOGET - Ignore the tie.
    DAVIDSON - Use Davidson's method to handle ties.
    """

    ADD = "add"
    SPREAD = "spread"
    FORGET = "forget"
    DAVIDSON = "davidson"


DEFAULT_PROPERTIES = (
    ReportedProperty.MEAN,
    ReportedProperty.DELTA,
    ReportedProperty.ABOVE_50,
    ReportedProperty.IN_ROPE,
    ReportedProperty.WEAK_INTERPRETATION,
)
