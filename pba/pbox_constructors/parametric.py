"""
.. _pbox_constructors.sp:

Automatically generated
_______________________

P-boxes based on scipy.stats distributions.

These p-boxes are generated using the scipy.stats distributions. The p-boxes are generated by taking the minimum and maximum values of the inverse CDF of the distribution for a given set of parameters.
"""

from ..interval import *
from ..pbox import *

import numpy as np
from itertools import product
from scipy import stats
import sys
from copy import deepcopy

# * The commented out distributions are overwritten in dists
dist = [
    "alpha",
    "anglit",
    "arcsine",
    "argus",
    # "beta",
    "betaprime",
    "bradford",
    "burr",
    "burr12",
    "cauchy",
    "chi",
    "chi2",
    "cosine",
    "crystalball",
    "dgamma",
    "dweibull",
    "erlang",
    "expon",
    "exponnorm",
    "exponweib",
    "exponpow",
    "f",
    "fatiguelife",
    "fisk",
    "foldcauchy",
    # "foldnorm",
    "genlogistic",
    "gennorm",
    "genpareto",
    "genexpon",
    "genextreme",
    "gausshyper",
    "gamma",
    "gengamma",
    "genhalflogistic",
    "geninvgauss",
    "gilbrat",
    "gompertz",
    "gumbel_r",
    "gumbel_l",
    "halfcauchy",
    "halflogistic",
    "halfnorm",
    "halfgennorm",
    "hypsecant",
    "invgamma",
    "invgauss",
    "invweibull",
    "johnsonsb",
    "johnsonsu",
    "kappa4",
    "kappa3",
    "ksone",
    "kstwobign",
    "laplace",
    "levy",
    "levy_l",
    "levy_stable",
    "logistic",
    "loggamma",
    "loglaplace",
    # "lognorm",
    "loguniform",
    "lomax",
    "maxwell",
    "mielke",
    "moyal",
    "nakagami",
    "ncx2",
    "ncf",
    "nct",
    "norm",
    "norminvgauss",
    "pareto",
    "pearson3",
    "powerlaw",
    "powerlognorm",
    "powernorm",
    "rdist",
    "rayleigh",
    "rice",
    "recipinvgauss",
    "semicircular",
    "skewnorm",
    "t",
    # "trapz",
    "triang",
    "truncexpon",
    "truncnorm",
    "tukeylambda",
    # "uniform",
    "vonmises",
    "vonmises_line",
    "wald",
    "weibull_min",
    "weibull_max",
    "wrapcauchy",
    "bernoulli",
    "betabinom",
    "binom",
    "boltzmann",
    "dlaplace",
    "geom",
    "hypergeom",
    "logser",
    "nbinom",
    "planck",
    "poisson",
    "randint",
    "skellam",
    "zipf",
    "yulesimon",
]


class parametric:

    def __init__(self, function_name):
        self.function_name = function_name

    def make_pbox(self, *args, steps: int = Pbox.STEPS, **kwargs) -> Pbox:
        f"""
        Generate a P-box from a scipy.stats.{self.function_name} distribution.
        """
        args = list(args)
        for i, a in enumerate(args):
            if not isinstance(a, Interval):
                args[i] = Interval(a)

        # define support
        x = np.linspace(Pbox._MN, 1 - Pbox._MN, steps)

        # get bound arguments
        new_args = list(product(*args))

        bounds = np.empty((len(new_args), steps))
        means = []
        variances = []

        for i, a in enumerate(new_args):

            bounds[i, :] = stats.__dict__[self.function_name].ppf(x, *a, **kwargs)
            m, v = stats.__dict__[self.function_name].stats(*a, **kwargs, moments="mv")

            means.append(m)
            variances.append(v)

        Left = np.array([min([b[i] for b in bounds]) for i in range(steps)])
        Right = np.array([max([b[i] for b in bounds]) for i in range(steps)])

        mean = Interval(min(means), max(means))
        var = Interval(min(variances), max(variances))

        return Pbox(
            left=Left,
            right=Right,
            mean=mean,
            var=var,
            shape=self.function_name,
            check_moments=False,
        )


for d in dist:
    locals()[d] = parametric(d).make_pbox

__all__ = deepcopy(dist)
