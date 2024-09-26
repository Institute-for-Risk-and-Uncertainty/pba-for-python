"""
This module contains functions to create p-boxes for various distributions. These functions have been hand-coded to ensure accuracy. All other distributions can be found in the :mod:`pba.pbox_constructors.parametric` module.
"""

__all__ = [
    "beta",
    "foldnorm",
    "normal",
    "N",
    "unif",
    "U",
    "KN",
    "KM",
    "lognormal",
    "lognorm",
    "trapz",
    "uniform",
    "weibull",
]


from scipy import stats
import numpy as np
from itertools import product

from typing import *
from warnings import *

from ..interval import Interval
from ..pbox import Pbox
from ..logical import sometimes
from .parametric import parametric, norm, weibull_max, weibull_min


def lognormal(mean, var, steps=200):
    """
    Creates a p-box for the lognormal distribution

    *Note: the parameters used are the mean and variance of the lognormal distribution not the mean and variance of the underlying normal*
    See:
    `[1]<https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters>`
    `[2]<https://stackoverflow.com/questions/51906063/distribution-mean-and-standard-deviation-using-scipy-stats>`


    Parameters
    ----------
    mean :
        mean of the lognormal distribution
    var :
        variance of the lognormal distribution

    Returns
    ----------
    Pbox

    """
    if steps > 1000:
        x = np.linspace(1 / steps, 1 - 1 / steps, steps)
    else:
        x = np.linspace(0.001, 0.999, steps)

    if mean.__class__.__name__ != "Interval":
        mean = Interval(mean, mean)
    if var.__class__.__name__ != "Interval":
        var = Interval(var, var)

    def __lognorm(mean, var):

        sigma = np.sqrt(np.log1p(var / mean**2))
        mu = np.log(mean) - 0.5 * sigma * sigma

        return stats.lognorm(sigma, loc=0, scale=np.exp(mu))

    bounds = np.array(
        [
            __lognorm(mean.left, var.left).ppf(x),
            __lognorm(mean.right, var.left).ppf(x),
            __lognorm(mean.left, var.right).ppf(x),
            __lognorm(mean.right, var.right).ppf(x),
        ]
    )
    print(bounds)
    Left = np.min(bounds, axis=0)
    Right = np.max(bounds, axis=0)

    Left = np.array(Left)
    Right = np.array(Right)
    return Pbox(Left, Right, steps=steps, shape="lognormal")


lognorm = lognormal


def beta(a, b, steps=200):
    """
    Beta distribution.
    """
    args = list(args)
    if not isinstance(a, Interval):
        a = Interval(a)
    if not isinstance(b, Interval):
        b = Interval(b)

    if sometimes(a < 0):
        raise ValueError("a must be always greater than 0")
    if sometimes(b < 0):
        raise ValueError("b must be always greater than 0")

    if a.left == 0:
        a.left = 1e-5
    if b.left == 0:
        b.left = 1e-5

    return parametric("beta").make_pbox(a, b, steps=steps, support=Interval(0, 1))


def foldnorm(mu, s, steps=200):

    x = np.linspace(0.0001, 0.9999, steps)
    if mu.__class__.__name__ != "Interval":
        mu = Interval(mu)
    if s.__class__.__name__ != "Interval":
        s = Interval(s)

    new_args = [
        [mu.lo() / s.lo(), 0, s.lo()],
        [mu.hi() / s.lo(), 0, s.lo()],
        [mu.lo() / s.hi(), 0, s.hi()],
        [mu.hi() / s.hi(), 0, s.hi()],
    ]

    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:

        bounds.append(stats.foldnorm.ppf(x, *a))
        bmean, bvar = stats.foldnorm.stats(*a, moments="mv")

        if bmean < mean_lo:
            mean_lo = bmean
        if bmean > mean_hi:
            mean_hi = bmean
        if bvar > var_hi:
            var_hi = bvar
        if bvar < var_lo:
            var_lo = bvar

    Left = [min([b[i] for b in bounds]) for i in range(steps)]
    Right = [max([b[i] for b in bounds]) for i in range(steps)]

    var = Interval(np.float64(var_lo), np.float64(var_hi))
    mean = Interval(np.float64(mean_lo), np.float64(mean_hi))

    Left = np.array(Left)
    Right = np.array(Right)

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="foldnorm",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


# def frechet_r(*args, steps = 200):
#     args = list(args)
#     for i in range(0,len(args)):
#         if args[i].__class__.__name__ != 'Interval':
#             args[i] = Interval(args[i])

#     Left, Right, mean, var = __get_bounds('frechet_r',steps,*args)

#     return Pbox(
#           Left,
#           Right,
#           steps      = steps,
#           shape      = 'frechet_r',
#           mean_left  = mean.left,
#           mean_right = mean.right,
#           var_left   = var.left,
#           var_right  = var.right
#           )

# def frechet_l(*args, steps = 200):
#     args = list(args)
#     for i in range(0,len(args)):
#         if args[i].__class__.__name__ != 'Interval':
#             args[i] = Interval(args[i])

#     Left, Right, mean, var = __get_bounds('frechet_l',steps,*args)

#     return Pbox(
#           Left,
#           Right,
#           steps      = steps,
#           shape      = 'frechet_l',
#           mean_left  = mean.left,
#           mean_right = mean.right,
#           var_left   = var.left,
#           var_right  = var.right
#           )


def trapz(a, b, c, d, steps=200):
    if a.__class__.__name__ != "Interval":
        a = Interval(a)
    if b.__class__.__name__ != "Interval":
        b = Interval(b)
    if c.__class__.__name__ != "Interval":
        c = Interval(c)
    if d.__class__.__name__ != "Interval":
        d = Interval(d)

    x = np.linspace(0.0001, 0.9999, steps)
    left = stats.trapz.ppf(
        x, *sorted([b.lo() / d.lo(), c.lo() / d.lo(), a.lo(), d.lo() - a.lo()])
    )
    right = stats.trapz.ppf(
        x, *sorted([b.hi() / d.hi(), c.hi() / d.hi(), a.hi(), d.hi() - a.hi()])
    )

    return Pbox(left, right, steps=steps, shape="trapz")


def uniform(a, b, steps=200):

    x = np.linspace(0, 1, steps)

    if a.__class__.__name__ != "Interval":
        a = Interval(a, a)
    if b.__class__.__name__ != "Interval":
        b = Interval(b, b)

    Left = np.linspace(a.left, b.left)
    Right = np.linspace(a.right, b.right)

    mean = 0.5 * (a + b)
    var = ((b - a) ** 2) / 12

    return Pbox(
        Left,
        Right,
        steps=steps,
        shape="uniform",
        mean_left=mean.left,
        mean_right=mean.right,
        var_left=var.left,
        var_right=var.right,
    )


def weibull(*args, steps=200):

    wm = weibull_max(*args)
    wl = weibull_min(*args)

    return Pbox(left=wl.left, right=wm.right)


### Other distributions
def KM(k, m, steps=200):
    with catch_warnings():
        simplefilter("ignore")
        return beta(Interval(k, k + 1), Interval(m, m + 1), steps=steps)


def KN(k, n, steps=200):
    return KM(k, n - k, steps=steps)


### Alternate names
normal = norm
N = normal
unif = uniform
U = uniform
