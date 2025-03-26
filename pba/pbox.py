from decimal import DivisionByZero
import itertools
from typing import *
from warnings import *

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as spi

from .interval import Interval
from .copula import Copula

__all__ = ["Pbox", "mixture", "truncate", "imposition", "NotIncreasingError"]


### Local functions ###
def _check_increasing(arr):
    return np.all(np.diff(arr) >= 0)


def _interpolate(left, right, steps, method):
    if method == "linear":
        nleft = np.interp(np.linspace(0, 1, steps), np.linspace(0, 1, len(left)), left)
        nright = np.interp(
            np.linspace(0, 1, steps), np.linspace(0, 1, len(right)), right
        )
    elif method == "cubicspline":
        nleft = spi.CubicSpline(np.linspace(0, 1, len(left)), left)(
            np.linspace(0, 1, steps)
        )
        nright = spi.CubicSpline(np.linspace(0, 1, len(right)), right)(
            np.linspace(0, 1, steps)
        )
    elif method == "step":
        percentiles = {
            i: Interval(j, k)
            for i, j, k in zip(np.linspace(0, 1, len(left)), left, right)
        }
        nleft = []
        nright = []
        for i in np.linspace(0, 1, steps):
            left_key = max(key for key in percentiles.keys() if key <= i)
            right_key = min(key for key in percentiles.keys() if key >= i)
            nleft.append(percentiles[left_key].left)
            nright.append(percentiles[right_key].right)
    else:
        raise ValueError(
            "Invalid interpolation method. Must be one of: linear, cubicspline, step"
        )

    return nleft, nright


class NotIncreasingError(Exception):
    pass


def _check_div_by_zero(pbox):
    if 0 <= min(pbox.left) and 0 >= max(pbox.right):
        raise DivisionByZero("Pbox contains 0")


def _interval_list_to_array(l, left=True):
    if left:
        f = lambda x: x.left if isinstance(x, Interval) else x
    else:  # must be right
        f = lambda x: x.right if isinstance(x, Interval) else x

    return np.array([f(i) for i in l])


def _check_steps(a, b):
    if a.steps > b.steps:
        warn(
            "Pboxes have different number of steps. Interpolating {b.__name__} to {a.steps} steps"
        )
        b.interpolate(a.steps, inplace=True)
    elif a.steps < b.steps:
        warn(
            "Pboxes have different number of steps. Interpolating {a.__name__} to {b.steps} steps"
        )
        a.interpolate(b.steps, inplace=True)

    return a, b


def _check_moments(left, right, steps, mean, var):

    def _sideVariance(w, mu):
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        if mu is None:
            mu = np.mean(w)
        return max(0, np.mean((w - mu) ** 2))

    cmean = Interval(np.mean(left), np.mean(right))
    if not mean.equiv(cmean) and not mean.equiv(Interval(-np.inf, np.inf)):
        warn("Mean specified does not match calculated mean. Using calculated mean")
        print(f"Specified mean: {mean}, calculated mean: {cmean}")
        mean = cmean

    if np.any(np.isinf(left)) or np.any(np.isinf(right)):
        cvar = Interval(0, np.inf)

    if np.all(right[0] == right) and np.all(left[0] == left):
        cvar = Interval(0, (right[0] - left[0]) ** (2 / 4))

    vr = _sideVariance(left, np.mean(left))
    w = np.copy(left)
    n = len(left)

    for i in reversed(range(n)):
        w[i] = right[i]
        v = _sideVariance(w, np.mean(w))

        if np.isnan(vr) or np.isnan(v):
            vr = np.inf
        elif vr < v:
            vr = v

    if left[n - 1] <= right[0]:
        vl = 0.0
    else:
        x = right
        vl = _sideVariance(w, np.mean(w))

        for i in reversed(range(n)):
            w[i] = left[i]
            here = w[i]

            if 1 < i:
                for j in reversed(range(i - 1)):
                    if w[i] < w[j]:
                        w[j] = here

            v = _sideVariance(w, np.mean(w))

            if np.isnan(vl) or np.isnan(v):
                vl = 0
            elif v < vl:
                vl = v

    cvar = Interval(vl, vr)

    if not var.equiv(cvar) and not var.equiv(Interval(0, np.inf)):
        warn(
            "Variance specified does not match calculated variance. Using calculated variance"
        )
        print(f"Specified variance: {var}, calculated variance: {cvar}")
        var = cvar

    return cmean, cvar


### Arithmetic Functions ###
# ** This is prefered as means can only test once for +, -, *, /, **#
def _arithmetic(a, b, method, op, enforce_steps=True, interpolation_method="linear"):
    # * If enforce_steps is True, the number of steps in the returned p-box is the maximum of the number of steps in a and b.

    if b.__class__.__name__ == "Interval":
        other = Pbox(other, steps=a.steps)

    if b.__class__.__name__ == "Pbox":

        a, b = _check_steps(a, b)

        if method == "f":
            with catch_warnings():
                simplefilter("ignore")
                nleft, nright = _f_arithmetic(a, b, op)

        elif method == "p":
            with catch_warnings():
                simplefilter("ignore")
                nleft, nright = _p_arithmetic(a, b, op)

        elif method == "o":
            with catch_warnings():
                simplefilter("ignore")
                nleft, nright = _o_arithmetic(a, b, op)

        elif method == "i":
            with catch_warnings():
                simplefilter("ignore")
                nleft, nright = _i_arithmetic(a, b, op)

        else:
            raise ArithmeticError("Calculation method unkown")

        nleft.sort()
        nright.sort()

        if enforce_steps:
            # Steps needs to match the maximum number of steps in a
            return Pbox(
                left=nleft,
                right=nright,
                steps=max(a.steps, b.steps),
                interpolation=interpolation_method,
            )
        else:
            return Pbox(left=nleft, right=nright)

    else:
        try:
            # Try adding constant
            if a.shape in ["uniform", "normal", "cauchy", "triangular", "skew-normal"]:
                s = a.shape
            else:
                s = ""

            return Pbox(
                left=op(a.left, other), right=op(a.right, other), shape=s, steps=a.steps
            )

        except:
            return NotImplemented


def _f_arithmetic(a, b, op):
    nleft = np.empty(a.steps)
    nright = np.empty(a.steps)

    for i in range(0, a.steps):
        j = np.array(range(i, a.steps))
        k = np.array(range(a.steps - 1, i - 1, -1))

        nright[i] = np.min(op(a.right[j], b.right[k]))

        jj = np.array(range(0, i + 1))
        kk = np.array(range(i, -1, -1))

        nleft[i] = np.max(op(a.left[jj], b.left[kk]))

    return nleft, nright


def _i_arithmetic(a, b, op):

    nleft = np.array([op(i, j) for i, j in itertools.product(a.left, b.left)])
    nright = np.array([op(i, j) for i, j in itertools.product(a.right, b.right)])

    return nleft, nright


def _o_arithmetic(a, b, op):
    nleft = op(a.left, np.flip(b.right))
    nright = op(a.right, np.flip(b.left))
    return nleft, nright


def _p_arithmetic(a, b, op):
    nleft = op(a.left, b.left)
    nright = op(a.right, b.right)
    return nleft, nright


### Pbox Class ###
class Pbox:
    r"""
    A probability distribution is a mathematical function that gives the probabilities of occurrence for diﬀerent possible values of a variable. Probability boxes (p-boxes) represent interval bounds on probability distributions. The simplest kind of p-box can be expressed mathematically as

    .. math::

        \mathcal{F}(x) = [\underline{F}(x),\overline{F}(x)], \ \underline{F}(x)\geq \overline{F}(x)\ \forall x \in \mathbb{R}


    where :math:`\underline{F}(x)` is the function that defines the left bound of the p-box and :math:`\overline{F}(x)` defines the right bound of the p-box. In PBA the left and right bounds are each stored as a NumPy array containing the percent point function (the inverse of the cumulative distribution function) for `steps` evenly spaced values between 0 and 1. P-boxes can be defined using all the probability distributions that are available through SciPy's statistics library,

    Naturally, precise probability distributions can be defined in PBA by defining a p-box with precise inputs. This means that within probability bounds analysis probability distributions are considered a special case of a p-box with zero width. Resultantly, all methodology that applies to p-boxes can also be applied to probability distributions.

    Distribution-free p-boxes can also be generated when the underlying distribution is unknown but parameters such as the mean, variance or minimum/maximum bounds are known. Such p-boxes make no assumption about the shape of the distribution and instead return bounds expressing all possible distributions that are valid given the known information. Such p-boxes can be constructed making use of Chebyshev, Markov and Cantelli inequalities from probability theory.

    .. attention::

        It is usually better to define p-boxes using distributions or non-parametric methods (see ). This constructor is provided for completeness and for the construction of p-boxes with precise inputs.

    :arg left: Left bound of the p-box. Can be a list, NumPy array, Interval or numeric type.
    :arg right: Right bound of the p-box. Can be a list, NumPy array, Interval or numeric type.
    :arg steps: Number of steps to discretize the p-box into. Default is None.
    :arg shape: The shap eof the distribution used to construct the p-box. The shape is defined by the p-box constructor. Default is `None`.
    :arg mean: Interval containing the mean of the p-box. Default is `Interval(-np.inf,np.inf)`.
    :arg var: Interval containing the variance of the p-box. Default is `Interval(-np.inf,np.inf)`.
    :arg interpolation: Interpolation method to use. See interpolation for more details. Default is `linear`.
    :arg check_moments: If `True`, the mean and variance of the p-box are checked and recalculated if necessary. Default is `True`.

    .. important::

        If steps is not specified, left and right must be arrays of the same length and steps is set at that value.

        If steps is specified, both left and right are interpolated to the specified number of steps using the specified interpolation method (see interpolation_). In this case if steps is less than the length of left or right, a warning is raised and steps is set to the length of left or right.

    .. warning::

        The statistic values be specified by constructor function are also calculated automatically. If specified values differ from the calculated values, the calculated values are used and a warning is raised.

        If check_moments is set to ``True`` and mean and/or var are specified, if the calculated values differ from the specified values, the calculated values are used and a warning is raised.

    .. error::

        If the left and right bounds are not increasing, a `NotIncreasingError` is raised.
        ``ValueError`` is raised if left and right are not the same length.


    """

    STEPS = 200
    _MN = 1e-4

    def __init__(
        self,
        left: Union[list, np.ndarray],
        right: Union[list, np.ndarray] = None,
        steps: int = None,
        shape: str = "",
        mean: Interval = Interval(-np.inf, np.inf),
        var: Interval = Interval(0, np.inf),
        mean_left: float = None,
        mean_right: float = None,
        var_left: float = None,
        var_right: float = None,
        interpolation: str = "linear",
        check_moments: bool = True,
    ):

        #!! TO BE DEPRECATED !!#
        if mean_left is not None or mean_right is not None:
            mean = Interval(mean_left, mean_right)
        if var_left is not None or var_right is not None:
            var = Interval(var_left, var_right)

        if right is None:
            right = left

        if not hasattr(left, "__iter__"):
            raise ValueError("left must be an iterable")
        if not hasattr(right, "__iter__"):
            raise ValueError("right must be an iterable")

        if isinstance(left, Interval):
            if steps is None:
                raise ValueError("steps must be specified if left is an Interval")
            left = np.array([left.left] * steps)

        if isinstance(right, Interval):
            if steps is None:
                raise ValueError("steps must be specified if right is an Interval")
            right = np.array([right.right] * steps)

        if len(left) != len(right):
            raise ValueError("left and right must be the same number of steps")

        if not _check_increasing(left) or not _check_increasing(right):
            raise NotIncreasingError("Left and right arrays must be increasing")

        if steps is None:
            steps = len(left)

        if isinstance(left, list):
            left = _interval_list_to_array(left)
        elif not isinstance(left, np.ndarray):
            left = np.array(left)

        if isinstance(right, list):
            right = _interval_list_to_array(right, left=False)
        elif not isinstance(right, np.ndarray):
            right = np.array(right)

        if steps != len(left):
            warn(
                f"Number of steps does not match length of left. Interpolating left/right to {steps} steps"
            )
            left, right = _interpolate(left, right, steps, interpolation)

        l, r = zip(*[(min(i), max(i)) for i in zip(left, right)])
        self.left = np.array(l)
        self.right = np.array(r)
        self.steps = steps
        self.shape = shape

        if (
            mean.equiv(Interval(-np.inf, np.inf))
            or var.equiv(Interval(-np.inf, np.inf))
            or check_moments
        ):

            self.mean, self.var = _check_moments(l, r, steps, mean, var)

        else:
            self.mean = mean
            self.var = var

    def __repr__(self):

        return f"Pbox: ~ {self.shape}(range={Interval(self.lo(),self.hi()):g}, mean={self.mean:g}, var={self.var:g})"

    __str__ = __repr__

    def __iter__(self):
        for val in np.array([self.left, self.right]).flatten():
            yield val

    def __neg__(self):
        if self.shape in ["uniform", "normal", "cauchy", "triangular", "skew-normal"]:
            s = self.shape
        else:
            s = ""

        return Pbox(
            left=sorted(-np.flip(self.right)),
            right=sorted(-np.flip(self.left)),
            steps=len(self.left),
            shape=s,
            mean=-self.mean,
            var=self.var,
            check_moments=False,
        )

    def __lt__(self, other):
        return self.lt(other, method="f")

    def __rlt__(self, other):
        return self.ge(other, method="f")

    def __le__(self, other):
        return self.le(other, method="f")

    def __rle__(self, other):
        return self.gt(other, method="f")

    def __gt__(self, other):
        return self.gt(other, method="f")

    def __rgt__(self, other):
        return self.le(other, method="f")

    def __ge__(self, other):
        return self.ge(other, method="f")

    def __rge__(self, other):
        return self.lt(other, method="f")

    def __and__(self, other):
        return self.logicaland(other, method="f")

    def __rand__(self, other):
        return self.logicaland(other, method="f")

    def __or__(self, other):
        return self.logicalor(other, method="f")

    def __ror__(self, other):
        return self.logicalor(other, method="f")

    def __add__(self, other):
        return self.add(other, method="f")

    def __radd__(self, other):
        return self.add(other, method="f")

    def __sub__(self, other):
        return self.sub(other, method="f")

    def __rsub__(self, other):
        self = -self
        return self.add(other, method="f")

    def __mul__(self, other):
        return self.mul(other, method="f")

    def __rmul__(self, other):
        return self.mul(other, method="f")

    def __pow__(self, other):
        return self.pow(other, method="f")

    def __rpow__(self, other):
        if not hasattr(other, "__iter__"):
            other = np.array((other))

        b = Pbox(other)
        return b.pow(self, method="f")

    def __truediv__(self, other):

        return self.div(other, method="f")

    def __rtruediv__(self, other):

        try:
            return other * self.recip()
        except:
            return NotImplemented

    def lo(self):
        """
        Returns the left-most value in the interval
        """
        return self.left[0]

    def hi(self):
        """
        Returns the right-most value in the interval
        """
        return self.right[-1]

    def unary(self, func, *args, **kwargs):
        """
        Allows for unary operations to be performed on a p-box.
        This is acheived by applying the function to each interval in the p-box.

        **Arguments:**
            ``func`` (``function``): Function to apply to each interval in the p-box.
            ``args`` (``tuple``): Arguments to pass to the function.
            ``kwargs`` (``dict``): Keyword arguments to pass to the function.

        .. important::

            The function must accept an Interval object as its first argument and return an Interval object. ``args`` and ``kwargs`` are passed to the function as additional arguments.

            >>> func(Interval(l,r),*args, **kwargs)

            The function must return an Interval object. Behaviour may be unpredictable if the endpoints of the inputted interval do not correspond to the endpoints of the outputted p-box.

        """
        ints = [
            func(Interval(l, r), *args, **kwargs) for l, r in zip(self.left, self.right)
        ]

        return Pbox(
            left=np.array([i.left for i in ints]),
            right=np.array([i.right for i in ints]),
        )

    def interpolate(self, steps: int, method: str = "linear", inplace=True) -> None:
        """
        Function to interpolate a p-box to a new number of steps.

        **Arguments:**
            ``steps`` (``int``): Number of steps to interpolate to.
            ``method`` (``str``): Interpolation method to use. Must be one of ``linear``, ``cubicspline`` or ``step``.
            ``inplace`` (``bool``): If ``True``, the p-box is interpolated in place. If ``False``, a new p-box is returned.

        .. note::

            ``method = linear`` uses ``numpy.interp``
            ``method = cubicspline`` uses ``scipy.interpolate.CubicSpline``
            ``method = step`` uses a step interpolation method.

        .. example::

            .. image:: https://github.com/Institute-for-Risk-and-Uncertainty/pba-for-python/blob/master/docs/images/interpolation.png?raw=true

        """
        if steps < self.steps:
            raise ValueError(
                "New number of steps must be greater than current number of steps"
            )

        new_left, new_right = _interpolate(self.left, self.right, steps, method)

        if inplace:
            self.left = new_left
            self.right = new_right
            self.steps = steps

        else:
            return Pbox(left=new_left, right=new_right, steps=steps, shape=self.shape)

    ### Arithmetic Functions
    def add(
        self, other: Union["Pbox", Interval, float, int], method="f", enforce_steps=True
    ) -> "Pbox":
        """
        Adds other to Pbox to other using the defined dependency method
        """
        try:
            return _arithmetic(
                self, other, method, op=lambda x, y: x + y, enforce_steps=enforce_steps
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"Addition of {other.__class__.__name__} to Pbox not implemented"
            )
        except:
            raise Exception(f"Addition of {other.__class__.__name__} to Pbox failed")

    def sub(self, other, method="f"):
        """
        Subtracts other from Pbox using the defined dependency method
        """
        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        return self.add(-other, method)

    def mul(self, other, method="f"):
        """
        Multiplies other and Pbox using the defined dependency method
        """
        try:
            return _arithmetic(self, other, method, op=lambda x, y: x * y)
        except NotImplementedError:
            raise NotImplementedError(
                f"Multiplication of {other.__class__.__name__} and Pbox not implemented"
            )
        except:
            raise Exception(
                f"Multiplication of {other.__class__.__name__} to Pbox failed"
            )

    def div(self, other, method="f"):
        """
        Divides Pbox by other using the defined dependency method
        """
        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        if isinstance(other, (Interval, Pbox)):
            return self.mul(other.recip(), method)
        else:
            return self.mul(1 / other, method)

    def pow(self, other: Union["Pbox", Interval, float, int], method="f") -> "Pbox":
        """
        Raises a p-box to the power of other using the defined dependency method
        """
        try:
            return _arithmetic(self, other, method, op=lambda x, y: x**y)
        except NotImplementedError:
            raise NotImplementedError(
                f"Power of {other.__class__.__name__} to Pbox not implemented"
            )
        except:
            raise Exception(f"Power of {other.__class__.__name__} to Pbox failed")

    def exp(self):
        return self.unary(function=lambda x: x.exp())

    def sqrt(self):
        return self.unary(function=lambda x: x.sqrt())

    def recip(self):
        """
        Calculates the reciprocal of a p-box.

        .. error::

            ``DivisionByZero`` is raised if the p-box contains 0.

        """
        _check_div_by_zero(self)
        return Pbox(
            left=1 / np.flip(self.right), right=1 / np.flip(self.left), steps=self.steps
        )

    def lt(self, other, method="f"):
        b = self.add(-other, method)
        return b.get_probability(
            0
        )  # return (self.add(-other, method)).get_probability(0)

    def le(self, other, method="f"):
        b = self.add(-other, method)
        return b.get_probability(
            0
        )  # how is the "or equal to" affecting the calculation?

    def gt(self, other, method="f"):
        self = -self
        b = self.add(other, method)
        return b.get_probability(0)  # maybe 1-prob ?

    def ge(self, other, method="f"):
        self = -self
        b = self.add(other, method)
        return b.get_probability(0)

    def min(self, other, method="f"):
        """
        Returns a new Pbox object that represents the element-wise minimum of two Pboxes.

        **Arguments**:

            ``other``: Another Pbox, Interval or a numeric value.
            ``method``: Calculation method to determine the minimum. Can be one of 'f', 'p', 'o', 'i'.

        **Returns**:

            ``Pbox``

        """
        if isinstance(other, (Interval, Pbox)):
            if method == "f":
                return _arithmetic(
                    self, other, method, op=lambda x, y: min(list(x) + list(y))
                )
            else:
                return _arithmetic(
                    self, other, method, op=lambda x, y: np.minimum(x, y)
                )
        else:
            try:
                return Pbox(
                    left=np.array([i if i < other else other for i in self.left]),
                    right=np.array([i if i < other else other for i in self.right]),
                )
            except:
                return NotImplemented(
                    f"Minimum of {other.__class__.__name__} and Pbox not implemented"
                )

    def max(self, other, method="f"):
        """
        Returns a new Pbox object that represents the element-wise minimum of two Pboxes.

        **Arguments**:

            ``other``: Another Pbox, Interval or a numeric value.
            ``method``: Calculation method to determine the minimum. Can be one of 'f', 'p', 'o', 'i'.

        **Returns**:

            ``Pbox``

        """
        if isinstance(other, (Interval, Pbox)):
            if method == "f":
                return _arithmetic(
                    self, other, method, op=lambda x, y: max(list(x) + list(y))
                )
            else:
                return _arithmetic(
                    self, other, method, op=lambda x, y: np.maximum(x, y)
                )
        else:
            try:
                return Pbox(
                    left=np.array([i if i > other else other for i in self.left]),
                    right=np.array([i if i > other else other for i in self.right]),
                )
            except:
                return NotImplemented(
                    f"Minimum of {other.__class__.__name__} and Pbox not implemented"
                )

    def truncate(
        self, a: Union[Interval, float, int], b: Union[float, int] = None, method="f"
    ):
        """
        Truncates a p-box to the interval [a,b], or a if b is not specified and a is an Interval.

        **Arguments:**

            ``a`` (``Interval``, ``float``, ``int``): The lower bound of the truncation interval.
            ``b`` (``float``, ``int``): The upper bound of the truncation interval. If not specified, the upper bound of ``a`` is used.
            ``method`` (``str``): The dependency method to use. Can be one of ``f``, ``p``, ``o``, ``i``.

        .. admonition:: Implementation

            >>> self.min(a).max(b)

        .. error::

            ``ValueError`` is raised if ``b`` is not specified and ``a`` is not an ``Interval``.

        """
        if isinstance(a, Interval):
            a, b = a.left, a.right
        else:
            if b is None:
                raise ValueError("b must be specified if a is not an Interval")

        return self.min(a, method=method).max(b, method=method)

    def logicaland(self, other, method="f"):  # conjunction
        if method == "i":
            return self.mul(other, method)  # independence a * b
        elif method == "p":
            return self.min(other, method)  # perfect min(a, b)
        elif method == "o":
            return max(self.add(other, method) - 1, 0)  # opposite max(a + b – 1, 0)
        elif method == "+":
            return self.min(other, method)  # positive env(a * b, min(a, b))
        elif method == "-":
            return self.min(other, method)  # negative env(max(a + b – 1, 0), a * b)
        else:
            return self.min(other, method).env(max(0, self.add(other, method) - 1))

    def logicalor(self, other, method="f"):  # disjunction
        if method == "i":
            return 1 - (1 - self) * (1 - other)  # independent 1 – (1 – a) * (1 – b)
        elif method == "p":
            return self.max(other, method)  # perfect max(a, b)
        elif method == "o":
            return min(self.add(other, method), 1)  # opposite min(1, a + b)
        # elif method=='+':
        #    return(env(,min(self.add(other,method),1))  # positive env(max(a, b), 1 – (1 – a) * (1 – b))
        # elif method=='-':
        #    return()  # negative env(1 – (1 – a) * (1 – b), min(1, a + b))
        else:
            return self.env(self.max(other, method), min(self.add(other, method), 1))

    def env(self, other):
        """
        .. _pbox.env:

        Computes the envelope of two Pboxes.

        **Arguments**:

            ``other``: Another Pbox, Interval or a numeric value.

        **Returns**:

            ``Pbox``

        .. error::

            ``NotImplementedError`` is raised if ``other`` is not a Pbox. Imputation of other needs to be done manually

        """

        if other.__class__.__name__ == "Pbox":
            if self.steps != other.steps:
                raise ArithmeticError("Both Pboxes must have the same number of steps")
        else:
            other = Pbox(other, steps=self.steps)

        nleft = np.minimum(self.left, other.left)
        nright = np.maximum(self.right, other.right)

        return Pbox(left=nleft, right=nright, steps=self.steps)

    def show(
        self,
        figax=None,
        now=True,
        title="",
        xlabel="x",
        ylabel=r"$\Pr(x \leq X)$",
        left_col="red",
        right_col="black",
        label=None,
        **kwargs,
    ):
        r"""
        Plots the p-box

        **Arguments:**

            ``figax`` (``tuple``): Tuple containing a matplotlib figure and axis object. If not specified, a new figure and axis object are created.
            ``now`` (``bool``): If ``True``, the figure is shown. If ``False``, the figure is returned.
            ``title`` (``str``): Title of the plot.
            ``xlabel`` (``str``): Label for the x-axis.
            ``ylabel`` (``str``): Label for the y-axis.
            ``label`` (``str``): Label for the plot (for use in legend).
            ``left_col`` (``str``): Colour of the left bound of the p-box.
            ``right_col`` (``str``): Colour of the right bound of the p-box.
            ``kwargs`` (``dict``): Additional keyword arguments to pass to ``matplotlib.pyplot.plot``.

        **Example:**

        .. code-block:: python

            >>> p = pba.N([-1,1],1)
            >>> fig, ax = plt.subplots()
            >>> p.show(figax = (fig,ax), now = True, title = 'Example', xlabel = 'x', ylabel = r'$\Pr(x \leq X)$',left_col = 'red',right_col = 'black')

        """
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        # now respects discretization
        L = self.left
        R = self.right
        steps = self.steps

        LL = np.concatenate((L, L, np.array([R[-1]])))
        RR = np.concatenate((np.array([L[0]]), R, R))
        ii = (
            np.concatenate(
                (np.arange(steps), np.arange(1, steps + 1), np.array([steps]))
            )
            / steps
        )
        jj = (
            np.concatenate((np.array([0]), np.arange(steps + 1), np.arange(1, steps)))
            / steps
        )

        ii.sort()
        jj.sort()
        LL.sort()
        RR.sort()

        if "color" in kwargs.keys():
            ax.plot(LL, ii, label=label, **kwargs)
            ax.plot(RR, jj, **kwargs)
        else:
            ax.plot(LL, ii, color=left_col, label=label, **kwargs)
            ax.plot(RR, jj, color=right_col, **kwargs)

        if title != "":
            ax.set_title(title, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if now:
            fig.show()
        else:
            return fig, ax

    plot = show

    def get_interval(self, p: float) -> Interval:
        """
        Returns the interval for the given probability
        """
        assert 0 < p < 1

        y = np.linspace(Pbox._MN, 1 - Pbox._MN, self.steps)

        left_val = self.left[np.abs(y - p).argmin()]
        right_val = self.right[np.abs(y - p).argmin()]

        return Interval(left_val, right_val)

    def get_probability(self, val) -> Interval:
        """
        Returns the interval
        """

        p = np.append(np.insert(np.linspace(0, 1, self.steps), 0, 0), 1)

        i = 0
        while i < self.steps and self.left[i] < val:
            i += 1

        ub = p[i]

        j = 0

        while j < self.steps and self.right[j] < val:
            j += 1

        lb = p[j]

        return Interval(lb, ub)

    def summary(self) -> str:
        """
        Returns a summary of the p-box
        """
        s = "Pbox Summary\n"
        s += "------------\n"
        if self.shape != "":
            s += f"Shape: {self.shape}\n"
        s += f"Range: {self.support()}\n"
        s += f"Mean: {self.mean}\n"
        s += f"Variance: {self.var}\n"
        s += f"Steps: {self.steps}\n"
        return s

    def support(self) -> Interval:
        """
        Returns the range of the pbox

        .. admonition:: Implementation

            >>> Interval(self.lo(),self.hi())

        """
        return Interval(min(self.left), max(self.right))

    def mean(self) -> Interval:
        """
        Returns the mean of the pbox
        """
        return self.mean

    def median(self) -> Interval:
        """
        Returns the median of the distribution
        """
        return Interval(np.median(self.left), np.median(self.right))

    def straddles(self, N, endpoints=True) -> bool:
        """
        Checks whether a number is within the p-box's support

        **Arguments:**

            ``N`` (``float``): Number to check
            ``endpoints`` (``bool``): If ``True``, the endpoints of the p-box are included in the check.

        **Returns:**

            ``bool``
        """
        return self.support().straddles(N, endpoints)

    def straddles_zero(self, endpoints=True) -> bool:
        """
        Checks whether :math:`0` is within the p-box
        """
        return self.straddles(0, endpoints)

    def imp(self, other):
        """
        Returns the imposition of self with other
        """
        if other.__class__.__name__ != "Pbox":
            try:
                pbox = Pbox(pbox)
            except:
                raise TypeError(
                    "Unable to convert %s object (%s) to Pbox" % (type(pbox), pbox)
                )

        u = []
        d = []

        assert self.steps == other.steps

        for sL, sR, oL, oR in zip(self.left, self.right, other.left, other.right):

            if max(sL, oL) > min(sR, oR):
                raise Exception("Imposition does not exist")

            u.append(max(sL, oL))
            d.append(min(sR, oR))

        return Pbox(left=u, right=d)


def left_list(implist, verbose=False):
    if not hasattr(implist, "__iter__"):
        return np.array(implist)

    return np.array([imp.lo() for imp in implist])


def right_list(implist, verbose=False):
    if not hasattr(implist, "__iter__"):
        return np.array(implist)

    return np.array([imp.hi() for imp in implist])


def truncate(pbox, min, max):
    return pbox.truncate(min, max)


def imposition(*args: Union[Pbox, Interval, float, int]):
    """
    Returns the imposition of the p-boxes in *args

    Parameters
    ----------
    *args :
        Number of p-boxes or objects to be mixed

    Returns
    ----------
    Pbox
    """
    x = []
    for pbox in args:
        if pbox.__class__.__name__ != "Pbox":
            try:
                pbox = Pbox(pbox)
            except:
                raise TypeError(
                    "Unable to convert %s object (%s) to Pbox" % (type(pbox), pbox)
                )
        x.append(pbox)

    p = x[0]

    for i in range(1, len(x)):
        p.imp(x[i])

    return p


def mixture(
    *args: Union[Pbox, Interval, float, int],
    weights: List[Union[float, int]] = [],
    steps: int = Pbox.STEPS,
) -> Pbox:
    """
    Mixes the pboxes in *args
    Parameters
    ----------
    *args :
        Number of p-boxes or objects to be mixed
    weights:
        Right side of box

    Returns
    ----------
    Pbox
    """
    # TODO: IMPROVE READBILITY

    x = []
    for pbox in args:
        if pbox.__class__.__name__ != "Pbox":
            try:
                pbox = Pbox(pbox)
            except:
                raise TypeError(
                    "Unable to convert %s object (%s) to Pbox" % (type(pbox), pbox)
                )
        x.append(pbox)

    k = len(x)
    if weights == []:
        weights = [1] * k

    # temporary hack
    # k = 2
    # x = [self, x]
    # w = [1,1]

    if k != len(weights):
        return "Need same number of weights as arguments for mixture"
    weights = [i / sum(weights) for i in weights]  # w = w / sum(w)
    u = []
    d = []
    n = []
    ml = []
    mh = []
    m = []
    vl = []
    vh = []
    v = []
    for i in range(k):
        u = u + list(x[i].left)
        d = np.append(d, x[i].right)
        n = (
            n + [weights[i] / x[i].steps] * x[i].steps
        )  # w[i]*rep(1/x[i].steps,x[i].steps))

        # mu = mean(x[i])
        # ml = ml + [mu.left()]
        # mh = mh + [mu.right()]
        # m = m + [mu]               # don't need?
        # sigma2 = var(x[[i]])  ### !!!! shouldn't be the sample variance, but the population variance
        # vl = vl + [sigma2.left()]
        # vh = vh + [sigma2.right()]
        # v = v + [sigma2]

        ML = x[i].mean.left
        MR = x[i].mean.right
        VL = x[i].var.left
        VR = x[i].var.right
        m = m + [Interval(ML, MR)]
        v = v + [Interval(VL, VR)]
        ml = ml + [ML]
        mh = mh + [MR]
        vl = vl + [VL]
        vh = vh + [VR]

    n = [_ / sum(n) for _ in n]  # n = n / sum(n)
    su = sorted(u)
    su = [su[0]] + su
    pu = [0] + list(
        np.cumsum([n[i] for i in np.argsort(u)])
    )  #  pu = c(0,cumsum(n[order(u)]))
    sd = sorted(d)
    sd = sd + [sd[-1]]
    pd = list(np.cumsum([n[i] for i in np.argsort(d)])) + [
        1
    ]  #  pd = c(cumsum(n[order(d)]),1)
    u = []
    d = []
    j = len(pu) - 1
    for p in reversed(
        np.arange(steps) / steps
    ):  # ii = np.arange(steps))/steps  #    ii = 0: (Pbox$steps-1) / Pbox$steps
        while p < pu[j]:
            j = j - 1  # repeat {if (pu[j] <= p) break; j = j - 1}
        u = [su[j]] + u
    j = 0
    for p in (
        np.arange(steps) + 1
    ) / steps:  # jj = (np.arange(steps)+1)/steps #  jj =  1: Pbox$steps / Pbox$steps
        while pd[j] < p:
            j = j + 1  # repeat {if (p <= pu[j]) break; j = j + 1}
        d = d + [sd[j]]
    mu = Interval(
        np.sum([W * M for M, W in zip(weights, ml)]),
        np.sum([W * M for M, W in zip(weights, mh)]),
    )
    s2 = 0
    for i in range(k):
        s2 = s2 + weights[i] * (v[i] + m[i] ** 2)
    s2 = s2 - mu**2

    return Pbox(
        np.array(u),
        np.array(d),
        mean=Interval(mu.left, mu.right),
        var=Interval(s2.left, s2.right),
        steps=steps,
        check_moments=False,
    )


def change_default_arithmetic_method(method):
    """
    Changes the default arithmetic method for p-boxes

    **Arguments:**

        ``method`` (``str``): Method to use. Must be one of ``f``, ``p``, ``o``, ``i``.

    """

    if method not in ["f", "p", "o", "i"]:
        raise ValueError("Method must be one of 'f', 'p', 'o', 'i'")

    def nadd(
        self,
        other: Union["Pbox", Interval, float, int],
        method=method,
        enforce_steps=True,
    ) -> "Pbox":

        try:
            return _arithmetic(
                self, other, method, op=lambda x, y: x + y, enforce_steps=enforce_steps
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"Addition of {other.__class__.__name__} to Pbox not implemented"
            )
        except:
            raise Exception(f"Addition of {other.__class__.__name__} to Pbox failed")

    def nsub(self, other, method=method):

        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        return self.add(-other, method)

    def nmul(self, other, method=method):

        try:
            return _arithmetic(self, other, method, op=lambda x, y: x * y)
        except NotImplementedError:
            raise NotImplementedError(
                f"Multiplication of {other.__class__.__name__} and Pbox not implemented"
            )
        except:
            raise Exception(
                f"Multiplication of {other.__class__.__name__} to Pbox failed"
            )

    def ndiv(self, other, method=method):

        if method == "o":
            method = "p"
        elif method == "p":
            method = "o"

        if isinstance(other, (Interval, Pbox)):
            return self.mul(other.recip(), method)
        else:
            return self.mul(1 / other, method)

    def npow(self, other: Union["Pbox", Interval, float, int], method=method) -> "Pbox":

        try:
            return _arithmetic(self, other, method, op=lambda x, y: x**y)
        except NotImplementedError:
            raise NotImplementedError(
                f"Power of {other.__class__.__name__} to Pbox not implemented"
            )
        except:
            raise Exception(f"Power of {other.__class__.__name__} to Pbox failed")

    Pbox.__add__ = nadd
    Pbox.__sub__ = nsub
    Pbox.__mul__ = nmul
    Pbox.__truediv__ = ndiv
    Pbox.__pow__ = npow
