if __name__ is not None and "." in __name__:
    from .interval import Interval
else:
    from interval import Interval

if __name__ is not None and "." in __name__:
    from .pbox import Pbox
else:
    from pbox import Pbox

import scipy.stats as sps
import numpy as np
#
# __all__ = [beta]

def beta(a,b, steps=200):

    def make_bound(x,a,b):
        bound = sps.beta.ppf(x, a, b)
        mean, var = sps.beta.stats(a,b, moments = 'mv')
        return bound, mean, var

    x = np.linspace(0,1,steps)

    if a.__class__.__name__ != 'Interval':
        a = Interval(a,a)
    if b.__class__.__name__ != 'Interval':
        b = Interval(b,b)

    LowerBound, LowerMean, LowerVar = make_bound(x, a.left(), b.right())
    UpperBound, UpperMean, UpperVar = make_bound(x, a.right(), b.left())

    mean_left = min(LowerMean, UpperMean)
    var_left  = min(LowerVar, UpperVar)

    mean_right = max(LowerMean, UpperMean)
    var_right  = max(LowerVar, UpperVar)

    return Pbox(LowerBound, UpperBound, steps = steps, shape=None, mean_left=mean_left, mean_right=mean_right, var_left=var_left, var_right=var_right)

def normal(mean, std, steps = 200):

    x = np.linspace(0.001,0.999,steps)

    if mean.__class__.__name__ != 'Interval':
        mean = Interval(mean,mean)
    if std.__class__.__name__ != 'Interval':
        std = Interval(std,std)

    bound0 = sps.norm.ppf(x, mean.left(), std.left())
    bound1 = sps.norm.ppf(x, mean.right(), std.left())
    bound2 = sps.norm.ppf(x, mean.left(), std.right())
    bound3 = sps.norm.ppf(x, mean.right(), std.right())

    LowerBound = [min(bound0[i],bound1[i],bound2[i],bound3[i]) for i in range(steps)]
    UpperBound = [max(bound0[i],bound1[i],bound2[i],bound3[i]) for i in range(steps)]

    LowerBound = np.array(LowerBound)
    UpperBound = np.array(UpperBound)
    return Pbox(LowerBound, UpperBound, steps = steps, shape='normal', mean_left=mean.left(), mean_right=mean.right(), var_left=std.left() ** 2, var_right=std.right() ** 2)
norm = normal
N = normal

def uniform(a , b, steps = 200):

    x = np.linspace(0,1,steps)

    if a.__class__.__name__ != 'Interval':
        a = Interval(a,a)
    if b.__class__.__name__ != 'Interval':
        b = Interval(b,b)

    LowerBound = np.linspace(a.left(),b.left())
    UpperBound = np.linspace(a.right(),b.right())

    mean = 0.5 * (a+b)
    var = ((b-a)**2 )/12

    return Pbox(LowerBound, UpperBound, steps = steps, shape=None, mean_left=mean.left(), mean_right=mean.right(), var_left=var.left(), var_right=var.right())

unif = uniform
U = uniform
