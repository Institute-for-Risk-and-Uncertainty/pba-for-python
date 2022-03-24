import json

import scipy.stats as sps

extra = {
    'lognorm': sps.lognorm,
    'foldnorm': sps.foldnorm,
    'trapz': sps.trapz,
    'truncnorm': sps.truncnorm,
    'uniform': sps.uniform,
}

dists = []
for k, v in simple_dists.items():
    dists.append(k)

d = {'dists':dists}
with open('/Users/dominiccalleja/pba-for-python/pba/data.json', 'w') as j:
    json.dump(d, j)


def __get_bounds(function_name=None, steps=200, *args):

    # define support
    x = np.linspace(0.0001, 0.9999, steps)

    #get bound arguments
    new_args = itertools.product(*args)

    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:

        bounds.append(dists[function_name].ppf(x, *a))
        bmean, bvar = dists[function_name].stats(*a, moments='mv')

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

    return Left, Right, mean, var


class Parametric():

    def __init__():

    def _pdf():

    def _cdf():

    def _ppf():
    
    def 