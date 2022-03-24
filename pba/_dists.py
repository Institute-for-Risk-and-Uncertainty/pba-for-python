if __name__ is not None and "." in __name__:
    from .interval import Interval
else:
    from interval import Interval
import json

import scipy.stats as sps

extra = {
    'lognorm': sps.lognorm,
    'foldnorm': sps.foldnorm,
    'trapz': sps.trapz,
    'truncnorm': sps.truncnorm,
    'uniform': sps.uniform,
}

dist = read_json('data.json')['dists']

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
    """
    A parametric Pbox is defined where parameters of a named distribtuion are specified as
    Intervals. 
    
    Parametric can be created using::
        pba.I(left,right)
        pba.Interval(left,right)
        
    Parameters
    ----------
    left : numeric
        left side of interval
    right : numeric
        right side of interval
        
    Attributes
    ----------
    left : numeric
        left side of interval
    right : numeric
        right side of interval

    """
    params = []

    def __init__(self, distribution, pbox=True, *args, **kwargs):
        self._parameter_list = list_parameters(distribution)
        self.distribution = distribution

        if args:
            self.set_from_args(*args)
        if kwargs:
            self.set_parameters(**kwargs)

        self._pbox = self._pba_constructor(*args)
        return None

    def _pdf(self):
        return None

    def _cdf(self):
        return None

    def _ppf(self):
        return None

    def _logpdf(self):
        return None


    def _pba_constructor(self, *args):
        args = list(args)

        for i in range(len(args)):
            if args[i].__class__.__name__ != 'Interval':
                args[i] = Interval(args[i])

        Left, Right, mean, var = __get_bounds(self.distribution, steps, *args)
        return Pbox(
            Left,
            Right,
            steps      = steps,
            shape      = self.distribution,
            mean_left  = mean.left,
            mean_right = mean.right,
            var_left   = var.left,
            var_right  = var.right
            )
    def set_from_args(self,*args):
        self._parameter_list
        args = list(args)
        for i, v in enumerate(args):
            d = {self._parameter_list[i]:v}
            self._set_parameter(**d)

    def set_parameters(self, **kwargs):
        if kwargs:
            self._set_parameter(**kwargs)

    def _set_parameter(self, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                assert k in self._parameter_list, '{} not in param list: {}'.format(
                    k, param)
                if not isinstance(v, Interval):
                    v = Interval(v)
                setattr(self, k, v)
    
    def __getattr__(self, name):
        try:
            return getattr(self._pbox, name)
        except AttributeError:
            raise AttributeError("Constructor' object has no attribute '%s'" % name)

def check_implimentation(distribution):
    if distribution in dist:
        return True
    else:
        sys.exit('{} not implimented, choose form : \n\n{}'.format(
            distribution, availiable_distributions()))
        return False

def availiable_distributions():
    return [' {},'.format(d) for d in dist]

def list_parameters(distribution):
    """List parameters for stats.distribution.
    
    Parameters
    ----------
        distribution: a string or stats distribution object.
    Output
    ------
        A list of distribution parameter strings.
    """
    if isinstance(distribution, str):
        try:
            distribution = getattr(sps, distribution)
        except:
            check_implimentation(distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(',')]
    else:
        parameters = []
    if distribution.name in sps._discrete_distns._distn_names:
        parameters.insert(0,'loc')
    elif distribution.name in sps._continuous_distns._distn_names:
        parameters.insert(0,'loc')
        parameters.insert(1,'scale')
    return parameters


def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data
