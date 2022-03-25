if __name__ is not None and "." in __name__:
    from .interval import Interval
    from .pbox import Pbox
else:
    from interval import Interval
    from pbox import Pbox
import json

import scipy.stats as sps
import numpy as np
import itertools

extra = {
    'lognorm': sps.lognorm,
    'foldnorm': sps.foldnorm,
    'trapz': sps.trapz,
    'truncnorm': sps.truncnorm,
    'uniform': sps.uniform,
    'beta': sps.beta
}

def read_json(file_name):
    f = open(file_name)
    data = json.load(f)
    return data

dist = read_json('data.json')['dists']

class Bounds():
    STEPS=200

    def __init__(self, shape, *args, n_subinterval=5):
        
        self.bounds = get_distributions(self.shape, *args, n_subinterval=n_subinterval)
        self.pbox = self._pba_constructor(*args)

    def _pba_constructor(self, *args):
        # args = list(args)

        # for i in range(len(args)):
        #     if args[i].__class__.__name__ != 'Interval':
        #         args[i] = Interval(args[i])

        Left, Right, mean, var = get_bounds(self.shape, self.STEPS, *args)
        return Pbox(
            Left,
            Right,
            steps=self.STEPS,
            shape=self.shape,
            mean_left=mean.left,
            mean_right=mean.right,
            var_left=var.left,
            var_right=var.right
        )

    def __getattr__(self, name):
        dist_methods = ['cdf', 'dist', 'entropy', 'expect', 'interval', 'isf',
                        'kwds', 'logcdf', 'logpdf', 'logpmf', 'logsf', 'mean',
                        'median', 'moment', 'pdf', 'pmf', 'ppf', 'random_state',
                        'rvs', 'sf', 'stats', 'std', 'support', 'var']
        try:
            if name in dist_methods:
                m = {}
                for k, v in self.bounds.items():
                    m[k] = getattr(v['dist'], name)

                def f(x):
                    l = [g(x) for j, g in m.items()]
                    return Interval(min(l), max(l))

                return f
            else:
                return getattr(self.pbox, name)
        except AttributeError:
            raise AttributeError(
                    "Bounds' object has no attribute '%s'" % name)

#TODO: Create a wrapper to allow instanciation with distribution name function wrappers.

class Parametric(Bounds):
    """
    A parametric Pbox is defined where parameters of a named distribtuion are specified as
    Intervals. This class wraps the scipy.stats library and supports all scipy methods such as
    pdf, cdf, survival function etc. 
    
    Parametric can be created using any combination of the following styles:
        pba.Parametric('norm', [0,1], [1,2])
        pba.Parametric('cauchy', Interval[0,1], 1)
        pba.Parametric('beta', a = Interval[0,.5], b=0.5)
        
        
    Parameters
    ----------
    shape : numeric
        left side of interval
    **args : float, list, np.array, Interval
        set of distribution parameters
    **kwargs :
        set of key value pairs for the distribution parameters
        
    Attributes
    ----------
    left : numeric
        left side of interval
    right : numeric
        right side of interval

    """
    params = []
    __pbox__=True

    def __init__(self, shape, *args, n_subinterval=5, **kwargs):
        self.params = list_parameters(shape)
        self.shape = shape

        if args:
            args = args2int(*args)
            self.set_from_args(*args)
        if kwargs:
            self.set_parameters(**kwargs)
            args = [v for i, v in kwargs.items()]
        
        super().__init__(self.shape,*args, n_subinterval= n_subinterval)
        

    def get_parameter_values(self):
        return [getattr(self, k) for k in self.params]

    def set_from_args(self,*args):
        self.params
        args = list(args)
        for i, v in enumerate(args):
            # if not isinstance(v, Interval):
            #     v = Interval(v)
            d = {self.params[i]:v}
            self._set_parameter(**d)

    def set_parameters(self, **kwargs):
        if kwargs:
            self._set_parameter(**kwargs)

    def _set_parameter(self, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                assert k in self.params, '{} not in param list: {}'.format(
                    k, param)
                if not isinstance(v, Interval):
                    v = Interval(v)
                setattr(self, k, v)
    
    # def __getattr__(self, name):
    #     try:
    #         return getattr(self.pbox, name)
    #     except:
    #         try:
    #             return getattr(self.bounds, name)
    #         except AttributeError:
    #             raise AttributeError("Parametric' object has no attribute '%s'" % name) 
    

def args2int(*args):
    args = list(args)
    for i, a in enumerate(args):
        if not isinstance(a, Interval):
            args[i] = Interval(a)
    return args

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

def get_distributions(distribution, *args, n_subinterval=5):
    
    args = list(args)
    if n_subinterval:
        args = [subintervalise(i, n_subinterval) for i in args]
    new_args = itertools.product(*args)

    bounds={}
    i=0
    for a in new_args:
        bounds[i] = {}
        bounds[i]['dist'] = getattr(sps, distribution)(*a)
        bounds[i]['param'] = a
        i+=1
    return bounds

def get_bounds(distribution, support=[1E-5, 1-1E-5], *args):
    # define support
    steps=200
    x = np.linspace(1E-5, 1-1E-5, 200)
    #get bound arguments
    new_args = itertools.product(*args)

    bounds = []
    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:
        bounds.append(getattr(sps, distribution).ppf(x, *a))
        bmean, bvar = getattr(sps, distribution).stats(*a, moments='mv')
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

def subintervalise(interval, n):
    xi = np.linspace(interval.left, interval.right, n)
    x = np.hstack([xi[:-1],xi[1:]])
    return x.T

class Normal(Parametric): 
    def __init__(self,*args, **kwargs):
        super().__init__('norm', *args, **kwargs)        

class t(Parametric): 
    def __init__(self,*args, **kwargs):
        super().__init__('t', *args, **kwargs)        


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = Parametric('norm', Interval(3,5), Interval(1,4), n_subinterval=10)
    N.plot()

    Xi = np.linspace(-15,15,200)
    pdf = [N.pdf(i) for i in Xi]
    L, R = zip(*pdf)

    plt.figure()
    plt.title('PBox Density')
    plt.plot(Xi, L)
    plt.plot(Xi, R)
    plt.show()

    cdf = [N.cdf(i) for i in Xi]
    L, R = zip(*cdf)

    plt.figure()
    plt.title('PBox Cumulative Parametric Compute')
    plt.plot(Xi, L)
    plt.plot(Xi, R)
    plt.show()

    sf = [N.sf(i) for i in Xi]
    L, R = zip(*sf)

    plt.figure()
    plt.title('PBox Survival Function')
    plt.plot(Xi, L)
    plt.plot(Xi, R)
    plt.show()

    logcdf = [N.logcdf(i) for i in Xi]
    L, R = zip(*logcdf)

    # Is this right?
    plt.figure()
    plt.title('PBox Log CDF Function')
    plt.plot(Xi, L)
    plt.plot(Xi, R)
    plt.show()

    logpdf = [N.logpdf(i) for i in Xi]
    L, R = zip(*logpdf)

    # Is this right?
    plt.figure()
    plt.title('PBox Log PDF Function')
    plt.plot(Xi, L)
    plt.plot(Xi, R)
    plt.show()

    alpha = np.linspace(0,.99,200)
    CI=[N.interval(i) for i in alpha]
    L, R=zip(*CI)

    # Is this right?
    plt.figure()
    plt.title('PBox CI Function')
    plt.plot(L,alpha)
    plt.plot(R,alpha)
    plt.show()

    print('Expected Value: {}'.format(N.expect(None)))
