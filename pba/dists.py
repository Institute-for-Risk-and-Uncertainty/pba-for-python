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
import itertools

from typing import *
from warnings import *

dists = {
    'alpha' : sps.alpha,
    'anglit' : sps.anglit,
    'arcsine' : sps.arcsine,
    'argus' : sps.argus,
    'beta' : sps.beta,
    'betaprime' : sps.betaprime,
    'bradford' : sps.bradford,
    'burr' : sps.burr,
    'burr12' : sps.burr12,
    'cauchy' : sps.cauchy,
    'chi' : sps.chi,
    'chi2' : sps.chi2,
    'cosine' : sps.cosine,
    'crystalball' : sps.crystalball,
    'dgamma' : sps.dgamma,
    'dweibull' : sps.dweibull,
    'erlang' : sps.erlang,
    'expon' : sps.expon,
    'exponnorm' : sps.exponnorm,
    'exponweib' : sps.exponweib,
    'exponpow' : sps.exponpow,
    'f' : sps.f,
    'fatiguelife' : sps.fatiguelife,
    'fisk' : sps.fisk,
    'foldcauchy' : sps.foldcauchy,
    'foldnorm' : sps.foldnorm,
    # 'frechet_r' : sps.frechet_r,
    # 'frechet_l' : sps.frechet_l,
    'genlogistic' : sps.genlogistic,
    'gennorm' : sps.gennorm,
    'genpareto' : sps.genpareto,
    'genexpon' : sps.genexpon,
    'genextreme' : sps.genextreme,
    'gausshyper' : sps.gausshyper,
    'gamma' : sps.gamma,
    'gengamma' : sps.gengamma,
    'genhalflogistic' : sps.genhalflogistic,
    'geninvgauss' : sps.geninvgauss,
    'gilbrat' : sps.gilbrat,
    'gompertz' : sps.gompertz,
    'gumbel_r' : sps.gumbel_r,
    'gumbel_l' : sps.gumbel_l,
    'halfcauchy' : sps.halfcauchy,
    'halflogistic' : sps.halflogistic,
    'halfnorm' : sps.halfnorm,
    'halfgennorm' : sps.halfgennorm,
    'hypsecant' : sps.hypsecant,
    'invgamma' : sps.invgamma,
    'invgauss' : sps.invgauss,
    'invweibull' : sps.invweibull,
    'johnsonsb' : sps.johnsonsb,
    'johnsonsu' : sps.johnsonsu,
    'kappa4' : sps.kappa4,
    'kappa3' : sps.kappa3,
    'ksone' : sps.ksone,
    'kstwobign' : sps.kstwobign,
    'laplace' : sps.laplace,
    'levy' : sps.levy,
    'levy_l' : sps.levy_l,
    'levy_stable' : sps.levy_stable,
    'logistic' : sps.logistic,
    'loggamma' : sps.loggamma,
    'loglaplace' : sps.loglaplace,
    'lognorm' : sps.lognorm,
    'loguniform' : sps.loguniform,
    'lomax' : sps.lomax,
    'maxwell' : sps.maxwell,
    'mielke' : sps.mielke,
    'moyal' : sps.moyal,
    'nakagami' : sps.nakagami,
    'ncx2' : sps.ncx2,
    'ncf' : sps.ncf,
    'nct' : sps.nct,
    'norm' : sps.norm,
    'norminvgauss' : sps.norminvgauss,
    'pareto' : sps.pareto,
    'pearson3' : sps.pearson3,
    'powerlaw' : sps.powerlaw,
    'powerlognorm' : sps.powerlognorm,
    'powernorm' : sps.powernorm,
    'rdist' : sps.rdist,
    'rayleigh' : sps.rayleigh,
    'rice' : sps.rice,
    'recipinvgauss' : sps.recipinvgauss,
    'semicircular' : sps.semicircular,
    'skewnorm' : sps.skewnorm,
    't' : sps.t,
    'trapz' : sps.trapz,
    'triang' : sps.triang,
    'truncexpon' : sps.truncexpon,
    'truncnorm' : sps.truncnorm,
    'tukeylambda' : sps.tukeylambda,
    'uniform' : sps.uniform,
    'vonmises' : sps.vonmises,
    'vonmises_line' : sps.vonmises_line,
    'wald' : sps.wald,
    'weibull_min' : sps.weibull_min,
    'weibull_max' : sps.weibull_max,
    'wrapcauchy' : sps.wrapcauchy,
    'bernoulli' : sps.bernoulli,
    'betabinom' : sps.betabinom,
    'binom' : sps.binom,
    'boltzmann' : sps.boltzmann,
    'dlaplace' : sps.dlaplace,
    'geom' : sps.geom,
    'hypergeom' : sps.hypergeom,
    'logser' : sps.logser,
    'nbinom' : sps.nbinom,
    'planck' : sps.planck,
    'poisson' : sps.poisson,
    'randint' : sps.randint,
    'skellam' : sps.skellam,
    'zipf' : sps.zipf,
    'yulesimon' : sps.yulesimon
}

def __get_bounds(function_name = None,steps = 200,*args):

    # define support
    x = np.linspace(0.0001,0.9999,steps)

    #get bound arguments
    new_args = itertools.product(*args)

    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:

        bounds.append(dists[function_name].ppf(x,*a))
        bmean, bvar = dists[function_name].stats(*a, moments = 'mv')

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

    var  = Interval(np.float64(var_lo),np.float64(var_hi))
    mean = Interval(np.float64(mean_lo),np.float64(mean_hi))

    Left = np.array(Left)
    Right = np.array(Right)

    return Left, Right, mean, var


def lognormal(mean, var, steps = 200):
    '''
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
        
    '''
    if steps > 1000:
        x = np.linspace(1/steps,1-1/steps,steps)
    else:
        x = np.linspace(0.001,0.999,steps)

    if mean.__class__.__name__ != 'Interval':
        mean = Interval(mean,mean)
    if var.__class__.__name__ != 'Interval':
        var = Interval(var,var)
    
    def __lognorm(mean, var):
    
        sigma = np.sqrt(np.log1p(var/mean**2))
        mu = np.log(mean) - 0.5*sigma*sigma
    
        return sps.lognorm(sigma, loc = 0,scale = np.exp(mu))
        
    bound0 = __lognorm(mean.left,var.left).ppf(x)
    bound1 = __lognorm(mean.right,var.left).ppf(x)
    bound2 = __lognorm(mean.left,var.right).ppf(x)
    bound3 = __lognorm(mean.right,var.right).ppf(x)

    Left = [min(bound0[i],bound1[i],bound2[i],bound3[i]) for i in range(steps)]
    Right = [max(bound0[i],bound1[i],bound2[i],bound3[i]) for i in range(steps)]

    Left = np.array(Left)
    Right = np.array(Right)
    return Pbox(
        Left,
        Right,
        steps = steps,
        shape='lognormal'
        )
    
    
lognorm = lognormal

def alpha(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('alpha',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'alpha',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def anglit(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('anglit',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'anglit',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def arcsine(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('arcsine',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'arcsine',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def argus(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('argus',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'argus',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def beta(*args, steps = 200):
    '''
    Beta distribution
    '''
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])
        if args[i].left == 0:
            args[i].left = 1e-5
        if args[i].right == 0:
            args[i].right = 1e-5
            
    Left, Right, mean, var = __get_bounds('beta',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'beta',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def betaprime(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('betaprime',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'betaprime',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def bradford(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('bradford',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'bradford',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def burr(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('burr',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'burr',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def burr12(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('burr12',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'burr12',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def cauchy(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('cauchy',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'cauchy',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def chi(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('chi',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'chi',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def chi2(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('chi2',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'chi2',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def cosine(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('cosine',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'cosine',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def crystalball(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('crystalball',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'crystalball',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def dgamma(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('dgamma',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'dgamma',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def dweibull(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('dweibull',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'dweibull',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def erlang(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('erlang',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'erlang',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def expon(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('expon',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'expon',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def exponnorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('exponnorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'exponnorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def exponweib(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('exponweib',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'exponweib',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def exponpow(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('exponpow',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'exponpow',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def f(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('f',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'f',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def fatiguelife(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('fatiguelife',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'fatiguelife',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def fisk(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('fisk',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'fisk',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def foldcauchy(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('foldcauchy',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'foldcauchy',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def foldnorm(mu,s, steps = 200):

    x = np.linspace(0.0001,0.9999,steps)
    if mu.__class__.__name__ != 'Interval':
        mu = Interval(mu)
    if s.__class__.__name__ != 'Interval':
        s = Interval(s)

    new_args = [
        [mu.lo()/s.lo(),0,s.lo()],
        [mu.hi()/s.lo(),0,s.lo()],
        [mu.lo()/s.hi(),0,s.hi()],
        [mu.hi()/s.hi(),0,s.hi()]
    ]


    bounds = []

    mean_hi = -np.inf
    mean_lo = np.inf
    var_lo = np.inf
    var_hi = 0

    for a in new_args:

        bounds.append(sps.foldnorm.ppf(x,*a))
        bmean, bvar = sps.foldnorm.stats(*a, moments = 'mv')

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

    var  = Interval(np.float64(var_lo),np.float64(var_hi))
    mean = Interval(np.float64(mean_lo),np.float64(mean_hi))

    Left = np.array(Left)
    Right = np.array(Right)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'foldnorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
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

def genlogistic(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('genlogistic',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'genlogistic',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gennorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gennorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gennorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def genpareto(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('genpareto',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'genpareto',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def genexpon(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('genexpon',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'genexpon',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def genextreme(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('genextreme',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'genextreme',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gausshyper(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gausshyper',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gausshyper',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gamma(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gamma',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gamma',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gengamma(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gengamma',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gengamma',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def genhalflogistic(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('genhalflogistic',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'genhalflogistic',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def geninvgauss(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('geninvgauss',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'geninvgauss',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gilbrat(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gilbrat',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gilbrat',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gompertz(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gompertz',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gompertz',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gumbel_r(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gumbel_r',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gumbel_r',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def gumbel_l(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('gumbel_l',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'gumbel_l',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def halfcauchy(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('halfcauchy',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'halfcauchy',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def halflogistic(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('halflogistic',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'halflogistic',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def halfnorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('halfnorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'halfnorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def halfgennorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('halfgennorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'halfgennorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def hypsecant(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('hypsecant',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'hypsecant',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def invgamma(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('invgamma',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'invgamma',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def invgauss(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('invgauss',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'invgauss',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def invweibull(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('invweibull',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'invweibull',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def johnsonsb(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('johnsonsb',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'johnsonsb',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def johnsonsu(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('johnsonsu',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'johnsonsu',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def kappa4(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('kappa4',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'kappa4',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def kappa3(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('kappa3',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'kappa3',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def ksone(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('ksone',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'ksone',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def kstwobign(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('kstwobign',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'kstwobign',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def laplace(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('laplace',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'laplace',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def levy(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('levy',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'levy',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def levy_l(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('levy_l',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'levy_l',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def levy_stable(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('levy_stable',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'levy_stable',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def logistic(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('logistic',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'logistic',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def loggamma(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('loggamma',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'loggamma',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def loglaplace(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('loglaplace',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'loglaplace',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def lognorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('lognorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'lognorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def loguniform(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('loguniform',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'loguniform',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def lomax(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('lomax',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'lomax',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def maxwell(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('maxwell',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'maxwell',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def mielke(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('mielke',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'mielke',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def moyal(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('moyal',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'moyal',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def nakagami(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('nakagami',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'nakagami',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def ncx2(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('ncx2',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'ncx2',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def ncf(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('ncf',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'ncf',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def nct(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('nct',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'nct',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def norm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('norm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'norm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def norminvgauss(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('norminvgauss',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'norminvgauss',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def pareto(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('pareto',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'pareto',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def pearson3(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('pearson3',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'pearson3',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def powerlaw(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('powerlaw',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'powerlaw',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def powerlognorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('powerlognorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'powerlognorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def powernorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('powernorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'powernorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def rdist(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('rdist',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'rdist',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def rayleigh(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('rayleigh',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'rayleigh',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def rice(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('rice',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'rice',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def recipinvgauss(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('recipinvgauss',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'recipinvgauss',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def semicircular(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('semicircular',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'semicircular',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def skewnorm(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('skewnorm',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'skewnorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def t(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('t',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 't',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def trapz(a,b,c,d , steps = 200):
    if a.__class__.__name__ != 'Interval':
        a = Interval(a)
    if b.__class__.__name__ != 'Interval':
        b = Interval(b)
    if c.__class__.__name__ != 'Interval':
        c = Interval(c)
    if d.__class__.__name__ != 'Interval':
        d = Interval(d)

    x = np.linspace(0.0001,0.9999,steps)
    left = sps.trapz.ppf(x,*sorted([b.lo()/d.lo(),c.lo()/d.lo(),a.lo(),d.lo()-a.lo()]))
    right = sps.trapz.ppf(x,*sorted([b.hi()/d.hi(),c.hi()/d.hi(),a.hi(),d.hi()-a.hi()]))

    return Pbox(
          left,
          right,
          steps      = steps,
          shape      = 'trapz'
          )

def triang(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('triang',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'triang',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def truncexpon(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('truncexpon',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'truncexpon',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def truncnorm(left,right,mean = None,stddev = None, steps = 200):
    
    if left.__class__.__name__ != 'Interval':
        left = Interval(left)
    if right.__class__.__name__ != 'Interval':
        right = Interval(right)
    if mean.__class__.__name__ != 'Interval':
        mean = Interval(mean)
    if stddev.__class__.__name__ != 'Interval':
        stddev = Interval(stddev)
    
    a,b = (left - mean)/stddev, (right - mean)/stddev
    

    Left, Right, mean, var = __get_bounds('truncnorm',steps,a,b,mean,stddev)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'truncnorm',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def tukeylambda(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('tukeylambda',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'tukeylambda',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )


def uniform(a, b, steps = 200):

    x = np.linspace(0,1,steps)

    if a.__class__.__name__ != 'Interval':
        a = Interval(a,a)
    if b.__class__.__name__ != 'Interval':
        b = Interval(b,b)

    Left = np.linspace(a.left,b.left)
    Right = np.linspace(a.right,b.right)

    mean = 0.5 * (a+b)
    var = ((b-a)**2 )/12

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'uniform',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def vonmises(*args, steps = Pbox.STEPS):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('vonmises',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'vonmises',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def vonmises_line(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('vonmises_line',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'vonmises_line',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def wald(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('wald',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'wald',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def weibull_min(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('weibull_min',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'weibull_min',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def weibull_max(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('weibull_max',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'weibull_max',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def wrapcauchy(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('wrapcauchy',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'wrapcauchy',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def bernoulli(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('bernoulli',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'bernoulli',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def betabinom(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('betabinom',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'betabinom',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def binom(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('binom',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'binom',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def boltzmann(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('boltzmann',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'boltzmann',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def dlaplace(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('dlaplace',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'dlaplace',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def geom(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('geom',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'geom',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def hypergeom(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('hypergeom',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'hypergeom',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def logser(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('logser',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'logser',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def nbinom(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('nbinom',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'nbinom',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def planck(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('planck',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'planck',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def poisson(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('poisson',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'poisson',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def randint(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('randint',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'randint',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def skellam(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('skellam',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'skellam',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def zipf(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('zipf',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'zipf',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )

def yulesimon(*args, steps = 200):
    args = list(args)
    for i in range(0,len(args)):
        if args[i].__class__.__name__ != 'Interval':
            args[i] = Interval(args[i])

    Left, Right, mean, var = __get_bounds('yulesimon',steps,*args)

    return Pbox(
          Left,
          Right,
          steps      = steps,
          shape      = 'yulesimon',
          mean_left  = mean.left,
          mean_right = mean.right,
          var_left   = var.left,
          var_right  = var.right
          )


### Other distributions
def KM(k,m,steps = 200):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return beta(Interval(k,k+1),Interval(m,m+1),steps = steps)

def KN(k,n,steps = 200):
    return KM(k,n-k,steps=steps)


### None-Distribution Pboxes 

def box(
        a: Union[Interval,float,int],
        b: Union[Interval,float,int] = None,
        steps = Pbox.STEPS
        ) -> Pbox:
    '''
    Returns Box interval
    
    
    Parameters
    ----------
        a :
            Left side of box
        b:
            Right side of box
    
    Returns
    ----------
        Pbox:
            p-box
        
    '''
    if b == None:
        b = a
    i = Interval(a,b)
    return Pbox(
        left = np.repeat(i.left,steps),
        right = np.repeat(i.right,steps),
        mean_left = i.left,
        mean_right= i.right,
        var_left = 0,
        var_right=((i.right-i.left)**2)/4,
        steps = steps
    )

min_max = box

def min_mean(
        minimum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        steps = Pbox.STEPS
    ) -> Pbox:
    jjj = [j/steps for j in range(1,steps-1)] + [1-1/steps]
    right = [((mean-minimum)/(1-j) + minimum) for j in jjj]
    return Pbox(
        left = np.repeat(minimum,steps),
        right = right
    )
def min_max_mean(    
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
    ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum and mean of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    mean : 
        mean value of the variable
    
    Returns
    ----------
    Pbox
    '''
    print(steps)
    mid = (maximum-mean)/(maximum-minimum)
    ii = [i/steps for i in range(steps)]
    left = [minimum if i <= mid else ((mean-maximum)/i + maximum) for i in ii]
    jj = [j/steps for j in range(1,steps+1)]
    right = [maximum if mid <= j else (mean - minimum * j) / (1 - j) for j in jj]
    print(len(left))
    return Pbox(
        left = np.array(left),
        right = np.array(right),
        steps = steps)


def min_max_mode(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        mode: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum, and mode of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    mode : 
        mode value of the variable
    
    Returns
    ----------
    Pbox

    '''
    if minimum == maximum:
        return box(minimum, maximum)

    ii = np.array([i/steps for i in range(steps)])
    jj = np.array([j/steps for j in range(1,steps+1)])
    
    return Pbox(
        left = ii * (mode - minimum) + minimum,
        right = jj * (maximum - mode) + mode,
        mean_left = (minimum+mode)/2,
        mean_right = (mode+maximum)/2,
        var_left = 0,
        var_right = (maximum-minimum)*(maximum-minimum)/12
    )


def min_max_median(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        median: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum and median of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    median : 
        median value of the variable
    
    Returns
    ----------
    Pbox

    '''
    if minimum == maximum:
        return box(minimum, maximum)

    ii = np.array([i/steps for i in range(steps)])
    jj = np.array([j/steps for j in range(1,steps+1)])
    
    
    return Pbox(
        left = np.array([p if p>0.5 else minimum for p in ii]),
        right = np.array([p if p<=0.5 else minimum for p in jj]),
        mean_left = (minimum+median)/2,
        mean_right = (median+maximum)/2,
        var_left = 0,
        var_right = (maximum-minimum)*(maximum-minimum)/4
    )


def min_max_mean_std(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        stddev: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum, mean and standard deviation of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    mean : 
        mean value of the variable
    stddev :
        standard deviation of the variable 
    
    Returns
    ----------
    Pbox

    '''
    if minimum == maximum:
        return box(minimum, maximum)
    def _left(x): 
        if type(x) in [int,float]:
            return x
        if x.__class__.__name__ == "Interval":
            return x.left
        if x.__class__.__name__ == "Pbox":
            return min(x.left)
        
    def _right(x): 
        if type(x) in [int,float]:
            return x
        if x.__class__.__name__ == "Interval":
            return x.right
        if x.__class__.__name__ == "Pbox":
            return max(x.right)           
           
    def _imp(a,b) : 
        return Interval(max(_left(a),_left(b)),min(_right(a),_right(b)))
    def _env(a,b) : 
        return Interval(min(_left(a),_left(b)),max(_right(a),_right(b)))    
    
    def _constrain(a, b, msg):
        if ((_right(a) < _left(b)) or (_right(b) < _left(a))) : 
            print("Math Problem: impossible constraint", msg)
        return _imp(a,b)
    
    zero = 0.0                          
    one = 1.0
    ran = maximum - minimum;
    m = _constrain(mean, Interval(minimum,maximum), "(mean)");
    s = _constrain(stddev, _env(Interval(0.0),(abs(ran*ran/4.0 - (maximum-mean-ran/2.0)**2))**0.5)," (dispersion)")
    ml = (m.left-minimum)/ran
    sl = s.left/ran
    mr = (m.right-minimum)/ran
    sr = s.right/ran
    z = box(minimum, maximum)
    n  = len(z.left)
    L = [0.0] * n
    R = [1.0] * n
    for i in range(n) :
        p = i / n
        if (p <= zero) : 
            x2 = zero
        else : x2 = ml - sr * (one / p - one)**0.5
        if (ml + p <= one) :
            x3 = zero
        else : 
            x5 = p*p + sl*sl - p
            if (x5 >= zero) :                  
                      x4 = one - p + x5**0.5
                      if (x4 < ml) : x4 = ml
            else : x4 = ml
            x3 = (p + sl*sl + x4*x4 - one) / (x4 + p - one)
        if ((p <= zero) or (p <= (one - ml))) : x6 = zero
        else : x6 = (ml - one) / p + one
        L[i] = max(max(max(x2,x3),x6),zero) * ran + minimum;
    
        p = (i+1)/n
        if (p >= one) : x2 = one
        else : x2 = mr + sr * (one/(one/p - one))**0.5
        if (mr + p >= one) : x3 = one
        else :
               x5 = p*p + sl*sl - p
               if (x5 >= zero) :                  
                      x4 = one - p - x5**0.5
                      if (x4 > mr) : x4 = mr                  
               else : x4 = mr 
               x3 = (p + sl*sl + x4*x4 - one) / (x4 + p - one) - one
             
        if (((one - mr) <= p) or (one <= p)) : x6 = one
        else : x6 = mr / (one - p)
        R[i] = min(min(min(x2,x3),x6),one) * ran + minimum
  
    v = s**2
    return Pbox(
        left = np.array(L),
        right = np.array(R),
        mean_left = _left(m),
        mean_right = _right(m),
        var_left = _left(v),
        var_right = _right(v),
        steps = steps)

def min_max_mean_var(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        var: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum, mean and standard deviation of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    mean : 
        mean value of the variable
    var :
        variance of the variable 
    
    Returns
    ----------
    Pbox

    '''
    return min_max_mean_std(minimum,maximum,mean,np.sqrt(var))

def mmms(*args):
    raise Exception('Depreciated use: min_max_mean_std(*args)')


### Alternate names
normal = norm
N = normal
unif = uniform
U = uniform

### ML-ME
def MLnorm(data): 
    return norm(np.mean(data),np.std(data))

def ME_min_max_mean_std(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        stddev: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    
    μ = ((mean- minimum) / (maximum - minimum))
    
    σ = (stddev/(maximum - minimum) )
    
    a = ((1-μ)/(σ**2) - 1/μ)*μ**2
    b = a*(1/μ - 1)
    
    return beta(a,b,steps=steps)* (maximum - minimum) + minimum

def betapert(minimum, maximum, mode):
    mu = (minimum + maximum + 4*mode)/6
    alpha1 = (mu - minimum)*(2*mode - minimum - maximum)/((mode - mu)*(maximum - minimum))
    alpha2 = alpha1*(maximum - mu)/(mu - minimum)
    return minimum + (maximum - minimum) * beta(alpha1, alpha2)