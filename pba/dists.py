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
    'frechet_r' : sps.frechet_r,
    'frechet_l' : sps.frechet_l,
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

def __get_bounds(function_name = None,steps = 200,*args,):

    x = np.linspace(0.0001,0.9999,steps)

    arg_list = [list(args)]
    complete = False
    # while not complete:
    for i in arg_list:
        for j in range(len(i)):
            if i[j].__class__.__name__ == 'Interval':

                k1 = [i[j].left() if l == i[j] else l for l in i]
                k2 = [i[j].right() if l == i[j] else l for l in i]

                arg_list.append(k1)
                arg_list.append(k2)

    new_args = []
    for k in arg_list:
        to_delete = False

        for l in k:

            if l.__class__.__name__ == 'Interval':
                to_delete = True

        if not to_delete and k not in new_args:
            new_args.append(k)

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


    LowerBound = [min([b[i] for b in bounds]) for i in range(steps)]
    UpperBound = [max([b[i] for b in bounds]) for i in range(steps)]

    var  = Interval(np.float64(var_lo),np.float64(var_hi))
    mean = Interval(np.float64(mean_lo),np.float64(mean_hi))

    LowerBound = np.array(LowerBound)
    UpperBound = np.array(UpperBound)

    return LowerBound, UpperBound, mean, var

def beta(a,b, steps=200):

    def make_bound(x,a,b):
        bound = sps.beta.ppf(x, a, b)
        mean, var = sps.beta.stats(a,b, moments = 'mv')
        return bound, mean, var

    x = np.linspace(0,1,steps)

    if a.__class__.__name__ != 'Interval':
        a = Interval(a)
    if b.__class__.__name__ != 'Interval':
        b = Interval(b)

    LowerBound, LowerMean, LowerVar = make_bound(x, a.left(), b.right())
    UpperBound, UpperMean, UpperVar = make_bound(x, a.right(), b.left())

    mean_left = min(LowerMean, UpperMean)
    var_left  = min(LowerVar, UpperVar)

    mean_right = max(LowerMean, UpperMean)
    var_right  = max(LowerVar, UpperVar)

    return Pbox(LowerBound, UpperBound, steps = steps, shape=None, mean_left=mean_left, mean_right=mean_right, var_left=var_left, var_right=var_right)

def KM(k,m):
    return beta(I(k,k+1),I(m,m+1))

def KN(k,n):
    assert k<n ['%i out of %i does not make sense. Did you mean to use KM?' %(k,n)]
    return KM(k,n-k)


def normal(mean, std, steps = 200):

    x = np.linspace(0.0001,0.9999,steps)

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

    return Pbox(
        LowerBound,
        UpperBound,
        steps = steps,
        shape='normal',
        mean_left=mean.left(),
        mean_right=mean.right(),
        var_left=std.left() ** 2,
        var_right=std.right() ** 2)
norm = normal
N = normal

def normal2(mean, std, steps = 200):
    if mean.__class__.__name__ != 'Interval':
        mean = Interval(mean,mean)
    if std.__class__.__name__ != 'Interval':
        std = Interval(std,std)

    LowerBound, UpperBound, mean, var = __get_bounds('norm',steps,mean,std)

    return Pbox(
        LowerBound,
        UpperBound,
        steps = steps,
        shape='normal',
        mean_left=mean.left(),
        mean_right=mean.right(),
        var_left=var.left(),
        var_right=var.right())


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

def lognormal(mean, var, steps = 200):

    x = np.linspace(0,1,steps)

    if mean.__class__.__name__ != 'Interval':
        mean = Interval(mean,mean)
    if var.__class__.__name__ != 'Interval':
        var = Interval(var,var)

    bound0 = sps.lognorm.ppf(x, mean.left(), var.left())
    bound1 = sps.lognorm.ppf(x, mean.right(), var.left())
    bound2 = sps.lognorm.ppf(x, mean.left(), var.right())
    bound3 = sps.lognorm.ppf(x, mean.right(), var.right())

    LowerBound = [min(bound0[i],bound1[i],bound2[i],bound3[i]) for i in range(steps)]
    UpperBound = [max(bound0[i],bound1[i],bound2[i],bound3[i]) for i in range(steps)]

    LowerBound = np.array(LowerBound)
    UpperBound = np.array(UpperBound)
    return Pbox(
        LowerBound,
        UpperBound,
        steps = steps,
        shape='normal',
        mean_left=mean.left(),
        mean_right=mean.right(),
        var_left=var.left(),
        var_right=var.right())

lognorm = lognormal

# def alpha(a, steps = 200):

def alpha(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('alpha',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'alpha',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def anglit(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('anglit',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'anglit',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def arcsine(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('arcsine',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'arcsine',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def argus(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('argus',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'argus',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def beta(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('beta',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'beta',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def betaprime(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('betaprime',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'betaprime',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def bradford(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('bradford',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'bradford',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def burr(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('burr',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'burr',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def burr12(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('burr12',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'burr12',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def cauchy(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('cauchy',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'cauchy',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def chi(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('chi',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'chi',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def chi2(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('chi2',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'chi2',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def cosine(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('cosine',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'cosine',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def crystalball(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('crystalball',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'crystalball',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def dgamma(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('dgamma',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'dgamma',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def dweibull(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('dweibull',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'dweibull',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def erlang(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('erlang',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'erlang',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def expon(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('expon',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'expon',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def exponnorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('exponnorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'exponnorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def exponweib(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('exponweib',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'exponweib',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def exponpow(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('exponpow',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'exponpow',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def f(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('f',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'f',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def fatiguelife(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('fatiguelife',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'fatiguelife',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def fisk(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('fisk',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'fisk',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def foldcauchy(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('foldcauchy',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'foldcauchy',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def foldnorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('foldnorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'foldnorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def frechet_r(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('frechet_r',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'frechet_r',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def frechet_l(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('frechet_l',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'frechet_l',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def genlogistic(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('genlogistic',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'genlogistic',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gennorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gennorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gennorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def genpareto(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('genpareto',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'genpareto',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def genexpon(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('genexpon',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'genexpon',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def genextreme(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('genextreme',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'genextreme',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gausshyper(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gausshyper',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gausshyper',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gamma(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gamma',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gamma',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gengamma(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gengamma',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gengamma',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def genhalflogistic(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('genhalflogistic',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'genhalflogistic',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def geninvgauss(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('geninvgauss',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'geninvgauss',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gilbrat(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gilbrat',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gilbrat',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gompertz(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gompertz',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gompertz',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gumbel_r(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gumbel_r',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gumbel_r',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def gumbel_l(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('gumbel_l',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'gumbel_l',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def halfcauchy(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('halfcauchy',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'halfcauchy',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def halflogistic(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('halflogistic',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'halflogistic',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def halfnorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('halfnorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'halfnorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def halfgennorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('halfgennorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'halfgennorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def hypsecant(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('hypsecant',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'hypsecant',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def invgamma(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('invgamma',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'invgamma',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def invgauss(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('invgauss',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'invgauss',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def invweibull(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('invweibull',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'invweibull',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def johnsonsb(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('johnsonsb',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'johnsonsb',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def johnsonsu(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('johnsonsu',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'johnsonsu',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def kappa4(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('kappa4',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'kappa4',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def kappa3(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('kappa3',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'kappa3',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def ksone(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('ksone',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'ksone',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def kstwobign(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('kstwobign',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'kstwobign',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def laplace(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('laplace',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'laplace',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def levy(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('levy',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'levy',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def levy_l(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('levy_l',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'levy_l',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def levy_stable(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('levy_stable',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'levy_stable',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def logistic(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('logistic',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'logistic',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def loggamma(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('loggamma',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'loggamma',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def loglaplace(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('loglaplace',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'loglaplace',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def lognorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('lognorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'lognorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def loguniform(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('loguniform',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'loguniform',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def lomax(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('lomax',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'lomax',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def maxwell(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('maxwell',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'maxwell',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def mielke(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('mielke',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'mielke',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def moyal(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('moyal',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'moyal',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def nakagami(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('nakagami',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'nakagami',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def ncx2(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('ncx2',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'ncx2',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def ncf(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('ncf',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'ncf',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def nct(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('nct',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'nct',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def norm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('norm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'norm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def norminvgauss(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('norminvgauss',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'norminvgauss',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def pareto(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('pareto',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'pareto',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def pearson3(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('pearson3',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'pearson3',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def powerlaw(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('powerlaw',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'powerlaw',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def powerlognorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('powerlognorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'powerlognorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def powernorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('powernorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'powernorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def rdist(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('rdist',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'rdist',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def rayleigh(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('rayleigh',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'rayleigh',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def rice(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('rice',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'rice',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def recipinvgauss(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('recipinvgauss',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'recipinvgauss',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def semicircular(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('semicircular',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'semicircular',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def skewnorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('skewnorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'skewnorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def t(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('t',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 't',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def trapz(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('trapz',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'trapz',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def triang(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('triang',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'triang',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def truncexpon(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('truncexpon',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'truncexpon',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def truncnorm(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('truncnorm',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'truncnorm',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def tukeylambda(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('tukeylambda',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'tukeylambda',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def uniform(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('uniform',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'uniform',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def vonmises(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('vonmises',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'vonmises',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def vonmises_line(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('vonmises_line',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'vonmises_line',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def wald(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('wald',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'wald',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def weibull_min(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('weibull_min',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'weibull_min',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def weibull_max(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('weibull_max',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'weibull_max',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def wrapcauchy(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('wrapcauchy',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'wrapcauchy',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def bernoulli(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('bernoulli',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'bernoulli',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def betabinom(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('betabinom',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'betabinom',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def binom(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('binom',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'binom',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def boltzmann(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('boltzmann',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'boltzmann',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def dlaplace(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('dlaplace',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'dlaplace',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def geom(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('geom',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'geom',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def hypergeom(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('hypergeom',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'hypergeom',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def logser(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('logser',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'logser',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def nbinom(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('nbinom',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'nbinom',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def planck(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('planck',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'planck',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def poisson(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('poisson',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'poisson',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def randint(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('randint',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'randint',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def skellam(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('skellam',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'skellam',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def zipf(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('zipf',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'zipf',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )

def yulesimon(*args, steps = 200):
    for arg in args:
        if arg.__class__.__name__ != 'Interval':
            arg = Interval(arg)

    LowerBound, UpperBound, mean, var = __get_bounds('yulesimon',steps,*args)

    return Pbox(
          LowerBound,
          UpperBound,
          steps      = steps,
          shape      = 'yulesimon',
          mean_left  = mean.left(),
          mean_right = mean.right(),
          var_left   = var.left(),
          var_right  = var.right()
          )
