###
#   Defines copula functions. For plotting and use in convolutions
#
#   To Do:  
#           -> Optimise gaussian copula. Currently requires multiple calls to mutivariate normal cdf, also produces NaNs at extremes
#           -> Clayton and Frank copulas should be defined in terms of generator functions
#           -> Gaussian, Clayton and Frank should return π, M and W at particular values
#           -> Density estimator with H-volume. Matlab implementation: https://github.com/AnderGray/Hvolume-Matlab
#           -> Copula plotter
#
#           -> Once completed, the Sigma, Tau and Rho convolutions may be defined
#
#
#       By: Ander Gray, University of Liverpool, ander.gray@liverpool.ac.uk
###

import numpy as np
from .interval import *
from scipy.stats import multivariate_normal, norm 
#import copulae as cops

class Copula(object):

    def __init__(self, cdf=None, func=None, param = None):

        self.cdf = cdf          # Could also include density, however expensive to compute
        self.func = func        # If the functional form is known
        self.param = param      # parameter for func

    def __repr__(self):

        statement1 = "Arbitrary"
        statement2 = ""

        if self.func is not None:
            func = self.func
            if func == indep: func = "π"
            if func == perf: func = "M"
            if func == opp: func = "W"
            if func == Gau: func = "Gau"
            if func == Cla: func = "Clayton"
            if func == F: func = "Frank"
            statement1 = f'{func}'

        if self.param is not None:
            func = self.func
            parName = "par"
            if func == Gau: parName = "r"
            if func == F: parName = "s"
            if func == Cla: parName = "t"
            statement2 = f'{parName}={self.param}'   
        
        return f'Copula ~ {statement1}({statement2})'

    def getcdf(self, x,y):

        if self.func is not None:   # If function is known, return it 
            if self.param is not None: return self.func(x, y, self.param)
            return self.func(x,y)

        else:   # Simple inner/outter interpolation. Returns interval for cdf value
            xsize, ysize = self.cdf.shape

            xIndexLower = int(np.floor(x * (xsize-1)))
            yIndexLower = int(np.floor(y * (ysize-1)))

            xIndexUpper = int(np.ceil(x * (xsize-1)))
            yIndexUpper = int(np.ceil(y * (ysize-1)))

            return Interval(self.cdf[xIndexLower, yIndexLower], self.cdf[xIndexUpper, yIndexUpper])

###
#   Copula functions and constructors
###
def indep(x,y): return x*y
def perf(x,y): return min(x,y)
def opp(x,y): return max(x+y-1,0)
def F(x,y,s = 1): return np.log(1+(s^x-1)*(s^y-1)/(s-1))/np.log(s)      # Bugged
def Cla(x,y, t = 0): return max((x^(-t)+y^(-t)-1)^(-1/t),0)             # Bugged
def Gau(x,y,r=0): return multivariate_normal.cdf([norm.ppf(x), norm.ppf(y)], mean = [0, 0], cov=r) # Previous implementation


# Copula constructors
def pi(steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[xs * ys for xs in x] for ys in y])
    return Copula(cdf, indep)

def M(steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[perf(xs, ys) for xs in x] for ys in y])
    return Copula(cdf, perf)

def W(steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[opp(xs, ys) for xs in x] for ys in y])
    return Copula(cdf, opp)

def Frank(s = 1, steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[F(xs, ys, s) for xs in x] for ys in y])
    return Copula(cdf, F, s)

def Clayton(t = 0, steps = 200):        
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[Cla(xs, ys) for xs in x] for ys in y])
    return Copula(cdf, Cla, t)

def Gaussian(r = 0, steps = 200):       # Slow, and produced NaNs
    x = y = np.linspace(0, 1, num=steps)
    cdf = np.array([[Gau(xs, ys,r) for xs in x] for ys in y])
    return Copula(cdf, Gau, r)

'''
def Gaussian(r = 0, steps = 200):
    x = y = np.linspace(0, 1, num=steps)
    g_cop = cops.GaussianCopula()
    g_cop.params = r
    cdf = np.array([[g_cop.cdf([xs, ys]) for xs in x] for ys in y])
    return Copula(cdf, g_cop, r)
'''