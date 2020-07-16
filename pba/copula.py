###
#   Defines copula functions. For plotting and use in convolutions
#
#   To Do:  
#           -> Optimise gaussian copula. Currently requires multiple calls to mutivariate normal cdf, also produces NaNs at extremes
#           -> Clayton and Frank copulas should be defined in terms of generator functions
#           -> Gaussian, Clayton and Frank should return π, M and W at particular values
#
#           -> Once completed, the Sigma, Tau and Rho convolutions may be defined
#
#
#       By: Ander Gray, University of Liverpool, ander.gray@liverpool.ac.uk
###

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .interval import *
from scipy.stats import multivariate_normal, norm 

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

    def get_cdf(self, x, y):    # x, y are points on the unit square

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

    def get_mass(self, x, y):   # x, y are intervals on the unit square

        C22 = self.get_cdf(x.hi(), y.hi())
        C21 = self.get_cdf(x.hi(), y.lo())
        C12 = self.get_cdf(x.lo(), y.hi())
        C11 = self.get_cdf(x.lo(), y.lo())

        return C22 - C21 - C12 + C11

    def show(self, pn = 50, fontsize = 20, cols = cm.RdGy):
        ##
        #   All the extra stuff is so that no more than 200 elements are plotted
        ##
        A = self.cdf; m = len(A)

        if m < pn: pn = m

        x = y = np.linspace(0, 1, num = pn)
        X, Y = np.meshgrid(x,y)

        nm = round(m/pn)
        Z = A[::nm,::nm]    # Skip over evelemts

        fig = plt.figure("SurfacPlots",figsize=(10,10))
        ax = fig.add_subplot(1,1,1,projection="3d")
        ax.plot_surface(X, Y, Z, rstride=2,edgecolors="k", cstride=2, alpha=0.8, linewidth=0.25, cmap = cols)
        plt.xlabel("X",fontsize = fontsize); plt.ylabel("Y", fontsize = fontsize)

        plt.show()

    def showContour(self, fontsize = 20, cols = cm.coolwarm):
        ##
        #   All the extra stuff is so that no more than 200 elements are plotted
        ##

        pn = 200    # Max plot number
        A = self.cdf; m = len(A)

        if m < pn: pn = m

        x = y = np.linspace(0, 1, num = pn)
        X, Y = np.meshgrid(x,y)

        nm = round(m/pn)
        Z = A[::nm,::nm]    # Skip over evelemts

        fig = plt.figure("SurfacPlots",figsize=(10,10))
        ax = fig.add_subplot(2,1,1,projection="3d")
        ax.plot_surface(X, Y, Z, rstride=2,edgecolors="k", cstride=2, alpha=0.8, linewidth=0.25, cmap = cols)
        plt.xlabel("X",fontsize = fontsize); plt.ylabel("Y", fontsize = fontsize)
        plt.title("Surface Plot", fontsize = fontsize)

        ax1 = fig.add_subplot(2,1,2)
        cp = ax1.contour(X, Y, Z, cmap = cols, levels = 15)
        ax1.clabel(cp, inline=1, fontsize=10)
        plt.xlabel("X", fontsize = fontsize); plt.ylabel("Y", fontsize = fontsize)
        plt.title("Contour Plot", fontsize = fontsize)

        plt.tight_layout()

        plt.show()



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
