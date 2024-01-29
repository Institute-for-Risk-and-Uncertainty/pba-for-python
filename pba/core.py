from typing import *
from .interval import *
from .pbox import Pbox
# from .cbox import *

import numpy as np
import warnings

def envelope(*args: Union[Interval, Pbox, float]) -> Union[Interval, Pbox]:
    '''
    Allows the envelope to be calculated for intervals and p-boxes.
    
    The envelope is the smallest interval/pbox that contains all values within the arguments.
    
    Parameters:
        The arguments for which the envelope needs to be calculated. The arguments can be intervals, p-boxes, or floats.
    
    Returns:
        The envelope of the given arguments, which can be an interval or a p-box.
    
    Raises:
        ValueError: If less than two arguments are given.
        ValueError: If none of the arguments are intervals or p-boxes.    
    '''
    # args = list(*args)
    
    # Raise error if <2 arguments are given
    if len(args) < 2:
        raise ValueError('At least two arguments are required')
    
    # get the type of all arguments
    types = [arg.__class__.__name__ for arg in args]
    
    # check if all arguments are intervals or pboxes
    if 'Interval' not in types and 'Pbox' not in types:
        raise ValueError('At least one argument needs to be an Interval or Pbox')
    # check if there is a p-box in the arguments
    elif 'Pbox' in types:
        # find first p-box
        i = types.index('Pbox')
        # move previous values to the end
        args = args[i:] + args[:i]
    else: #Intervals only
        # find first interval
        i = types.index('Interval')
        # move previous values to the end
        args = args[i:] + args[:i]
        
    e = args[0].env(args[1])
    for arg in args[2:]:
        e = e.env(arg)
    
    return e

def env(*args):
    '''
    Deprecated function, use envelope() instead.
    '''
    warnings.warn('env() is deprecated, use envelope() instead', DeprecationWarning)
    return envelope(*args)

def min(x,y):
    if x.__class__.__name__ == 'Pbox':
        return x.min(y)
    if y.__class__.__name__ == 'Pbox':
        return y.min(x)
    else:
        raise NotImplementedError('At least one argument needs to be a Pbox')

def max(x,y):
    if x.__class__.__name__ == 'Pbox':
        return x.max(y)
    if y.__class__.__name__ == 'Pbox':
        return y.max(x)
    else:
        raise NotImplementedError('At least one argument needs to be a Pbox')

def sum(l: Union[list,tuple] ,method = 'f'):
    '''
    Allows the sum to be calculated for intervals and p-boxes
    
    Parameters
    ----------
        l : list of pboxes or intervals
        method : pbox addition method to be used
    
    Output
    ------
        sum of interval or pbox objects within l
    
    '''
    s = 0
    for o in l:
        if o.__class__.__name__ == 'Pbox':
            s = o.add(s,method = method)
        else:
            s += o
    return s

def mean(l: Union[list,tuple] ,method = 'f'):
    '''    
    Allows the mean to be calculated for intervals and p-boxes
    
    Parameters
    ----------
        l : list of pboxes or intervals
        method : pbox addition method to be used
    
    Output
    ------
        mean of interval or pbox objects within l
    
    '''
    s = sum(l,method = method)
    
    return s/len(l)

def mul(*args, method = None):
    for i,arg in enumerate(args):
        if i == 0:
            n = arg
        elif n.__class__.__name__ == 'Interval':
            if arg.__class__.__name__ == 'Interval':
                if method is None:
                    n *= arg
                elif method == 'p':
                    n = n.pmul(arg)
                elif method == 'o':
                    n = n.omul(arg)
                else:
                    raise Exception(f"Method {method} unknown for Interval * Interval calculation")
            elif arg.__class__.__name__ == 'Pbox':
                n = arg.mul(n,method = method)
            else:
                n *= arg
        elif n.__class__.__name__ == 'Pbox':
            if method is None:
                n *= arg
            else:
                n = n.mul(arg,method = method)
        else:
            n *= arg
    return n
                
def sqrt(a):
    if a.__class__.__name__ == 'Interval':
        return Interval(np.sqrt(a.left),np.sqrt(a.right))
    elif a.__class__.__name__ == 'PBox':
        return Pbox(
            left = np.sqrt(a.left),
            right = np.sqrt(a.right),
        )
    else: 

       return np.sqrt(a)
