'''
Non-parametric p-box generators
-------------------------------
'''
__all__ = [
    'what_I_know',
    'box',
    'min_max',
    'min_max_mean',
    'min_max_mean_std',
    'min_max_mean_var',
    'min_max_mode',
    'min_max_median',
    'min_max_median_is_mode',
    'mean_std',
    'mean_var',
    'pos_mean_std',
    'min_max_percentile',
    'symmetric_mean_std'
    ]

from .pbox import Pbox, imposition
from .interval import Interval
from .dists import *
from typing import *
import numpy as np

def what_I_know(
        minimum: Optional[Union[Interval,float,int]] = None,
        maximum: Optional[Union[Interval,float,int]] = None,
        mean: Optional[Union[Interval,float,int]] = None,
        median: Optional[Union[Interval,float,int]] = None,
        mode: Optional[Union[Interval,float,int]] = None,
        std: Optional[Union[Interval,float,int]] = None,
        var: Optional[Union[Interval,float,int]] = None,
        cv: Optional[Union[Interval,float,int]] = None,
        # percentiles: Optional[Union[Interval,float,int]] = None,
        # coverages: Optional[Union[Interval,float,int]] = None,
        # shape: Optional[Literal['unimodal', 'symmetric', 'positive', 'nonnegative', 'concave', 'convex', 'increasinghazard', 'decreasinghazard', 'discrete', 'integervalued', 'continuous', '', 'normal', 'lognormal']] = None,
        # data: Optional[list] = None,
        # confidence: Optional[float] = 0.95,
        debug: bool = False,
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution free p-box based upon the information given. This function works by calculating every possible non-parametric p-box that can be generated using the information provided. The returned p-box is the imposition of these p-boxes.
    
    Parameters
    ----------
        minimum
        maximum
        mean
        median
        mode
        std
        var
        cv
        percentiles
        
    Returns
    ----------
        Pbox:
            Imposition of possible p-boxes
    
    '''
    
    def _print_debug(skk): print("\033[93m {}\033[00m" .format(skk),end=' ')
    
    def _get_pbox(func,*args,steps = steps,debug=False): 
        if debug: _print_debug(func.__name__)
        try: 
            return func(*args,steps=steps)
        except:
            raise Exception(f'Unable to generate {func.__name__} pbox')
    
    # if 'positive' in shape:
    #     if minimum is None:
    #         minimum = 0
    #     else:
    #         minimum = max(0,minimum)
            
    #     if debug: _print_debug("Shape is positive")
        
    # if 'negative' in shape:
    #     if maximum is None:
    #         maximum = 0
    #     else:
    #         maximum = min(0,maximum)

    #     if debug: _print_debug("Shape is negative")
        
    imp = []
    
    if minimum is not None and maximum is not None: imp += _get_pbox(min_max,minimum,maximum,debug=debug)
        
    if minimum is not None and mean is not None: imp += _get_pbox(min_mean,minimum,mean,debug=debug)
    
    if maximum is not None and mean is not None: imp += _get_pbox(max_mean,maximum,mean,debug=debug)
    
    if minimum is not None and maximum is not None and mean is not None: imp += _get_pbox(min_max_mean,minimum,maximum,mean,debug=debug)
    
    if minimum is not None and maximum is not None and mode is not None: imp += _get_pbox(min_max_mode,minimum,maximum,mode,debug=debug)
    
    if minimum is not None and maximum is not None and median is not None: imp += _get_pbox(min_max_median,minimum,maximum,median,debug=debug)
    
    if minimum is not None and mean is not None and std is not None: imp += minimum+_get_pbox(pos_mean_std,mean-minimum,std,debug=debug)
    
    if maximum is not None and mean is not None and std is not None: imp += _get_pbox(pos_mean_std,maximum-mean,std,debug=debug) - maximum
    
    if minimum is not None and maximum is not None and mean is not None and std is not None: imp += _get_pbox(min_max_mean_std,minimum,maximum,mean,std,debug=debug) 
    
    if minimum is not None and maximum is not None and mean is not None and var is not None: imp += _get_pbox(min_max_mean_var,minimum,maximum,mean,var,debug=debug) 
    
    if mean is not None and std is not None: imp += _get_pbox(mean_std,mean,std,debug=debug) 
    
    if mean is not None and var is not None: imp += _get_pbox(mean_var,mean,var,debug=debug) 
    
    if mean is not None and cv is not None: imp += _get_pbox(mean_std,mean,cv*mean,debug=debug) 

    if len(imp) == 0:
        raise Exception("No valid p-boxes found")
    return imposition(imp)
    
def box(
        a: Union[Interval,float,int],
        b: Union[Interval,float,int] = None,
        steps = Pbox.STEPS
        ) -> Pbox:
    '''
    Returns Box Pbox
    
    
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

def min_mean(
        minimum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        steps = Pbox.STEPS
    ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum and mean of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    mean : 
        mean value of the variable
    
    Returns
    ----------
    Pbox
    '''
    jjj = np.array([j/steps for j in range(1,steps-1)] + [1-1/steps])

    right = [((mean-minimum)/(1-j) + minimum) for j in jjj]
    return Pbox(
        left = np.repeat(minimum,steps),
        right = right,
        mean_left = mean,
        mean_right = mean,
        steps=steps
    )

def max_mean(
        maximum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        steps = Pbox.STEPS
    ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum and mean of the variable
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    mean : 
        mean value of the variable
    
    Returns
    ----------
    Pbox
    '''
    return min_mean(-maximum,-mean).__neg__()
    


def mean_std(
        mean: Union[Interval,float,int],
        std: Union[Interval,float,int],
        steps = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the mean and standard deviation of the variable
 
    Parameters
    ----------
    mean : 
        mean of the variable
    std : 
        standard deviation of the variable
    
    Returns
    ----------
    Pbox

    '''
    iii = [1/steps] + [i/steps for i in range(1,steps-1)]
    jjj = [j/steps for j in range(1,steps-1)] + [1-1/steps]
    
    left = [mean - std * np.sqrt(1/i - 1) for i in iii] 
    right = [mean + std * np.sqrt(j / (1 - j)) for j in jjj]
    
    return Pbox(
        left = left,
        right = right,
        steps = steps,
        mean_left = mean,
        mean_right =mean,
        var_left = std**2,
        var_right = std**2
    )

def mean_var(
        mean: Union[Interval,float,int],
        var: Union[Interval,float,int],
        steps = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the mean and variance of the variable
    
    Equivalent to `mean_std(mean,np.sqrt(var))`
 
    Parameters
    ----------
    mean : 
        mean of the variable
    var : 
        variance of the variable
    
    Returns
    ----------
    Pbox

    '''
    return mean_std(mean,np.sqrt(var),steps)

def pos_mean_std(
        mean: Union[Interval,float,int],
        std: Union[Interval,float,int],
        steps = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a positive distribution-free p-box based upon the mean and standard deviation of the variable
 
    Parameters
    ----------
    mean : 
        mean of the variable
    std : 
        standard deviation of the variable
    
    Returns
    ----------
    Pbox

    '''
    iii = [1/steps] + [i/steps for i in range(1,steps-1)]
    jjj = [j/steps for j in range(1,steps-1)] + [1-1/steps]
    
    left = [max((0,mean - std * np.sqrt(1/i - 1))) for i in iii] 
    right = [min((mean/(1-j), mean + std * np.sqrt(j / (1 - j)))) for j in jjj]
    
    return Pbox(
        left = left,
        right = right,
        steps = steps,
        mean_left = mean,
        mean_right =mean,
        var_left = std**2,
        var_right = std**2
    )

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

def min_max_median_is_mode(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        m: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum and median/mode of the variable when median = mode.
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    m : 
        m = median = mode value of the variable
    
    Returns
    ----------
    Pbox

    '''
    ii = np.array([i/steps for i in range(steps)])
    jjj = [j/steps for j in range(1,steps-1)] + [1-1/steps]
    
    u = [p*2*(m-minimum) + minimum if p <= 0.5 else m for p in ii]
    
    d = [(p-0.5)*2*(maximum - m) + m if p > 0.5 else m for p in jjj]

    
    return Pbox(
        left = u,
        right = d,
        mean_left=(minimum + 3 + m)/4,
        mean_right=(3*m +maximum)/4,
        var_left=0,
        var_right=(maximum - minimum)*(maximum - minimum)/4,
    )

def min_max_percentile(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        fraction: Union[list,float,int], 
        percentile: Union[list,float],
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a distribution-free p-box based upon the minimum, maximum and a percentile of the distribution.
 
    Parameters
    ----------
    minimum : 
        minimum value of the variable
    maximum : 
        maximum value of the variable
    fraction : 
        fraction that the percentile(s) is/are at
    percentile :
        the percentile(s) at fraction
    
    Returns
    ----------
    Pbox

    '''
    ii = [i/steps for i in range(steps)]
    jjj = [j/steps for j in range(1,steps-1)] + [1-1/steps]
    
    if hasattr(fraction,'__iter__'):
        
        if len(fraction) != len(percentile):
            raise Exception('fraction and percentile must be the same length')

        fraction = np.array([0] + list(fraction) + [1])
        percentile = np.array([minimum] + list(percentile) + [maximum])
        
        u = [percentile[fraction<=x][-1] for x in ii]
        d = [percentile[fraction>=x][0] for x in jjj]
        
    else:
        u = [minimum if x <= fraction else percentile for x in ii]
        d = [maximum if x > fraction else percentile for x in jjj]
    
    return Pbox(
        left = u,
        right = d
    )

def symmetric_mean_std(
        mean: Union[Interval,float,int], 
        std: Union[Interval,float,int], 
        steps: int = Pbox.STEPS
        ) -> Pbox:
    '''
    Generates a symmetrix distribution-free p-box based upon the mean and standard deviation of the variable
 
    Parameters
    ----------
    mean : 
        mean value of the variable
    std :
        standard deviation of the variable 
    
    Returns
    ----------
    Pbox
    '''
    iii = [1/steps] + [i/steps for i in range(1,steps-1)]
    jjj = [j/steps for j in range(1,steps-1)] + [1-1/steps]
    
    u = [mean - std / np.sqrt(2 * p) if p <= 0.5 else mean for p in iii]
    d = [mean + std / np.sqrt(2 * (1 - p)) if p > 0.5 else mean for p in jjj]
    
    return Pbox(
        left = u,
        right = d,
        mean_left=mean,
        mean_right=mean,
        var_left=std**2,
        var_right=std**2
    )
    


def min_max_mean_std(
        minimum: Union[Interval,float,int], 
        maximum: Union[Interval,float,int], 
        mean: Union[Interval,float,int], 
        std: Union[Interval,float,int], 
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
    std :
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
    s = _constrain(std, _env(Interval(0.0),(abs(ran*ran/4.0 - (maximum-mean-ran/2.0)**2))**0.5)," (dispersion)")
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
    
    Equivalent to min_max_mean_std(minimum,maximum,mean,np.sqrt(var))
 
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