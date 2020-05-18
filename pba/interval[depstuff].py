# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:57:55 2019

@author: sggdale (with 'inspiration' from Marco)

nick: Sorry
"""

#testcommit

import numpy as np
import random as r


class Interval():

    def __repr__(self): # return
        return "[%g, %g]"%(self.Left,self.Right)

    def __str__(self): # print
        return "[%g, %g]"%(self.Left,self.Right)

    def __init__(self,Left = None, Right = None, dep = dict()):

        # kill complex nums
        assert not isinstance(Left, np.complex) or not isinstance(Right, np.complex), "Inputs must be real numbers"

        # assume vaccous if no inputs
        if Left is None and Right is None:
            Right = np.inf
            Left = np.inf

        # If only one input assume zero width
        elif Left is None and Right is not None:
            Left = Right
        elif Left is not None and Right is None:
            Right = Left

        # if iterable, find endpoints
        if hasattr(Left, '__iter__') and hasattr(Right, '__iter__'):

            LL = min(Left)
            UL = min(Right)
            LU = max(Left)
            UU = max(Right)

            Left = min(LL,LU)
            Right = max(LU,UU)

        elif hasattr(Left, '__iter__'):

            LL = min(Left)
            LU = max(Left)

            Left = min(LL,LU)


        elif hasattr(Right, '__iter__'):

            UL = min(Right)
            UU = max(Right)

            Right = max(LU,UU)


        if Left > Right:
            LowerUpper = [Left, Right]
            Left = min(LowerUpper)
            Right = max(LowerUpper)

        self.Left = Left
        self.Right = Right

        if not dep == {}:
            self.dependencies = dep
        else:
            self.dependencies = {id(self): lambda s: Interval(s,s)}

    def __iter__(self):
        for bound in [self.Left, self.Right]:
            yield bound

    def __len__(self):
        return 2

    def __add__(self,other):

        if other.__class__.__name__ == 'Interval':
            if id(other) in self.dependencies.keys():
                lo = Interval(
                    self.dependencies[id(other)](other.Left).Left + other.Left,
                    self.dependencies[id(other)](other.Left).Right + other.Left
                )
                hi = Interval(
                    self.dependencies[id(other)](other.Right).Left + other.Right,
                    self.dependencies[id(other)](other.Right).Right + other.Right
                )
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            elif id(self) in other.dependencies:
                lo = Interval(
                    self.Left + other.dependencies[id(self)](self.Left).Left,
                    self.Left + other.dependencies[id(self)](self.Left).Right
                )
                hi = Interval(
                    self.Right + other.dependencies[id(self)](self.Right).Left,
                    self.Right + other.dependencies[id(self)](self.Right).Right
                )
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            else:
                lo = self.Left + other.Left
                hi = self.Right + other.Right
            return self.Build_Dependence(other, Interval(lo, hi), 'add')

        elif other.__class__.__name__ == 'Pbox':
            # Perform Pbox addition assuming independance
            return other.add(self, method = 'i')
        else:
            try:
                lo = self.Left + other
                hi  = self.Right + other
            except:
                return NotImplemented       

        return Interval(lo, hi)

    def __radd__(self,left):
        return self.__add__(left)

    def __sub__(self, other):

        if other.__class__.__name__ == "Interval":
            if id(other) == id(self):
                return Interval(0,0)
            elif id(other) in self.dependencies.keys():
                lo = Interval(
                    self.dependencies[id(other)](other.Left).Left - other.Left,
                    self.dependencies[id(other)](other.Left).Right - other.Left
                )
                hi = Interval(
                    self.dependencies[id(other)](other.Right).Left - other.Right,
                    self.dependencies[id(other)](other.Right).Right - other.Right
                )
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            elif id(self) in other.dependencies:
                lo = Interval(
                    self.Left - other.dependencies[id(self)](self.Left).Left,
                    self.Left - other.dependencies[id(self)](self.Left).Right
                )
                hi = Interval(
                    self.Right - other.dependencies[id(self)](self.Right).Left,
                    self.Right - other.dependencies[id(self)](self.Right).Right
                )
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            else:
                lo = self.Left - other.Right
                hi = self.Right - other.Left  
            return self.Build_Dependence(other, Interval(lo, hi), 'sub')

        elif other.__class__.__name__ == "Pbox":
            # Perform Pbox subtractnion assuming independance
            return other.rsub(self)
        else:
            try:
                lo = self.Left - other
                hi  = self.Right - other
            except:
                return NotImplemented
        return Interval(lo, hi)

    def __rsub__(self, other):
        if other.__class__.__name__ == "Interval":
            # should be overkill
            lo = other.Right - self.Left
            hi = other.Right - self.Right

        elif other.__class__.__name__ == "Pbox":
            # shoud have be caught by Pbox.__sub__()
            return other.__sub__(self)
        else:
            try:
                lo = other - self.Right
                hi = other - self.Left

            except:
                return NotImplemented

        return Interval(lo,hi)

    def __mul__(self, other):
        if other.__class__.__name__ == "Interval":
            if id(other) == id(self) and self.straddles(0):
                lo = 0
                hi = max(self.Left**2, self.Right**2)
            elif id(other) in self.dependencies.keys():
                lo = Interval(
                    self.dependencies[id(other)](other.Left).Left * other.Left,
                    self.dependencies[id(other)](other.Left).Right * other.Left
                )
                hi = Interval(
                    self.dependencies[id(other)](other.Right).Left * other.Right,
                    self.dependencies[id(other)](other.Right).Right * other.Right
                )
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            elif id(self) in other.dependencies:
                lo = Interval(
                    self.Left * other.dependencies[id(self)](self.Left).Left,
                    self.Left * other.dependencies[id(self)](self.Left).Right
                )
                hi = Interval(
                    self.Right * other.dependencies[id(self)](self.Right).Left,
                    self.Right * other.dependencies[id(self)](self.Right).Right
                )
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            else:
                b1 = self.lo() * other.lo()
                b2 = self.lo() * other.hi()
                b3 = self.hi() * other.lo()
                b4 = self.hi() * other.hi()

                lo = min(b1,b2,b3,b4)
                hi = max(b1,b2,b3,b4)
            return self.Build_Dependence(other, Interval(lo, hi), 'mul')

        elif other.__class__.__name__ == "Pbox":

            return other.mul(self)

        else:

            try:

                lo = self.lo() * other
                hi = self.hi() * other

            except:
                
                return NotImplemented
        return Interval(lo, hi)

        

    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):

        if other.__class__.__name__ == "Interval":
            if id(other) == id(self):
                return Interval(1,1)
            elif id(other) in self.dependencies.keys():
                if other.Left != 0:
                    lo = Interval(
                        self.dependencies[id(other)](other.Left).Left / other.Left,
                        self.dependencies[id(other)](other.Left).Right / other.Left
                        )
                else:
                    lo = Interval()
                if other.Right != 0 and not other.straddles(0):
                    hi = Interval(
                        self.dependencies[id(other)](other.Right).Left / other.Right,
                        self.dependencies[id(other)](other.Right).Right / other.Right
                        )
                else:
                    hi = Interval()
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            elif id(self) in other.dependencies:
                if other.Left != 0:
                    lo = Interval(
                        self.Left / other.dependencies[id(self)](self.Left).Left,
                        self.Left / other.dependencies[id(self)](self.Left).Right
                        )
                else:
                    lo = Interval()
                if other.Right != 0 and not other.straddles(0):
                    hi = Interval(
                        self.Right / other.dependencies[id(self)](self.Right).Left,
                        self.Right / other.dependencies[id(self)](self.Right).Right
                        )
                else:
                    hi = Interval()
                [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]
            else:
                if other.straddles(0):
                    # Cant divide by zero, but allow for vacuous positive and negative intervals, assuming that they will be used in ways which cancel each other out. Otherwise output will also be a vacuous interval. Also, the limit of dividing by a zero-straddling interval will be infinity, even if it is undefined at zero.
                    try: b1 = self.lo() / other.lo()
                    except: b1 = np.inf
                    try: b2 = self.lo() / other.hi()
                    except: b2 = np.inf
                    try: b3 = self.hi() / other.lo()
                    except: b3 = np.inf
                    try: b4 = self.hi() / other.hi()
                    except: b4 = np.inf
                    #raise ZeroDivisionError()
                else:
                    b1 = self.lo() / other.lo()
                    b2 = self.lo() / other.hi()
                    b3 = self.hi() / other.lo()
                    b4 = self.hi() / other.hi()

                lo = min(b1,b2,b3,b4)
                hi = max(b1,b2,b3,b4)
            
            return self.Build_Dependence(other, Interval(lo, hi), 'truediv')

        else:
            try:
                lo = self.lo()/other
                hi = self.hi()/other
            except:

                return NotImplemented
        return Interval(lo, hi)

        


    def __rtruediv__(self,other):

        try:
            return other * self.recip()
        except:
            return NotImplemented


    def __pow__(self,other):
        if other.__class__.__name__ == "Interval":
            if id(other) in self.dependencies.keys():
                lo = Interval(
                    self.dependencies[id(other)](other.Left).Left ** other.Left,
                    self.dependencies[id(other)](other.Left).Right ** other.Left
                    )
                hi = Interval(
                    self.dependencies[id(other)](other.Right).Left ** other.Right,
                    self.dependencies[id(other)](other.Right).Right ** other.Right
                    )
                
            elif id(self) in other.dependencies:
                lo = Interval(
                    self.Left ** other.dependencies[id(self)](self.Left).Left,
                    self.Left ** other.dependencies[id(self)](self.Left).Right
                    )
                hi = Interval(
                    self.Right ** other.dependencies[id(self)](self.Right).Left,
                    self.Right ** other.dependencies[id(self)](self.Right).Right
                    )

            else:
                    pow1 = self.Left ** other.Left
                    pow2 = self.Left ** other.Right
                    pow3 = self.Right ** other.Left
                    pow4 = self.Right ** other.Right
                    lo = min(pow1,pow2,pow3,pow4)
                    hi = max(pow1,pow2,pow3,pow4)

            if self.straddles(0) and lo != hi:
                lo.Left = min(0, lo.Left)
            if other.straddles(0):
                hi.Right = max(1, hi.Right, lo.Right)
            [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]   
            return self.Build_Dependence(other, Interval(lo, hi), 'pow')

        elif other.__class__.__name__ in ("int", "float"):
            pow1 = self.Left ** other
            pow2 = self.Right ** other
            lo = min(pow1,pow2)
            hi = max(pow1,pow2)
            if self.straddles(0) and lo != hi:
                lo = 0
        return Interval(lo, hi)

        
    
    def __mod__(self, other):
        assert not(isinstance(other, Interval)), 'Can\'t do that yet. Modulo by a float or integer please'
        if self.Right - self.Left >= other:
            lo = 0
            hi = other
        else:

            lo = self.Left % other
            if other - lo >= (self.Right - self.Left):
                lo = 0
                hi = other
            else:
                hi = min(self.Right - self.Left, other)
        NewInt = Interval(lo, hi)
        NewInt.dependencies[id(self)] = lambda s: s%other
        return NewInt

    def __rpow__(self, other):
        if other.__class__.__name__ == "Interval":
            if id(other) in self.dependencies.keys():
                lo = Interval(
                    other.Left ** self.dependencies[id(other)](other.Left).Left,
                    other.Left ** self.dependencies[id(other)](other.Left).Right
                    )
                hi = Interval(
                    other.Right ** self.dependencies[id(other)](other.Right).Left,
                    other.Right ** self.dependencies[id(other)](other.Right).Right
                    )
            elif id(self) in other.dependencies:
                lo = Interval(
                    other.dependencies[id(self)](self.Left).Left ** self.Left,
                    other.dependencies[id(self)](self.Left).Right ** self.Left
                    )
                hi = Interval(
                    other.dependencies[id(self)](self.Right).Left ** self.Right,
                    other.dependencies[id(self)](self.Right).Right ** self.Right
                    )
                
            else:
                    pow1 = other.Left ** other.Left
                    pow2 = other.Left ** other.Right
                    pow3 = other.Right ** other.Left
                    pow4 = other.Right ** other.Right
                    lo = min(pow1,pow2,pow3,pow4)
                    hi = max(pow1,pow2,pow3,pow4)
            if other.straddles(0) and lo != hi:
                lo.Left = min(0, lo.Left)
            if other.straddles(0):
                hi.Right = max(1, hi.Right, lo.Right)
            
            [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]

            return self.Build_Dependence(other, Interval(lo, hi), 'rpow')
            

        elif other.__class__.__name__ in ("int", "float"):
            pow1 = other.Left ** other
            pow2 = other.Right ** other
            lo = min(pow1,pow2)
            hi = max(pow1,pow2)
            if other.straddles(0) and lo != hi:
                lo.Left = min(0, lo.Left)
        [lo, hi] = [min(lo.Left, hi.Left), max(lo.Right, hi.Right)]

        return Interval(lo, hi)
        """ 

    def __le__(self, other):
        if isinstance(other, Interval):
            return self.Left <= other.Left
        elif isinstance(other, (int, float)):
            return self.Left <= other

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.Left < other.Left
        elif isinstance(other, (int, float)):
            return self.Left < other
    def __ge__(self, other):
        if isinstance(other, Interval):
            return self.Right >= other.Right
        elif isinstance(other, (int, float)):
            return self.Right >= other

    def __gt__(self, other):
        if isinstance(other, Interval):
            return self.Right > other.Right
        elif isinstance(other, (int, float)):
            return self.Right > other

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.Left == other.Left and self.Right == other.Right
        elif isinstance(other, (int, float)):
            return self.Left == other and self.Right == other """


    def left(self):
        return self.Left

    def right(self):
        return self.Right

    lo = left
    hi = right

    def mean(*args):
        LSum = 0
        USum = 0
        DataLen = len(args)
        for x in args:
            if x.__class__.__name__ in ("int","float"):
                x = Interval(x)
            if x.__class__.__name__ in ("list","tuple"):
                DataLen = DataLen + (len(x) - 1)
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LSum = LSum + y.Left
                    USum = USum + y.Right
            if x.__class__.__name__ == "Interval":
                LSum = LSum + x.Left
                USum = USum + x.Right
            LMean = LSum / DataLen
            UMean = USum / DataLen
        return Interval(LMean, UMean)

    def median(*args):
        LBounds = []
        LSorted = []
        UBounds = []
        USorted = []

        for x in [*args]:
            if x.__class__.__name__ in ("int","float"):
                x = Interval(x)
                LBounds.append(x.Left)
                UBounds.append(x.Right)
            if x.__class__.__name__ in ("list","tuple"):
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LBounds.append(y.Left)
                    UBounds.append(y.Right)
            if x.__class__.__name__ == "Interval":
                LBounds.append(x.Left)
                UBounds.append(x.Right)
        while (len(LBounds) > 0):
            MinL = min(LBounds)
            LSorted.append(MinL)
            LBounds.remove(MinL)
        while (len(UBounds) > 0):
            MinU = min(UBounds)
            USorted.append(MinU)
            UBounds.remove(MinU)

        if (len(LSorted) % 2) != 0:
            LMedian = LSorted[len(LSorted)//2]
            UMedian = USorted[len(USorted)//2]
        else:
            LMedian = (LSorted[len(LSorted)//2] + LSorted[(len(LSorted)//2)-1])/2
            UMedian = (USorted[len(USorted)//2] + USorted[(len(USorted)//2)-1])/2
        return Interval(LMedian,UMedian)

    def variance(*args):
        dataMean = Interval.mean(*args)
        LBounds = []
        UBounds = []
        LDev = []
        UDev = []
        DataLen = len(args)
        for x in [*args]:
            if x.__class__.__name__ in ("int","float"):
                x = Interval(x)
                LBounds.append(x.Left)
                UBounds.append(x.Right)
            if x.__class__.__name__ in ("list","tuple"):
                DataLen = DataLen + (len(x) - 1)
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LBounds.append(y.Left)
                    UBounds.append(y.Right)

        for y in LBounds:
            LDev.append(abs(y - dataMean.Left)**2)
        for z in UBounds:
            UDev.append(abs(z - dataMean.Right)**2)

        LSDev = (sum(LDev))/DataLen
        USDev = (sum(UDev))/DataLen
        return Interval(LSDev, USDev)

    def mode(*args):
        NotImplemented

    def straddles(self,N):
        if isinstance(N, Interval):
            if self.Left <= N.Left and self.Right >= N.Right:
                return True
            else:
                return False
        else:
            if (not np.isfinite(self.Left) or self.Left <= N) and (not np.isfinite(self.Right) or self.Right >= N):
                return True
            else:
                return False

    def straddles_zero(self):
        self.straddles(0)


    def recip(self):

        if self.straddles_zero():
            # Cant divide by zero
            raise ZeroDivisionError()

        elif 1/self.hi() < 1/self.lo():
            return Interval(1/self.hi(), 1/self.lo())
        else:
            return Interval(1/self.lo(), 1/self.hi())

    def conjunction(self, other):
        if isinstance(other, Interval):
            return I(max([x.Left for x in [self, other]]), min([x.Right for x in [self, other]]))
        else:
            return other

    def Dynamic_Func(self, target, func):
        if func == 'mul':
            return target.__mul__
        elif func == 'add':
            return target.__add__
        elif func == 'sub':
            return target.__sub__
        elif func == 'truediv':
            return target.__truediv__
        elif func == 'pow':
            return target.__pow__


    def Build_Dependence(self, other, NewInt, func):
        def self_dep(s):
            ss = self.Dynamic_Func(other, func)(s)
            ss.dependencies = dict()
            return ss
        NewInt.dependencies[id(self)] = self_dep
        if isinstance(other, Interval):
            def other_dep(o):
                oo = self.Dynamic_Func(self, func)(o)
                if isinstance(oo, Interval):
                    oo.dependencies = dict()
                    return oo
                else: 
                    return Interval()
            NewInt.dependencies[id(other)] = other_dep
        return NewInt
        

    """ def dependence(self, other, struct, structinv = []):
        if other.__class__.__name__ == "Interval":
            if struct == 'M':
                newstruct = lambda x: I(x,x)

            elif struct == 'W':
                newstruct = lambda x: I(self.Right-x,self.Right-x)

            elif struct == 'I':
                newstruct = lambda x: I(self.Left,self.Right)

            elif struct == '+':
                newstruct = lambda x: (
                I(self.Left, 0.5*(self.Right - self.Left)) if I(self.Left, 0.5*(self.Right - self.Left)).straddles(x) else (
                    I(0.5*(self.Right - self.Left), self.Right) if I(0.5*(self.Right - self.Left), self.Right).straddles(x) else self.Left))

            elif struct == '-':
                newstruct = lambda x: (
                I(0.5*(self.Right - self.Left), self.Right) if I(self.Left, 0.5*(self.Right - self.Left)).straddles(x) else (
                    I(self.Left, 0.5*(self.Right - self.Left)) if I(0.5*(self.Right - self.Left), self.Right).straddles(x) else 0))
            else:
                assert hasattr(struct, '__name__'), "Struct must be a function returning an interval, or a standard copula (M, W, I, +, -)."

                newstruct = struct

            if not hasattr(self, 'dependencies'):
                self.dependencies = {}

            self.dependencies[id(other)] = newstruct """

# a = Interval(1,2)
# b = Interval(3,4)
# c = Interval(-2,5)
# d = Interval(-7,-4)

##.sort() function to sort numbers for median
##list1.count(x) function to help with mode


I = Interval
