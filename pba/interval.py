# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:57:55 2019

@author: sggdale (with 'inspiration' from Marco)

nick: Sorry
"""

#testcommit

import numpy as np
import random as r

from .logic import Logical

class Interval():

    def __repr__(self): # return
        return "[%g, %g]"%(self.Left,self.Right)

    def __str__(self): # print
        return "[%g, %g]"%(self.Left,self.Right)

    def __init__(self,Left = None, Right = None, dep = None):

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
        if dep is None:
            self.DepFuncs = dict()
            self.DepFuncs[id(self)] = lambda x: Interval(x, x, dep = {})
            self.Dependencies = {id(self): self}
        else:
            self.DepFuncs = dep
            if dep == {}:
                self.Dependencies = dict()
            else:
                self.DepFuncs[id(self)] = lambda x: Interval(x,x, dep = {})
                self.Dependencies = {id(self): self}
        

    def __iter__(self):
        for bound in [self.Left, self.Right]:
            yield bound

    def __len__(self):
        return 2

    def __add__(self,other):

        if isinstance(other, Interval):
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'add'), 'add')
            else:
                lo = self.Left + other.Left
                hi = self.Right + other.Right
        elif other.__class__.__name__ == 'Pbox':
            # Perform Pbox addition assuming independance
            return other.add(self, method = 'i')
        else:
            try:
                lo = self.Left + other
                hi  = self.Right + other
            except:
                return NotImplemented

        return self.Build_Dependence(other, Interval(lo, hi), 'add')

    def __radd__(self,left):
        return self.__add__(left)

    def __sub__(self, other):

        if other.__class__.__name__ == "Interval":
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'sub'), 'sub')
            else:

                lo = self.Left - other.Right
                hi = self.Right - other.Left
        elif other.__class__.__name__ == "Pbox":
            # Perform Pbox subtractnion assuming independance
            return other.rsub(self)
        else:
            try:
                lo = self.Left - other
                hi  = self.Right - other
            except:
                return NotImplemented

        return self.Build_Dependence(other, Interval(lo, hi), 'sub')

    def __rsub__(self, other):
        if other.__class__.__name__ == "Interval":
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'rsub'), 'rsub')
            else:
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

        return self.Build_Dependence(other, Interval(lo, hi), 'rsub')

    def __mul__(self,other):
        if other.__class__.__name__ == "Interval":
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'mul'), 'mul')
            else:

                b1 = self.lo() * other.lo()
                b2 = self.lo() * other.hi()
                b3 = self.hi() * other.lo()
                b4 = self.hi() * other.hi()

                lo = min(b1,b2,b3,b4)
                hi = max(b1,b2,b3,b4)

        elif other.__class__.__name__ == "Pbox":

            return other.mul(self)

        else:

            try:

                lo = self.lo() * other
                hi = self.hi() * other

            except:

                return NotImplemented

        return self.Build_Dependence(other, Interval(lo, hi), 'mul')

    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):

        if other.__class__.__name__ == "Interval":
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'truediv'), 'truediv')
            else:

                if other.straddles_zero():
                    # Cant divide by zero
                    # raise ZeroDivisionError()

                    # But dividing by an interval which crosses 0 has an infinite limit, even if the zero division point is undefined.
                    return I()
                

                b1 = self.lo() / other.lo()
                b2 = self.lo() / other.hi()
                b3 = self.hi() / other.lo()
                b4 = self.hi() / other.hi()

                lo = min(b1,b2,b3,b4)
                hi = max(b1,b2,b3,b4)

        else:
            try:
                lo = self.lo()/other
                hi = self.hi()/other
            except:

                return NotImplemented

        return self.Build_Dependence(other, Interval(lo, hi), 'truediv')


    def __rtruediv__(self,other):

        try:
            return other * self.recip()
        except:
            return NotImplemented


    def __pow__(self,other):
        if other.__class__.__name__ == "Interval":
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'pow'), 'pow')
            else:
                pow1 = self.Left ** other.Left
                pow2 = self.Left ** other.Right
                pow3 = self.Right ** other.Left
                pow4 = self.Right ** other.Right
                lo = min(pow1,pow2,pow3,pow4)
                hi = max(pow1,pow2,pow3,pow4)
        elif other.__class__.__name__ in ("int", "float"):
            pow1 = self.Left ** other
            pow2 = self.Right ** other
            lo = min(pow1,pow2)
            hi = max(pow1,pow2)
            if (self.Right >= 0) and (self.Left <= 0) and (other % 2 == 0):
                lo = 0
        return self.Build_Dependence(other, Interval(lo, hi), 'pow')

    def __rpow__(self,left):
        if left.__class__.__name__ == "Interval":
            if set(self.Dependencies).intersection(set(other.Dependencies)) != set():
                return self.Build_Dependence(other, self.Dep_Operation(other, 'pow'), 'pow')
            else:
                pow1 = left.Left ** self.Left
                pow2 = left.Left ** self.Right
                pow3 = left.Right ** self.Left
                pow4 = left.Right ** self.Right
                lo = min(pow1,pow2,pow3,pow4)
                hi = max(pow1,pow2,pow3,pow4)

        elif left.__class__.__name__ in ("int", "float"):
            pow1 = left ** self.Left
            pow2 = left ** self.Right
            lo = min(pow1,pow2)
            hi = max(pow1,pow2)

        return self.Build_Dependence(other, Interval(lo, hi), 'rpow')


    def __lt__(self,other):
        # <
        if other.__class__.__name__ == 'Interval':
            if self.Right < other.Left:
                return Logical(1,1)
            elif self.Left > other.Right:
                return Logical(0,0)
            elif self.straddles(other.Left,endpoints = False) or self.straddles(other.Right,endpoints = False):
                return Logical(0,1)
            else:
                return Logical(0,0)
        else:
            try:
                if self.Right < other:
                    return Logical(1,1)
                elif self.straddles(other,endpoints = False):
                    return Logical(0,1)
                else:
                    return Logical(0,0)
            except Exception as e:
                raise ValueError

    def __eq__(self,other):
        # ==
        if other.__class__.__name__ == 'Interval':
            if self.straddles(other.Left) or self.straddles(other.Right):
                return Logical(0,1)
            else:
                return Logical(0,0)
        else:
            try:
                if self.straddles(other):
                    return Logical(0,1)
                else:
                    return Logical(0,0)
            except:
                raise ValueError


    def __gt__(self,other):
        # >
        if other.__class__.__name__ == 'Interval':
            if self.Right < other.Left:
                return Logical(0,0)
            elif self.Left > other.Right:
                return Logical(1,1)
            elif self.straddles(other.Left,endpoints = False) or self.straddles(other.Right,endpoints = False):
                return Logical(0,1)
            else:
                return Logical(0,0)
        else:
            try:
                if self.Right > other:
                    return Logical(1,1)
                elif self.straddles(other,endpoints = False):
                    return Logical(0,1)
                else:
                    return Logical(0,0)
            except Exception as e:
                raise ValueError

    def __ne__(self,other):
        # !=
        if other.__class__.__name__ == 'Interval':
            if self.straddles(other.Left) or self.straddles(other.Right):
                return Logical(0,1)
            else:
                return Logical(1,1)
        else:
            try:
                if self.straddles(other):
                    return Logical(0,1)
                else:
                    return Logical(1,1)
            except:
                raise ValueError

    def __le__(self,other):
        # <=
        if other.__class__.__name__ == 'Interval':
            if self.Right <= other.Left:
                return Logical(1,1)
            elif self.Left >= other.Right:
                return Logical(0,0)
            elif self.straddles(other.Left,endpoints = True) or self.straddles(other.Right,endpoints = True):
                return Logical(0,1)
            else:
                return Logical(0,0)
        else:
            try:
                if self.Right <= other:
                    return Logical(1,1)
                elif self.straddles(other,endpoints = True):
                    return Logical(0,1)
                else:
                    return Logical(0,0)
            except Exception as e:
                raise ValueError

    def __ge__(self,other):
        if other.__class__.__name__ == 'Interval':
            if self.Right <= other.Left:
                return Logical(0,0)
            elif self.Left >= other.Right:
                return Logical(1,1)
            elif self.straddles(other.Left,endpoints = True) or self.straddles(other.Right,endpoints = True):
                return Logical(0,1)
            else:
                return Logical(0,0)
        else:
            try:
                if self.Right > other:
                    return Logical(1,1)
                elif self.straddles(other,endpoints = True):
                    return Logical(0,1)
                else:
                    return Logical(0,0)
            except Exception as e:
                raise ValueError

    def __bool__(self):
        print(Logical(self.Left,self.Right))
        try:
            if Logical(self.Left,self.Right):

                return True
            else:
                return False
        except:
            raise ValueError("Truth value of Interval %s is ambiguous" %self)

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

    def straddles(self,N, endpoints = True):

        if endpoints:
            if self.Left <= N and self.Right >= N:
                return True
        else:
            if self.Left < N and self.Right > N:
                return True

        return False

    def straddles_zero(self,endpoints = True):
        return self.straddles(0,endpoints)

    def recip(self):

        if self.straddles_zero():
            # Cant divide by zero
            raise ZeroDivisionError()

        elif 1/self.hi() < 1/self.lo():
            return Interval(1/self.hi(), 1/self.lo())
        else:
            return Interval(1/self.lo(), 1/self.hi())

    def intersection(self, other):
        if isinstance(other, Interval):
            return I(max([x.Left for x in [self, other]]), min([x.Right for x in [self, other]]))
        else:
            return other

    def Dynamic_Func(self, func, other = None):
        if not other is None:
            target = other
        else:
            target = self
        if func == 'mul':
            return target.__mul__
        elif func == 'rmul':
            return target.__rmul__
        elif func == 'add':
            return target.__add__
        elif func == 'radd':
            return target.__radd__
        elif func == 'sub':
            return target.__sub__
        elif func == 'rsub':
            return target.__rsub__
        elif func == 'truediv':
            return target.__truediv__
        elif func == 'rtruediv':
            return target.__rtruediv__
        elif func == 'pow':
            return target.__pow__
        elif func == 'rpow':
            return target.__rpow__

    def Dep_Operation(self, other, func):
        Shared_Deps = set(self.Dependencies).intersection(set(other.Dependencies))
        lo = [self.DepFuncs[Dep](other.Dependencies[Dep].Left).Dynamic_Func(func)(other.DepFuncs[Dep](self.Dependencies[Dep].Left)) for Dep in Shared_Deps]
        hi = [self.DepFuncs[Dep](self.Dependencies[Dep].Right).Dynamic_Func(func)(other.DepFuncs[Dep](other.Dependencies[Dep].Right)) for Dep in Shared_Deps]
        lo = min([L.Left if isinstance(L, Interval) else L for L in lo])
        hi = max([H.Right if isinstance(H, Interval) else H for H in hi])
        return Interval(lo, hi)

    def Build_Dependence(self, other, NewInt, func):
        if isinstance(other, Interval):
            Shared_Deps = set(self.Dependencies).intersection(set(other.Dependencies))
            for DepShared in Shared_Deps:
                if DepShared == id(self):
                    NewInt.DepFuncs[DepShared] = lambda x: I(x, x).Dynamic_Func(func)(other.DepFuncs[DepShared](I(x, x))) if not isinstance(x, Interval) else x.Dynamic_Func(func)(other.DepFuncs[DepShared](x))
                elif DepShared == id(other):
                    NewInt.DepFuncs[DepShared] = lambda x: self.DepFuncs[DepShared](I(x, x)).Dynamic_Func(func)(I(x, x)) if not isinstance(x, Interval) else self.DepFuncs[DepShared](x).Dynamic_Func(func)(x)
                else:
                    NewInt.DepFuncs[DepShared] = lambda x: self.DepFuncs[DepShared](I(x, x)).Dynamic_Func(func)(other.DepFuncs[DepShared](I(x, x))) if not isinstance(x, Interval) else self.DepFuncs[DepShared](x).Dynamic_Func(func)(other.DepFuncs[DepShared](x))
                NewInt.Dependencies[DepShared] = self.Dependencies[DepShared]
            if not self.Dependencies == {} or other.Dependencies == {}: 
                for DepSelf in set(self.DepFuncs).difference(other.DepFuncs):
                    
                    if DepSelf == id(self):
                        NewInt.DepFuncs[DepSelf] = lambda x: I(x, x).Dynamic_Func(func)(other) if not isinstance(x, Interval) else x.Dynamic_Func(func)(other)
                    else:
                        TargFunc = self.DepFuncs[DepSelf]
                        NewInt.DepFuncs[DepSelf] = lambda x: TargFunc(I(x, x)).Dynamic_Func(func)(other) if not isinstance(x, Interval) else TargFunc(x).Dynamic_Func(func)(other)
                    NewInt.Dependencies[DepSelf] = self.Dependencies[DepSelf]
                for DepOther in set(other.DepFuncs).difference(self.DepFuncs):
                    
                    if DepOther == id(other):
                        NewInt.DepFuncs[DepOther] = lambda x: I(x, x).Dynamic_Func([func[1:] if func[0]=='r' else 'r' + func][0])(self) if not isinstance(x, Interval) else x.Dynamic_Func([func[1:] if func[0]=='r' else 'r' + func][0])(self)
                    else:
                        TargFunc = other.DepFuncs[DepOther]
                        NewInt.DepFuncs[DepOther] = lambda x: TargFunc(I(x, x)).Dynamic_Func([func[1:] if func[0]=='r' else 'r' + func][0])(self) if not isinstance(x, Interval) else TargFunc(x).Dynamic_Func([func[1:] if func[0]=='r' else 'r' + func][0])(self)
                    NewInt.Dependencies[DepOther] = other.Dependencies[DepOther]
            
        else:
            NewInt.DepFuncs[id(self)] = lambda x: I(x, x).Dynamic_Func(func)( other)
            NewInt.Dependencies[id(self)] = self
        return NewInt


# a = Interval(1,2)
# b = Interval(3,4)
# c = Interval(-2,5)
# d = Interval(-7,-4)

##.sort() function to sort numbers for median
##list1.count(x) function to help with mode


I = Interval
