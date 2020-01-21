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

    def __init__(self,Left = None, Right = None):

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

    def __iter__(self):
        for bound in [self.Left, self.Right]:
            yield bound

    def __len__(self):
        return 2

    def __add__(self,other):

        if other.__class__.__name__ == 'Interval':
            lo = self.Left + other.Left
            hi = self.Right + other.Right
        elif other.__class__.__name__ == 'Pbox':
            # Perform Pbox addition assuming independance
            return other.add(self, method = 'i')
        else:
            try:
                lo = self.Left + other
                hi  = seld.Right + other
            except:
                raise ValueError('unsupported operand type(s) for +: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

    def __radd__(self,left):
        return self.__add__(left)

    def __sub__(self, other):

        if other.__class__.__name__ == "Interval":

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
                raise ValueError('unsupported operand type(s) for -: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

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
                raise ValueError('unsupported operand type(s) for -: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

    def __mul__(self,other):
        if other.__class__.__name__ == "Interval":

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
                raise ValueError('unsupported operand type(s) for *: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):

        if other.__class__.__name__ == "Interval":

            if other.straddles_zero():
                # Cant divide by zero
                raise ZeroDivisionError()

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

                raise ValueError('unsupported operand type(s) for /: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)


    def __rtruediv__(self,other):

        try:
            return other * self.recip()
        except:
            raise ValueError('unsupported operand type(s) for /: \'Interval\' and \'%s\'' %other.__class__.__name__)


    def __pow__(self,other):
        if other.__class__.__name__ == "Interval":
            pow1 = self.Left ** other.Left
            pow2 = self.Left ** other.Right
            pow3 = self.Right ** other.Left
            pow4 = self.Right ** other.Right
            powUp = max(pow1,pow2,pow3,pow4)
            powLow = min(pow1,pow2,pow3,pow4)
        elif other.__class__.__name__ in ("int", "float"):
            pow1 = self.Left ** other
            pow2 = self.Right ** other
            powUp = max(pow1,pow2)
            powLow = min(pow1,pow2)
            if (self.Right >= 0) and (self.Left <= 0) and (other % 2 == 0):
                powLow = 0
        return Interval(powLow,powUp)

    def __rpow__(self,left):
        if left.__class__.__name__ == "Interval":
            pow1 = left.Left ** self.Left
            pow2 = left.Left ** self.Right
            pow3 = left.Right ** self.Left
            pow4 = left.Right ** self.Right
            powUp = max(pow1,pow2,pow3,pow4)
            powLow = min(pow1,pow2,pow3,pow4)

        elif left.__class__.__name__ in ("int", "float"):
            pow1 = left ** self.Left
            pow2 = left ** self.Right
            powUp = max(pow1,pow2)
            powLow = min(pow1,pow2)

        return Interval(powLow,powUp)


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
        if self.Left <= N and self.Right >= N:
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


# a = Interval(1,2)
# b = Interval(3,4)
# c = Interval(-2,5)
# d = Interval(-7,-4)

##.sort() function to sort numbers for median
##list1.count(x) function to help with mode


I = Interval
