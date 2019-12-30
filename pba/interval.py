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
        return "[%g, %g]"%(self.LowerBound,self.UpperBound)

    def __str__(self): # print
        return "[%g, %g]"%(self.LowerBound,self.UpperBound)

    def __init__(self,LowerBound = None, UpperBound = None):

        # kill complex nums
        assert not isinstance(LowerBound, np.complex) or not isinstance(UpperBound, np.complex), "Inputs must be real numbers"

        # assume vaccous if no inputs
        if LowerBound is None and UpperBound is None:
            UpperBound = np.inf
            LowerBound = np.inf

        # If only one input assume zero width
        elif LowerBound is None and UpperBound is not None:
            LowerBound = UpperBound
        elif LowerBound is not None and UpperBound is None:
            UpperBound = LowerBound

        # if iterable, find endpoints
        if hasattr(LowerBound, '__iter__') and hasattr(UpperBound, '__iter__'):

            LL = min(LowerBound)
            UL = min(UpperBound)
            LU = max(LowerBound)
            UU = max(UpperBound)

            LowerBound = min(LL,LU)
            UpperBound = max(LU,UU)

        elif hasattr(LowerBound, '__iter__'):

            LL = min(LowerBound)
            LU = max(LowerBound)

            LowerBound = min(LL,LU)


        elif hasattr(UpperBound, '__iter__'):

            UL = min(UpperBound)
            UU = max(UpperBound)

            UpperBound = max(LU,UU)


        if LowerBound > UpperBound:
            LowerUpper = [LowerBound, UpperBound]
            LowerBound = min(LowerUpper)
            UpperBound = max(LowerUpper)

        self.LowerBound = LowerBound
        self.UpperBound = UpperBound

    def __iter__(self):
        for bound in [self.LowerBound, self.UpperBound]:
            yield bound

    def __len__(self):
        return 2

    def __add__(self,other):

        if other.__class__.__name__ == 'Interval':
            lo = self.LowerBound + other.LowerBound
            hi = self.UpperBound + other.UpperBound
        elif other.__class__.__name__ == 'Pbox':
            # Perform Pbox addition assuming independance
            return other.add(self, method = 'i')
        else:
            try:
                lo = self.LowerBound + other
                hi  = seld.UpperBound + other
            except:
                raise ValueError('unsupported operand type(s) for +: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

    def __radd__(self,left):
        return self.__add__(left)

    def __sub__(self, other):

        if other.__class__.__name__ == "Interval":

            lo = self.LowerBound - other.UpperBound
            hi = self.UpperBound - other.LowerBound
        elif other.__class__.__name__ == "Pbox":
            # Perform Pbox subtractnion assuming independance
            return other.rsub(self)
        else:
            try:
                lo = self.LowerBound - other
                hi  = self.UpperBound - other
            except:
                raise ValueError('unsupported operand type(s) for -: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

    def __rsub__(self, other):
        if other.__class__.__name__ == "Interval":
            # should be overkill
            lo = other.UpperBound - self.LowerBound
            hi = other.UpperBound - self.UpperBound

        elif other.__class__.__name__ == "Pbox":
            # shoud have be caught by Pbox.__sub__()
            return other.__sub__(self)
        else:
            try:
                lo = other - self.UpperBound
                hi = other - self.LowerBound

            except:
                raise ValueError('unsupported operand type(s) for -: \'Interval\' and \'%s\'' %other.__class__.__name__)

        return Interval(lo,hi)

    def __mul__(self,other):
        if other.__class__.__name__ == "Interval":

            b1 = self.lo * other.lo
            b2 = self.lo * other.hi
            b3 = self.hi * other.lo
            b4 = self.hi * other.hi

            lo = min(b1,b2,b3,b4)
            hi = max(b1,b2,b3,b4)

        elif other.__class__.__name__ == "Pbox":

            return other.mul(self)

        else:

            try:

                lo = self.lo * other
                hi = self.hi * other

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

            b1 = self.lo * other.lo
            b2 = self.lo * other.hi
            b3 = self.hi * other.lo
            b4 = self.hi * other.hi

            lo = min(b1,b2,b3,b4)
            hi = max(b1,b2,b3,b4)

        else:
            try:
                return self * 1/other
            except:
                raise ValueError('unsupported operand type(s) for /: \'Interval\' and \'%s\'' %other.__class__.__name__)


    def __rtruediv__(self,other):

        try:
            return other * self.recip()
        except:
            raise ValueError('unsupported operand type(s) for /: \'Interval\' and \'%s\'' %other.__class__.__name__)


    def __pow__(self,other):
        if other.__class__.__name__ == "Interval":
            pow1 = self.LowerBound ** other.LowerBound
            pow2 = self.LowerBound ** other.UpperBound
            pow3 = self.UpperBound ** other.LowerBound
            pow4 = self.UpperBound ** other.UpperBound
            powUp = max(pow1,pow2,pow3,pow4)
            powLow = min(pow1,pow2,pow3,pow4)
        elif other.__class__.__name__ in ("int", "float"):
            pow1 = self.LowerBound ** other
            pow2 = self.UpperBound ** other
            powUp = max(pow1,pow2)
            powLow = min(pow1,pow2)
            if (self.UpperBound >= 0) and (self.LowerBound <= 0) and (other % 2 == 0):
                powLow = 0
        return Interval(powLow,powUp)

    def __rpow__(self,left):
        if left.__class__.__name__ == "Interval":
            pow1 = left.LowerBound ** self.LowerBound
            pow2 = left.LowerBound ** self.UpperBound
            pow3 = left.UpperBound ** self.LowerBound
            pow4 = left.UpperBound ** self.UpperBound
            powUp = max(pow1,pow2,pow3,pow4)
            powLow = min(pow1,pow2,pow3,pow4)

        elif left.__class__.__name__ in ("int", "float"):
            pow1 = left ** self.LowerBound
            pow2 = left ** self.UpperBound
            powUp = max(pow1,pow2)
            powLow = min(pow1,pow2)

        return Interval(powLow,powUp)


    def left(self):
        return self.LowerBound

    def right(self):
        return self.UpperBound

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
                    LSum = LSum + y.LowerBound
                    USum = USum + y.UpperBound
            if x.__class__.__name__ == "Interval":
                LSum = LSum + x.LowerBound
                USum = USum + x.UpperBound
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
                LBounds.append(x.LowerBound)
                UBounds.append(x.UpperBound)
            if x.__class__.__name__ in ("list","tuple"):
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LBounds.append(y.LowerBound)
                    UBounds.append(y.UpperBound)
            if x.__class__.__name__ == "Interval":
                LBounds.append(x.LowerBound)
                UBounds.append(x.UpperBound)
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
                LBounds.append(x.LowerBound)
                UBounds.append(x.UpperBound)
            if x.__class__.__name__ in ("list","tuple"):
                DataLen = DataLen + (len(x) - 1)
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LBounds.append(y.LowerBound)
                    UBounds.append(y.UpperBound)

        for y in LBounds:
            LDev.append(abs(y - dataMean.LowerBound)**2)
        for z in UBounds:
            UDev.append(abs(z - dataMean.UpperBound)**2)

        LSDev = (sum(LDev))/DataLen
        USDev = (sum(UDev))/DataLen
        return Interval(LSDev, USDev)

    def mode(*args):
        NotImplemented

    def straddles(self,N):
        if self.LowerBound <= N and self.UpperBound >= N:
            return True
        else:
            return False

    def straddles_zero(self):
        self.straddles(0)


    def recip(self):

        if self.straddles_zero():
            # Cant divide by zero
            raise ZeroDivisionError()

        elif 1/self.hi < 1/self.lo:
            return Interval(1/self.hi, 1/self.lo)
        else:
            return Interval(1/self.lo, 1/self.hi)


# a = Interval(1,2)
# b = Interval(3,4)
# c = Interval(-2,5)
# d = Interval(-7,-4)

##.sort() function to sort numbers for median
##list1.count(x) function to help with mode


I = Interval
