# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:57:55 2019

@author: sggdale (with 'inspiration' from Marco)
"""

#testcommit


import random as r


global iDict
global inf
inf = float('inf')
iDict = {}

class Interval():
    global inf
    def __repr__(self): # return
        return "[%g, %g]"%(self.__LowerBound,self.__UpperBound)

    def __str__(self): # print
        return "[%g, %g]"%(self.__LowerBound,self.__UpperBound)

    def __init__(self,*args):
        self.name = ""
        self.id = r.randint(1,1e12)
        iDict[self.id] = self
        if len(args) == 0:
            LowerBound = -inf
            UpperBound = inf
            MidPoint = 0
        if len(args) == 1:
            LowerBound = args[0]
            UpperBound = args[0]
            MidPoint = args[0]
        if len(args) == 2:
            LowerBound = min(args)
            UpperBound = max(args)
            MidPoint = (LowerBound + UpperBound)/2
        if len(args) > 2:
            raise "ERROR: Too Many Arguments!"

        self.__LowerBound = LowerBound
        self.__UpperBound = UpperBound
        self.__MidPoint = MidPoint


    def __add__(self,other):
        IDcheck = self.id + other.id
        if other.__class__.__name__ in ("int","float"):
            addLow = self.__LowerBound + other
            addUp = self.__UpperBound + other
        elif other.__class__.__name__ == "Interval":
            if IDcheck in iDict:
                return iDict[IDcheck]
            else:
                addLow = self.__LowerBound + other.__LowerBound
                addUp = self.__UpperBound + other.__UpperBound
        result = Interval(addLow,addUp)
        result.id = self.id + other.id
        iDict[self.id + other.id] = result
        return result

    def __radd__(self,left):
        if left.__class__.__name__ in ("int", "float"):
            addLow = left + self.__LowerBound
            addUp = left + self.__UpperBound
        elif left.__class__.__name__ == "Interval":
            addLow = self.__LowerBound + left.__LowerBound
            addUp = self.__UpperBound + left.__UpperBound
        result = Interval(addLow,addUp)
        result.id = self.id + left.id
        iDict[self.id + left.id] = result
        return result

    def __sub__(self, other):
        IDcheck = self.id - other.id
        if other.__class__.__name__ in ("int", "float"):
            subLow = self.__LowerBound - other
            subUp = self.__UpperBound - other
        elif other.__class__.__name__ == "Interval":
            if self.id == other.id:
                subLow = 0
                subUp = 0
            elif IDcheck in iDict:
                return iDict[IDcheck]
            else:
                subLow = self.__LowerBound - other.__UpperBound
                subUp = self.__UpperBound - other.__LowerBound
        result = Interval(subLow,subUp)
        result.id = self.id - other.id
        iDict[self.id - other.id] = result
        return result


    def __rsub__(self, left):
        if left.__class__.__name__ in ("int", "float"):
            subLow = left - self.__LowerBound
            subUp = left - self.__UpperBound
        elif left.__class__.__name__ == "Interval":
            if self.id == left.id:
                subLow = 0
                subUp = 0
            else:
                subLow = left.__UpperBound - self.__LowerBound
                subUp = left.__UpperBound - self.__UpperBound
        return Interval(subLow,subUp)

    def __mul__(self,other):
        if other.__class__.__name__ in ("int", "float"):
            if other>0:
                mulLow = self.__LowerBound * other
                mulUp = self.__UpperBound * other
            else:
                mulLow = self.__UpperBound * other
                mulUp = self.__LowerBound * other
        elif other.__class__.__name__ == "Interval":
            mul1 = self.__LowerBound * other.__LowerBound
            mul2 = self.__LowerBound * other.__UpperBound
            mul3 = self.__UpperBound * other.__LowerBound
            mul4 = self.__UpperBound * other.__UpperBound
            mulLow = min(mul1,mul2,mul3,mul4)
            mulUp = max(mul1,mul2,mul3,mul4)
        result = Interval(mulLow,mulUp)
        result.id = self.id * other.id
        iDict[self.id * other.id] = result
        return result

    def __rmul__(self,left):
        if left.__class__.__name__ in ("int", "float"):
            if left>0:
                mulLow = self.__LowerBound * left
                mulUp = self.__UpperBound * left
            else:
                mulLow = self.__UpperBound * left
                mulUp = self.__LowerBound * left
        elif left.__class__.__name__ == "Interval":
            mul1 = self.__LowerBound * left.__LowerBound
            mul2 = self.__LowerBound * left.__UpperBound
            mul3 = self.__UpperBound * left.__LowerBound
            mul4 = self.__UpperBound * left.__UpperBound
            mulLow = min(mul1,mul2,mul3,mul4)
            mulUp = max(mul1,mul2,mul3,mul4)
        return Interval(mulLow,mulUp)

    def __truediv__(self,other):
        IDcheck = self.id / other.id
        if other.__class__.__name__ in ("int", "float"):
            if other>0:
                divLow = self.__LowerBound / other
                divUp = self.__UpperBound / other
            elif other<0:
                divLow = self.__UpperBound / other
                divUp = self.__LowerBound / other
            return Interval(divLow,divUp)

        elif other.__class__.__name__ == "Interval":

            if self.id == other.id:
                if (self.__UpperBound == 0) and (self.__LowerBound == 0):
                    return "Undefined"
                else:
                    return Interval(1)
            elif IDcheck in iDict:
                return iDict[IDcheck]
            elif (other.__UpperBound >= 0) and (other.__LowerBound <= 0):
                divintLowLow = -inf
                divintUpUp = inf
                divLow1 = self.__LowerBound * divintLowLow
                divLow2 = self.__LowerBound * (1/other.__LowerBound)
                divLow3 = self.__UpperBound * divintLowLow
                divLow4 = self.__UpperBound * (1/other.__LowerBound)
                divLowLow = min(divLow1,divLow2,divLow3,divLow4)
                divLowUp = max(divLow1,divLow2,divLow3,divLow4)

                divUp1 = self.__LowerBound * divintUpUp
                divUp2 = self.__LowerBound * (1/other.__UpperBound)
                divUp3 = self.__UpperBound * divintUpUp
                divUp4 = self.__UpperBound * (1/other.__UpperBound)
                divUpLow = min(divUp1,divUp2,divUp3,divUp4)
                divUpUp = max(divUp1,divUp2,divUp3,divUp4)

                resultlow = Interval(divLowLow,divLowUp)
                resultup = Interval(divUpLow,divUpUp)
                resultlow.id = self.id / other.id
                resultup.id = self.id / other.id
                iDict[self.id/other.id] = resultlow, resultup
                return resultlow, resultup

            else:
                div1 = self.__LowerBound / other.__LowerBound
                div2 = self.__LowerBound / other.__UpperBound
                div3 = self.__UpperBound / other.__LowerBound
                div4 = self.__UpperBound / other.__UpperBound
                divLow = min(div1,div2,div3,div4)
                divUp = max(div1,div2,div3,div4)
                result = Interval(divLow,divUp)
                result.id = self.id / other.id
                iDict[self.id/other.id] = result
                return result




    def __rtruediv__(self,left):
        if left.__class__.__name__ in ("int", "float"):
            if left>0:
                divLow = left / self.__UpperBound
                divUp = left / self.__LowerBound
            elif left<0:
                divLow = left / self.__LowerBound
                divUp = left / self.__UpperBound
            return Interval(divLow,divUp)

        elif left.__class__.__name__ == "Interval":
            if self.id == left.id:
                if (self.__UpperBound == 0) and (self.__LowerBound == 0):
                    return "Undefined"
                else:
                    return Interval(1)
            elif (self.__UpperBound >= 0) and (self.__LowerBound <= 0):
                divintLowLow = -inf
                divintUpUp = inf
                divLow1 = left.__LowerBound * divintLowLow
                divLow2 = left.__LowerBound * (1/self.__LowerBound)
                divLow3 = left.__UpperBound * divintLowLow
                divLow4 = left.__UpperBound * (1/self.__LowerBound)
                divLowLow = min(divLow1,divLow2,divLow3,divLow4)
                divLowUp = max(divLow1,divLow2,divLow3,divLow4)

                divUp1 = left.__LowerBound * divintUpUp
                divUp2 = left.__LowerBound * (1/self.__UpperBound)
                divUp3 = left.__UpperBound * divintUpUp
                divUp4 = left.__UpperBound * (1/self.__UpperBound)
                divUpLow = min(divUp1,divUp2,divUp3,divUp4)
                divUpUp = max(divUp1,divUp2,divUp3,divUp4)

                return [Interval(divLowLow,divLowUp),Interval(divUpLow,divUpUp)]

            else:
                div1 = left.__LowerBound / self.__LowerBound
                div2 = left.__LowerBound / self.__UpperBound
                div3 = left.__UpperBound / self.__LowerBound
                div4 = left.__UpperBound / self.__UpperBound
                divLow = min(div1,div2,div3,div4)
                divUp = max(div1,div2,div3,div4)
                return Interval(divLow,divUp)



    def __pow__(self,other):
        if other.__class__.__name__ == "Interval":
            pow1 = self.__LowerBound ** other.__LowerBound
            pow2 = self.__LowerBound ** other.__UpperBound
            pow3 = self.__UpperBound ** other.__LowerBound
            pow4 = self.__UpperBound ** other.__UpperBound
            powUp = max(pow1,pow2,pow3,pow4)
            powLow = min(pow1,pow2,pow3,pow4)
        elif other.__class__.__name__ in ("int", "float"):
            pow1 = self.__LowerBound ** other
            pow2 = self.__UpperBound ** other
            powUp = max(pow1,pow2)
            powLow = min(pow1,pow2)
            if (self.__UpperBound >= 0) and (self.__LowerBound <= 0) and (other % 2 == 0):
                powLow = 0
        return Interval(powLow,powUp)

    def __rpow__(self,left):
        if left.__class__.__name__ == "Interval":
            pow1 = left.__LowerBound ** self.__LowerBound
            pow2 = left.__LowerBound ** self.__UpperBound
            pow3 = left.__UpperBound ** self.__LowerBound
            pow4 = left.__UpperBound ** self.__UpperBound
            powUp = max(pow1,pow2,pow3,pow4)
            powLow = min(pow1,pow2,pow3,pow4)

        elif left.__class__.__name__ in ("int", "float"):
            pow1 = left ** self.__LowerBound
            pow2 = left ** self.__UpperBound
            powUp = max(pow1,pow2)
            powLow = min(pow1,pow2)

        return Interval(powLow,powUp)
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
                    LSum = LSum + y.__LowerBound
                    USum = USum + y.__UpperBound
            if x.__class__.__name__ == "Interval":
                LSum = LSum + x.__LowerBound
                USum = USum + x.__UpperBound
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
                LBounds.append(x.__LowerBound)
                UBounds.append(x.__UpperBound)
            if x.__class__.__name__ in ("list","tuple"):
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LBounds.append(y.__LowerBound)
                    UBounds.append(y.__UpperBound)
            if x.__class__.__name__ == "Interval":
                LBounds.append(x.__LowerBound)
                UBounds.append(x.__UpperBound)
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
                LBounds.append(x.__LowerBound)
                UBounds.append(x.__UpperBound)
            if x.__class__.__name__ in ("list","tuple"):
                DataLen = DataLen + (len(x) - 1)
                for y in x:
                    if y.__class__.__name__ in ("int","float"):
                        y = Interval(y)
                    LBounds.append(y.__LowerBound)
                    UBounds.append(y.__UpperBound)

        for y in LBounds:
            LDev.append(abs(y - dataMean.__LowerBound)**2)
        for z in UBounds:
            UDev.append(abs(z - dataMean.__UpperBound)**2)

        LSDev = (sum(LDev))/DataLen
        USDev = (sum(UDev))/DataLen
        return Interval(LSDev, USDev)

    def mode(*args):
        NotImplemented





# a = Interval(1,2)
# b = Interval(3,4)
# c = Interval(-2,5)
# d = Interval(-7,-4)

##.sort() function to sort numbers for median
##list1.count(x) function to help with mode
