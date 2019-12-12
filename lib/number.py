#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:59:25 2017
Modified on Thu Oct 10 18:45:37 2019

@author: marco
"""


class interval():
    def __repr__(self): # return
        return "[%g, %g]"%(self.__lo,self.__hi)

    def __str__(self): # print
        return "[%g, %g]"%(self.__lo,self.__hi)

    def __init__(self,*args):
        self.name = "" # initialise this with an empty string

        if (args is None) | (len(args)==0):
            from numpy import inf
            lo = -inf
            hi = inf
            mid = 0
            rad = inf
            width = inf
        if len(args)==1:
            lo, hi = args[0], args[0]
            mid = (lo + hi)/2
            rad = 0
            width = 0
        if len(args)==2:
            lo, hi = args[0], args[1]
            mid = (lo + hi)/2
            rad = (hi - lo)/2
            width = hi - lo

        self.__lo = lo
        self.__hi = hi
        self.__m = mid
        self.__r = rad
        self.__w = width

        self.__zeroin = False
        if (lo <= 0) & (hi >= 0):
            self.__zeroin = True

     # Some handy methods
    def inf(self):
        return self.__lo
    def sup(self):
        return self.__hi
    def mid(self):
        return self.__m
    def rad(self):
        return self.__r
    def width(self):
        return self.__w
    def onzero(self):
        return self.__zeroin
    def slider(self,p):
        return self.__lo + p * self.__w

    def contains(self,other):
        if other.__class__.__name__ != "interval":
            other = interval(other,other)
        return (self.inf() <= other.inf()) & (self.sup() >= other.sup())
    def inside(self,other):
        if other.__class__.__name__ != "interval":
            other = interval(other,other)
        return (self.inf() >= other.inf()) & (self.sup() <= other.sup())

    #------------------------------------------------------------------------------------------------------
    # Override arithmetical operations START
    #------------------------------------------------------------------------------------------------------
    def __add__(self,other):
        if other.__class__.__name__ in ("int", "float"):
            addL = self.__lo + other
            addH = self.__hi + other
            return interval(addL,addH)
        elif other.__class__.__name__ == "interval":
            addL = self.__lo + other.__lo
            addH = self.__hi + other.__hi
            return interval(addL,addH)
        else:
            return NotImplemented

    def __radd__(self, left):
        if left.__class__.__name__ in ("int", "float"):
            addL = left + self.__lo
            addH = left + self.__hi
            return self.__add__(left)
        else:
            return NotImplemented #print("Error: addition is allowed only between intervals, integers and floats")

    def __sub__(self, other):
        if other.__class__.__name__ in ("int", "float"):
            subL = self.__lo - other
            subH = self.__hi - other
            return interval(subL,subH)
        elif other.__class__.__name__ == "interval":
            subL = self.__lo - other.__hi
            subH = self.__hi - other.__lo
            return interval(subL,subH)

    def __rsub__(self, left):
        if left.__class__.__name__ in ("int", "float"):
            subL = left - self.__hi
            subH = left - self.__lo
            return interval(subL,subH)
        else:
            return NotImplemented #print("Error: subtraction is allowed only between intervals, integers and floats")


    def __mul__(self,other):
        if other.__class__.__name__ in ("int", "float"):
            if other>0:
                mulL = self.__lo * other
                mulH = self.__hi * other
            else:
                mulL = self.__hi * other
                mulH = self.__lo * other
        elif other.__class__.__name__ == "interval":
            if (self.__lo>=0) & (other.__lo>=0):
                mulL = self.__lo * other.__lo
                mulH = self.__hi * other.__hi
            elif (self.__lo>=0) & ((other.__lo<0) & (other.__hi>0)):
                mulL = self.__hi * other.__lo
                mulH = self.__hi * other.__hi
            elif (self.__lo>=0) & (other.__hi<=0):
                mulL = self.__hi * other.__lo
                mulH = self.__lo * other.__hi
            elif ((self.__lo<0) & (self.__hi>0)) & (other.__lo>=0):
                mulL = self.__lo * other.__hi
                mulH = self.__hi * other.__hi
            elif ((self.__lo<0) & (self.__hi>0)) & ((other.__lo<0) & (other.__hi>0)):
                mulL1 = self.__lo * other.__hi
                mulL2 = self.__hi * other.__lo
                mulL = min(mulL1,mulL2)
                mulH1 = self.__lo * other.__lo
                mulH2 = self.__hi * other.__hi
                mulH = max(mulH1,mulH2)
            elif ((self.__lo<0) & (self.__hi>0)) & (other.__hi<=0):
                mulL = self.__hi * other.__lo
                mulH = self.__lo * other.__lo
            elif (self.__hi<=0) & (other.__lo>=0):
                mulL = self.__lo * other.__hi
                mulH = self.__hi * other.__lo
            elif (self.__hi<=0) & ((other.__lo<0) & (other.__hi>0)):
                mulL = self.__lo * other.__hi
                mulH = self.__lo * other.__lo
            elif (self.__hi<=0) & (other.__hi<=0):
                mulL = self.__hi * other.__hi
                mulH = self.__lo * other.__lo
        return interval(mulL,mulH)

    def __rmul__(self, left):
        if left.__class__.__name__ in ("int", "float"):
            return self.__mul__(left)
        else:
            return NotImplemented


    def __truediv__(self,other):
        if other.__class__.__name__ in ("int", "float"):
            if other>0:
                divL = self.__lo / other
                divH = self.__hi / other
            elif other<0:
                divL = self.__hi / other
                divH = self.__lo / other
        elif other.__class__.__name__ == "interval":
            if other.onzero():
                raise Warning("Division by interval containing zero not allowed")
            if (self.__lo>=0) & (other.__lo>0):
                divL = self.__lo/other.__hi
                divH = self.__hi/other.__lo
            elif ((self.__lo<0) & (self.__hi>0)) & (other.__lo>0):
                divL = self.__lo/other.__lo
                divH = self.__hi/other.__lo
            elif (self.__hi<=0) & (other.__lo>0):
                divL = self.__lo/other.__lo
                divH = self.__hi/other.__hi
            elif (self.__lo>=0) & (other.__hi<0):
                divL = self.__hi/other.__hi
                divH = self.__lo/other.__lo
            elif ((self.__lo<0) & (self.__hi>0)) & (other.__hi<0):
                divL = self.__hi/other.__hi
                divH = self.__lo/other.__hi
            elif (self.__hi<=0) & (other.__hi<0):
                divL = self.__hi/other.__lo
                divH = self.__lo/other.__hi
        return interval(divL,divH)

    def __rtruediv__(self, left):
        if left.__class__.__name__ in ("int", "float"):
            if left>0:
                if (self.__lo>0):
                    divL = left / self.__hi
                    divH = left / self.__lo
                elif (self.__hi<0):
                    divL = left / self.__hi
                    divH = left / self.__lo
                else:
                    # this should not return an error, but rather an unbounded interval
                    print("Division is allowed for intervals not containing the zero")
                    raise ZeroDivisionError
            elif left<0:
                if (self.__lo>0):
                    divL = left / self.__lo
                    divH = left / self.__hi
                elif (self.__hi<0):
                    divL = left / self.__lo
                    divH = left / self.__hi
                else:
                    # this should not return an error, but rather an unbounded interval
                    print("Division is allowed for intervals not containing the zero")
                    raise ZeroDivisionError
            return interval(divL,divH)
        else:
            return NotImplemented

    def __pow__(self,other):
        if other.__class__.__name__ == "interval":
            return NotImplemented #print("Power elevation requires a new operator")
        elif other.__class__.__name__ in ("int", "float"):
            if (other%2==0) | (other%2==1):
                other = int(other)
            if other.__class__.__name__ == "int":
                if other > 0:
                    if other%2 == 0: # even power
                        if self.__lo >= 0:
                            powL = self.__lo ** other
                            powH = self.__hi ** other
                        elif self.__hi < 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        else: # interval contains zero
                            H = max(-self.__lo,self.__hi)
                            powL = 0
                            powH = H ** other
                    elif other%2 == 1: # odd power
                        powL = self.__lo ** other
                        powH = self.__hi ** other
                elif other < 0:
                    if other%2 == 0: # even power
                        if self.__lo >= 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        elif self.__hi < 0:
                            powL = self.__lo ** other
                            powH = self.__hi ** other
                        else: # interval contains zero
                            print("Error. \nThe interval contains zero, so negative powers should return \u00B1 Infinity")
                    elif other%2 == 1: # odd power
                        if self.__lo != 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        else: # interval contains zero
                            print("Error. \nThe interval contains zero, so negative powers should return \u00B1 Infinity")
            elif other.__class__.__name__ == "float":
                if self.__lo >= 0:
                    if other > 0:
                        powL = self.__lo ** other
                        powH = self.__hi ** other
                    elif other < 0:
                        powL = self.__hi ** other
                        powH = self.__lo ** other
                    elif other == 0:
                        powL = 1
                        powH = 1
        return interval(powL,powH)

    def __rpow__(self,left):
        return NotImplemented
#         if left < 0:
#             powL = left ** self.__hi
#             powH = left ** self.__lo
#         elif left > 0:
#             powL = left ** self.__lo
#             powH = left ** self.__hi
#         elif left == 0:
#             #if (self.__lo < 0) & (self.__hi > 0) | (self.__lo == 0) | (self.__hi == 0):
#             if slef.__lo < 0:
#                 raise NameError("Numbers containing zeros cannot be raised to a negative power")
#             powL, powH = 1
#         return interval(powL,powH)

    def __lt__(self, other):
        if other.__class__.__name__ == "interval":
            return self.sup() < other.inf()
        elif other.__class__.__name__ in ("int","float"):
            return self.sup() < other
    def __rlt__(self,left):
        if left.__class__.__name__ == "interval":
            return left.sup() < self.inf()
        elif left.__class__.__name__ in ("int","float"):
            return left < self.inf()

    def __gt__(self, other):
        if other.__class__.__name__ == "interval":
            return self.inf() > other.sup()
        elif other.__class__.__name__ in ("int","float"):
            return self.inf() > other
    def __rgt__(self, left):
        if left.__class__.__name__ == "interval":
            return left.inf() > self.sup()
        elif left.__class__.__name__ in ("int","float"):
            return left > self.sup()

    def __le__(self, other):
        if other.__class__.__name__ == "interval":
            return self.sup() <= other.inf()
        elif other.__class__.__name__ in ("int","float"):
            return self.sup() <= other
    def __rle__(self,left):
        if left.__class__.__name__ == "interval":
            return left.sup() <= self.inf()
        elif left.__class__.__name__ in ("int","float"):
            return left <= self.inf()

    def __ge__(self, other):
        if other.__class__.__name__ == "interval":
            return self.inf() >= other.sup()
        elif other.__class__.__name__ in ("int","float"):
            return self.inf() >= other
    def __rge__(self, left):
        if left.__class__.__name__ == "interval":
            return left.inf() >= self.sup()
        elif left.__class__.__name__ in ("int","float"):
            return left >= self.sup()

    def __eq__(self, other):
        if other.__class__.__name__ == "interval":
            return (self.sup() == other.sup()) & (self.inf() == other.inf())
        elif other.__class__.__name__ in ("int","float"):
            return (self.sup() == other) & (self.inf() == other)

    def __ne__(self, other):
        if other.__class__.__name__ == "interval":
            return (self.sup() != other.sup()) & (self.inf() != other.inf())
        else:
            return (self.sup() != other) & (self.inf() != other)

    #------------------------------------------------------------------------------------------------------
    # Override arithmetical operations END
    #------------------------------------------------------------------------------------------------------






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:24:34 2019
Modified on Thu Oct 29 19:21:52 2019

@author: marco de-angelis

The code below does some dependency tracking using AD
"""

class watch():
    # A class variable, counting the number of variables
    # this will be used in the "smart" tracking
    variables = 0
    rootVariables = 0
    constants = 0
    roots = {}
#     def __repr__(self):
#         return prettyprint(self)

#     def __str__(self):
#         return prettyprint(self)
    def __init__(self,**kwargs):
        self.name = ""
        self.__classes = ('watch','number','constant')
#         self.__bob = id(self)
#         self.__apu = hash(self)
        self.__tape = []
        self.__hashtable = {}
        self.__children = []
        for key, val in kwargs.items():
            if key == "name":
                self.name = val
                # self.__originalname = val
            if key == "value":
                self.__value = val
            if key == "kind":
                # this is mandatory default should be variable
                self.__type = val
            if key == "gradient":
                self.__gradient = val
            if key == "tracker":
                # self is not root, i.e. variable has nested variables on which it depends
                self.__hashtable = val
            if key == "tape":
                self.__tape = val
        self.__isroot = False
        if (len(self.__hashtable) == 0) & (self.__type == "variable"):
            # self is root, i.e. it does not depend on any other variable
            watch.rootVariables += 1 # increase the counter of independent/root variables
            self.__isroot =True # spell out with a boolean that self is a root
            self.__rootindex = watch.rootVariables  # # assign index to the root variable. This is a necessary step to be able to make sense of ordered outputs, such as the gradient.
            # The index coincides with the order of assignment: so first assigned variable has index 1, second assigned var 2, and so on.
            self.__bob = "root_"+str(self.__rootindex)
            self.__hashtable[self.__bob] = {}
            self.__hashtable[self.__bob]["value"] = self.__value
            self.__hashtable[self.__bob]["index"] = self.__rootindex
            self.__hashtable[self.__bob]["reps"] = 1
            self.__hashtable[self.__bob]["seed"] = 1
            self.__hashtable[self.__bob]["name"] = self.name
        elif self.__type == "variable":
            watch.variables += 1
            self.__index = watch.variables
            self.__bob = "var_"+str(watch.variables)
            # self.__fullname = "h:"+str(self.__apu)
        elif self.__type == "constant":
            watch.constants += 1
            self.__bob = "c_"+str(watch.constants)
            self.__index = 0
            self.__hashtable[self.__bob] = {}
            self.__hashtable[self.__bob]["value"] = self.__value
            self.__hashtable[self.__bob]["index"] = self.__index
            self.__hashtable[self.__bob]["reps"] = 0
            self.__hashtable[self.__bob]["seed"] = 0
            self.__hashtable[self.__bob]["name"] = self.name
        if len(self.__tape) == 0:
            self.__tape = [self.__bob]
        if self.name == "":
            self.name = self.__bob
        # self.__surname = self.name+" ("+str(self.__hashtable[self.__bob]["index"])+")"


    def gettype(self):
        return self.__type

    def derivative(self):
        return self.__der

    def getgradient(self):
        return self.__gradient

    def value(self):
        return self.__value

    def superclass(self):
        return self.__class__.__name__

    def getid(self):
        return self.__bob

    def getbob(self):
        return self.__bob

    def gettape(self):
        return self.__tape

    def isroot(self):
        return self.__isroot

    def getindex(self):
        return self.__index

    def gethashtable(self):
        return self.__hashtable

    def surname(self):
        return self.__surname

    def mergebobs(self,other):
        xd=self.gethashtable().copy()
        yd=other.gethashtable().copy()
        xset = {i for i in xd.keys()}
        yset = {i for i in yd.keys()}
        xyset = xset.union(yset)
        xyd = {}
        for t in xyset:
            for tt in xset.difference(yset):
                yd[tt] = {}
                yd[tt]["name"] = ""
                yd[tt]["value"] = xd[tt]["value"]
                yd[tt]["reps"] = 0
                yd[tt]["seed"] = 0
                yd[tt]["index"] = xd[tt]["index"]
            for tt in yset.difference(xset):
                xd[tt] = {}
                xd[tt]["name"] = ""
                xd[tt]["value"] = yd[tt]["value"]
                xd[tt]["reps"] = 0
                xd[tt]["seed"] = 0
                xd[tt]["index"] = yd[tt]["index"]
            xyd[t] = {}
            xyd[t]["reps"] = xd[t]["reps"] + yd[t]["reps"]
#             xyd[t]["value"] = xd[t]["value"] + yd[t]["value"]

            if (yd[t]["name"] == ""):
                xyd[t]["name"] = xd[t]["name"]
            else:
                xyd[t]["name"] = yd[t]["name"]
            #xyd[t]["name"] = xd[t]["name"] + yd[t]["name"]

            xyd[t]["index"] = xd[t]["index"] # or equivalently yd[t]["index"]
            xyd[t]["value"] = xd[t]["value"]
        if self.__type == "constant":
            del xyd[self.getbob()]
        elif other.__type == "constant":
            # set_trace()
            del xyd[other.getbob()]
        return xyd, xd, yd


    def tracker(self,other,op,side):
        # ops = ["+","-","*","/","**"]  # these are the currently supported operations
#         if side == "left":
#             other = left
        if (self.__type == "constant") & (other.__type == "constant"):
            if side == "right":
#                 tape = [self.getbob()] + [op] + [other.getbob()]
                tape = [str(self.value())] + [op] + [str(other.value())]
            elif side == "left":
                tape = [str(other.value())] + [op] + [str(self.value())]
        elif self.__type == "constant":
            if side == "right":
                tape = [str(self.value())] + [op] + other.gettape()
            elif side == "left":
                tape = other.gettape() + [op] + [str(self.value())]
        elif other.__type == "constant":
            if side == "right":
                tape = self.gettape() + [op] + [str(other.value())]
            elif side == "left":
                tape = [str(other.value())] + [op] + self.gettape()
        else:
            if side == "right":
                tape = self.gettape() + [op] + other.gettape()
            elif side == "left":
                tape = other.gettape() + [op] + self.gettape()
        return tape



    def __add__(self,other):
        if other.__class__.__name__ not in self.__classes:
            other = watch(value=other,kind="constant")
        tape = self.tracker(other,"+","right")  # this will also output the tree
        omni, sese, alia = self.mergebobs(other) # output the hashtable from the merging of the two hashtables
        val = self.__value + other.__value # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = sese[t]["seed"] + alia[t]["seed"]  # update the seeds and compute the derivative for addition
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def __radd__(self,left):
        if left.__class__.__name__ not in self.__classes:
            left = watch(value=left,kind="constant")
        tape = self.tracker(left,"+","left")  # this will also output the tree
        omni, sese, alia = self.mergebobs(left) # output the hashtable from the merging of the two objects
        val = left.__value + self.__value  # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = alia[t]["seed"] + sese[t]["seed"]   # compute the derivative for addition
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z

    def  __sub__(self,other):
        if other.__class__.__name__ not in self.__classes:
            other = watch(value=other,kind="constant")
        tape = self.tracker(other,"-","right")  # this will also output the tree
        omni, sese, alia = self.mergebobs(other) # output the hashtable from the merging of the two hashtables
        val = self.__value - other.__value # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = sese[t]["seed"] - alia[t]["seed"]  # update the seeds and compute the derivative
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def __rsub__(self,left):
        if left.__class__.__name__ not in self.__classes:
            left = watch(value=left,kind="constant")
        tape = self.tracker(left,"-","left")  # this will also output the tree
        omni, sese, alia = self.mergebobs(left) # output the hashtable from the merging of the two objects
        val = left.__value - self.__value  # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = alia[t]["seed"] - sese[t]["seed"]   # compute the derivative
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z

    def __mul__(self,other):
        if other.__class__.__name__ not in self.__classes:
            other = watch(value=other,kind="constant")
        tape = self.tracker(other,"*","right")  # this will also output the tree
        omni, sese, alia = self.mergebobs(other) # output the hashtable from the merging of the two hashtables
        val = self.__value * other.__value # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = self.__value * alia[t]["seed"] + sese[t]["seed"] * other.__value # update the seeds and compute the derivative
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def __rmul__(self,left):
        if left.__class__.__name__ not in self.__classes:
            left = watch(value=left,kind="constant")
        tape = self.tracker(left,"*","left")  # this will also output the tree
        omni, sese, alia = self.mergebobs(left) # output the hashtable from the merging of the two objects
        val = left.__value * self.__value  # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = self.__value * alia[t]["seed"] + sese[t]["seed"] * left.__value   # compute the derivative
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def __truediv__(self,other):
        if other.__class__.__name__ not in self.__classes:
            other = watch(value=other,kind="constant")
        tape = self.tracker(other,"/","right")  # this will also output the tree
        omni, sese, alia = self.mergebobs(other) # output the hashtable from the merging of the two hashtables
        val = self.__value / other.__value # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = (sese[t]["seed"] - (self.__value/other.__value) * alia[t]["seed"]) / other.__value  # update the seeds and compute the derivative
            #self.__value * alia[t]["seed"] + sese[t]["seed"] * other.__value
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def __rtruediv__(self,left):
        if left.__class__.__name__ not in self.__classes:
            left = watch(value=left,kind="constant")
        tape = self.tracker(left,"/","left")  # this will also output the tree
        omni, sese, alia = self.mergebobs(left) # output the hashtable from the merging of the two objects
        val = left.__value / self.__value  # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = (alia[t]["seed"] - (left.__value/self.__value) * sese[t]["seed"]) / self.__value  # update the seeds and compute the derivative
            # self.__value * alia[t]["seed"] * sese[t]["seed"] * left.__value   # compute the derivative for addition
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def  __pow__(self,other):
        if other.__class__.__name__ not in self.__classes:
            other = watch(value=other,kind="constant")
        tape = self.tracker(other,"^","right")  # this will also output the tree
        omni, sese, alia = self.mergebobs(other) # output the hashtable from the merging of the two hashtables
        val = self.__value ** other.__value # add the numbers
        for t in {i for i in omni.keys()}:
            omni[t]["seed"] = sese[t]["seed"] * other.__value * self.__value**(other.__value - 1)
            # sese[t]["seed"] - ((self.__value/other.__val) * alia[t]["seed"]) / other.__value  # update the seeds and compute the derivative
        indexes = [omni[i]["index"] for i in omni.keys()] # output the list of indexes
        indexes.sort() # this ensures that the gradient is an ordered list. The order reflects the creation of the variables. So the first variable being created has index 0, the second variable index 1 and so on..
        grad = [omni["root_"+str(indexes[i])]["seed"] for i in range(len(omni))] # populate the gradient with the derivatives
        z = watch(kind="variable",value=val,gradient=grad,tracker=omni,tape=tape)
        return z
    def __rpow__(self,left):
        raise Warning("Not implemented yet")
        return None


#https://docs.python.org/3.1/glossary.html





class number(watch):
    def __init__(self,*args):
        self.name = ''
        for arg in args:
            if arg.__class__.__name__=='str':
                self.name = arg
                self.__originalname = arg
            # elif arg.__class__.__name__== ('float' or 'int'):
            else:  # this can be anything for which binary operations are defined / allowed
                self.__value = arg

        super().__init__(kind='variable',value=self.__value,name=self.__originalname)

    def superclass(self):
        return self.__class__.__bases__[0].__name__


class constant(watch):
    def __init__(self,*args):
        self.name = ''
        for arg in args:
            if arg.__class__.__name__=='str':
                self.name = arg
                self.__originalname = arg
            # elif arg.__class__.__name__== ('float' or 'int'):
            elif arg.__class__.__name__ in ('int','float'):
                self.__value = arg
            else:
                raise Warning("Allowed types for constants are float or integer")

        super().__init__(kind='constant',value=self.__value,name=self.name)

    def superclass(self):
        return self.__class__.__bases__[0].__name__
