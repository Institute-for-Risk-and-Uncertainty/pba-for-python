import pytest

from pba import *
import warnings

def test_pbox_creation():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )
    
    b = Pbox(
        [Interval(0,1),Interval(1,2),Interval(2,3),Interval(2,4)],
    )
    assert is_same_as(b,Pbox(left = [0,1,2,2],right=[1,2,3,4],steps=4), exact_pbox = False )
    
def test_pbox_add():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3    
    )
    
    # Generic addition
    c = a + b
    ans = Pbox(
        left = [1,2,3],
        right = [5,6,7],
        steps = 3
    )
    assert is_same_as(c,ans, exact_pbox = False )

def test_pbox_add_f():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3    
    )
    # Frechet addition
    c = a.add(b,method='f')
    ans = Pbox(
        left = [1,2,3],
        right = [5,6,7],
        steps = 3
    )
    # same answer as generic addition
    assert is_same_as(c,ans, exact_pbox = False )

def test_pbox_add_p():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3    
    )
    # Perfect addition
    c = a.add(b,method='p')
    ans = Pbox(
        left = [1,3,5],
        right = [3,5,7],
        steps = 3
    )
    assert is_same_as(c,ans, exact_pbox = False )

def test_pbox_add_o():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3    
    )
    # Opposite addition
    c = a.add(b,method='o')
    ans = Pbox(
        left = [4,4,4],
        right = [4,4,4],
        steps = 3
    )
    assert is_same_as(c,ans, exact_pbox = False )
    
def test_pbox_add_i():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3    
    )
    # independence addition
    c = a.add(b,method='i')
    ans = Pbox(
        left = [1,3,5],
        right = [3,5,7],
        steps = 3
    )
    assert is_same_as(c,ans, exact_pbox = False )
        
    
def test_pbox_sub():
    a = Pbox(
        left = [0,0,0],
        right= [1,1,1],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3    
    )
    
    # Generic subtraction
    c = a - b
    ans = Pbox(
        left = [-4,-3,-2],
        right = [-2,-1,0],
        steps = 3
    )
    assert is_same_as(c,ans, exact_pbox = False )
    
def test_pbox_mul():
    a = Pbox(
        left = [0,0,0],
        right= [1,2,3],
        steps = 3
    )   
    b = Pbox(
        left = [1,2,3],
        right= [2,3,4],
        steps=3  
    )
    
    # Generic multiplication
    c = a * b
    ans = Pbox(
        left = [0,0,0],
        right = [4,8,12],
        steps = 3
    )
    assert is_same_as(c,ans, exact_pbox = False )