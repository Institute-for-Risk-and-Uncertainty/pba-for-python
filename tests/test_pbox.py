import pytest

from pba import *

def test_pbox_creation():
    a = Pbox(
        left = [0,1,2],
        right= [1,2,3],
        steps = 3
    )
    
    b = Pbox(
        [Interval(0,1),Interval(1,2),Interval(2,3),Interval(2,4)],
    )
    assert is_same_as(b,Pbox(left = [0,1,2,2],right=[1,2,3,4],steps=4))
    
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
    assert is_same_as(c,ans)
    
    # Frechet addition
    c = a.add(b,method='f')
    # same answer as generic addition
    assert is_same_as(c,ans)
    
    # Perfect addition
    c = a.add(b,method='p')
    ans = Pbox(
        left = [1,3,5],
        right = [3,5,7],
        steps = 3
    )
    assert is_same_as(c,ans)
    
    # Opposite addition
    c = a.add(b,method='o')
    ans = Pbox(
        left = [4,4,4],
        right = [4,4,4],
        steps = 3
    )
    assert is_same_as(c,ans)
    
    # independence addition
    c = a.add(b,method='i')
    ans = Pbox(
        left = [1,3,5],
        right = [3,5,7],
        steps = 3
    )
    assert is_same_as(c,ans)
    
def test_pbox_sub():
    a = Pbox(
        left = [10,11,12],
        right= [13,14,15],
        steps = 3
    )   
    b = Pbox(
        left = [0,1,2],
        right= [3,4,5],
        steps = 3    
    )
    
    # Generic subtraction
    c = a - b
    ans = Pbox(
        left = [5,6,7],
        right = [13,14,15],
        steps = 3
    )
    assert is_same_as(c,ans)
    
    # Frechet subtraction
    c = a.sub(b,method='f')
    # same answer as generic subtraction
    assert is_same_as(c,ans)
    
    # Perfect subtraction
    c = a.sub(b,method='p')
    ans = Pbox(
        left = [10,10,10],
        right = [10,10,10],
        steps = 3
    )
    assert is_same_as(c,ans)
    
    # Opposite subtraction
    c = a.sub(b,method='o')
    ans = Pbox(
        left = [5,7,9],
        right = [11,13,15],
        steps = 3
    )
    assert is_same_as(c,ans)
    
    # independence subtraction
    c = a.sub(b,method='i')
    ans = Pbox(
        left = [5,7,9],
        right = [11,13,15],
        steps = 3
    )
    assert is_same_as(c,ans)