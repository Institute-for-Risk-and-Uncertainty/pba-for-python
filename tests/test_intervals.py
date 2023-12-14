import pytest

from pba import Interval
from numpy.random import default_rng

rng = default_rng()

def test_Interval_creation():
    l = rng.integers(-100,100)
    r = rng.integers(-100,100)
    a = Interval(l,r)
    if l < r:
        assert a.left == l and a.right == r
    else:
        assert a.left == r and a.right == l
        
    
def test_Interval_add():
    a = Interval(10,30)
    b = Interval(-10,20)
    c = a + b
    assert c.left == 0 and c.right == 50
    
def test_Interval_sub():
    a = Interval(10,30)
    b = Interval(-10,30)
    c = a - b
    assert c.left == -20 and c.right == 40
    
def test_Interval_mul():
    a = Interval(7,50)
    b = Interval(-1,0.5)
    c = a * b
    assert c.left == -50 and c.right == 25

def test_Interval_div():
    a = Interval(4,50)
    b = Interval(2,5)
    c = a / b
    assert c.left == 0.8 and c.right == 25
    
def test_Interval_div0():
    a = Interval(4,50)
    b = Interval(-2,5)
    
    with pytest.raises(ZeroDivisionError):
        a / b
