import pytest

from pba import Interval, is_same_as, Logical, PM

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
        
    # test creation with Interval.pm
    a = PM(10,1)
    assert a.left == 9 and a.right == 11
        
    
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
        
def test_Interval_lt():
    a = Interval(0,1)
    b = Interval(2,3)
    c = Interval(0.5,2.5)
    d = Interval(1,2)
    
    assert a < b
    assert not b < a
    assert is_same_as(c < a, Logical(0,1))
    assert is_same_as(c < b, Logical(0,1))
    assert is_same_as(a < d, Logical(0,1))
    assert a < 1.1
    assert not a < -0.1
    assert is_same_as(a < 0.5, Logical(0,1))
    assert is_same_as(a < 1, Logical(0,1))
    # Also test __rgt__
    assert 1.1 > a
    assert not -0.1 > a
    assert is_same_as(0.5 > a, Logical(0,1))
    assert is_same_as(1 > a, Logical(0,1))
    
    with pytest.raises(TypeError):
        a < 'a' 
        
def test_Interval_gt():
    a = Interval(0,1)
    b = Interval(2,3)
    c = Interval(0.5,2.5)
    d = Interval(1,2)
    
    assert not a > b
    assert b > a
    assert is_same_as(c > a, Logical(0,1))
    assert is_same_as(c > b, Logical(0,1))
    assert is_same_as(a > d, Logical(0,1))
    assert not a > 1.1
    assert a > -0.1
    assert is_same_as(a > 0.5, Logical(0,1))
    assert is_same_as(a > 1, Logical(0,1))
    # Also test __rlt__
    assert 1.1 > a
    assert not -0.1 > a
    assert is_same_as(0.5 > a, Logical(0,1))
    assert is_same_as(1 > a, Logical(0,1))
    
    with pytest.raises(TypeError):
        a < 'a' 
        
def test_Interval_eq():
    a = Interval(0,1)
    b = Interval(1,2)
    c = Interval(1.5,2.5)

    assert is_same_as(a == b, Logical(0,1))
    assert is_same_as(b == c, Logical(0,1))
    assert not a == c

    assert is_same_as(1 == a, Logical(0,1))
    
def test_Interval_neq():
    a = Interval(0,1)
    b = Interval(1,2)
    c = Interval(1.5,2.5)

    assert is_same_as(a == b, Logical(0,1))
    assert is_same_as(b == c, Logical(0,1))
    assert not a == c

    assert is_same_as(1 == a, Logical(0,1))
    
def test_repr():
    import pba
    a = Interval(0,1)
    b = PM(0,1)
    pba.interval.pm_repr()
    assert a.__repr__() == 'Interval [0.5 ± 0.5]'
    assert b.__repr__() == 'Interval [0 ± 1]'
    pba.interval.lr_repr()
    assert a.__repr__() == 'Interval [0, 1]'
    assert b.__repr__() == 'Interval [-1, 1]'
    