import pytest

import pba 

def test_is_same_as():
    a = pba.N(0,1)
    b = a
    assert pba.is_same_as(a,b, deep = True)
    c = pba.I(0,1)
    d = c
    assert pba.is_same_as(d,c, deep = True)
    

    a = pba.N([-1,1],1)
    b = 1
    assert not pba.is_same_as(a,b, deep = False)

    c = pba.I(0,1)
    assert not pba.is_same_as(a,c, deep = False)
        

    a = pba.N([-1,1],1)
    b = pba.N([-1,1],1)
    assert pba.is_same_as(a,b, deep = False)

    c = pba.U([1,2],[3,4])
    assert not pba.is_same_as(a,c,deep = False)
    

    a = pba.I(0,1)
    b = pba.I(0,1)
    assert pba.is_same_as(a,b)

    c = pba.I(0,1)
    d = pba.I(-2,1)
    assert not pba.is_same_as(c,d)
    
def test_logical_always():
    
    a = pba.Logical(0,0)
    b = pba.Logical(0,1)
    c = pba.Logical(1,1)
    
    assert not pba.always(a)
    assert not pba.always(b)
    assert pba.always(c)
    del a,b,c

    a = pba.Interval(0,0)
    b = pba.Interval(0,0.3)
    c = pba.Interval(0.3,0.7)
    d = pba.Interval(.7,1)
    e = pba.Interval(1,1)
    f = pba.Interval(-.2,1)
    g = pba.Interval(.2,2)
    
    assert not pba.always(a)
    assert not pba.always(b)
    assert not pba.always(c)
    assert not pba.always(d)
    assert pba.always(e)
    with pytest.raises(ValueError): pba.always(f)
    with pytest.raises(ValueError): pba.always(g)
    del a,b,c,d,e,f,g
    
    a = 0
    b = .3
    c = 1
    d = False
    e = True    
    assert not pba.always(a)
    assert not pba.always(b)
    assert pba.always(c)
    assert not pba.always(d)
    assert pba.always(e)
    
def test_logical_never():
    
    a = pba.Logical(0,0)
    b = pba.Logical(0,1)
    c = pba.Logical(1,1)
    
    assert pba.never(a)
    assert not pba.never(b)
    assert not pba.never(c)
    del a,b,c

    a = pba.Interval(0,0)
    b = pba.Interval(0,0.3)
    c = pba.Interval(0.3,0.7)
    d = pba.Interval(.7,1)
    e = pba.Interval(1,1)
    f = pba.Interval(-.2,1)
    g = pba.Interval(.2,2)
    
    assert pba.never(a)
    assert not pba.never(b)
    assert not pba.never(c)
    assert not pba.never(d)
    assert not pba.never(e)
    with pytest.raises(ValueError): pba.never(f)
    with pytest.raises(ValueError): pba.never(g)
    del a,b,c,d,e,f,g
    
    a = 0
    b = .3
    c = 1
    d = False
    e = True    
    f = 4
    assert pba.never(a)
    assert not pba.never(b)
    assert not pba.never(c)
    assert pba.never(d)
    assert not pba.never(e)
    with pytest.raises(ValueError): pba.never(f)
    
   
def test_logical_sometimes():
    
    a = pba.Logical(0,0)
    b = pba.Logical(0,1)
    c = pba.Logical(1,1)
    
    assert not pba.sometimes(a)
    assert  pba.sometimes(b)
    assert  pba.sometimes(c)
    del a,b,c

    a = pba.Interval(0,0)
    b = pba.Interval(0,0.3)
    c = pba.Interval(0.3,0.7)
    d = pba.Interval(.7,1)
    e = pba.Interval(1,1)
    f = pba.Interval(-.2,1)
    g = pba.Interval(.2,2)
    
    assert not pba.sometimes(a)
    assert  pba.sometimes(b)
    assert  pba.sometimes(c)
    assert  pba.sometimes(d)
    assert  pba.sometimes(e)
    with pytest.raises(ValueError): pba.sometimes(f)
    with pytest.raises(ValueError): pba.sometimes(g)
    del a,b,c,d,e,f,g
    
    a = 0
    b = .3
    c = 1
    d = False
    e = True    
    f = 4
    assert not pba.sometimes(a)
    assert  pba.sometimes(b)
    assert  pba.sometimes(c)
    assert not pba.sometimes(d)
    assert  pba.sometimes(e)
    with pytest.raises(ValueError): pba.sometimes(f)

  
def test_logical_xtimes():
    
    a = pba.Logical(0,0)
    b = pba.Logical(0,1)
    c = pba.Logical(1,1)
    
    assert not pba.xtimes(a)
    assert  pba.xtimes(b)
    assert not pba.xtimes(c)
    del a,b,c

    a = pba.Interval(0,0)
    b = pba.Interval(0,0.3)
    c = pba.Interval(0.3,0.7)
    d = pba.Interval(.7,1)
    e = pba.Interval(1,1)
    f = pba.Interval(-.2,1)
    g = pba.Interval(.2,2)
    
    assert not pba.xtimes(a)
    assert not pba.xtimes(b)
    assert  pba.xtimes(c)
    assert not pba.xtimes(d)
    assert not pba.xtimes(e)
    with pytest.raises(ValueError): pba.xtimes(f)
    with pytest.raises(ValueError): pba.xtimes(g)
    del a,b,c,d,e,f,g
    
    a = 0
    b = .3
    c = 1
    d = False
    e = True    
    f = 4
    assert not pba.xtimes(a)
    assert  pba.xtimes(b)
    assert not pba.xtimes(c)
    assert not pba.xtimes(d)
    assert not pba.xtimes(e)
    with pytest.raises(ValueError): pba.xtimes(f)