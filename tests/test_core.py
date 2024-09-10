import pytest

from pba import *


def test_envelope():
    # Test case 1: Testing with intervals only
    i = [
        I(0,1),
        I(2,3),
        I(-1,2),
        I(5,6)
    ]
    result = envelope(*i)
    assert is_same_as(result, I(-1,6))

    # Test case 2: Testing with interval and floats
    i = [
        -2,
        I(0,1),
        I(2,3),
        7,
        I(5,6)
    ]
    result = envelope(*i)
    assert is_same_as(result, I(-2,7))
    
    # Test case 3: Testing with Pboxes
    i = [
        box(0,1),
        box(2,3),
        box(-1,2),
        box(5,6)
    ]
    result = envelope(*i)
    assert is_same_as(result, box(-1,6))
    
    # Test case 4: Testing with Pboxes, intervals and floats
    i = [
        2.34,
        3.45,
        box(0,1),
        I(2,3),
        box(-1,2),
        I(5,6)
    ]
    result = envelope(*i)
    assert is_same_as(result, box(-1,6))
    
    # Check for value error if no Pbox or Interval given
    i = [
        2.34,
        3.45,
        0.1,
        0.2
    ]
    with pytest.raises(ValueError):
        result = envelope(*i)
    
    # Check for value error if less than 2 arguments given
    i = [
        2.34
    ]
    with pytest.raises(ValueError):
        result = envelope(*i)