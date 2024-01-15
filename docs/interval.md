# Intervals

An interval is an uncertain number for which only the endpoints are known, {math}`x=[a,b]`.
This is interpreted as $x$ being between {math}`a` and {math}`b` but with no more information about the value of {math}`x`.
Intervals embody epistemic uncertainty within PBA.

Intervals can be created using either of the following:
```python
>>> import pba
>>> pba.Interval(0,1)
Interval [0,1]
>>> pba.I(2,3)
Interval [2,3]
```
## Arithmetic


## Class Documentation
```{eval-rst}
.. autoclass:: pba.interval.Interval
   :members:
```