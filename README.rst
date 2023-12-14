.. image:: https://readthedocs.org/projects/pba-for-python/badge/?version=latest
    :target: https://pba-for-python.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


PBA
========

PBA is a probability bound analysis library for Python that allows one to create and calculate with probability distributions, intervals, and probability boxes (p-boxes) within Python.

Probability distributions can be specified using ``pba.distname(**args)`` where *distname* is any of the named distributions that scipy.stats supports.  For instance,   pba.N(0,1) specifies a Normal distribution with mean 0 and variance 1. P-boxes can be created by using interval arguments for these distributions.  Intervals can be created using ``pba.I(left, right)`` where *left* and *right* are expressions for the lower and upper limits of the interval.

Features
--------

- Interval arithmetic (see https://en.wikipedia.org/wiki/Interval_arithmetic)
- P-box arithmetic (see https://en.wikipedia.org/wiki/Probability_bounds_analysis)

Installation
-------------

Install pba by running

    pip install pba

Contribute & Support
--------------------

If you are having issues or would like to help with development or have intersting use cases, please let us know.
You can email ngg@liv.ac.uk.

License
--------

The project is licensed under the MIT License.
