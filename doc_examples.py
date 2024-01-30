'''
File generates images for documentation.
'''

from matplotlib import pyplot as plt
import pba

perc = pba.from_percentiles(
    {0: 0,
    0.25: 0.5,
    0.5: pba.I(1,2),
    0.75: pba.I(1.5,2.5),
    1: 3}
)
f1,ax1 = plt.subplots()
perc.plot(figax = (f1,ax1))
f1.savefig('docs/images/from_percentiles.png')