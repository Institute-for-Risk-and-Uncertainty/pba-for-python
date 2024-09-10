#%%
### File generates images for documentation
from matplotlib import pyplot as plt
import pba
import numpy as np
import matplotlib

font = {'size'   : 16}

matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rc('font', **font)

#%% From percentiles
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

#%% Pbox interpolation
left = np.array([0,1,2,3,4])
right = np.array([1,2,4,8,16])

p1 = pba.Pbox(left,right,steps=2000,interpolation = 'linear')
p2 = pba.Pbox(left,right,steps=2000,interpolation = 'step')
p3 = pba.Pbox(left,right,steps=2000,interpolation = 'cubicspline')

fig,ax = plt.subplots()


p1.show(figax = (fig,ax),label = 'linear',color='red')
p2.show(figax = (fig,ax),label = 'step',color='blue')
p3.show(figax = (fig,ax),label = 'cubicspline',color='green')
ax.scatter(left, np.linspace(0,1,len(left)),marker='D',color='k',zorder=2,s=50)
ax.scatter(right, np.linspace(0,1,len(right)), marker='D', color='k', zorder=2, s=50)
ax.legend()

fig.savefig('docs/images/interpolation.png')
#%% default arithmetic test
p1 = pba.N([-1,1],1,steps = 1000)
p2 = pba.U([0,1],[1,2],steps = 1000)

c1 = p1+p2
pba.pbox.change_default_arithmetic_method('i')
c2 = p1+p2
pba.pbox.change_default_arithmetic_method('o')
c3 = p1+p2
pba.pbox.change_default_arithmetic_method('p')
c4 = p1+p2

fig,ax = plt.subplots()
c1.show(figax = (fig,ax),label = 'f',color='red')
c2.show(figax = (fig,ax),label = 'i',color='blue')
c3.show(figax = (fig,ax),label = 'o',color='green')
c4.show(figax = (fig,ax),label = 'p',color='purple')
ax.legend()


fig.savefig('docs/images/arithmetic.png')
