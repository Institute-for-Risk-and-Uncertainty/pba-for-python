from distutils.core import setup

setup(
    name='pba',
    version='v001dev',
    packages=['pba',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README').read(),
)

'''
python3 setup.py sdist
sudo pip uninstall pba
sudo pip install /Users/nickgray/Documents/PhD/code/pba.py/dist/pba-0dev.tar.gz
'''
