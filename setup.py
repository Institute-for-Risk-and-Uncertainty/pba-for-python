from distutils.core import setup

setup(
    name='pba',
    version='v003',
    packages=['pba',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README').read(),
    install_requires=['numpy','scipy','matplotlib'],
    url='https://gitlab.com/nickgray1995/pba-for-python',
    author='Nick Gray',
    author_email = 'nickgray@liv.ac.uk'
)
# RUN THIS CODE
'''
python3 setup.py sdist
pip uninstall pba
pip install /Users/nickgray/Documents/PhD/code/pba.py/dist/pba-dev.tar.gz --user
'''
