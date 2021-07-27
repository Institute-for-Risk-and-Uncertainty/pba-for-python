from distutils.core import setup

setup(
    name='pba',
    version='0.9.0',
    packages=['pba',],
    license='MIT License',
    long_description=open('README.rst').read(),
    install_requires=[
        'numpy>=1.21.1',
        'scipy>=1.7.0',
        'matplotlib>=3.3.2'],
    python_requires='>=3',
    url='https://github.com/Institute-for-Risk-and-Uncertainty/pba-for-python',
    author='Nick Gray',
    author_email = 'nickgray@liv.ac.uk'
)
# RUN THIS CODE
'''
python3 setup.py sdist
python3 -m twine upload dist/pba-

pip uninstall pba
pip install /Users/nickgray/Documents/PhD/pbacode/pba-for-python/dist/pba-0.8.11.tar.gz
'''
