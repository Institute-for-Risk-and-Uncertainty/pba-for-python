from distutils.core import setup

setup(
    name='pba',
    version='0.90.2',
    packages=['pba',],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy>=1.25.2',
        'scipy>=1.11.2',
        'matplotlib>=3.3.2'],
    url='https://github.com/Institute-for-Risk-and-Uncertainty/pba-for-python',
    author='Nick Gray',
    author_email = 'ngg@liv.ac.uk'
)
# RUN THIS CODE
'''
python3 setup.py sdist
python3 -m twine upload dist/pba-

cd docs && make html && cd ..
'''
