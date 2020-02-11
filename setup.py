from distutils.core import setup

setup(
    name='pba',
    version='0.4.4',
    packages=['pba',],
    license='MIT License',
    long_description=open('README.rst').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy>=1.17.3',
        'scipy>=1.4.0',
        'matplotlib>=3.0.0'],
    python_requires='>=3',
    url='https://gitlab.com/nickgray1995/pba-for-python',
    author='Nick Gray',
    author_email = 'nickgray@liv.ac.uk'
)
# RUN THIS CODE
'''
python3 setup.py sdist
pip uninstall pba
pip install /Users/nickgray/Documents/PhD/code/pba-for-python/dist/pba-0.4.dev4.tar.gz --user

TEST
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
python3 -m twine upload dist/*
'''
