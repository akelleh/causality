from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='causality',

    version='0.0.3',

    description='Tools for causal analysis',
    long_description=long_description,

    url='http://github.com/akelleh/causality',

    author='Adam Kelleher',
    author_email='akelleh@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='causality data analysis inference causal graphs DAG',

    packages=find_packages(exclude=['tests']),

    install_requires=['numpy', 'scipy', 'pandas',
                      'statsmodels', 'networkx', 'patsy', 
                      'pytz', 'python-dateutil', 'decorator',
                      'pytz', 'six']

)
