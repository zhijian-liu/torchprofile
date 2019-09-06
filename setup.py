from setuptools import setup, find_packages

from torchprofile import __version__

setup(
    name='torchprofile',
    version=__version__,
    packages=find_packages(exclude=['examples', 'tests']),
    url='https://github.com/mit-han-lab/torchprofile',
    install_requires=[
        'numpy>=1.14',
        'torch>=1.2',
        'torchvision>=0.4'
    ]
)
