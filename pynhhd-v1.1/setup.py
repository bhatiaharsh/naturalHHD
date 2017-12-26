from distutils.core import setup

setup(
    name='pynhhd',
    version='1.1.0',
    description='Python tool to compute the natural Helmholtz-Hodge decomposition.',
    author='Harsh Bhatia',
    author_email='hbhatia@llnl.gov',
    packages=['pynhhd'],
    package_dir = {'pynhhd' : 'pynhhd'}
)
