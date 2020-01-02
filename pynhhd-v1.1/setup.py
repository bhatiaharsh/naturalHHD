from distutils.core import setup

setup(
    name='pynhhd',
    version='1.1.0',
    description='Python tool to compute the natural Helmholtz-Hodge decomposition.',
    author='Harsh Bhatia',
    author_email='hbhatia@llnl.gov',
    url='https://github.com/bhatiaharsh/naturalHHD',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    license='BSD License',
    python_requires='>=3.6',
    packages=['pynhhd', 'pynhhd.utils'],
    package_dir = {'pynhhd' : 'pynhhd'}
)
