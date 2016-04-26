from distutils.core import setup

# Read the version number
with open("irt_parameter_estimation/_version.py") as f:
    exec(f.read())

setup(
    name='irt_parameter_estimation',
    version=__version__, # use the same version that's in _version.py
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['irt_parameter_estimation'],
    scripts=[],
    url='http://pypi.python.org/pypi/irt_parameter_estimation/',
    license='LICENSE.txt',
    description='Parameter estimation routines for logistic Item Characteristic Curves (ICC) from Item Response Theory (IRT)',
    long_description=open('README.md').read(),
    install_requires=['numpy>=1.0',
                      'scipy>=0.8'
                     ],
)
