from setuptools import setup, find_packages

install_requires = ['rdkit', 'networkx']

setup(
    name='UniDockTools',
    version='1.0.1',
    author='DP BayMax',
    url='https://github.com/UR-Free/UniDockTools',
    description="A data processing tool for UniDock input and output",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    
    package_data={
        "UniDock": ["data"],
    },
    entry_points={
        'console_scripts': [
            'Unidock = UniDockTools.unidock:main',
        ],
    },
)
