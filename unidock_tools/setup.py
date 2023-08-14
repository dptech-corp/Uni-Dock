from setuptools import setup, find_packages

setup(
    name='unidock_tools',
    version='1.1.2',
    author="DP Technology",
    author_email="yanghe@dp.tech",
    url='https://github.com/UR-Free/unidock_tools',
    description="A data processing tool for Uni-Dock input and output",
    packages=find_packages(),
    install_requires=['rdkit', 'networkx', 'numpy'],
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'Unidock = unidock_tools.unidock:main',
        ],
    },
)
