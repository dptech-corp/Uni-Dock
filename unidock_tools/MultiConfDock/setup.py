from setuptools import setup, find_packages

setup(
    name='mcdock',
    version='1.0.0',
    author='Hang Zheng',
    author_email="zhengh@dp.tech",
    description=('Multi-Conformation Docking'),
    keywords='docking',
    install_requires=[
        # "rdkit", 
        "networkx", 
        "numpy"
    ],
    packages=find_packages(),
    zip_safe=False,
    entry_points={'console_scripts': [
        "mcdock=mcdock.MultiConfDock:main_cli"
    ]},
    include_package_data=True
)
