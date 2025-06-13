from setuptools import setup, find_packages

setup(
    name='hygraph_core',       # Name of your package
    version='0.3.9.2',                 # Version of your package
    description='A package for managing and analyzing HyGraph data',
    long_description=open('README.md').read(),  # Read the content of README.md
    long_description_content_type='text/markdown',
    author='Mouna',
    author_email='',
    url='https://github.com/',  # URL to the project repository
    packages=find_packages(),        # Automatically find packages in the directory
    install_requires=[
        'numpy',  # For numerical operations
        'pandas',  # For timestamp management and data manipulation
        'xarray',  # For handling multidimensional time series
        'scipy',  # For scientific computations, e.g., distance metrics
        'fastdtw',  # For Dynamic Time Warping similarity computation
        'networkx',  # For graph management and operations
        'matplotlib',  # Optional, for visualization of time series or graphs
        'scikit-learn',  # Optional, for advanced similarity metrics or clustering (cosine similarity, etc.)
        'igraph',
        'statsmodels',
        'ijson',
        'polars'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',         # Specify required Python version
)
