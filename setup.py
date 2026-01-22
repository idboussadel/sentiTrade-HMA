"""
Setup file for SentiTrade-HMA v2.0
"""
from setuptools import setup, find_packages

setup(
    name='sentitrade-hma',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'yfinance',
        'newsapi-python',
        'pyyaml',
        'python-dotenv',
        'matplotlib',
        'seaborn',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)
