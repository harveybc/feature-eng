from setuptools import setup, find_packages

setup(
    name='feature-eng',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'feature_eng=app.main:main'
        ],
        'feature_eng.plugins': [
            'default=app.plugins.technical_indicator:Plugin',
            'ssa=app.plugins.ssa:Plugin',
            'fft=app.plugins.fft:Plugin'
        ]
    },
    install_requires=[
            'numpy',
            'pandas',
            'h5py',
            'scipy',
            'build',
            'pytest',
            'pdoc3'
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A timeseries prediction system that supports dynamic loading of predictor plugins for processing time series data.'
)
