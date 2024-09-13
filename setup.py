from setuptools import setup, find_packages

setup(
    name='feature_eng',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'feature_eng=app.main:main'
        ],
        'feature_eng.plugins': [
            'default=app.plugins.tech_indicator:Plugin',
            'tech_indicator=app.plugins.tech_indicator:Plugin',
            'technical_indicator=app.plugins.technical_indicator:Plugin',
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
    description='aa.'
)
