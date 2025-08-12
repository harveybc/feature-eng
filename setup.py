from setuptools import setup, find_packages

setup(
    name='feature_eng',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'feature_eng=app.main:main'
        ],
        'feature_eng.features': [
            'default=plugins.features.tech_indicator:Plugin',
            'tech_indicator=plugins.features.tech_indicator:Plugin',
            'technical_indicator=plugins.features.tech_indicator:Plugin',
            'ssa=plugins.features.ssa:Plugin',
            'fft=plugins.features.fft:Plugin'
        ],
        'feature_eng.post_processors': [
            'decomposition=app.plugins.post_processors.decomposition_post_processor:DecompositionPostProcessor'
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
