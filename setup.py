from setuptools import setup, find_packages

setup(
    name='feature_eng',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console'
        '_scripts': [
            'feature_eng=app.main:main'
        ],
        'feature_eng.pipeline': [
            'default=plugins.pipeline.default_pipeline:PipelinePlugin'
        ],
        'feature_eng.features': [
            'default=plugins.features.tech_indicator:FeaturePlugin',
            'tech_indicator=plugins.features.tech_indicator:FeaturePlugin'
        ],
        'feature_eng.aligner': [
            'default=plugins.aligner.default_aligner:AlignerPlugin'
        ],
        'feature_eng.post_processor': [
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
