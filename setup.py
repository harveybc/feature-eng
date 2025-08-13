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
            'default=fe_plugins.pipeline.default_pipeline:PipelinePlugin'
        ],
        'feature_eng.features': [
            'default=fe_plugins.features.base_features:FeaturePlugin',
            'base_features=fe_plugins.features.base_features:BaseFeaturePlugin',
            'technical_features=fe_plugins.features.technical_features:TechnicalFeaturePlugin',
            'fundamental_features=fe_plugins.features.fundamental_features:FundamentalFeaturePlugin',
            'seasonal_features=fe_plugins.features.seasonal_features:SeasonalFeaturePlugin',
            'high_frequency_features=fe_plugins.features.high_frequency_features:HighFreqFeaturePlugin'
        ],
        'feature_eng.aligner': [
            'default=fe_plugins.aligner.default_aligner:AlignerPlugin'
        ],
        'feature_eng.post_processor': [
            'decomposition=fe_plugins.post_processor.decomposition_post_processor:DecompositionPostProcessor'
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
