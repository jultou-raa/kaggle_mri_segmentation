from setuptools import setup, find_packages


setup(
    name='demo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'albumentations',
        'natsort',
        'numpy',
        'pandas',
        'opencv-python-headless',
        'plotly',
        'lightning',
        'scikit-learn',
        'streamlit',
        'torch',
        'torchmetrics',
        'torchvision',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    # Added deployment requirements
    setup_requires=[
        'pytest',
        'pytest-cov',
        'pytest-sugar',
        'pylint'
    ]
)