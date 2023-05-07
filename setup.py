from setuptools import setup, find_packages

setup(
    name='my-package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'lightning>=2.0.2',
        'nibabel>=3.0.2',
        'numpy>=1.22.4',
        'opencv_contrib_python>=4.7.0.72',
        'opencv_python>=4.7.0.72',
        'opencv_python_headless>=4.7.0.72',
        'pandas>=1.5.3',
        'Requests>=2.30.0',
        'scikit_learn>=1.2.2',
        'torch>=2.0.0+cu118',
        'torchmetrics>=0.11.4',
        'torchvision>=0.15.1+cu118',
        'HD_BET @ git+https://github.com/MIC-DKFZ/HD-BET.git',
        'batchgenerators @ git+https://github.com/MIC-DKFZ/batchgenerators.git'
    ]
)