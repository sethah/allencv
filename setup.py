from setuptools import find_packages, setup

setup(
    name='allencv',
    packages=find_packages(),
    version='0.1.4-unreleased',
    description='',
    author='sethah',
    license='MIT',
    python_requires='>=3.6.1',
    install_requires=[
        'torch>=0.4.1',
        'torchvision==0.3.0',
        'allennlp',
        'numpy',
        'overrides',
        'ipython',
        'albumentations',
        'Pillow',
        'overrides',
        'numpy',
      ],
    tests_require=[
        'pytest'
    ]
)
