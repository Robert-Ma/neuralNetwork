from setuptools import setup, find_packages


setup(
    name='neuralnetwork',
    version='1.0.0',
    description='Neural Network',
    author='Wei Ma',
    author_email='Ma-Wei@outlook.com',
    packages=find_packages(),
    include_package_data=False,
    keywords=['neural', 'network'],
    license='MIT License',
    install_requires=[
        'numpy',
        'scipy',
    ],
)
