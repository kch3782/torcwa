from setuptools import setup, find_packages

setup(
    name='torcwa',
    version='0.1.4.2',
    description='GPU-accelerated Fourier modal method with automatic differentiation',
    author='Changhyun Kim',
    author_email='kch3782@snu.ac.kr',
    license='LGPL',
    url='https://github.com/kch3782/torcwa',
    install_requires=['torch>=1.10.1'],
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    packages=find_packages(),
    keywords='torcwa',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ]
)