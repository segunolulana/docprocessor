from setuptools import setup, find_packages

setup(
    name='smy',
    packages=find_packages(exclude=['test.seg']),
    include_package_data=True,
    install_requires=[
        'binaryornot==0.4.4',
        'gensim==3.8.3',
        'icecream==2.0.0',
        'line-profiler==3.0.2',
        'pandas==1.0.3',
        'summa==1.2.0'
    ],
)
