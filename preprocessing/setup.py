#!/usr/bin/python
from setuptools import find_packages
from setuptools import setup

setup(
    name='twitter-project',
    version='0.2',
    author='nikeNano',
    author_email='',
    install_requires=['tensorflow==1.11.0',
                      'tensorflow-transform==0.11.0'],
    packages=find_packages(exclude=['data']),
    description='GCP ml engine/dataflow/pubsub/bigquery demp',
    url=''
)