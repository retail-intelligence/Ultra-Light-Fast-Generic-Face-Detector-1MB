# from distutils.core import setup
from setuptools import find_packages
from setuptools import setup 

setup(name='facedetector',
      version='1.0',
      description='Retail Intelligence face_detector',
      author='nautec',
      install_requires=[
      'numpy>=1.24.2',
      'torch>=1.12.1',
      'torchvision>=0.13.1',
      'opencv_python>=4.7.0.72',
      'matplotlib>=3.5.3',
      'wget>=3.2'
      ],
      packages=find_packages(),
      url='https://github.com/retail-intelligence/Ultra-Light-Fast-Generic-Face-Detector-1MB',
    )