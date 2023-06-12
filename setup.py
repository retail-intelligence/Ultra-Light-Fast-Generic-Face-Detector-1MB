from setuptools import find_packages
from setuptools import setup

setup(
    name="facedetector",
    version="1.0",
    description="Retail Intelligence face_detector",
    author="nautec",
    url="https://github.com/retail-intelligence/Ultra-Light-Fast-Generic-Face-Detector-1MB",
    packages=find_packages(),
    py_modules=["ulfg_face_detector"],
)
