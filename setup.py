
from setuptools import setup, find_packages

from imenhance import __version__

classifiers = '''Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name='image_enhancement_tools',
    packages=find_packages(include=('imenhance',)),
    include_package_data=True,
    version=__version__,
    description='A collection of tools for real-time image enhancement',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=classifiers.splitlines(),
    install_requires=requirements,
    python_requires='>=3.6',
)