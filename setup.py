from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sphericaldistortion',
    version='0.1',
    author='Dominique Piché-Meunier',
    author_email='dominique.piche-meunier.1@ulaval.ca',
    description='Lens distortion using the spherical distortion model',
    packages=['sphericaldistortion'],
    install_requires=requirements
)