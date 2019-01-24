from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='pytorch-hrvvi-ext',
      version='1.0',
      author='HrvvI',
      author_email='sbl1996@126.com',
      packages=['hutil'],
      install_requires=requirements,
      )
