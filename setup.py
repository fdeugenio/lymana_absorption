import subprocess
from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as file:
        return(file.read())

setup(name='lymana_absorption',
      version='0.0',
      description='Spectral fitting of Lyman alpha absorption',
      long_description=readme(),
      classifiers=[
        'Development Status :: 0 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://github.com/joriswitstok/lymana_absorption',
      author='Joris Witstok',
      author_email='tbd',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'matplotlib',
        'numpy>=1.23.2',
        'scipy>=1.9.0',
        'astropy>=5.1',
        'spectres>=2.2.0',
        'corner',
        'seaborn',
        'pymultinest',
        'tqdm',
      ],
      python_requires='>=3.10',
      #entry_points={
      #  'console_scripts': [
      #      'nirspecxf      = nirspecxf.samisimsig_script:main',
      #      ],
      #},
     )
