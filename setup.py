from setuptools import setup, find_packages
from glob import glob
__version__ = "0.0.1"

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='RFoT',
    version=__version__,
    author='Maksim E. Eren',
    author_email='maksim@umbc.edu',
    description='Random Forest of Tensors (RFoT)',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_dir={'RFoT': 'RFoT/'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    setup_requires=[
        'joblib', 'matplotlib', 'numpy',
        'pandas', 'scikit-learn', 'scipy', 'seaborn',
        'tqdm', 'sparse'
    ],
    url='https://github.com/MaksimEkin/RFoT',
    install_requires=['pyCP_APR @ https://github.com/lanl/pyCP_APR/tarball/main#egg=pyCP_APR-1.0.1'],
    dependency_links=['https://github.com/lanl/pyCP_APR/tarball/main#egg=pyCP_APR-1.0.1'],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8.5',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.8.5',
    install_requires=INSTALL_REQUIRES,
    license='License :: BSD3 License',
)
