from setuptools import setup, find_packages

# Read in the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyblmm",
    version="0.1.0",
    author="Tom Maullin",
    author_email="TomMaullin@gmail.com",
    description="The Big Linear Mixed Models Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tommaullin/blmm",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements, 
    entry_points={
        'console_scripts': [
            'blmm=blmm.blmm_cluster:_main',
        ],
    },
    python_requires='>=3.6',
)
