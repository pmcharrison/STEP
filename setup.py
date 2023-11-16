import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


PACKAGE_NAME = "step"


setuptools.setup(
    name="step",
    version="0.0.1",
    author="Pol van Rijn, Harin Lee, Nori Jacoby",
    author_email="pol.van-rijn@ae.mpg.de",
    description="Sequential Transmission Evaluation Pipeline (STEP)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polvanrijn/STEP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.0",
    include_package_data=True,
    package_data={"internat": ["VERSION"]},
    install_requires=[
        "psynet",
        "dallinger",
    ],
)

# python3.7 setup.py sdist bdist_wheel
