import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlprep", # Replace with your own username
    version="0.0.3",
    author="Anoop Sharma",
    author_email="asanoop24@gmail.com",
    description="A utility package for data preprocessing before the modelling exercise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asanoop24/ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas",
        "scikit-learn"
    ],
    python_requires='>=3.6',
)