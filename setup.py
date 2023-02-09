from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme:
    long_description = readme.read()


setup(
    name='flax-trainer',
    version= "0.0.0.4",
    description='Korean AI Project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hkjeon13/flax-trainer",
    author="Eddie",
    author_email="hkjeo13@gmail.com",
    zip_safe=False,
    license="MIT",

    py_modules=["flax_trainer"],

    python_requires=">=3",

    packages=find_packages("."),
    package_data= {"": ["*.json"]},
    include_package_data=True,
    install_requires=[
            "transformers",
            "datasets",
            "evaluate",
            "optax"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)