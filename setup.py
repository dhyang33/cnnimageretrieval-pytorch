import setuptools


setuptools.setup(
    name="cirtorch",
    version=0.1,
    install_requires=[
        "numpy",
        "scipy",
        "torchvision",
    ],
    packages=setuptools.find_packages(),
)
