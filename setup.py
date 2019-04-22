import setuptools


setuptools.setup(
    name="sheet-id-cirtorch",
    version=1.0,
    install_requires=[
        "numpy",
        "scipy",
        "pytorch",
        "torchvision",
    ],
    packages=setuptools.find_packages(),
)
