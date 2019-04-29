import setuptools

setuptools.setup(
    name="cnc",
    version="1.0.0",
    author="Stefan Petrovic",
    author_email="stef.ptr@protonmail.com",
    description="CNC cutting path optimization package.",
    url="https://github.com/stefan-ptrvch/cnc_path_optimization.git",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'bokeh', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
