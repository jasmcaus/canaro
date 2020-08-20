import setuptools

DESCRIPTION = """A Python library including support for Deep Learning models built using the Keras framework."""

LONG_DESCRIPTION = DESCRIPTION + """ This repository is actively being maintained. If there are any issues, kindly open a thread in the 'Issues' pane on the official Github repository. """

setuptools.setup(
    name="caer-models",
    version="0.0.3",
    author="Jason Dsouza",
    author_email="jasmcaus@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/jasmcaus/caer-models",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=['tensorflow'],
    keywords=['computer vision', 'deep learning', 'tensorflow', 'keras', 'convolutional neural networks', 'opencv', 'matplotlib'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
)