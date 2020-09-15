import setuptools
from setuptools_rust import RustExtension, Binding


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="nlplease",
    version="0.0.1",
    author="Piotr Bajger",
    description="Python NLP package with Rust extensions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/piotrbajger/nplplease",
    python_requires='>=3.6',
    setup_requires=[
        "setuptools-rust",
        "wheel"
    ],
    install_requires=[
        "scipy",
    ],
    rust_extensions=[RustExtension("nlplease.nlplease", binding=Binding.PyO3)]
)
