from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pylingua",
    version="0.0.1",
    description="A user toolkit for analyzing and interfacing with Large Language Models (LLMs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='python nlp machine-learning deep-learning distributed-computing neural-networks tensor llm',
    requires_python = ">=3.7",
    url="https://github.com/VectorInstitute/lingua",
    author=["Vector AI Engineering"],
    author_email="ai_engineering@vectorinstitute.ai",
    license="MIT",
    packages=["lingua"],
    install_requires=[
        "certifi==2022.12.7",
        "charset-normalizer==3.0.1",
        "idna==3.4",
        "requests==2.28.2",
        "torch==1.13.1",
        "typing_extensions==4.4.0",
        "urllib3==1.26.14",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
