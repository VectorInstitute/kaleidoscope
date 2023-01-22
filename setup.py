from setuptools import setup

setup(
    name="lingua",
    version="0.0.1",
    description="A user toolkit for analyzing and interfacing with Large Language Models (LLMs)",
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
        "Intended Audience :: Science/Research/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: UBUNTU :: Linux",
        "Programming Language :: Python :: 3.8" "Programming Language :: Python :: 3.9",
    ],
)
