# Lingua
![PyPI](https://img.shields.io/pypi/v/pylingua)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pylingua)
![GitHub](https://img.shields.io/github/license/VectorInstitute/lingua)
![DOI](https://img.shields.io/badge/DOI-in--progress-blue)
[![Documentation](https://img.shields.io/badge/api-reference-lightgrey.svg)](https://lingua-sdk.readthedocs.io/en/latest/)

A user toolkit for analyzing and interfacing with Large Language Models (LLMs)

## Overview

``lingua`` provides a few high-level APIs, namely:

* `generate_text` - Returns an LLM text generation based on prompt input
* `module_names` - Returns all modules in the LLM neural network
* `instances` - Returns all active LLMs instantiated by the model service

``lingua`` is composed of the following components:

* Python SDK - A command line tool wrapping the gateway service API endpoints
* Web service - A front-end web application tool sending requests to the gateway service
* Model service - A backend utility that loads models into GPU memory and exposes an interface to recieve requests


## Getting Started
Instructions for setting up gateway service.

### Install
```bash
git clone https://github.com/VectorInstitute/lingua.git
```

### Start app
```bash
sudo docker compose -f lingua/docker-compose.yaml up
```

### Install Lingua SDK Toolkit
The Lingua SDK toolkit is a Python module that provides a programmatic
interface for interfacing with the services found here. You can download and
install the SDK from its own repository:
https://github.com/VectorInstitute/lingua-sdk

## Documentation
Full documentation and API reference are available at: http://lingua-sdk.readthedocs.io.

## Contributing
Contributing to lingua is welcomed. See [Contributing](CONTRIBUTING) for
guidelines.

## License
[MIT](LICENSE)

## Citation
Reference to cite when you use Lingua in a project or a research paper:
```
Sivaloganathan, J., Coatsworth, M., Willes, J., Choi, M., & Shen, G. (2022). Lingua. http://VectorInstitute.github.io/lingua. computer software, Vector Institute for Artificial Intelligence. Retrieved from https://github.com/VectorInstitute/lingua.git.
```
