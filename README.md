![Kaleidoscope](https://user-images.githubusercontent.com/72175053/229659242-27090171-2005-4d1c-9de3-0a5b02ef8ceb.png)
-----------------
# Kaleidoscope
![PyPI](https://img.shields.io/pypi/v/kscope)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kscope)
![GitHub](https://img.shields.io/github/license/VectorInstitute/kaleidoscope)
![DOI](https://img.shields.io/badge/DOI-in--progress-blue)
[![Documentation](https://img.shields.io/badge/api-reference-lightgrey.svg)](https://kaleidoscope-sdk.readthedocs.io/en/latest/)

A fast inference service for serving and querying Large Language Models (LLMs)

## Overview

``kaleidoscope`` provides a few high-level APIs, namely:

* `model_instances` - Shows a list of all active LLMs instantiated by the model service
* `load_model` - Loads an LLM via the model service
* `generate` - Returns an LLM text generation based on prompt input

``kaleidoscope`` is composed of the following components:

* Python SDK - A frontend Python library for interacting with LLMs, available in a separate repository at https://github.com/VectorInstitute/kaleidoscope-sdk
* Model Service - A backend utility that loads models into GPU memory and exposes an interface to recieve requests
* Gateway Service - A controller service that interfaces between the frontend user tools and model service


## Getting Started
Instructions for setting up gateway service.

### Install
```bash
git clone https://github.com/VectorInstitute/kaleidoscope.git
```

### Start gateway container
```bash
cp kaleidoscope/web/.env-example kaleidoscope/web/.env
sudo docker compose -f kaleidoscope/web/docker-compose.yaml up
```

### Install Kaleidoscope SDK Toolkit
The Kaleidoscope SDK toolkit is a Python module that provides a programmatic
interface for interfacing with the services found here. You can download and
install the SDK from its own repository:
https://github.com/VectorInstitute/kaleidoscope-sdk

## Contributing
Contributing to kaleidoscope is welcomed. See [Contributing](CONTRIBUTING.md) for
guidelines.

## License
[MIT](LICENSE)

## Citation
Reference to cite when you use Kaleidoscope in a project or a research paper:
```
Sivaloganathan, J., Coatsworth, M., Willes, J., Choi, M., & Shen, G. (2022). Kaleidoscope. http://VectorInstitute.github.io/kaleidoscope. computer software, Vector Institute for Artificial Intelligence. Retrieved from https://github.com/VectorInstitute/kaleidoscope.git.
```
