# Lingua
A user toolkit for analyzing and interfacing with Large Language Models (LLMs)

<!--
[![PyPI]()]()
[![code checks]()]()
[![integration tests]()]()
[![docs]()]()
[![codecov]()
[![license]()]()
-->

## Overview

``lingua`` provides a few high-level APIs namely:

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

## [Documentation](https://vectorinstitute.github.io/lingua/)
More information can be found on the Lingua documentation site.

## Contributing
Contributing to lingua is welcomed. See [Contributing](https://github.com/VectorInstitute/lingua/blob/main/doc/CONTRIBUTING.md) for
guidelines.

## License
[MIT](LICENSE)

## Citation
Reference to cite when you use Lingua in a project or a research paper:
```
Sivaloganathan, J., Coatsworth, M., Willes, J., Choi, M., & Shen, G. (2022). Lingua. http://VectorInstitute.github.io/lingua. computer software, Vector Institute for Artificial Intelligence. Retrieved from https://github.com/VectorInstitute/lingua.git.
```
