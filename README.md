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

* `model_instances` - Shows a list of all active LLMs instantiated by the model service
* `load_model` - Loads an LLM via the model service
* `generate` - Returns an LLM text generation based on prompt input
* `module_names` - Returns all modules names in the LLM neural network
* `get_activations` - Retrieves all activations for a set of modules

``lingua`` is composed of the following components:

* Python SDK - A frontend Python library for interacting with LLMs, available in a separate repository at https://github.com/VectorInstitute/lingua-sdk
* Model Service - A backend utility that loads models into GPU memory and exposes an interface to recieve requests
* Gateway Service - A controller service that interfaces between the frontend user tools and model service


## Getting Started
Instructions for setting up gateway service.

### Install
```bash
git clone https://github.com/VectorInstitute/lingua.git
```

### Start gateway container
```bash
cp lingua/web/.env-example lingua/web/.env
sudo docker compose -f lingua/web/docker-compose.yaml up
```

### Install Lingua SDK Toolkit
The Lingua SDK toolkit is a Python module that provides a programmatic
interface for interfacing with the services found here. You can download and
install the SDK from its own repository:
https://github.com/VectorInstitute/lingua-sdk


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
