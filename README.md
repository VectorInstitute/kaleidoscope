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

### Install
```bash
git clone https://github.com/VectorInstitute/lingua.git
```

### Start app
```bash
sudo docker compose -f lingua/docker-compose.yaml up
```

<!-- 
### Installing lingua using pip

```bash
python3 -m pip install pycyclops
```
-->

## Developing
The development environment has been tested on ``python = 3.9.10``.

### SDK sample
```python
import lingua
remote_model= lingua.RModel('llm.cluster.local', 3001, 'OPT', 'YOUR AUTH KEY FROM THE WEB SERVICE')
remote_model.model_name # get currently initialized model name
remote_model.module_names # get module names
remote_model.auth_key # get current auth key
remote_model.get_models() # get model instances

# sample text generation w/ input parameters
text_gen= remote_model.generate_text('hello world', max_tokens=5, top_k=4, top_p=3, rep_penalty=1, temperature=0.5) 
dir(text_gen) # for list of methods associated with generated text object
text_gen.text # display only text
text_gen.logprobs # display logprobs
text_gen.tokens # display tokens
```

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
```
