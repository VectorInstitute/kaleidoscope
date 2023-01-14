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

<!-- 
### Installing lingua using pip

```bash
python3 -m pip install lingua
```
-->

## SDK Developing
The development environment has been tested on ``python = 3.9``

### If your environment is ``python < 3.8``
>Download Conda: https://conda.io/projects/conda/en/stable/user-guide/install/download.html
>### Create a conda environment with ``python >= 3.8``
>```bash
>conda create -n venv python=3.9
>```
>
>### Activate conda environment
>```bash
>conda activate venv
>```

### If your environment is ``python >= 3.8``
>### Create virtual environment named ``env``
>```bash
>python3 -m venv env
>```
>
>### Activate virtual environment 
>```bash
>source env/bin/activate
>```
>
>### Update PIP
>```bash
>pip install --upgrade pip
>```


### Install Lingua
```bash
pip install git+https://github.com/VectorInstitute/lingua.git
```

### Retrieve personal auth key from http://llm.cluster.local:3001
A sample text generation submission from the web may be required to sign-in and generate an updated authentication key.
![Auth_demo_pic](https://user-images.githubusercontent.com/72175053/210878149-c142e36c-d61b-4b44-984f-3c0f8dec13de.png)

### Sample
```python
import lingua
remote_model= lingua.RModel('llm.cluster.local', 3001, 'OPT', 'YOUR AUTH KEY FROM WEB SERVICE')
remote_model.model_name # get current initialized model name
remote_model.module_names # get module names
remote_model.auth_key # get current auth key
remote_model.get_models() # get model instances

# sample text generation w/ input parameters
text_gen= remote_model.generate_text('What is the answer to life, the universe, and everything?', max_tokens=5, top_k=4, top_p=3, rep_penalty=1, temperature=0.5) 
dir(text_gen) # display methods associated with generated text object
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
Sivaloganathan, J., Coatsworth, M., Willes, J., Choi, M., & Shen, G. (2022). Lingua. http://VectorInstitute.github.io/lingua. computer software, Vector Institute for Artificial Intelligence. Retrieved from https://github.com/VectorInstitute/lingua.git. 
```
