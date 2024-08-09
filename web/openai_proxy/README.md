# OpenAI-compatible Proxy

## Setup

On a machine with SLURM access, run the following:

- Set env var `TELEMETRY_CALLBACK_URL` and point to http://hostname:port/worker_callback
- Launch proxy Flask application.
- Point OpenAI client SDK to use http://hostname:port/v1 as API base.

```bash
PROXY_PORT=25567
export PROXY_BASE_URL=http://`hostname`:${PROXY_PORT}
export SSH_USERNAME=llm
export SSH_HOST=vremote
export SSH_PORT=22
export WORKER_SLURM_QOS=llm
export MIN_UPDATE_INTERVAL_SECONDS=10
export MAX_NUM_HISTORIC_RECORDS=360
echo Set OpenAI API Base URL to ${PROXY_BASE_URL}/v1
python3 -m web.openai_proxy.routes
```

## Background

vLLM provides support for OpenAI ChatCompletion API. Features and limitations:

- Supports batch processing
- Supports streaming output
- Requests are not stateful- suitable for load balancing across instances
- Does not support authentication

API Reference:

- OpenAI API Specification: [API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- vLLM Implementation of the OpenAI ChatCompletion API: [link to source](https://github.com/vllm-project/vllm/blob/3eeb148f467e3619e8890b1a5ebe86a173f91bc9/vllm/entrypoints/openai/serving_chat.py#L68)

## Motivation

Streamline researcher and developer experience by abstracting away details about model launching and the SLURM cluster. Implement seamless support for the OpenAI API Client without any additional package install on the client side.

## Proposed workflow

User obtain "OpenAI API Key" and configurations from KScope via browser.

- User authenticate to KScope web interface via LDAP and http-simple-login
- KScope provide an API key valid for many days
- KScope provide a list of models available.
- Additionally, KScope web page provides instructions and example code for setting up the OpenAI API Client
- Since it might take an indefinite amount of time before the server starts, users might find it helpful to run non-time-sensitive API requests (e.g., batch processing) in an infinite while loop.

User configure official OpenAI client (`pip install openai`) for KScope.

- Save API Key from KScope as an API key environment variable
- Set OpenAI API Base URL to URL of KScope gateway.
- Set model name appropriately.

KScope Gateway authenticates and reverse-proxy requests to vLLM OpenAI-compatible Backend, launching SLURM jobs as needed.

- KScope validates API Key and user permissions on each request
- For each model name, KScope maintains a list of currently active vLLM jobs
  - Include a background task to periodically check on the state of each instance
  - Possible enhancement: load monitoring, automated scaling (both scaling up and scaling down)

If model is not running at the moment, KScope signals the OpenAI Client immediately.

- Returns an [OpenAI-compatible error](https://platform.openai.com/docs/guides/error-codes/python-library-error-types)
- Example: 503 - The engine is currently overloaded, please try again later, where the recommendation is to "Please retry your requests after a brief wait."

If the vLLM job is ready, KScope forwards the request to the vLLM and returns the response to the client. Possible approach:

- Send HTTP request to upstream Python `requests` library
  - Enable "streaming" mode to obtain a generator object and support token streaming (see [Discussion](https://stackoverflow.com/a/57498146) and API [reference](https://requests.readthedocs.io/en/latest/user/advanced/#body-content-workflow) for requests)
- Redirect response "stream" (a Python generator) to the user
  - Similarly, enable "streaming" by setting `direct_passthrough` to True in the Flask Response (see [Discussion](https://stackoverflow.com/a/5166423))

### Minimum Viable Product

Limitations

- API Key is not validated
- No automatic scaling (scale-up)- supports up to one vLLM instance per model
- vLLM instances are not stopped (scaled-down) automatically

Proposed Scheme for Minimum Viable Product

- Implement Flask route for `/v1/chat/completion`
- Implement unit tests for both streaming and non-streaming examples.
