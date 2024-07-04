from flask import current_app
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
import typing
import ast
from enum import Enum
from config import Config


class Task(Enum):
    """Task enum"""
    GENERATE = 0
    GET_ACTIVATIONS = 1
    EDIT_ACTIVATIONS = 2

def _param(dtype, value, batch_size):
    if bool(value):
        return np.ones((batch_size, 1), dtype=dtype) * value
    else:
        return np.zeros((batch_size, 1), dtype=dtype) + value

def _str_list2numpy(str_list: typing.List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)
    return np.char.encode(str_ndarray, "utf-8")

def prepare_prompts_tensor(prompts):
    name, value =  "prompts", prompts

    triton_dtype = "BYTES"
    input = _str_list2numpy(value)
    # np.array(value, dtype=bytes)

    tensor = httpclient.InferInput(name, input.shape, triton_dtype)
    tensor.set_data_from_numpy(input)
    return tensor

def prepare_param_tensor(input, inputs_config, batch_size):
    name, value = input
    input_config = [input_config for input_config in inputs_config if input_config['name'] == name][0]
    triton_dtype = input_config['data_type'].split('_')[1]
    if triton_dtype == "STRING":
        triton_dtype = "BYTES"
        value = _str_list2numpy(value)
    input = _param(triton_to_np_dtype(triton_dtype), value, batch_size)

    tensor = httpclient.InferInput(name, input.shape, triton_dtype)
    tensor.set_data_from_numpy(input)
    return tensor

def prepare_inputs(inputs, inputs_config):
    """
    Prepare inputs for Triton
    """
    inputs = inputs.copy()
    prompts = [[prompt] for prompt in inputs['prompts']]
    batch_size = len(prompts)
    inputs.pop('prompts')

    inputs_wrapped = [prepare_prompts_tensor(prompts)]
    
    for input in inputs.items():
        try:
            inputs_wrapped.append(prepare_param_tensor(input, inputs_config, batch_size))
        except Exception as err:
            return (err, input)

    return inputs_wrapped


class TritonClient():

    def __init__(self, host):
        self._client = httpclient.InferenceServerClient(host, concurrency=1, verbose=False, network_timeout=Config.TRITON_INFERENCE_TIMEOUT)

    def infer(self, model_name, inputs, task=Task.GENERATE):
        task_config = self._client.get_model_config(model_name)
        
        inputs['task'] = task.value
        inputs_wrapped = prepare_inputs(inputs, task_config['input'])
        if isinstance(inputs_wrapped, tuple):
            return inputs_wrapped

        try:
            response = self._client.infer(model_name, inputs_wrapped)
        except Exception as err:
            return err
        sequences = np.char.decode(response.as_numpy("sequences").astype("bytes"), "utf-8").tolist()
        tokens = []
        logprobs = []

        try:
            tokens = np.char.decode(response.as_numpy("tokens").astype("bytes"), "utf-8").tolist()
        except Exception as err:
            pass

        try:
            logprobs = np.char.decode(response.as_numpy("logprobs").astype("bytes"), "utf-8").tolist()
        except Exception as err:
            pass
        
        # Logprobs need special treatment because they are encoded as bytes
        # Regular np float arrays don't work, each element has a different number of items
        for i in range(len(logprobs)):
            # Logprobs sometime get returned as a single string of floats, instead of a list of byte objects
            # If this is the case, reformat them
            if isinstance(logprobs[i], str):
                logprobs[i] = logprobs[i][1:-1].split(', ')
            logprobs[i] = [float(prob) if prob!="None" else None for prob in logprobs[i]]

        result = {
            "sequences": sequences,
            "tokens": tokens,
            "logprobs": logprobs
        }
        
        return result

    def is_model_ready(self, model_name):
        return self._client.is_model_ready(model_name)
