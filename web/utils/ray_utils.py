import requests
import numpy as np
import ast
from enum import Enum


class Task(Enum):
    """Task enum"""
    GENERATE = 0
    GET_ACTIVATIONS = 1
    EDIT_ACTIVATIONS = 2


def prepare_inputs(inputs):
    """
    Prepare inputs for Ray
    """
    inputs = inputs.copy()
    prompts = [[prompt] for prompt in inputs['prompts']]
    batch_size = len(prompts)
    inputs.pop('prompts')

    inputs_dict = dict()
    inputs_dict['prompts'] = prompts
    
    for name, value in inputs.items():
        try:
            inputs_dict[name] = [[value]]*batch_size
        except Exception as err:
            return (err, input)

    return inputs_dict


class RayClient():

    def __init__(self, host):
        self._host = host

    def infer(self, model_name, inputs, task=Task.GENERATE):

        inputs['task'] = task.value

        inputs_dict = prepare_inputs(inputs)
        if isinstance(inputs_dict, tuple):
            return inputs_dict
        
        try:
            response = requests.get(self._host, params=inputs_dict).json()
        except Exception as err:
            return err
        
        sequences = np.char.decode(response["sequences"].astype("bytes"), "utf-8").tolist()
        tokens = []
        logprobs = []

        try:
            tokens = np.char.decode(response["tokens"].astype("bytes"), "utf-8").tolist()
        except Exception as err:
            pass

        try:
            logprobs = np.char.decode(response["logprobs"].astype("bytes"), "utf-8").tolist()
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
        
        if task in [Task.GET_ACTIVATIONS, Task.EDIT_ACTIVATIONS]:
            activations = np.char.decode(response["activations"].astype("bytes"), "utf-8").tolist()
            for idx in range(len(activations)):
                activations[idx] = ast.literal_eval(activations[idx])
            result.update({"activations": activations})

        return result

    def is_model_ready(self, model_name):
        # TODO - Implement model ready check using Ray, set to True for time being
        return True