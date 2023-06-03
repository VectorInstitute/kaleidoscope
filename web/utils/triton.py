import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

DTYPE_MAP = {
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "bool": bool,
        "object": object
    }

def prepare_tensor(name, input):
    tensor = httpclient.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    tensor.set_data_from_numpy(input)
    return tensor

def prepare_inputs(input_data, cfg):
    input_data = np.array(input_data).astype(object)
    inputs = [prepare_tensor(cfg["input_alias"], input_data)]
    params = cfg["parameters"]
    for _, p_dict in params.items():
        if isinstance(p_dict["value"], list):
            p_input = np.array([p_dict["value"]] * input_data.shape[0], dtype=DTYPE_MAP[p_dict["type"]])
        elif isinstance(p_dict["value"], str):
            raise NotImplementedError
        else:
            p_input = (p_dict["value"] * np.ones([input_data.shape[0], 1])).astype(DTYPE_MAP[p_dict["type"]])
        inputs.append(
            prepare_tensor(p_dict["alias"], p_input)
        )
    return inputs

def update_param_cfg(param_cfg, input_gen_cfg):
    new_param_cfg = param_cfg.copy()
    for p_name, p_dict in new_param_cfg["parameters"].items():
        p_dict.update({
            "value": input_gen_cfg.get(p_name, p_dict["default"])
            })
    return new_param_cfg


class TritonClient():

    def __init__(self, host):
        self._client = httpclient.InferenceServerClient(host, concurrency=1, verbose=True)

    def infer(self, model_name, inputs, config, task="generation"):

        model_bind_name = f'{model_name}_{task}'
        # if config[model_name][task] is None:
        task_config = self._client.get_model_config(model_bind_name)

        inputs = prepare_inputs(inputs, task_config)
        return self._client.infer(model_name, inputs)

    def is_model_ready(self, model_name, task="generation"):
        model_bind_name = f'{model_name}_{task}'
        print(model_bind_name)
        is_model_ready = self._client.is_model_ready(model_bind_name)
        return is_model_ready
        

    # Only for GPT-J
    # MODEl_GPTJ_FASTERTRANSFORMER = "ensemble" 
    
    # client = httpclient.InferenceServerClient(host,
    #                                           concurrency=1,
    #                                           verbose=False)
    
    # inputs = [[elm] for elm in prompts]
    # param_config = json.load(open("../../models/GPT-J/config.json", "r")) # TODO - Query model service to fetch param config
    # param_config = update_param_cfg(param_config, generation_config)
    # from pprint import pprint
    # pprint(param_config)
    # inputs = prepare_inputs(inputs, param_config)

    # result = client.infer(MODEl_GPTJ_FASTERTRANSFORMER, inputs)
    # output0 = result.as_numpy("OUTPUT_0")
    # print(output0.shape)
    # print(output0)
