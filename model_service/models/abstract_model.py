import abc

from pytriton.decorators import batch


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def load(self, device, model_path):
        pass

    # @abc.abstractmethod
    # def module_names(self):
    #     pass

    @abc.abstractmethod
    @batch
    def generate(self, **inputs):
        pass

    @abc.abstractmethod
    def get_activations(self, request):
        pass
