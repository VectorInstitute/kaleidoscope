import abc


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def load(self, device, model_path):
        pass

    @abc.abstractmethod
    def module_names(self):
        pass

    @abc.abstractmethod
    def generate(self, request):
        pass

    @abc.abstractmethod
    def get_activations(self, request):
        pass

    @abc.abstractmethod
    def edit_activations(self, request):
        pass
