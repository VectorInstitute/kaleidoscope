import abc

class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def module_names(self):
        pass

    @abc.abstractmethod
    def generate_text(self, request):
        pass