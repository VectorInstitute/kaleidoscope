from torch.utils.data import IterableDataset

from .io_utils import stream_jsonl


class PromptsDataset(IterableDataset):

    def __init__(self, file_path):
        super(PromptsDataset).__init__()
        self.file_path = file_path

    def __iter__(self):
        return iter(stream_jsonl(self.file_path))
