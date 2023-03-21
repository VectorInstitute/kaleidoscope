import json


def read_txt(file_name):
    """
    Reads content from .txt file into a list.
    """
    with open(file_name, "r") as f:
        content = f.readlines()
    return content


def read_json(file_name):
    """
    Reads content from a json file as a dict.
    """
    with open(file_name, "r") as f:
        content = json.load(f)
    return content


def read_jsonl(file_name):
    """
    Reads objects from jsonl file to list.
    """
    content = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            content.append(json.loads(line.strip("\n|\r")))
    return content


def write_txt(content, file_name):
    """
    Writes content to a .txit file. Creates a newline seperated file 
    if content is a list.
    """
    with open(file_name, "w") as f:
        if isinstance(content, list):
            for item in content:
                f.write(f"{item}\n")
        else:
            f.write(content)


def write_jsonl(content, file_name):
    """
    Writes list of objects to jsonl file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        for item in content:
            json_item = json.dumps(item)
            f.write(f"{json_item}\n")
