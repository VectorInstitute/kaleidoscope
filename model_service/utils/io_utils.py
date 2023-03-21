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
