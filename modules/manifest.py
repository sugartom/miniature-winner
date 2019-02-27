
Text2Text = {
    "input": TextProto,
    "output": TextProto,
    "computational_device": 1,
    "library_code_path": "text2text.py"
}


Text2Speech = {
    "input": TextProto,
    "output": AudioProto,
    "computational_device": 1,
    "library_code_path": "text2speech.py"
}


Speech2Text = {
    "input": AudioProto,
    "output": TextProto,
    "computational_device": 1,
    "library_code_path": "text2speech.py"
}

TextEncoder = {
    "input": TextProto,
    "output": TextProto,
    "computational_device": 0,
}

TextDecoder = {
    "input": TextProto,
    "output": TextProto,
    "computational_device": 0,
}
