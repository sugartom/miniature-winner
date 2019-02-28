import modules_pb2

Text2Text = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.TEXT,
    "computational_device": 1,
    "library_code_path": "text2text.py"
}


Text2Speech = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.AUDIO,
    "computational_device": 1,
    "library_code_path": "text2speech.py"
}


Speech2Text = {
    "input": modules_pb2.AUDIO,
    "output": modules_pb2.TEXT,
    "computational_device": 1,
    "library_code_path": "text2speech.py"
}

TextEncoder = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.TEXT,
    "computational_device": 0,
}

TextDecoder = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.TEXT,
    "computational_device": 0,
}
