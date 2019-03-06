import modules_pb2

Translation = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.TEXT,
    "computational_device": 1,
    "library_code_path": "Transformer.py"
}


SpeechSynthesis = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.AUDIO,
    "computational_device": 1,
    "library_code_path": "Tacotron.py"
}


SpeechRecogniztion = {
    "input": modules_pb2.AUDIO,
    "output": modules_pb2.TEXT,
    "computational_device": 1,
    "library_code_path": "Deepspeech2.py"
}

TextEncoder = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.TEXT,
    "computational_device": 0,
    "library_code_path": "text_encoder.py"
}

TextDecoder = {
    "input": modules_pb2.TEXT,
    "output": modules_pb2.TEXT,
    "computational_device": 0,
    "library_code_path": "text_decoder.py"
}
