import modules_pb2

modules = [
    {
        "module_name": "Translation",
        "input": modules_pb2.TEXT,
        "output": modules_pb2.TEXT,
        "computational_device": 1,
        "library_code_path": "Transformer.py"
    },


    {
        "module_name": "SpeechSynthesis",
        "input": modules_pb2.TEXT,
        "output": modules_pb2.AUDIO,
        "computational_device": 1,
        "library_code_path": "Tacotron.py"
    },


    {
        "module_name": "SpeechRecogniztion",
        "input": modules_pb2.AUDIO,
        "output": modules_pb2.TEXT,
        "computational_device": 1,
        "library_code_path": "Deepspeech2.py"
    },

    {
        "module_name": "TextEncoder",
        "input": modules_pb2.TEXT,
        "output": modules_pb2.TEXT,
        "computational_device": 0,
        "library_code_path": "text_encoder.py"
    },

    {
        "module_name": "TextDecoder",
        "input": modules_pb2.TEXT,
        "output": modules_pb2.TEXT,
        "computational_device": 0,
        "library_code_path": "text_decoder.py"
    },

    {
        "module_name": "Resample",
        "input": modules_pb2.AUDIO,
        "output": modules_pb2.AUDIO,
        "computational_device": 0,
        "library_code_path": "audio_resample.py"
    }

]
