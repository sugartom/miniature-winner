import modules_pb2

import runpy
manifest = runpy.run_path("./manifest.py")

modules = manifest['modules']
print(modules)
